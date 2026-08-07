"""Regression tests for `data/generate_qot_dataset.py`'s pure graph logic.

Covers Bug C (segment splitter emitting overlapping/gapped segments because
`chosen` regen nodes were sorted by node *value* instead of their *position*
in `path`). No GNPy / topology loading needed — `split_path_into_segments`
only touches plain Python lists.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from diffopt.topology import Edge
from generate_qot_dataset import (
    build_edge_lookup,
    compute_duplication_stats,
    format_duplication_report,
    get_k_shortest_paths,
    path_to_edges,
    split_path_into_segments,
)


def _tiles_exactly(path: list, segments: list) -> bool:
    """True iff `segments` tile `path` with no gaps and no duplicated interior nodes."""
    if segments[0][0] != path[0] or segments[-1][-1] != path[-1]:
        return False
    for i in range(len(segments) - 1):
        if segments[i][-1] != segments[i + 1][0]:
            return False
        if len(segments[i]) < 2:
            return False
    # Reconstruct path by concatenating segments, dropping each segment's
    # leading node except the very first (it's the previous segment's tail).
    reconstructed = segments[0][:]
    for seg in segments[1:]:
        reconstructed.extend(seg[1:])
    return reconstructed == path


class _FixedSampleRng(random.Random):
    """A Random whose .sample() always returns a pre-set value, ignoring input.

    Lets a test force a specific `chosen` set (bypassing the randint/sample
    calls inside split_path_into_segments) while still exercising the real
    sort-by-position fix.
    """

    def __init__(self, fixed_sample, fixed_randint=None):
        super().__init__()
        self._fixed_sample = fixed_sample
        self._fixed_randint = fixed_randint

    def sample(self, population, k):
        return list(self._fixed_sample)

    def randint(self, a, b):
        if self._fixed_randint is not None:
            return self._fixed_randint
        return super().randint(a, b)


def test_bug_c_exact_failure_trace():
    """Reproduce the exact trace from task-6-brief.md Bug C.

    path=[3, 9, 5, 7], candidates chosen={5, 9}. Under the old
    sorted-by-node-value bug, sorted({5, 9}) == [5, 9] (already numeric
    order) which is NOT path order (9 comes before 5 in path). The fix sorts
    by path.index(), giving [9, 5] instead.
    """
    path = [3, 9, 5, 7]
    rng = _FixedSampleRng(fixed_sample={5, 9}, fixed_randint=2)
    segments = split_path_into_segments(path, regen_nodes=[5, 9], rng=rng)

    assert _tiles_exactly(path, segments)
    # breakpoints should be [3, 9, 5, 7] (chosen sorted by path position: 9 then 5)
    assert segments == [[3, 9], [9, 5], [5, 7]]


def test_bug_c_single_regen_node():
    path = [0, 1, 2, 3, 4]
    rng = _FixedSampleRng(fixed_sample={2}, fixed_randint=1)
    segments = split_path_into_segments(path, regen_nodes=[2], rng=rng)

    assert _tiles_exactly(path, segments)
    assert segments == [[0, 1, 2], [2, 3, 4]]


def test_bug_c_two_regen_nodes_in_path_order():
    """Chosen nodes {1, 9} are split at their path-position order: 1 (index 2) then 9 (index 3).

    This happens to coincide with numeric order for this path, so it doesn't
    exercise the reverse-order scenario Bug C was about — that regression
    coverage lives in `test_bug_c_exact_failure_trace` and
    `test_real_rng_many_trials_always_tiles` above/below. Kept here as a
    valid (if redundant) tiling check.
    """
    path = [0, 8, 1, 9, 2]
    rng = _FixedSampleRng(fixed_sample={1, 9}, fixed_randint=2)
    segments = split_path_into_segments(path, regen_nodes=[1, 9], rng=rng)

    assert _tiles_exactly(path, segments)
    assert segments == [[0, 8, 1], [1, 9], [9, 2]]


def test_n_regen_zero_returns_whole_path():
    path = [0, 1, 2, 3]
    rng = _FixedSampleRng(fixed_sample=set(), fixed_randint=0)
    segments = split_path_into_segments(path, regen_nodes=[1, 2], rng=rng)
    assert segments == [path]


def test_real_rng_many_trials_always_tiles():
    """Fuzz with the real (unmocked) rng across many seeds/paths."""
    path = [10, 3, 7, 1, 9, 4, 2]
    candidates = [3, 7, 1, 9, 4]
    for seed in range(200):
        rng = random.Random(seed)
        segments = split_path_into_segments(path, regen_nodes=candidates, rng=rng)
        assert _tiles_exactly(path, segments), (seed, segments)


def test_get_k_shortest_paths_returns_top_k_in_order():
    """Verify get_k_shortest_paths returns k shortest paths in weight order.

    Build a small graph with multiple simple paths between src and dst,
    enumerate all paths by hand, and verify the first k returned by
    get_k_shortest_paths match the k paths with smallest total weights.

    Graph: 0 --1-- 1 --1-- 3
                   |       |
                   +--1-2--+

    Paths from 0 to 3 (by total weight):
      1. 0-1-3: weight 1+1 = 2
      2. 0-1-2-3: weight 1+1+1 = 3
    """
    G = nx.Graph()
    # Add nodes
    for node in [0, 1, 2, 3]:
        G.add_node(node)

    # Add edges with weights
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 3, weight=1.0)
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 3, weight=1.0)

    # Get k=2 shortest paths
    paths = get_k_shortest_paths(G, src=0, dst=3, k=2)

    assert len(paths) == 2, f"Expected 2 paths, got {len(paths)}"
    assert paths[0] == [0, 1, 3], f"Expected first path [0, 1, 3], got {paths[0]}"
    assert paths[1] == [0, 1, 2, 3], f"Expected second path [0, 1, 2, 3], got {paths[1]}"

    # Verify the weights are in order
    weight_0_1_3 = sum(G[paths[0][i]][paths[0][i+1]]["weight"] for i in range(len(paths[0])-1))
    weight_0_1_2_3 = sum(G[paths[1][i]][paths[1][i+1]]["weight"] for i in range(len(paths[1])-1))
    assert weight_0_1_3 <= weight_0_1_2_3, f"Paths not in order: {weight_0_1_3} vs {weight_0_1_2_3}"


def test_get_k_shortest_paths_no_path():
    """Verify get_k_shortest_paths returns empty list when no path exists."""
    G = nx.Graph()
    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_edge(0, 1, weight=1.0)

    # No path from 0 to 2 (2 is disconnected)
    paths = get_k_shortest_paths(G, src=0, dst=2, k=5)
    assert paths == [], f"Expected empty list, got {paths}"


def test_get_k_shortest_paths_k_exceeds_available():
    """Verify get_k_shortest_paths returns fewer than k paths if fewer exist."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=1.0)

    # Only 1 simple path from 0 to 2: [0, 1, 2]
    paths = get_k_shortest_paths(G, src=0, dst=2, k=5)
    assert len(paths) == 1, f"Expected 1 path, got {len(paths)}"
    assert paths[0] == [0, 1, 2], f"Expected [0, 1, 2], got {paths[0]}"


def test_get_k_shortest_paths_truncates_more_than_k_paths():
    """Verify get_k_shortest_paths truncates when >k simple paths exist.

    This test catches off-by-one mutations like islice(gen, k+1) or
    islice(gen, k-1). The graph has 6+ distinct simple paths from src to dst;
    we request k=2 and verify:
      (a) exactly 2 results are returned (not 1, not 3)
      (b) they are the 2 cheapest by hand-computed weight
      (c) a heavier path known to exist is NOT in results (proves truncation)

    Graph structure (src=0, dst=4):
      0-1: weight 1
      0-2: weight 1
      0-3: weight 2
      1-4: weight 1
      2-4: weight 1
      3-4: weight 1
      1-2: weight 0.1  (creates detours)
      1-3: weight 1

    All simple paths from 0 to 4 (by weight):
      1. 0-1-4: weight 1+1 = 2         ← top 1
      2. 0-2-4: weight 1+1 = 2         ← top 2
      3. 0-1-2-4: weight 1+0.1+1 = 2.1 ← NOT returned when k=2
      4. 0-2-1-4: weight 1+0.1+1 = 2.1
      5. 0-1-3-4: weight 1+1+1 = 3
      6. 0-3-4: weight 2+1 = 3
      ... and possibly more via other orderings
    """
    G = nx.Graph()
    # Add all nodes
    for node in [0, 1, 2, 3, 4]:
        G.add_node(node)

    # Add edges to create multiple paths
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(0, 2, weight=1.0)
    G.add_edge(0, 3, weight=2.0)
    G.add_edge(1, 4, weight=1.0)
    G.add_edge(2, 4, weight=1.0)
    G.add_edge(3, 4, weight=1.0)
    G.add_edge(1, 2, weight=0.1)  # Creates detours
    G.add_edge(1, 3, weight=1.0)  # Creates more paths

    # Request only k=2 shortest paths
    paths = get_k_shortest_paths(G, src=0, dst=4, k=2)

    # (a) Exactly k results returned
    assert len(paths) == 2, f"Expected exactly 2 paths, got {len(paths)}: {paths}"

    # Compute weights of returned paths
    def path_weight(path):
        return sum(G[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1))

    weights = [path_weight(p) for p in paths]

    # (b) Both returned paths are the top 2 by weight
    # The two shortest are 0-1-4 and 0-2-4, both weight 2.0
    for w in weights:
        assert w == 2.0, f"Returned path has weight {w}, expected 2.0 (top 2 shortest)"

    # (c) A heavier path known to exist (e.g., 0-1-2-4 with weight 2.1) is NOT returned
    # If islice(gen, k-1) or islice(gen, k+1) mutation happened, this would fail
    detour_path = [0, 1, 2, 4]
    assert detour_path not in paths, (
        f"Heavier path {detour_path} (weight 2.1) should not be in top-2, "
        f"but it is. This suggests truncation did not occur correctly."
    )


# ---------------------------------------------------------------------------
# build_edge_lookup / path_to_edges
# ---------------------------------------------------------------------------

class _FakeTopology:
    """Minimal stand-in exposing only what build_edge_lookup reads —
    avoids constructing a real Topology (GNPy/populate_optical) for a test
    of pure dict-building logic."""

    def __init__(self, edges):
        self.undirected_edges = edges


def _make_edge(src: int, dst: int, length_km: float = 80.0) -> Edge:
    return Edge(
        src=src, dst=dst, length_km=length_km, num_spans=1,
        span_lengths_km=[length_km], fiber_type="SSMF", amplifier_nf_db=[5.0],
    )


def test_build_edge_lookup_and_path_to_edges_bidirectional():
    """edge_lookup must resolve a path in either direction to the same Edge objects
    (this is what let path_to_edges drop its own per-call rebuild — the lookup is
    built once via build_edge_lookup and reused for both path directions)."""
    edges = [_make_edge(0, 1, 80.0), _make_edge(1, 2, 100.0)]
    lookup = build_edge_lookup(_FakeTopology(edges))

    forward = path_to_edges(lookup, [0, 1, 2])
    assert [e.length_km for e in forward] == [80.0, 100.0]
    assert forward[0] is edges[0] and forward[1] is edges[1]

    reverse = path_to_edges(lookup, [2, 1, 0])
    assert reverse[0] is edges[1] and reverse[1] is edges[0]


def test_path_to_edges_raises_on_missing_edge():
    edges = [_make_edge(0, 1, 80.0)]
    lookup = build_edge_lookup(_FakeTopology(edges))
    with pytest.raises(ValueError, match="No edge between"):
        path_to_edges(lookup, [0, 1, 5])


# ---------------------------------------------------------------------------
# compute_duplication_stats / format_duplication_report
# ---------------------------------------------------------------------------

def _dup_test_frame(rows: list) -> pd.DataFrame:
    """rows: list of (span_features_0, span_features_1, n_spans) tuples.
    Only 2 feature columns — the function matches any `span_features_*`
    prefix, so this exercises the same logic as production's 300 columns."""
    return pd.DataFrame(
        [{"span_features_0": a, "span_features_1": b, "n_spans": n, "gsnr_db": 15.0}
         for a, b, n in rows]
    )


def test_compute_duplication_stats_known_pattern():
    """Hand-worked duplication pattern, verified by construction:

    train (5 rows, keys by (f0, f1, n_spans)):
      (1,2,1) x2  <- internal dup
      (3,4,1) x1  <- unique
      (5,6,2) x2  <- internal dup
    -> 3 unique vectors, 4/5 rows are internal duplicates.

    val (4 rows):
      (1,2,1) x2  <- matches train AND duplicates within val
      (7,8,1) x1  <- matches nothing
      (3,4,1) x1  <- matches train, unique within val
    -> 3 unique val vectors; 2/4 val rows share a val-internal duplicate;
       3/4 val rows have a train duplicate; 2/3 unique val vectors seen in train.
    """
    train_df = _dup_test_frame([(1, 2, 1), (1, 2, 1), (3, 4, 1), (5, 6, 2), (5, 6, 2)])
    val_df = _dup_test_frame([(1, 2, 1), (1, 2, 1), (7, 8, 1), (3, 4, 1)])

    stats = compute_duplication_stats(train_df, val_df)

    assert stats["train_rows"] == 5
    assert stats["train_unique_vectors"] == 3
    assert stats["train_internal_dup_rows"] == 4
    assert stats["train_internal_dup_rate"] == pytest.approx(0.8)

    assert stats["val_rows"] == 4
    assert stats["val_unique_vectors"] == 3
    assert stats["val_internal_dup_rows"] == 2
    assert stats["val_internal_dup_rate"] == pytest.approx(0.5)

    assert stats["val_rows_with_train_duplicate"] == 3
    assert stats["val_in_train_dup_rate"] == pytest.approx(0.75)
    assert stats["val_unique_vectors_in_train"] == 2
    assert stats["val_unique_in_train_rate"] == pytest.approx(2 / 3)


def test_compute_duplication_stats_no_duplicates():
    """A dataset with no repeats anywhere reports all-zero duplication."""
    train_df = _dup_test_frame([(1, 1, 1), (2, 2, 1), (3, 3, 1)])
    val_df = _dup_test_frame([(4, 4, 1), (5, 5, 1)])

    stats = compute_duplication_stats(train_df, val_df)

    assert stats["train_internal_dup_rows"] == 0
    assert stats["val_internal_dup_rows"] == 0
    assert stats["val_rows_with_train_duplicate"] == 0
    assert stats["val_in_train_dup_rate"] == 0.0
    assert stats["val_unique_vectors_in_train"] == 0


def test_format_duplication_report_warns_above_50_percent():
    high_dup_stats = compute_duplication_stats(
        _dup_test_frame([(1, 2, 1), (1, 2, 1), (3, 4, 1), (5, 6, 2), (5, 6, 2)]),
        _dup_test_frame([(1, 2, 1), (1, 2, 1), (7, 8, 1), (3, 4, 1)]),
    )
    report = format_duplication_report(high_dup_stats)
    assert "WARNING" in report
    assert "75.0%" in report  # val_in_train_dup_rate


def test_format_duplication_report_no_warning_below_50_percent():
    low_dup_stats = compute_duplication_stats(
        _dup_test_frame([(1, 1, 1), (2, 2, 1), (3, 3, 1)]),
        _dup_test_frame([(4, 4, 1), (5, 5, 1)]),
    )
    report = format_duplication_report(low_dup_stats)
    assert "WARNING" not in report
