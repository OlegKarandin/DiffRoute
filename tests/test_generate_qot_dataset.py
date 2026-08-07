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
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from generate_qot_dataset import split_path_into_segments, get_k_shortest_paths


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


def test_bug_c_two_regen_nodes_reverse_numeric_order():
    """Chosen nodes appear in path in the OPPOSITE order of their numeric value."""
    path = [0, 8, 1, 9, 2]
    rng = _FixedSampleRng(fixed_sample={1, 9}, fixed_randint=2)
    segments = split_path_into_segments(path, regen_nodes=[1, 9], rng=rng)

    assert _tiles_exactly(path, segments)
    # path order of {1, 9} is 9 (index 3) then 1 (index 2)... wait: path index
    # of 9 is 3, of 1 is 2, so path order is 1 (idx 2) then 9 (idx 3).
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
