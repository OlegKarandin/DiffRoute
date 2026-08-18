"""Generate segment-level QoT training data using real GNPy (diffopt.qot.optical_bridge).

No analytical fallback: `optical_bridge.segment_gsnr_db` raises loudly on any
physics failure rather than silently substituting an approximation (see
`diffopt/qot/optical_bridge.py` module docstring).
"""
from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffopt.topology import load_topology, Topology
from diffopt.qot.optical_bridge import oms_sequence_for_node_path, segment_gsnr_db
from diffopt.qot.span_features import SPAN_FEATURE_DIM, span_feature_rows


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def build_nx_graph(topology: Topology) -> nx.Graph:
    """Build undirected NetworkX graph with length_km as edge weight."""
    G = nx.Graph()
    for node in range(topology.num_nodes):
        G.add_node(node)
    for i, edge in enumerate(topology.undirected_edges):
        G.add_edge(edge.src, edge.dst, weight=edge.length_km, edge_idx=i)
    return G


def get_k_shortest_paths(G: nx.Graph, src: int, dst: int, k: int):
    """Return up to k shortest paths by total edge weight."""
    try:
        paths = list(itertools.islice(nx.shortest_simple_paths(G, src, dst, weight="weight"), k))
        return paths
    except nx.NetworkXNoPath:
        return []


def build_edge_lookup(topology: Topology) -> dict:
    """(src, dst) and (dst, src) -> Edge, built once per topology load.

    `path_to_edges` used to rebuild this from `topology.undirected_edges`
    (a property that walks every OMS element chain) on every call — cheap
    per call in isolation, but it adds up: measured ~0.37ms/call, and with
    1-3 segments per sample that's tens of seconds of pure waste over a
    50k+-sample generation run. Build it once here instead.
    """
    edge_lookup = {}
    for edge in topology.undirected_edges:
        edge_lookup[(edge.src, edge.dst)] = edge
        edge_lookup[(edge.dst, edge.src)] = edge
    return edge_lookup


def path_to_edges(edge_lookup: dict, path: list) -> list:
    """Convert node path to list of Edge objects."""
    edges = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge = edge_lookup.get((u, v))
        if edge is None:
            raise ValueError(f"No edge between {u} and {v}")
        edges.append(edge)
    return edges


def build_edge_id_lookup(topology: Topology) -> dict:
    """(src, dst) and (dst, src) -> index into topology.undirected_edges.

    Parallel to build_edge_lookup's Edge-keyed dict, built once per topology
    load and reused for every sample -- needed because
    diffopt.qot.span_features.span_feature_rows takes edge IDs (indices into
    topology.undirected_edges), the same convention diffopt.pipeline uses,
    rather than Edge objects directly.
    """
    edge_id_lookup = {}
    for i, edge in enumerate(topology.undirected_edges):
        edge_id_lookup[(edge.src, edge.dst)] = i
        edge_id_lookup[(edge.dst, edge.src)] = i
    return edge_id_lookup


def path_to_edge_ids(edge_id_lookup: dict, path: list) -> list:
    """Convert node path to list of edge IDs into topology.undirected_edges."""
    ids = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        eid = edge_id_lookup.get((u, v))
        if eid is None:
            raise ValueError(f"No edge between {u} and {v}")
        ids.append(eid)
    return ids


def split_path_into_segments(
    path: list,
    regen_nodes: list,
    rng: random.Random,
    edge_lookup: dict,
    min_reach_km: float = 250.0,
    max_reach_km: float = 3700.0,
) -> list:
    """Split path into transparent segments targeting a realistic regen reach.

    Real coherent regeneration reach is bitrate-dependent, not a fixed node
    count. Measured directly against this project's own
    configs/modulation_formats.yaml via real GNPy on a synthetic 80km-span
    chain: reach ranges from ~320km (800 Gbps, 15.1dB threshold) to beyond 3600km
    (300 Gbps, 4.8dB threshold) at full (48-channel) loading. `min_reach_km`/
    `max_reach_km` default to that measured range with a small margin
    (250-3700km), spanning the full set of `bitrate_options` this project
    actually uses rather than a generic rule of thumb.

    Greedily walks the path, accumulating real per-edge distance from
    `edge_lookup`. Splits at the first regen-candidate node reached once
    accumulated distance since the last split is >= a target reach sampled
    fresh (uniform in [min_reach_km, max_reach_km]) for each segment — so
    segment lengths vary sample to sample across the whole realistic range,
    rather than correlating every segment on one path to a single draw. If
    the path ends before any candidate is reached past the target, the
    final segment simply runs to the destination (a genuine consequence of
    topology sparsity when no regen site exists soon enough — not faked
    around; still bounded by `max_spans_per_segment` downstream).

    Returns list of sub-paths (each sub-path is a list of node IDs).
    """
    regen_set = set(regen_nodes)
    target_reach = rng.uniform(min_reach_km, max_reach_km)

    segments = []
    current_segment = [path[0]]
    accum_km = 0.0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        accum_km += edge_lookup[(u, v)].length_km
        current_segment.append(v)

        is_last_edge = (i == len(path) - 2)
        if not is_last_edge and v in regen_set and accum_km >= target_reach:
            segments.append(current_segment)
            current_segment = [v]
            accum_km = 0.0
            target_reach = rng.uniform(min_reach_km, max_reach_km)

    segments.append(current_segment)

    # Invariant: segments must tile `path` exactly — no dropped nodes, no
    # duplicated interior coverage, every boundary connects to the next
    # segment's start. Holds by construction (each split shares its
    # boundary node with both segments) but kept as an explicit regression
    # guard, same as the prior node-count-based implementation.
    assert segments[0][0] == path[0] and segments[-1][-1] == path[-1]
    for i in range(len(segments) - 1):
        assert segments[i][-1] == segments[i + 1][0]
        assert len(segments[i]) >= 2  # every segment has at least one edge

    return segments


def generate_sample(
    topology: Topology,
    G: nx.Graph,
    edge_lookup: dict,
    edge_id_lookup: dict,
    regen_candidates: list,
    cfg: dict,
    rng: random.Random,
    max_spans: int,
    mode_id: str,
) -> list:
    """Generate one or more segment samples from a single demand.

    Returns list of dicts, one per segment.
    """
    n = topology.num_nodes
    src = rng.randint(0, n - 1)
    dst = rng.randint(0, n - 1)
    while dst == src:
        dst = rng.randint(0, n - 1)

    paths = get_k_shortest_paths(G, src, dst, cfg.get("k_paths", 5))
    if not paths:
        return []

    path = rng.choice(paths)

    # Split into segments, targeting a realistic per-segment regen reach
    segments = split_path_into_segments(
        path, regen_candidates, rng, edge_lookup,
        min_reach_km=cfg.get("min_regen_reach_km", 250.0),
        max_reach_km=cfg.get("max_regen_reach_km", 3700.0),
    )

    samples = []
    for seg_path in segments:
        # Get edges for this segment
        try:
            seg_edges = path_to_edges(edge_lookup, seg_path)
            seg_edge_ids = path_to_edge_ids(edge_id_lookup, seg_path)
        except ValueError:
            continue

        n_spans = sum(edge.num_spans for edge in seg_edges)
        if n_spans == 0 or n_spans > max_spans:
            continue

        # Sample number of WDM channels (same for all spans in segment)
        n_channels = rng.randint(1, cfg.get("num_channels_cband", 48))
        channel_loading_fraction = n_channels / cfg.get("num_channels_cband", 48)

        # Simulate GSNR via real GNPy (diffopt.qot.optical_bridge). `mode_id`
        # is fixed once per run (see main()) — GSNR is mode-invariant given
        # the shared 87.5GBaud/0.15 roll-off across all formats, see
        # tests/test_optical_bridge.py's mode-invariance test.
        oms_sequence = oms_sequence_for_node_path(topology, seg_path)
        gsnr_db = segment_gsnr_db(topology, oms_sequence, mode_id, n_channels)

        # Build per-span features via the shared span_feature_rows — the
        # canonical [span_length_km, fiber_type_idx, amp_nf_db,
        # channel_loading_fraction, accum_dist_km] ordering
        # docs/architecture/invariants.md declares a fixed invariant, now
        # defined once in diffopt.qot.span_features instead of duplicated
        # here.
        span_feature_list = span_feature_rows(
            topology, seg_edge_ids,
            channel_loading_fraction=channel_loading_fraction,
        )

        # Pad to max_spans
        padded = span_feature_list + [[0.0] * SPAN_FEATURE_DIM] * (max_spans - n_spans)
        flat = [v for span in padded for v in span]

        row = {f"span_features_{i}": flat[i] for i in range(max_spans * SPAN_FEATURE_DIM)}
        row["n_spans"] = n_spans
        row["gsnr_db"] = gsnr_db
        samples.append(row)

    return samples


def _feature_vector_keys(df: pd.DataFrame) -> pd.Series:
    """Hashable per-row key from span_features_* + n_spans (exact float64 equality).

    Two rows with an identical key have an identical (span geometry,
    fiber type, NF, channel loading) input to the physics model — on a
    bounded topology, real segments genuinely repeat (same OMS sequence +
    same sampled channel count), and GNPy is deterministic given identical
    inputs, so identical keys always carry identical `gsnr_db` labels too.
    """
    feature_cols = [c for c in df.columns if c.startswith("span_features_")] + ["n_spans"]
    return pd.Series(list(map(tuple, df[feature_cols].to_numpy())), index=df.index)


def compute_duplication_stats(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Measure exact-feature-vector duplication within and across splits.

    A small topology's segment/channel-loading space is finite (bounded by
    node/edge count x 48 quantized loading levels) — at high enough sample
    counts, duplication is an expected, real property of that bounded
    space, not necessarily a bug (see docs/investigations/CHANGELOG.md's
    Phase 1a milestone note for the investigation that established this for
    german_17). But it's
    invisible unless measured, and it directly affects how a `val_rmse`
    number should be interpreted (a val row that also appears in train
    isn't testing generalization to anything new). This makes it visible
    on every dataset-generation run instead of requiring a one-off
    manual investigation.
    """
    train_keys = _feature_vector_keys(train_df)
    val_keys = _feature_vector_keys(val_df)

    train_key_counts = train_keys.value_counts()
    val_key_counts = val_keys.value_counts()

    train_dup_rows = int((train_key_counts.reindex(train_keys).to_numpy() > 1).sum())
    val_dup_rows = int((val_key_counts.reindex(val_keys).to_numpy() > 1).sum())

    train_key_set = set(train_key_counts.index)
    val_unique_keys = set(val_key_counts.index)
    val_rows_with_train_dup = int(val_keys.isin(train_key_set).sum())
    val_unique_in_train = len(val_unique_keys & train_key_set)

    return {
        "train_rows": len(train_df),
        "train_unique_vectors": int(train_keys.nunique()),
        "train_internal_dup_rows": train_dup_rows,
        "train_internal_dup_rate": train_dup_rows / len(train_df) if len(train_df) else 0.0,
        "val_rows": len(val_df),
        "val_unique_vectors": int(val_keys.nunique()),
        "val_internal_dup_rows": val_dup_rows,
        "val_internal_dup_rate": val_dup_rows / len(val_df) if len(val_df) else 0.0,
        "val_rows_with_train_duplicate": val_rows_with_train_dup,
        "val_in_train_dup_rate": val_rows_with_train_dup / len(val_df) if len(val_df) else 0.0,
        "val_unique_vectors_in_train": val_unique_in_train,
        "val_unique_in_train_rate": (
            val_unique_in_train / len(val_unique_keys) if val_unique_keys else 0.0
        ),
    }


def format_duplication_report(stats: dict) -> str:
    """Human-readable duplication summary, printed after dataset generation."""
    lines = [
        "Duplicate-feature-vector report:",
        f"  train: {stats['train_rows']} rows, {stats['train_unique_vectors']} unique vectors "
        f"({stats['train_internal_dup_rate']:.1%} of rows share a vector with another train row)",
        f"  val:   {stats['val_rows']} rows, {stats['val_unique_vectors']} unique vectors "
        f"({stats['val_internal_dup_rate']:.1%} of rows share a vector with another val row)",
        f"  val rows whose vector also appears in train: "
        f"{stats['val_rows_with_train_duplicate']}/{stats['val_rows']} "
        f"({stats['val_in_train_dup_rate']:.1%})",
        f"  unique val vectors also seen in train: "
        f"{stats['val_unique_vectors_in_train']}/{stats['val_unique_vectors']} "
        f"({stats['val_unique_in_train_rate']:.1%})",
    ]
    if stats["val_in_train_dup_rate"] > 0.5:
        lines.append(
            "  WARNING: over half of validation rows have an exact duplicate in the "
            "training set. val_rmse on this split will overstate generalization to "
            "genuinely novel segments — consider a bigger topology (more distinct "
            "segment geometries) or measuring RMSE on just the never-seen-in-train "
            "subset for an honest generalization estimate."
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate QoT dataset")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    topology_path = cfg["topology"]
    topology = load_topology(topology_path, cfg["modulation_formats"])
    G = build_nx_graph(topology)
    edge_lookup = build_edge_lookup(topology)
    edge_id_lookup = build_edge_id_lookup(topology)
    regen_candidates = topology.regen_candidate_nodes

    # Fixed once for the whole run: GSNR is mode-invariant given the shared
    # 87.5GBaud/0.15 roll-off across all formats — see
    # tests/test_optical_bridge.py's mode-invariance test. Resampling per
    # segment would just add noise-free-but-pointless variance to the dataset.
    mode_id = topology.modes.list()[0].id

    max_spans = cfg.get("max_spans_per_segment", 60)

    output_dir = Path(cfg["dataset_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("seed", 42)

    dfs = {}
    for split, n_target in [("train", cfg["num_train_samples"]), ("val", cfg["num_val_samples"])]:
        print(f"\nGenerating {split} split ({n_target} samples)...")
        rng = random.Random(seed + (0 if split == "train" else 1))

        rows = []
        with tqdm(total=n_target) as pbar:
            attempts = 0
            while len(rows) < n_target and attempts < n_target * 20:
                new_samples = generate_sample(
                    topology, G, edge_lookup, edge_id_lookup, regen_candidates, cfg, rng, max_spans, mode_id
                )
                for s in new_samples:
                    if len(rows) < n_target:
                        rows.append(s)
                        pbar.update(1)
                attempts += 1

        if len(rows) < n_target:
            print(f"Warning: only generated {len(rows)}/{n_target} samples for {split}")

        df = pd.DataFrame(rows[:n_target])
        out_path = output_dir / f"{split}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Saved {len(df)} samples to {out_path}")
        dfs[split] = df

    print()
    print(format_duplication_report(compute_duplication_stats(dfs["train"], dfs["val"])))

    print("\nDataset generation complete.")


if __name__ == "__main__":
    main()
