"""Generate segment-level QoT training data using real GNPy (diffopt.qot.optical_bridge).

No analytical fallback: `optical_bridge.segment_gsnr_db` raises loudly on any
physics failure rather than silently substituting an approximation (see
`diffopt/qot/optical_bridge.py` module docstring).
"""
from __future__ import annotations

import argparse
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

from diffopt.topology import load_topology, Topology, FIBER_TYPE_INDEX
from diffopt.qot.optical_bridge import oms_sequence_for_node_path, segment_gsnr_db


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
        paths = list(nx.shortest_simple_paths(G, src, dst, weight="weight"))
        return paths[:k]
    except nx.NetworkXNoPath:
        return []


def path_to_edges(G: nx.Graph, topology: Topology, path: list) -> list:
    """Convert node path to list of Edge objects."""
    edges = []
    edge_lookup = {}
    for edge in topology.undirected_edges:
        edge_lookup[(edge.src, edge.dst)] = edge
        edge_lookup[(edge.dst, edge.src)] = edge

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge = edge_lookup.get((u, v))
        if edge is None:
            raise ValueError(f"No edge between {u} and {v}")
        edges.append(edge)
    return edges


def split_path_into_segments(path: list, regen_nodes: list, rng: random.Random) -> list:
    """Split path into transparent segments by random regen placement.

    Picks 0 to min(2, len(intermediate_nodes)) regen points from intermediate nodes
    that are also regen candidates (degree >= 3).

    Returns list of sub-paths (each sub-path is a list of node IDs).
    """
    intermediate = path[1:-1]
    candidates = [n for n in intermediate if n in set(regen_nodes)]

    # Sample 0 to min(2, len(candidates)) regen points
    max_regen = min(2, len(candidates))
    n_regen = rng.randint(0, max_regen)

    if n_regen == 0:
        return [path]

    # Bug C fix: sort chosen regen nodes by their POSITION in `path`, not by
    # node id. `path` comes from nx.shortest_simple_paths and is always a
    # simple path (no repeated nodes), so path.index() is unambiguous.
    # Sorting by node value instead (the original bug) could put a
    # numerically-smaller-but-later node before a numerically-larger-but-
    # earlier one, producing breakpoints out of path order — which silently
    # drops nodes into an empty segment and duplicates others into an
    # overlapping one (see task-6-brief.md Bug C for the exact trace).
    chosen = sorted(rng.sample(candidates, k=n_regen), key=lambda node: path.index(node))

    # Build segments
    segments = []
    breakpoints = [path[0]] + chosen + [path[-1]]
    for i in range(len(breakpoints) - 1):
        start = breakpoints[i]
        end = breakpoints[i + 1]
        # Find sub-path in original path
        si = path.index(start)
        ei = path.index(end)
        segments.append(path[si:ei + 1])

    # Invariant: segments must tile `path` exactly — no dropped nodes, no
    # duplicated interior coverage, every boundary connects to the next
    # segment's start.
    assert segments[0][0] == path[0] and segments[-1][-1] == path[-1]
    for i in range(len(segments) - 1):
        assert segments[i][-1] == segments[i + 1][0]
        assert len(segments[i]) >= 2  # every segment has at least one edge

    return segments


def generate_sample(
    topology: Topology,
    G: nx.Graph,
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

    # Split into segments
    segments = split_path_into_segments(path, regen_candidates, rng)

    samples = []
    for seg_path in segments:
        # Get edges for this segment
        try:
            seg_edges = path_to_edges(G, topology, seg_path)
        except ValueError:
            continue

        # Build span lists for this segment
        span_lengths = []
        amp_nf_dbs = []
        fiber_types = []
        for edge in seg_edges:
            span_lengths.extend(edge.span_lengths_km)
            amp_nf_dbs.extend(edge.amplifier_nf_db)
            fiber_types.extend([edge.fiber_type] * edge.num_spans)

        n_spans = len(span_lengths)
        if n_spans == 0 or n_spans > max_spans:
            continue

        # Sample number of WDM channels (same for all spans in segment)
        n_channels = rng.randint(1, cfg.get("num_channels_cband", 48))
        channel_loading_fraction = n_channels / cfg.get("num_channels_cband", 48)

        # Simulate GSNR via real GNPy (diffopt.qot.optical_bridge). `mode_id`
        # is fixed once per run (see main()) — GSNR is mode-invariant given
        # the shared 87.5GBaud/0.15 roll-off across all formats, see Task 4's
        # test_optical_bridge.py mode-invariance test.
        oms_sequence = oms_sequence_for_node_path(topology, seg_path)
        gsnr_db = segment_gsnr_db(topology, oms_sequence, mode_id, n_channels)

        # Build per-span features. Bug A fix: append using the pre-increment
        # accum_dist (distance at span START), then increment — matches
        # pipeline.py::_extract_span_features's convention exactly.
        accum_dist = 0.0
        span_feature_list = []
        for j, (sl, nf) in enumerate(zip(span_lengths, amp_nf_dbs)):
            # Bug B fix: look up each span's actual fiber type instead of a
            # hardcoded SSMF=0.0, matching pipeline.py's per-span lookup.
            ftype_idx = float(FIBER_TYPE_INDEX.get(fiber_types[j], 0))
            span_feature_list.append([
                sl,                        # span_length_km
                ftype_idx,                 # fiber_type_idx
                nf,                        # amp_nf_db
                channel_loading_fraction,  # channel_loading_fraction
                accum_dist,               # accum_dist_km
            ])
            accum_dist += sl

        # Pad to max_spans
        padded = span_feature_list + [[0.0] * 5] * (max_spans - n_spans)
        flat = [v for span in padded for v in span]

        row = {f"span_features_{i}": flat[i] for i in range(max_spans * 5)}
        row["n_spans"] = n_spans
        row["gsnr_db"] = gsnr_db
        samples.append(row)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate QoT dataset")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    topology_path = cfg["topology"]
    topology = load_topology(topology_path, cfg["modulation_formats"])
    G = build_nx_graph(topology)
    regen_candidates = topology.regen_candidate_nodes

    # Fixed once for the whole run: GSNR is mode-invariant given the shared
    # 87.5GBaud/0.15 roll-off across all formats — see Task 4's
    # test_optical_bridge.py mode-invariance test. Resampling per segment
    # would just add noise-free-but-pointless variance to the dataset.
    mode_id = topology.modes.list()[0].id

    max_spans = cfg.get("max_spans_per_segment", 60)

    output_dir = Path(cfg["dataset_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("seed", 42)

    for split, n_target in [("train", cfg["num_train_samples"]), ("val", cfg["num_val_samples"])]:
        print(f"\nGenerating {split} split ({n_target} samples)...")
        rng = random.Random(seed + (0 if split == "train" else 1))

        rows = []
        with tqdm(total=n_target) as pbar:
            attempts = 0
            while len(rows) < n_target and attempts < n_target * 20:
                new_samples = generate_sample(topology, G, regen_candidates, cfg, rng, max_spans, mode_id)
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

    print("\nDataset generation complete.")


if __name__ == "__main__":
    main()
