"""Generate segment-level QoT training data using GNPy (or analytical fallback)."""
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

from diffopt.topology import load_topology, Topology
from diffopt.qot.gnpy_bridge import simulate_segment


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def build_nx_graph(topology: Topology) -> nx.Graph:
    """Build undirected NetworkX graph with length_km as edge weight."""
    G = nx.Graph()
    for node in topology.nodes:
        G.add_node(node["id"])
    for i, edge in enumerate(topology.edges):
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
    for i, edge in enumerate(topology.edges):
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

    chosen = sorted(rng.sample(candidates, k=n_regen))

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

    return segments


def generate_sample(
    topology: Topology,
    G: nx.Graph,
    regen_candidates: list,
    cfg: dict,
    rng: random.Random,
    max_spans: int,
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

        # Simulate GSNR
        launch_power_dbm = cfg.get("launch_power_dbm", -1.0)
        seed_val = rng.randint(0, 2**31 - 1)

        gsnr_db = simulate_segment(
            span_lengths_km=span_lengths,
            amplifier_nf_db=amp_nf_dbs,
            fiber_type=fiber_types[0] if fiber_types else "SSMF",
            n_channels=n_channels,
            launch_power_dbm=launch_power_dbm,
            seed=seed_val,
        )

        # Build per-span features
        accum_dist = 0.0
        span_feature_list = []
        for j, (sl, nf) in enumerate(zip(span_lengths, amp_nf_dbs)):
            accum_dist += sl
            ftype_idx = 0.0  # SSMF=0
            span_feature_list.append([
                sl,                        # span_length_km
                ftype_idx,                 # fiber_type_idx
                nf,                        # amp_nf_db
                channel_loading_fraction,  # channel_loading_fraction
                accum_dist,               # accum_dist_km
            ])

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
    topology = load_topology(topology_path)
    G = build_nx_graph(topology)
    regen_candidates = topology.regen_candidate_nodes

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
                new_samples = generate_sample(topology, G, regen_candidates, cfg, rng, max_spans)
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
