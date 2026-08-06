"""Parse .dat topology files and generate topology JSONs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from multilayer_optical_mcp.model.optical_topology_import import split_link_into_spans


def parse_dat_file(path: str) -> Tuple[int, List[Tuple[int, int, float]]]:
    """Parse a .dat topology file.

    Returns:
        (num_nodes, edges) where edges is list of (src, dst, length_km)
        Only keeps edges where src < dst (deduplication).
    """
    p = Path(path)
    text = p.read_text()

    lines = text.splitlines()

    in_nodes = False
    in_fibers = False
    num_nodes = 0
    edges = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower() == "nodes":
            in_nodes = True
            in_fibers = False
            continue
        if line.lower() == "fibers":
            in_fibers = True
            in_nodes = False
            continue
        if line.startswith("#"):
            continue

        if in_nodes:
            num_nodes += 1
        elif in_fibers:
            # Format: link_id, src, dst, length_km
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                src = int(parts[1])
                dst = int(parts[2])
                length = float(parts[3])
            except ValueError:
                continue
            if src < dst:
                edges.append((src, dst, length))

    return num_nodes, edges


def build_topology_json(
    dat_path: str,
    output_path: str,
    target_span_km: float = 80.0,
    min_span_km: float = 20.0,
    fiber_type: str = "SSMF",
    amplifier_nf_db: float = 5.5,
) -> dict:
    """Parse .dat file and generate topology JSON, also writing it to output_path."""
    num_nodes, edges = parse_dat_file(dat_path)

    nodes = [{"id": i} for i in range(num_nodes)]

    edge_list = []
    for src, dst, length_km in sorted(edges):
        spans = split_link_into_spans(length_km, target_span_km, min_span_km)
        num_spans = len(spans)
        edge_list.append({
            "src": src,
            "dst": dst,
            "length_km": length_km,
            "num_spans": num_spans,
            "span_lengths_km": spans,
            "fiber_type": fiber_type,
            "amplifier_nf_db": [amplifier_nf_db] * num_spans,
        })

    topology = {"nodes": nodes, "edges": edge_list}

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(topology, indent=2))

    return topology


if __name__ == "__main__":
    import sys
    base = Path(__file__).parent.parent

    build_topology_json(
        str(base / "german_17.dat"),
        str(base / "configs/topology/german_17.json"),
    )
    print("Generated configs/topology/german_17.json")

    build_topology_json(
        str(base / "EU_19.dat"),
        str(base / "configs/topology/eu_19.json"),
    )
    print("Generated configs/topology/eu_19.json")
