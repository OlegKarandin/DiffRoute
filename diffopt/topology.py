"""Load topology JSON and expose PyTorch-friendly graph."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch


FIBER_TYPE_INDEX = {"SSMF": 0, "LEAF": 1, "TWRS": 2}


@dataclass
class Edge:
    src: int
    dst: int
    length_km: float
    num_spans: int
    span_lengths_km: List[float]
    fiber_type: str
    amplifier_nf_db: List[float]

    @property
    def mean_span_length_km(self) -> float:
        return sum(self.span_lengths_km) / len(self.span_lengths_km)

    @property
    def mean_amp_nf_db(self) -> float:
        return sum(self.amplifier_nf_db) / len(self.amplifier_nf_db)


@dataclass
class Topology:
    nodes: List[dict]
    edges: List[Edge]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def edge_index(self) -> torch.Tensor:
        """Shape (2, E) LongTensor with [src; dst] rows (undirected, src < dst)."""
        srcs = [e.src for e in self.edges]
        dsts = [e.dst for e in self.edges]
        return torch.tensor([srcs, dsts], dtype=torch.long)

    def edge_src(self, edge_id: int) -> int:
        return self.edges[edge_id].src

    def edge_dst(self, edge_id: int) -> int:
        return self.edges[edge_id].dst

    @property
    def regen_candidate_nodes(self) -> List[int]:
        """Nodes with degree >= 3 (as undirected graph)."""
        degree = [0] * self.num_nodes
        for e in self.edges:
            degree[e.src] += 1
            degree[e.dst] += 1
        return [i for i, d in enumerate(degree) if d >= 3]

    def get_edge_features(self) -> torch.Tensor:
        """Shape (E, 5): [mean_span_length_km, fiber_type_idx, mean_amp_nf_db, num_spans, total_length_km]."""
        rows = []
        for e in self.edges:
            ftype_idx = float(FIBER_TYPE_INDEX.get(e.fiber_type, 0))
            rows.append([
                e.mean_span_length_km,
                ftype_idx,
                e.mean_amp_nf_db,
                float(e.num_spans),
                e.length_km,
            ])
        return torch.tensor(rows, dtype=torch.float32)


def load_topology(path: str) -> Topology:
    """Load topology from JSON file."""
    data = json.loads(Path(path).read_text())
    nodes = data["nodes"]
    edges = [
        Edge(
            src=e["src"],
            dst=e["dst"],
            length_km=e["length_km"],
            num_spans=e["num_spans"],
            span_lengths_km=e["span_lengths_km"],
            fiber_type=e["fiber_type"],
            amplifier_nf_db=e["amplifier_nf_db"],
        )
        for e in data["edges"]
    ]
    return Topology(nodes=nodes, edges=edges)
