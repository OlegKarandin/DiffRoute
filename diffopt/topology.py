"""Topology built on top of the upstream ``multilayer_optical_network`` optical model.

``Topology`` subclasses ``OpticalNetworkModel`` and adds the PyTorch-friendly
views (``edge_index``, ``get_edge_features``, ``regen_candidate_nodes``, ...)
the differentiable pipeline needs, derived from the upstream model's ROADM /
fiber / amplifier / OMS bookkeeping instead of a hand-rolled dataclass graph.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import List, Union

import torch

from multilayer_optical_network.model.optical_network import OpticalNetworkModel
from multilayer_optical_network.model.optical_topology_import import (
    SSMF_LOSS_COEF_DB_PER_KM,
    populate_optical,
)
from multilayer_optical_network.model.modes import load_modulation_formats


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


class Topology(OpticalNetworkModel):
    """Optical model plus the tensor views the differentiable pipeline needs."""

    @classmethod
    def from_graph_json(
        cls,
        topology_path: Union[str, Path],
        modulation_formats_path: Union[str, Path],
    ) -> "Topology":
        modes = load_modulation_formats(Path(modulation_formats_path))
        self = cls(modes=modes)
        graph = json.loads(Path(topology_path).read_text())
        populate_optical(self, graph, SSMF_LOSS_COEF_DB_PER_KM)
        return self

    @property
    def num_nodes(self) -> int:
        """Number of nodes, derived from the ROADM ids (``roadm_{nid}``).

        Node ids must be contiguous ``0..N-1`` — downstream pipeline code
        (regen buffers, edge/node feature tensors) indexes by raw node id, so
        a gap must fail loudly here instead of corrupting silently downstream.
        """
        node_ids = sorted(int(rid[len("roadm_"):]) for rid in self._roadms.keys())
        n = len(node_ids)
        if node_ids != list(range(n)):
            raise ValueError(
                f"Topology node ids are not contiguous 0..{n - 1}: {node_ids}"
            )
        return n

    @cached_property
    def undirected_edges(self) -> List[Edge]:
        """One Edge per undirected pair (src < dst), derived from the OMS pairs.

        ``populate_optical`` creates two directed OMS objects per undirected
        edge (``oms_{src}_{dst}`` and ``oms_{dst}_{src}``); only the forward
        one (``int(src) < int(dst)``) becomes the canonical Edge.

        Cached (computed once, on first access) rather than a plain
        ``@property``: this list is rebuilt from the OMS/fiber/amplifier
        tables on every access (measured ~0.56ms/call on ``ind_132``'s 168
        edges), and several call sites now resolve individual edges by id
        once per transparent segment (``diffopt.qot.span_features.
        span_feature_rows``, used from both ``pipeline.py``'s per-training-
        step forward pass and ``generate_qot_dataset.py``'s per-sample
        generation loop) -- recomputing the whole list on every one of those
        calls would silently reintroduce the exact O(edges)
        per-segment cost ``generate_qot_dataset.py::build_edge_lookup``'s
        docstring already documents having fixed once. Safe to cache: a
        ``Topology`` is fully populated by ``populate_optical`` before any
        caller ever touches it (constructor -> populate -> use, never
        interleaved), so there is no code path that mutates the
        OMS/fiber/amplifier tables after this property's first read.
        """
        forward_oms = [
            oms for oms in self.list_oms()
            if int(oms.src_node_id) < int(oms.dst_node_id)
        ]
        forward_oms.sort(key=lambda oms: (int(oms.src_node_id), int(oms.dst_node_id)))

        edges = []
        for oms in forward_oms:
            fiber_ids = oms.elements[2::2]
            amp_ids = oms.elements[3::2]
            span_lengths_km = [self.get_fiber(fid).length_km for fid in fiber_ids]
            amplifier_nf_db = [self.get_amplifier(aid).nf_db for aid in amp_ids]
            fiber_type = self.get_fiber(fiber_ids[0]).type_variety
            edges.append(Edge(
                src=int(oms.src_node_id),
                dst=int(oms.dst_node_id),
                length_km=round(math.fsum(span_lengths_km), 2),
                num_spans=len(span_lengths_km),
                span_lengths_km=span_lengths_km,
                fiber_type=fiber_type,
                amplifier_nf_db=amplifier_nf_db,
            ))
        return edges

    @property
    def edge_index(self) -> torch.Tensor:
        """Shape (2, E) LongTensor with [src; dst] rows (undirected, src < dst)."""
        edges = self.undirected_edges
        srcs = [e.src for e in edges]
        dsts = [e.dst for e in edges]
        return torch.tensor([srcs, dsts], dtype=torch.long)

    def get_edge_features(self) -> torch.Tensor:
        """Shape (E, 5): [mean_span_length_km, fiber_type_idx, mean_amp_nf_db, num_spans, total_length_km]."""
        rows = []
        for e in self.undirected_edges:
            ftype_idx = float(FIBER_TYPE_INDEX.get(e.fiber_type, 0))
            rows.append([
                e.mean_span_length_km,
                ftype_idx,
                e.mean_amp_nf_db,
                float(e.num_spans),
                e.length_km,
            ])
        return torch.tensor(rows, dtype=torch.float32)

    @property
    def regen_candidate_nodes(self) -> List[int]:
        """Nodes with undirected degree >= 3."""
        degree: dict = {}
        for e in self.undirected_edges:
            degree[e.src] = degree.get(e.src, 0) + 1
            degree[e.dst] = degree.get(e.dst, 0) + 1
        return sorted(n for n, d in degree.items() if d >= 3)


def load_topology(topology_path: str, modulation_formats_path: str) -> Topology:
    """Thin alias for Topology.from_graph_json."""
    return Topology.from_graph_json(topology_path, modulation_formats_path)
