"""Precomputed per-edge analytical ASE noise proxy for the Vlastelica STE.

Reuses the per-span ASE term from diffopt.qot.gnpy_bridge.analytical_gsnr_db
(the h*nu physical constant is dropped — only relative magnitude across
edges matters, since the output is median-normalized below).
"""
from __future__ import annotations

import torch

from diffopt.qot.gnpy_bridge import SSMF_ALPHA_DB_KM
from diffopt.topology import Topology


def compute_edge_ase_noise(topology: Topology) -> torch.Tensor:
    """Per-edge linear ASE noise proxy, median-normalized.

    For each span: ase_span = nf_lin * (span_loss_lin - 1)
    Per edge: sum over its spans.
    Output is divided by its own median so the median edge has noise = 1.0.
    """
    raw: list[float] = []
    for edge in topology.edges:
        edge_noise = 0.0
        for span_km, nf_db in zip(edge.span_lengths_km, edge.amplifier_nf_db):
            # Uses SSMF_ALPHA_DB_KM regardless of edge.fiber_type (LEAF, TWRS
            # not handled). Matches the same SSMF-only limitation already
            # present in gnpy_bridge.analytical_gsnr_db; both shipped
            # topologies are 100% SSMF today.
            span_loss_lin = 10.0 ** (SSMF_ALPHA_DB_KM * span_km / 10.0)
            nf_lin = 10.0 ** (nf_db / 10.0)
            edge_noise += nf_lin * (span_loss_lin - 1.0)
        raw.append(edge_noise)

    noise = torch.tensor(raw, dtype=torch.float32)
    return noise / noise.median()
