"""Tests for the analytical per-edge ASE noise proxy used by the pipeline's STE."""
from __future__ import annotations

import torch

from diffopt.qot.edge_noise import compute_edge_ase_noise
from diffopt.topology import Edge, Topology


def _make_edge(src: int, dst: int, span_km: float, nf_db: float) -> Edge:
    return Edge(
        src=src, dst=dst, length_km=span_km, num_spans=1,
        span_lengths_km=[span_km], fiber_type="SSMF",
        amplifier_nf_db=[nf_db],
    )


def test_shape_and_positivity():
    topo = Topology(nodes=[{"id": i} for i in range(3)], edges=[
        _make_edge(0, 1, 80.0, 5.0),
        _make_edge(1, 2, 80.0, 5.0),
    ])
    noise = compute_edge_ase_noise(topo)
    assert noise.shape == torch.Size([2])
    assert torch.all(noise > 0)


def test_longer_span_has_higher_noise():
    topo = Topology(nodes=[{"id": i} for i in range(3)], edges=[
        _make_edge(0, 1, 40.0, 5.0),
        _make_edge(1, 2, 400.0, 5.0),
    ])
    noise = compute_edge_ase_noise(topo)
    assert noise[1].item() > noise[0].item()


def test_median_normalized():
    topo = Topology(nodes=[{"id": i} for i in range(4)], edges=[
        _make_edge(0, 1, 40.0, 5.0),
        _make_edge(1, 2, 80.0, 5.0),
        _make_edge(2, 3, 120.0, 5.0),
    ])
    noise = compute_edge_ase_noise(topo)
    assert abs(noise.median().item() - 1.0) < 1e-5


def test_distributed_amplification_has_less_noise_than_concentrated():
    """Two 80km spans accumulate less ASE noise than one 160km span of the
    same total length/NF (exponential convexity of the per-span ASE term).
    A shared reference edge is included in both topologies because a
    single non-reference edge always normalizes to exactly 1.0 by itself."""
    reference = _make_edge(8, 9, 80.0, 5.0)

    topo_concentrated = Topology(nodes=[{"id": i} for i in range(4)], edges=[
        reference,
        Edge(src=0, dst=1, length_km=160.0, num_spans=1,
             span_lengths_km=[160.0], fiber_type="SSMF", amplifier_nf_db=[5.0]),
    ])
    topo_distributed = Topology(nodes=[{"id": i} for i in range(4)], edges=[
        reference,
        Edge(src=0, dst=1, length_km=160.0, num_spans=2,
             span_lengths_km=[80.0, 80.0], fiber_type="SSMF", amplifier_nf_db=[5.0, 5.0]),
    ])

    noise_concentrated = compute_edge_ase_noise(topo_concentrated)[1].item()
    noise_distributed = compute_edge_ase_noise(topo_distributed)[1].item()
    assert noise_distributed < noise_concentrated
