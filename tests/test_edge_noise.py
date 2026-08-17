"""Tests for the analytical per-edge ASE noise proxy used by the pipeline's STE."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from diffopt.qot.edge_noise import compute_edge_ase_noise
from diffopt.topology import Topology

from multilayer_optical_network.model.assets import FiberType

BASE = Path(__file__).parent.parent
MODULATION_FORMATS_PATH = BASE / "configs/modulation_formats.yaml"


def _num_nodes(edges: list[dict]) -> int:
    return max(max(e["src"], e["dst"]) for e in edges) + 1


def _build_topology(tmp_path, edges: list[dict]) -> Topology:
    graph = {
        "nodes": [{"id": i} for i in range(_num_nodes(edges))],
        "edges": edges,
    }
    topo_path = tmp_path / "topo.json"
    topo_path.write_text(json.dumps(graph))
    return Topology.from_graph_json(str(topo_path), str(MODULATION_FORMATS_PATH))


def _edge_dict(src: int, dst: int, span_km: float, nf_db: float,
                fiber_type: str = "SSMF") -> dict:
    return {
        "src": src, "dst": dst, "length_km": span_km, "num_spans": 1,
        "span_lengths_km": [span_km], "fiber_type": fiber_type,
        "amplifier_nf_db": [nf_db],
    }


def test_shape_and_positivity(tmp_path):
    topo = _build_topology(tmp_path, [
        _edge_dict(0, 1, 80.0, 5.0),
        _edge_dict(1, 2, 80.0, 5.0),
    ])
    noise = compute_edge_ase_noise(topo)
    assert noise.shape == torch.Size([2])
    assert torch.all(noise > 0)


def test_longer_span_has_higher_noise(tmp_path):
    topo = _build_topology(tmp_path, [
        _edge_dict(0, 1, 40.0, 5.0),
        _edge_dict(1, 2, 400.0, 5.0),
    ])
    noise = compute_edge_ase_noise(topo)
    assert noise[1].item() > noise[0].item()


def test_median_normalized(tmp_path):
    topo = _build_topology(tmp_path, [
        _edge_dict(0, 1, 40.0, 5.0),
        _edge_dict(1, 2, 80.0, 5.0),
        _edge_dict(2, 3, 120.0, 5.0),
    ])
    noise = compute_edge_ase_noise(topo)
    assert abs(noise.median().item() - 1.0) < 1e-5


def test_distributed_amplification_has_less_noise_than_concentrated(tmp_path):
    """Two 80km spans accumulate less ASE noise than one 160km span of the
    same total length/NF (exponential convexity of the per-span ASE term).
    A shared reference edge is included in both topologies because a
    single non-reference edge always normalizes to exactly 1.0 by itself."""
    reference = _edge_dict(8, 9, 80.0, 5.0)

    concentrated_dir = tmp_path / "concentrated"
    distributed_dir = tmp_path / "distributed"
    concentrated_dir.mkdir()
    distributed_dir.mkdir()

    topo_concentrated = _build_topology(concentrated_dir, [
        reference,
        {"src": 0, "dst": 1, "length_km": 160.0, "num_spans": 1,
         "span_lengths_km": [160.0], "fiber_type": "SSMF",
         "amplifier_nf_db": [5.0]},
    ])
    topo_distributed = _build_topology(distributed_dir, [
        reference,
        {"src": 0, "dst": 1, "length_km": 160.0, "num_spans": 2,
         "span_lengths_km": [80.0, 80.0], "fiber_type": "SSMF",
         "amplifier_nf_db": [5.0, 5.0]},
    ])

    def _noise_for_edge(topo, src, dst):
        idx = next(
            i for i, e in enumerate(topo.undirected_edges)
            if e.src == src and e.dst == dst
        )
        return compute_edge_ase_noise(topo)[idx].item()

    noise_concentrated = _noise_for_edge(topo_concentrated, 0, 1)
    noise_distributed = _noise_for_edge(topo_distributed, 0, 1)
    assert noise_distributed < noise_concentrated


def test_different_fiber_types_use_different_loss_coefficients(tmp_path):
    """The old code used a hardcoded SSMF-only loss coefficient for every
    edge regardless of `edge.fiber_type`, so two edges with identical spans
    and NFs but different fiber types always produced identical noise. This
    test fails under that old behavior and passes once compute_edge_ase_noise
    looks up `topology.get_fiber_type(edge.fiber_type).loss_coef_db_per_km`
    per edge.

    `Topology.from_graph_json` -> `populate_optical` registers every distinct
    fiber_type name found in the graph edges, but all of them share the same
    loss coefficient passed in (there's no per-type physics in the graph JSON
    schema). To get two edges with genuinely different coefficients, we
    re-register the "LEAF" FiberType with a different loss_coef_db_per_km
    after construction. `register_fiber_type` stores into a plain dict keyed
    by type_variety (`self._fiber_types[ft.type_variety] = ft`), so it
    silently overwrites on a duplicate name rather than raising -- confirmed
    by reading OpticalNetworkModel.register_fiber_type's source. That makes
    this override edit only the "LEAF" entry, leaving "SSMF" untouched.
    """
    topo = _build_topology(tmp_path, [
        _edge_dict(0, 1, 80.0, 5.0, fiber_type="SSMF"),
        _edge_dict(2, 3, 80.0, 5.0, fiber_type="LEAF"),
    ])

    ssmf_coef = topo.get_fiber_type("SSMF").loss_coef_db_per_km
    leaf_coef = topo.get_fiber_type("LEAF").loss_coef_db_per_km
    assert ssmf_coef == leaf_coef  # both registered with the same default today

    higher_coef = ssmf_coef + 0.1
    topo.register_fiber_type(FiberType(type_variety="LEAF", loss_coef_db_per_km=higher_coef))
    assert topo.get_fiber_type("SSMF").loss_coef_db_per_km == ssmf_coef  # untouched
    assert topo.get_fiber_type("LEAF").loss_coef_db_per_km == higher_coef

    noise = compute_edge_ase_noise(topo)
    ssmf_edge_idx = next(
        i for i, e in enumerate(topo.undirected_edges) if e.fiber_type == "SSMF"
    )
    leaf_edge_idx = next(
        i for i, e in enumerate(topo.undirected_edges) if e.fiber_type == "LEAF"
    )
    # Same span length/NF, higher loss coefficient -> strictly higher noise.
    assert noise[leaf_edge_idx].item() > noise[ssmf_edge_idx].item()

    # And it should match the formula exactly, not just be "higher by some amount".
    span_km, nf_db = 80.0, 5.0
    nf_lin = 10.0 ** (nf_db / 10.0)
    expected_ssmf = nf_lin * (10.0 ** (ssmf_coef * span_km / 10.0) - 1.0)
    expected_leaf = nf_lin * (10.0 ** (higher_coef * span_km / 10.0) - 1.0)
    expected_ratio = expected_leaf / expected_ssmf
    actual_ratio = noise[leaf_edge_idx].item() / noise[ssmf_edge_idx].item()
    assert abs(actual_ratio - expected_ratio) < 1e-4
