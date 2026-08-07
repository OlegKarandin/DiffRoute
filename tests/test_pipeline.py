"""
Tests for the end-to-end DiffONet pipeline (Phase 1c).

Two in-memory topologies:

  Linear: 0—1—2—3—4  (4 edges, all degree ≤ 2, no regen candidates)
          Useful for single-segment identity test.

  Hub:    0—1—3—4   and   0—2—3—4   (5 edges, node 3 has degree 3)
          Node 3 is the sole regen candidate.
          Two paths 0→4 enable routing choice (EdgeWeightNet gradient test)
          and a boundary at node 3 (regen_logits gradient test).
          The two routes are physically asymmetric (different span lengths
          on eids 0,2 vs eids 1,3; the shared final hop eid 4 is unchanged)
          so that EdgeWeightNet does not output an identical weight for
          every edge — a degenerate case where a graph-symmetric,
          equal-length-route topology can make the aggregate surrogate
          gradient on EdgeWeightNet's parameters cancel to zero regardless
          of the routing signal (see test_edge_weight_net_grad_differs_
          with_and_without_ste_proxy).
"""
from __future__ import annotations

from pathlib import Path

import torch
import pytest

from diffopt.demands import Demand
from diffopt.loss import compute_loss
from diffopt.modulation import ModulationConfig
from diffopt.pipeline import DiffONetPipeline, segment_path
from diffopt.placement.regenerator import RegenPlacement
from diffopt.qot.model import SpanAttentionQoT
from diffopt.qot.segment_combiner import SegmentCombiner
from diffopt.routing.edge_weight_net import EdgeWeightNet
from diffopt.topology import Topology

from multilayer_optical_mcp.model.optical_topology_import import (
    SSMF_LOSS_COEF_DB_PER_KM, populate_optical,
)
from multilayer_optical_mcp.model.modes import load_modulation_formats


MODULATION_FORMATS_PATH = Path(__file__).parent.parent / "configs/modulation_formats.yaml"


# ---------------------------------------------------------------------------
# Topology factories
# ---------------------------------------------------------------------------

def _modes():
    return load_modulation_formats(MODULATION_FORMATS_PATH)


def _build_topology(num_nodes: int, edges: list[dict]) -> Topology:
    graph = {"nodes": [{"id": i} for i in range(num_nodes)], "edges": edges}
    topo = Topology(modes=_modes())
    populate_optical(topo, graph, SSMF_LOSS_COEF_DB_PER_KM)
    return topo


def _make_edge_dict(src: int, dst: int, length_km: float = 80.0) -> dict:
    return {
        "src": src, "dst": dst, "length_km": length_km, "num_spans": 1,
        "span_lengths_km": [length_km], "fiber_type": "SSMF",
        "amplifier_nf_db": [5.0],
    }


def make_linear_topology() -> Topology:
    """Chain 0—1—2—3—4. All degrees ≤ 2 → no regen candidates."""
    edges = [_make_edge_dict(i, i + 1) for i in range(4)]
    return _build_topology(5, edges)


def make_hub_topology() -> Topology:
    """
    Edges: 0-1 (eid 0), 0-2 (eid 1), 1-3 (eid 2), 2-3 (eid 3), 3-4 (eid 4)
    Degrees: 0→2, 1→2, 2→2, 3→3, 4→1  →  node 3 is sole regen candidate.
    Two paths 0→4: via node 1 (eids 0,2,4) and via node 2 (eids 1,3,4).

    The two routes use different (but still ≥20 km, realistic) span
    lengths on their non-shared edges — 60 km via node 1 (eids 0,2) vs
    100 km via node 2 (eids 1,3) — so EdgeWeightNet, which is a per-edge
    function of static span features, does not produce an identical
    weight for every edge. The shared final hop (eid 4) is left at the
    original 80 km. This breaks a physical-symmetry degeneracy without
    changing the graph structure (both routes remain 3 edges; node 3
    remains the sole degree-3 / regen-candidate node).
    """
    edges = [
        _make_edge_dict(0, 1, length_km=60.0),    # eid 0 — route via node 1
        _make_edge_dict(0, 2, length_km=100.0),   # eid 1 — route via node 2
        _make_edge_dict(1, 3, length_km=60.0),    # eid 2 — route via node 1
        _make_edge_dict(2, 3, length_km=100.0),   # eid 3 — route via node 2
        _make_edge_dict(3, 4, length_km=80.0),    # eid 4 — shared final hop
    ]
    return _build_topology(5, edges)


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def make_pipeline(topology: Topology) -> DiffONetPipeline:
    qot_model = SpanAttentionQoT(max_spans=60)
    segment_combiner = SegmentCombiner()  # stateless — temperature is now a forward()-time arg
    edge_weight_net = EdgeWeightNet()
    regen_placement = RegenPlacement(topology.num_nodes)
    return DiffONetPipeline(
        topology=topology,
        qot_model=qot_model,
        segment_combiner=segment_combiner,
        edge_weight_net=edge_weight_net,
        regen_placement=regen_placement,
    )


def make_mod_config() -> ModulationConfig:
    """Minimal ModulationConfig with a single 400 Gbps format."""
    return ModulationConfig(
        channel_spacing_ghz=100.0,
        symbol_rate_gbaud=64.0,
        num_channels_cband=48,
        cut_channel_index=24,
        formats=[{"bitrate_gbps": 400, "snr_threshold_db": 20.0}],
    )


# ---------------------------------------------------------------------------
# Test 1: forward pass shapes
# ---------------------------------------------------------------------------

def test_forward_pass_shapes():
    topo = make_linear_topology()
    pipeline = make_pipeline(topo)
    demands = [
        Demand(id=0, src=0, dst=4, bitrate_gbps=400.0),
        Demand(id=1, src=0, dst=3, bitrate_gbps=400.0),
        Demand(id=2, src=1, dst=4, bitrate_gbps=400.0),
    ]
    path_costs, gsnr_preds, path_indicators, regen_probs = pipeline(demands)

    assert len(gsnr_preds) == 3
    assert len(path_costs) == 3
    assert len(path_indicators) == 3

    for did in [0, 1, 2]:
        assert gsnr_preds[did].shape == torch.Size([])   # scalar
        assert path_costs[did].shape == torch.Size([])   # scalar
        assert path_indicators[did].shape == torch.Size([len(topo.undirected_edges)])

    assert regen_probs.shape == torch.Size([topo.num_nodes])


# ---------------------------------------------------------------------------
# Test 2: EdgeWeightNet gradient flows via path_cost_loss
# ---------------------------------------------------------------------------

def test_gradient_flow_edge_weight_net():
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    mod_cfg = make_mod_config()

    demands = [Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)]
    path_costs, gsnr_preds, _, regen_probs = pipeline(demands, lambda_=5.0)

    loss, _ = compute_loss(
        gsnr_preds=gsnr_preds,
        path_costs=path_costs,
        demands=demands,
        regen_probs=regen_probs,
        modulation_config=mod_cfg,
    )
    loss.backward()

    for name, param in pipeline.edge_weight_net.named_parameters():
        assert param.grad is not None, f"EdgeWeightNet param {name!r} has no grad"
        assert param.grad.abs().sum().item() > 0, \
            f"EdgeWeightNet param {name!r} has all-zero grad"


# ---------------------------------------------------------------------------
# Test 3: regen_logits gradient flows via boundary_probs
# ---------------------------------------------------------------------------

def test_gradient_flow_regen_logits():
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    mod_cfg = make_mod_config()

    # Demand 0→4 routes through node 3 (regen candidate) → boundary prob used
    demands = [Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)]
    path_costs, gsnr_preds, _, regen_probs = pipeline(demands, lambda_=5.0)

    loss, _ = compute_loss(
        gsnr_preds=gsnr_preds,
        path_costs=path_costs,
        demands=demands,
        regen_probs=regen_probs,
        modulation_config=mod_cfg,
    )
    loss.backward()

    logits = pipeline.regen_placement.regen_logits
    assert logits.grad is not None, "regen_logits.grad is None"
    assert logits.grad.abs().sum().item() > 0, "regen_logits.grad is all-zero"


# ---------------------------------------------------------------------------
# Test 4: QoT model parameters have no gradient after backward
# ---------------------------------------------------------------------------

def test_qot_frozen():
    topo = make_linear_topology()
    pipeline = make_pipeline(topo)
    mod_cfg = make_mod_config()

    demands = [Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)]
    path_costs, gsnr_preds, _, regen_probs = pipeline(demands)

    loss, _ = compute_loss(
        gsnr_preds=gsnr_preds,
        path_costs=path_costs,
        demands=demands,
        regen_probs=regen_probs,
        modulation_config=mod_cfg,
    )
    loss.backward()

    for name, param in pipeline.qot_model.named_parameters():
        assert param.grad is None, \
            f"QoT param {name!r} should have no grad but got {param.grad}"


# ---------------------------------------------------------------------------
# Test 5: single-segment identity — pipeline GSNR matches direct QoT call
# ---------------------------------------------------------------------------

def test_single_segment_identity():
    topo = make_linear_topology()
    # Linear topology has no regen candidates → every path is one segment
    # → SegmentCombiner just passes the single GSNR through unchanged
    # (temperature is irrelevant here: with one segment, SegmentCombiner
    # never reaches the soft_max branch at all)
    pipeline = make_pipeline(topo)

    demand = Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)
    path_costs, gsnr_preds, path_indicators, regen_probs = pipeline([demand])

    # Find which edges are on the path
    indicator = path_indicators[0].detach()
    active_eids = [e for e in range(indicator.shape[0]) if indicator[e].item() > 0.5]

    # Build the same span feature tensor the pipeline would have used
    rows = []
    accum_dist = 0.0
    for eid in active_eids:
        edge = topo.undirected_edges[eid]
        for span_idx in range(edge.num_spans):
            rows.append([
                edge.span_lengths_km[span_idx],
                0.0,   # SSMF → index 0
                edge.amplifier_nf_db[span_idx],
                0.5,   # default channel_loading_fraction
                accum_dist,
            ])
            accum_dist += edge.span_lengths_km[span_idx]

    n_spans = len(rows)
    span_feats = torch.zeros(1, 60, 5)
    span_feats[0, :n_spans] = torch.tensor(rows, dtype=torch.float32)
    padding_mask = torch.zeros(1, 60, dtype=torch.bool)
    padding_mask[0, :n_spans] = True

    with torch.no_grad():
        direct_gsnr = pipeline.qot_model(span_feats, padding_mask)[0]

    pipeline_gsnr = gsnr_preds[0].detach()
    assert abs(pipeline_gsnr.item() - direct_gsnr.item()) < 1e-4, (
        f"Pipeline GSNR {pipeline_gsnr.item():.6f} dB != "
        f"direct QoT {direct_gsnr.item():.6f} dB"
    )


# ---------------------------------------------------------------------------
# Test 6: backward produces no NaN or Inf gradients
# ---------------------------------------------------------------------------

def test_loss_backward_no_nan():
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    mod_cfg = make_mod_config()

    demands = [
        Demand(id=0, src=0, dst=4, bitrate_gbps=400.0),
        Demand(id=1, src=0, dst=3, bitrate_gbps=400.0),
    ]
    path_costs, gsnr_preds, _, regen_probs = pipeline(demands, lambda_=5.0)

    loss, _ = compute_loss(
        gsnr_preds=gsnr_preds,
        path_costs=path_costs,
        demands=demands,
        regen_probs=regen_probs,
        modulation_config=mod_cfg,
    )
    loss.backward()

    trainable = list(pipeline.edge_weight_net.parameters()) + \
                [pipeline.regen_placement.regen_logits]
    for param in trainable:
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), \
                f"Non-finite gradient found in param of shape {param.shape}"


# ---------------------------------------------------------------------------
# Test 7: STE preserves the QoT-accurate forward value across a multi-segment path
# ---------------------------------------------------------------------------

def test_ste_preserves_forward_value():
    """Segment GSNR from the STE blend numerically equals what a direct,
    no-STE combination of frozen QoT calls would produce, on a path with
    a real regen-candidate boundary (hub topology, node 3)."""
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    temperature = 0.01  # sharp, so soft_max ≈ true max — must match both calls below

    demand = Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)
    _, gsnr_preds, path_indicators, regen_probs = pipeline([demand], soft_max_temperature=temperature)

    indicator = path_indicators[0].detach()
    active_eids = [e for e in range(indicator.shape[0]) if indicator[e].item() > 0.5]

    segments, boundary_nodes = segment_path(
        active_eids, demand.src, pipeline._regen_candidate_set, pipeline._edges, demand.dst
    )

    direct_gsnrs = []
    for seg in segments:
        span_feats, padding_mask = pipeline._extract_span_features(seg, torch.device("cpu"))
        with torch.no_grad():
            direct_gsnrs.append(pipeline.qot_model(span_feats, padding_mask)[0])

    boundary_probs = [regen_probs[n].detach() for n in boundary_nodes]
    expected_gsnr = pipeline.segment_combiner(direct_gsnrs, boundary_probs, temperature=temperature)

    assert abs(gsnr_preds[0].item() - expected_gsnr.item()) < 1e-4, (
        f"STE-blended GSNR {gsnr_preds[0].item():.6f} dB != "
        f"direct QoT+combiner GSNR {expected_gsnr.item():.6f} dB"
    )


# ---------------------------------------------------------------------------
# Test 8: path_indicator gradient is not a uniform multiple of edge_weights
# ---------------------------------------------------------------------------

def test_path_indicator_gradient_not_proportional_to_edge_weights():
    """∂feasibility_loss/∂path_indicator must NOT be a scalar multiple of
    edge_weights on the active path. If it were, the Vlastelica perturbation
    c_target = w + lambda*grad would be a uniform rescaling of all edge
    costs, which preserves shortest-path ordering and makes the surrogate
    return the same path forever (the bug this plan fixes)."""
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    always_infeasible_cfg = ModulationConfig(
        channel_spacing_ghz=100.0, symbol_rate_gbaud=64.0,
        num_channels_cband=48, cut_channel_index=24,
        formats=[{"bitrate_gbps": 400, "snr_threshold_db": 1000.0}],
    )

    demand = Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)
    path_costs, gsnr_preds, path_indicators, regen_probs = pipeline([demand], lambda_=5.0)

    pi = path_indicators[0]
    pi.retain_grad()

    edge_feats = torch.cat([
        pipeline._topo_edge_features,
        regen_probs[pipeline._edge_src_ids].unsqueeze(1).detach(),
        regen_probs[pipeline._edge_dst_ids].unsqueeze(1).detach(),
    ], dim=1)
    edge_weights = pipeline.edge_weight_net(edge_feats).squeeze(-1).detach()

    loss, _ = compute_loss(
        gsnr_preds=gsnr_preds, path_costs=path_costs, demands=[demand],
        regen_probs=regen_probs, modulation_config=always_infeasible_cfg,
    )
    loss.backward()

    assert pi.grad is not None
    assert pi.grad.abs().sum().item() > 0

    active = pi.detach() > 0.5
    ratio = pi.grad[active] / edge_weights[active]
    assert ratio.std().item() > 1e-6, (
        "gradient is a uniform multiple of edge_weights on the active path — "
        "the Vlastelica perturbation would rescale all costs equally and "
        "never change the selected path"
    )


# ---------------------------------------------------------------------------
# Test 9: EdgeWeightNet gradient actually depends on the STE proxy term
# ---------------------------------------------------------------------------

def test_edge_weight_net_grad_differs_with_and_without_ste_proxy():
    """Zeroing the _edge_ase_noise buffer collapses proxy_noise to a constant
    (independent of path_indicator), reproducing today's pre-fix behavior.
    EdgeWeightNet's gradient must differ between the two cases, proving the
    proxy term (not just path_cost_loss) is contributing to the signal."""
    topo = make_hub_topology()
    always_infeasible_cfg = ModulationConfig(
        channel_spacing_ghz=100.0, symbol_rate_gbaud=64.0,
        num_channels_cband=48, cut_channel_index=24,
        formats=[{"bitrate_gbps": 400, "snr_threshold_db": 1000.0}],
    )
    demand = Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)

    def run(pipeline: DiffONetPipeline) -> torch.Tensor:
        path_costs, gsnr_preds, _, regen_probs = pipeline([demand], lambda_=5.0)
        loss, _ = compute_loss(
            gsnr_preds=gsnr_preds, path_costs=path_costs, demands=[demand],
            regen_probs=regen_probs, modulation_config=always_infeasible_cfg,
        )
        loss.backward()
        return torch.cat([p.grad.flatten() for p in pipeline.edge_weight_net.parameters()])

    torch.manual_seed(0)
    pipeline_with_ste = make_pipeline(topo)
    grad_with = run(pipeline_with_ste)

    torch.manual_seed(0)
    pipeline_without_ste = make_pipeline(topo)
    pipeline_without_ste._edge_ase_noise.zero_()
    grad_without = run(pipeline_without_ste)

    assert not torch.allclose(grad_with, grad_without, atol=1e-8), (
        "EdgeWeightNet gradient identical with and without the STE proxy — "
        "the proxy term is not contributing to the routing signal"
    )


# ---------------------------------------------------------------------------
# Test 10: feasible demand contributes zero feasibility-loss gradient
# ---------------------------------------------------------------------------

def test_feasible_demand_zero_feasibility_gradient():
    """When a demand's path already clears the GSNR threshold, relu's flat
    zero region means the feasibility term contributes zero gradient to
    path_indicator — the STE must not bypass this gating."""
    topo = make_linear_topology()
    pipeline = make_pipeline(topo)
    always_feasible_cfg = ModulationConfig(
        channel_spacing_ghz=100.0, symbol_rate_gbaud=64.0,
        num_channels_cband=48, cut_channel_index=24,
        formats=[{"bitrate_gbps": 400, "snr_threshold_db": -1000.0}],
    )

    demand = Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)
    _, gsnr_preds, path_indicators, _ = pipeline([demand])

    pi = path_indicators[0]
    pi.retain_grad()

    threshold = torch.tensor(
        always_feasible_cfg.required_snr_threshold(demand.bitrate_gbps),
        dtype=torch.float32,
    )
    feasibility_loss = torch.relu(threshold - gsnr_preds[demand.id])
    feasibility_loss.backward()

    assert feasibility_loss.item() == 0.0
    assert pi.grad is not None
    assert pi.grad.abs().sum().item() == 0.0
