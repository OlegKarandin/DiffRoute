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

from multilayer_optical_network.model.optical_topology_import import (
    SSMF_LOSS_COEF_DB_PER_KM, populate_optical,
)
from multilayer_optical_network.model.modes import load_modulation_formats


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
# Test 2: EdgeWeightNet gradient flows via path_noise_loss
# ---------------------------------------------------------------------------

def test_gradient_flow_edge_weight_net():
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    mod_cfg = make_mod_config()

    demands = [Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)]
    path_costs, gsnr_preds, _, regen_probs = pipeline(demands, lambda_=5.0)

    loss, _ = compute_loss(
        gsnr_preds=gsnr_preds,
        path_noise_costs=path_costs,
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
        path_noise_costs=path_costs,
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
        path_noise_costs=path_costs,
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
        path_noise_costs=path_costs,
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
        gsnr_preds=gsnr_preds, path_noise_costs=path_costs, demands=[demand],
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
    proxy term (not just path_noise_loss) is contributing to the signal."""
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
            gsnr_preds=gsnr_preds, path_noise_costs=path_costs, demands=[demand],
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


# ---------------------------------------------------------------------------
# Test 11: QoT calls are batched across demands/segments within one forward()
# ---------------------------------------------------------------------------

def test_qot_model_called_once_per_forward(monkeypatch):
    """pipeline.forward() must invoke the QoT model exactly once per call,
    regardless of how many demands or segments it processes internally.

    Profiling (docs/investigations/open_followups.md #1) found ~920 QoT
    calls/epoch at batch size 1 accounting for 59% of pipeline.forward's
    wall-clock, almost entirely PyTorch per-op dispatch overhead rather than
    arithmetic. Two demands that each cross the hub topology's regen
    candidate (node 3) produce 4 segments total (2 per demand); an
    unbatched implementation calls the QoT model 4 times, a batched one
    exactly once."""
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)

    call_count = 0
    original_forward = SpanAttentionQoT.forward

    def counting_forward(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(SpanAttentionQoT, "forward", counting_forward)

    demands = [
        Demand(id=0, src=0, dst=4, bitrate_gbps=400.0),  # via node 3 -> 2 segments
        Demand(id=1, src=1, dst=4, bitrate_gbps=400.0),  # via node 3 -> 2 segments
    ]
    pipeline(demands)

    assert call_count == 1, (
        f"QoT model forward() called {call_count} times for 2 demands "
        "producing 4 segments total — expected exactly 1 batched call"
    )


# ---------------------------------------------------------------------------
# Test 12: batched QoT output is scattered back to the correct demand/segment
# ---------------------------------------------------------------------------

def test_batched_qot_matches_per_segment_direct_calls():
    """With multiple demands each producing multiple segments, every
    demand's end-to-end GSNR must match what direct (unbatched, no-STE)
    per-segment QoT calls + SegmentCombiner would produce for that same
    demand's segments — i.e. batching must not cross-wire results between
    demands or segments."""
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    temperature = 0.01  # sharp, so soft_max ~= true max

    demands = [
        Demand(id=0, src=0, dst=4, bitrate_gbps=400.0),  # via node 3 -> 2 segments
        Demand(id=1, src=1, dst=4, bitrate_gbps=400.0),  # via node 3 -> 2 segments
    ]
    _, gsnr_preds, path_indicators, regen_probs = pipeline(
        demands, soft_max_temperature=temperature
    )

    for demand in demands:
        indicator = path_indicators[demand.id].detach()
        active_eids = [e for e in range(indicator.shape[0]) if indicator[e].item() > 0.5]

        segments, boundary_nodes = segment_path(
            active_eids, demand.src, pipeline._regen_candidate_set,
            pipeline._edges, demand.dst,
        )

        direct_gsnrs = []
        for seg in segments:
            span_feats, padding_mask = pipeline._extract_span_features(seg, torch.device("cpu"))
            with torch.no_grad():
                direct_gsnrs.append(pipeline.qot_model(span_feats, padding_mask)[0])

        boundary_probs = [regen_probs[n].detach() for n in boundary_nodes]
        expected_gsnr = pipeline.segment_combiner(direct_gsnrs, boundary_probs, temperature=temperature)

        assert abs(gsnr_preds[demand.id].item() - expected_gsnr.item()) < 1e-4, (
            f"demand {demand.id}: batched GSNR {gsnr_preds[demand.id].item():.6f} dB != "
            f"direct per-segment GSNR {expected_gsnr.item():.6f} dB"
        )


# ---------------------------------------------------------------------------
# Test 13: batched QoT padding trims to the batch's true max span count
# ---------------------------------------------------------------------------

def test_qot_batch_trims_padding_to_true_max_spans(monkeypatch):
    """The batched QoT call must pad every segment only up to the widest
    real segment in the current batch, not the architectural max_spans=60
    — trimming the wasted padding columns is what eliminates the ~98%
    masked-out attention compute profiled in
    docs/investigations/open_followups.md #1 (real segments run <=12 spans,
    median ~7, vs the 60-wide architectural ceiling). The hub topology's
    0->4 path crosses regen candidate node 3, splitting into a 2-edge/
    2-span segment (0-1, 1-3 or 0-2, 2-3) and a 1-edge/1-span segment
    (3-4) — batch max is 2, both segments share that one batched call."""
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)

    captured_seq_lens: list[int] = []
    original_forward = SpanAttentionQoT.forward

    def capturing_forward(self, span_features, padding_mask):
        captured_seq_lens.append(span_features.shape[1])
        return original_forward(self, span_features, padding_mask)

    monkeypatch.setattr(SpanAttentionQoT, "forward", capturing_forward)

    demands = [Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)]
    pipeline(demands)

    assert captured_seq_lens == [2], (
        f"expected the single batched QoT call padded to 2 spans (the "
        f"batch's true max, from the 2-edge first segment), got "
        f"seq_len(s) {captured_seq_lens}"
    )


# ---------------------------------------------------------------------------
# Test 14/15: edge-weight scale degeneracy (docs/investigations/
# edge_weight_scale_collapse.md). These tests verify Task 1's renormalisation
# specifically: once EdgeWeightNet's raw output is renormalised to unit mean
# with a live (non-detached) divisor, the loss is homogeneous of degree 0 in
# that raw output for ANY downstream loss, so the scale-direction gradient is
# exactly zero and "shrink every weight" is not a descent direction. This
# property holds regardless of what the path-cost term happens to be
# denominated in — it would still pass even under a regression back to
# Task 2's bug (path_cost_loss reading edge_weights instead of
# edge_ase_noise). These tests therefore CANNOT detect that regression;
# test_path_noise_cost_equals_ase_noise_along_route is the dedicated guard
# for Task 2's property.
# ---------------------------------------------------------------------------

class _ScaledNet(torch.nn.Module):
    """Wraps EdgeWeightNet and multiplies its output by a constant.

    Used to simulate an arbitrarily collapsed (or inflated) weight scale
    without retraining, so the scale-invariance property can be asserted at
    the magnitudes that actually occur in a real run.
    """

    def __init__(self, inner: torch.nn.Module, scale: float) -> None:
        super().__init__()
        self.inner = inner
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x) * self.scale


def test_scale_direction_gradient_is_zero():
    """Euler check: for a degree-0 homogeneous loss, sum_i u_i * dL/du_i == 0
    exactly, where u is EdgeWeightNet's raw pre-normalisation output.

    This is the fix stated as an equation. It is also the .detach() guard:
    detaching the mean in pipeline.forward deletes autograd's correction term
    and this assertion fails immediately.
    """
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    mod_cfg = make_mod_config()

    captured = {}

    def hook(_module, _inputs, output):
        output.retain_grad()
        captured["raw"] = output

    handle = pipeline.edge_weight_net.register_forward_hook(hook)
    demands = [Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)]
    path_costs, gsnr_preds, _, regen_probs = pipeline(demands, lambda_=5.0)
    handle.remove()

    loss, _ = compute_loss(
        gsnr_preds=gsnr_preds,
        path_noise_costs=path_costs,
        demands=demands,
        regen_probs=regen_probs,
        modulation_config=mod_cfg,
    )
    loss.backward()

    raw = captured["raw"]
    assert raw.grad is not None, "raw EdgeWeightNet output received no gradient"

    radial = (raw.detach() * raw.grad).sum().item()
    # Relative tolerance against the magnitude of the terms being summed —
    # the identity is exact, so any residual is float error only.
    term_scale = (raw.detach().abs() * raw.grad.abs()).sum().item()
    assert abs(radial) <= 1e-5 * max(term_scale, 1.0), (
        f"scale-direction gradient is {radial:.6e} (term scale {term_scale:.6e}) "
        "— expected ~0. The loss is not degree-0 in EdgeWeightNet's raw output, "
        "so 'shrink every weight' is still a free descent direction."
    )


def test_total_loss_invariant_to_edge_weight_scale():
    """Scaling EdgeWeightNet's output by any positive constant must leave the
    total loss and every chosen path bit-identical.

    Includes 1e-11 — the magnitude weights actually collapsed to in the
    ind_132 run — applying CLAUDE.md's lesson from correction #8, where the
    original soft_max tests passed only because they never covered the
    production scale.
    """
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    mod_cfg = make_mod_config()
    demands = [Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)]

    def run(scale: float):
        inner = pipeline.edge_weight_net
        pipeline.edge_weight_net = _ScaledNet(inner, scale)
        try:
            path_costs, gsnr_preds, path_indicators, regen_probs = pipeline(
                demands, lambda_=5.0
            )
            loss, _ = compute_loss(
                gsnr_preds=gsnr_preds,
                path_noise_costs=path_costs,
                demands=demands,
                regen_probs=regen_probs,
                modulation_config=mod_cfg,
            )
            return loss.item(), path_indicators[0].detach().clone()
        finally:
            pipeline.edge_weight_net = inner

    loss_1, path_1 = run(1.0)
    loss_big, path_big = run(1000.0)
    # 1e-11 is close to where pipeline.forward's `clamp_min(1e-12)` on the
    # unit-mean divisor engages — one more order of magnitude down and the
    # clamp would trigger, silently breaking scale-invariance with no warning
    # path. Not hypothetical: the pre-fix production run had median raw
    # weights collapse to ~5.7e-11, i.e. within a decade of this floor.
    loss_collapsed, path_collapsed = run(1e-11)

    assert abs(loss_1 - loss_big) < 1e-5, (
        f"loss changed under 1000x weight scaling: {loss_1:.6f} -> {loss_big:.6f}"
    )
    assert abs(loss_1 - loss_collapsed) < 1e-5, (
        f"loss changed under 1e-11 weight scaling: {loss_1:.6f} -> {loss_collapsed:.6f}"
    )
    assert torch.equal(path_1, path_big), "routing changed under 1000x scaling"
    assert torch.equal(path_1, path_collapsed), "routing changed under 1e-11 scaling"


# ---------------------------------------------------------------------------
# Test 16: the path-cost term is denominated in physical ASE noise, not in
# learned edge weights (docs/investigations/edge_weight_scale_collapse.md).
# ---------------------------------------------------------------------------

def test_path_noise_cost_equals_ase_noise_along_route():
    """The path-cost term must be the sum of edge_ase_noise over the chosen
    route. Fixed coefficients make it degree-0 in edge_weights by
    construction, so the term cannot be reduced by deflating the weights.
    """
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    demands = [Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)]

    path_noise_costs, _, path_indicators, _ = pipeline(demands, lambda_=5.0)

    indicator = path_indicators[0].detach()
    expected = (indicator * pipeline._edge_ase_noise).sum()

    assert abs(path_noise_costs[0].item() - expected.item()) < 1e-6, (
        f"path cost {path_noise_costs[0].item():.6f} != accumulated ASE noise "
        f"{expected.item():.6f} along the route"
    )
    # Sanity: a real route has strictly positive accumulated noise.
    assert path_noise_costs[0].item() > 0.0


# ---------------------------------------------------------------------------
# Test 17: EdgeWeightNet's static topology inputs are standardised.
# Unnormalised inputs (length spanning 19-597 km against regen_prob features
# in [0,1]) gave the random init a backwards prior — corr(w, length_km) =
# -0.463 on ind_132, i.e. longer edges priced cheaper.
# ---------------------------------------------------------------------------

def test_topology_edge_features_are_standardised():
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    feats = pipeline._topo_edge_features

    assert feats.shape == (len(topo.undirected_edges), 5)
    assert torch.isfinite(feats).all(), "standardisation produced NaN or inf"

    # On the hub topology, columns 1 (fiber_type_idx), 2 (mean_amp_nf_db) and
    # 3 (num_spans) are constant across edges. A clamped std must map them to
    # exactly 0 rather than dividing by ~0.
    for col in (1, 2, 3):
        assert torch.allclose(feats[:, col], torch.zeros_like(feats[:, col]), atol=1e-6), (
            f"constant column {col} did not map to zero: {feats[:, col]}"
        )

    # Columns 0 (mean_span_length_km) and 4 (total_length_km) vary across the
    # hub topology's 60/100/80 km edges, so they must come out standardised.
    for col in (0, 4):
        assert abs(feats[:, col].mean().item()) < 1e-5, (
            f"column {col} mean is {feats[:, col].mean().item():.6e}, expected ~0"
        )
        assert abs(feats[:, col].std(unbiased=False).item() - 1.0) < 1e-5, (
            f"column {col} std is {feats[:, col].std(unbiased=False).item():.6f}, expected ~1"
        )

    # The statistics themselves are exposed for diagnostics.
    assert pipeline._topo_feat_mean.shape == (1, 5)
    assert pipeline._topo_feat_std.shape == (1, 5)


# ---------------------------------------------------------------------------
# Test 18: weights must not collapse over a real (if short) training loop.
# End-to-end statement of the bug: on ind_132 the median raw weight fell
# ~8.2e7x over 60 epochs while rank-corr with init stayed at +0.999.
# ---------------------------------------------------------------------------

def test_segment_longer_than_max_spans_raises_a_named_error():
    """A route whose segment exceeds max_spans must fail loudly.

    SpanAttentionQoT's positional embedding is nn.Embedding(max_spans); going
    past it otherwise raises a bare IndexError from inside torch.
    """
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    pipeline.max_spans = 1  # every real segment here has more than 1 span

    demands = [Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)]

    with pytest.raises(ValueError, match="max_spans"):
        pipeline(demands, lambda_=5.0)


def test_edge_weights_do_not_collapse_over_training():
    torch.manual_seed(0)
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    mod_cfg = make_mod_config()
    # lr=0.1 / 200 steps *explodes* the raw median weight ~150x even on the
    # healthy, committed pipeline.py on this 5-edge toy fixture -- the
    # fixture is small enough that an aggressive optimizer setting
    # overwhelms any signal from the correction-#9 fix either way. Swept
    # lr in {0.01, 0.02, 0.03, 0.05} x steps in {50, 100, 200}: lr=0.01/50
    # steps is the combination where the healthy pipeline's raw median
    # weight stays close to its starting value (ratio ~1.01, comfortably
    # inside the 2x band below), while reverting correction #9's
    # divisor-detach fix in pipeline.py (`.mean().clamp_min(1e-12)` ->
    # `.mean().clamp_min(1e-12).detach()`) collapses the same measurement
    # by ~20 orders of magnitude (7.1e-01 -> 3.5e-20) in the same 50 steps
    # -- an unambiguous discriminator, not a coin flip. Verified by hand:
    # mutate pipeline.py that one line, `pytest -k collapse` fails with
    # exactly that number, `git checkout -- diffopt/pipeline.py`, passes
    # again.
    optimizer = torch.optim.Adam(pipeline.edge_weight_net.parameters(), lr=0.01)

    demands = [
        Demand(id=0, src=0, dst=4, bitrate_gbps=400.0),
        Demand(id=1, src=1, dst=4, bitrate_gbps=400.0),
    ]

    def median_raw_weight() -> float:
        """Median of EdgeWeightNet's RAW output — the quantity that collapsed.
        Measured pre-normalisation, since the unit-mean division would hide
        any scale drift by construction."""
        with torch.no_grad():
            regen_probs = pipeline.regen_placement.get_regen_probs(1.0)
            feats = torch.cat([
                pipeline._topo_edge_features,
                regen_probs[pipeline._edge_src_ids].unsqueeze(1),
                regen_probs[pipeline._edge_dst_ids].unsqueeze(1),
            ], dim=1)
            return pipeline.edge_weight_net(feats).squeeze(-1).median().item()

    before = median_raw_weight()

    for _ in range(50):
        optimizer.zero_grad()
        path_noise_costs, gsnr_preds, _, regen_probs = pipeline(demands, lambda_=5.0)
        loss, _ = compute_loss(
            gsnr_preds=gsnr_preds,
            path_noise_costs=path_noise_costs,
            demands=demands,
            regen_probs=regen_probs,
            modulation_config=mod_cfg,
        )
        loss.backward()
        optimizer.step()

    after = median_raw_weight()

    assert before > 0.0, "degenerate fixture: initial median weight is zero"
    assert after > before / 2.0, (
        f"median raw edge weight collapsed {before:.6e} -> {after:.6e} "
        f"({before / max(after, 1e-30):.2e}x) over 50 steps at lr=0.01"
    )
    assert after < before * 2.0, (
        f"median raw edge weight exploded {before:.6e} -> {after:.6e}"
    )
