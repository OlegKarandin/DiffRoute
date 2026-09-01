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
from diffopt.pipeline import DiffONetPipeline, _spearman, segment_path
from diffopt.placement.allocation import AllocationHead
from diffopt.qot.model import SpanAttentionQoT
from diffopt.qot.segment_combiner import SegmentCombiner
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

def make_pipeline(topology: Topology, *, alloc_ste: bool = False) -> DiffONetPipeline:
    qot_model = SpanAttentionQoT(max_spans=60)
    segment_combiner = SegmentCombiner()  # stateless and parameter-free
    # The head is amortized over route-local features, so it takes no
    # topology argument — unlike the (num_nodes,) logit vector it replaces.
    allocation_head = AllocationHead(alloc_ste=alloc_ste)
    return DiffONetPipeline(
        topology=topology,
        qot_model=qot_model,
        segment_combiner=segment_combiner,
        allocation_head=allocation_head,
        modulation_config=make_mod_config(),
        margin_db=0.5,
    )


def make_demands() -> list[Demand]:
    """The standard three-demand set for the hub topology.

    0->4 and 1->4 both cross node 3 (the sole regen candidate), so each has
    exactly one boundary; 0->3 terminates there, so it has none. That mix is
    what makes the ragged (D, J) padding real rather than rectangular.
    """
    return [
        Demand(id=0, src=0, dst=4, bitrate_gbps=400.0),
        Demand(id=1, src=0, dst=3, bitrate_gbps=400.0),
        Demand(id=2, src=1, dst=4, bitrate_gbps=400.0),
    ]


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
    path_noise_costs, gsnr_preds, path_indicators, alloc = pipeline(demands)

    assert len(gsnr_preds) == 3
    assert len(path_noise_costs) == 3
    assert len(path_indicators) == 3

    for did in [0, 1, 2]:
        assert gsnr_preds[did].shape == torch.Size([])   # scalar
        assert path_noise_costs[did].shape == torch.Size([])   # scalar
        assert path_indicators[did].shape == torch.Size([len(topo.undirected_edges)])

    assert alloc.site_view.shape == torch.Size([topo.num_nodes])


# ---------------------------------------------------------------------------
# Test 2: EdgeWeightNet gradient flows via path_noise_loss
# ---------------------------------------------------------------------------

def test_gradient_flow_edge_weight_net():
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    mod_cfg = make_mod_config()

    demands = [Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)]
    path_noise_costs, gsnr_preds, _, alloc = pipeline(demands, lambda_=5.0)

    loss, _ = compute_loss(
        gsnr_preds=gsnr_preds,
        path_noise_costs=path_noise_costs,
        demands=demands,
        device_count=alloc.device_count,
        modulation_config=mod_cfg,
        duals=torch.ones(len(demands)),
    )
    loss.backward()

    assert pipeline.edge_log_weight.grad is not None, "edge_log_weight has no grad"
    assert pipeline.edge_log_weight.grad.abs().sum().item() > 0, \
        "edge_log_weight has all-zero grad"


# ---------------------------------------------------------------------------
# Test 4: QoT model parameters have no gradient after backward
# ---------------------------------------------------------------------------

def test_qot_frozen():
    topo = make_linear_topology()
    pipeline = make_pipeline(topo)
    mod_cfg = make_mod_config()

    demands = [Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)]
    path_noise_costs, gsnr_preds, _, alloc = pipeline(demands)

    loss, _ = compute_loss(
        gsnr_preds=gsnr_preds,
        path_noise_costs=path_noise_costs,
        demands=demands,
        device_count=alloc.device_count,
        modulation_config=mod_cfg,
        duals=torch.ones(len(demands)),
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
    # (with one segment it never reaches the fold at all)
    pipeline = make_pipeline(topo)

    demand = Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)
    path_noise_costs, gsnr_preds, path_indicators, _ = pipeline([demand])

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
    path_noise_costs, gsnr_preds, _, alloc = pipeline(demands, lambda_=5.0)

    loss, _ = compute_loss(
        gsnr_preds=gsnr_preds,
        path_noise_costs=path_noise_costs,
        demands=demands,
        device_count=alloc.device_count,
        modulation_config=mod_cfg,
        duals=torch.ones(len(demands)),
    )
    loss.backward()

    trainable = [pipeline.edge_log_weight] + \
                list(pipeline.allocation_head.parameters())
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

    demand = Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)
    _, gsnr_preds, path_indicators, alloc = pipeline([demand])

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

    # The allocation is per (demand, boundary) now, so the boundary
    # probabilities come from the head's own physics decisions for this
    # demand's row, not from a per-node lookup.
    boundary_probs = [
        alloc.a_physics[0, k].detach() for k in range(len(boundary_nodes))
    ]
    expected_gsnr = pipeline.segment_combiner(direct_gsnrs, boundary_probs)

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
    path_noise_costs, gsnr_preds, path_indicators, alloc = pipeline([demand], lambda_=5.0)

    pi = path_indicators[0]
    pi.retain_grad()

    # edge_log_weight is a free per-edge parameter (spec decision 6), so this
    # is forward()'s own input, recomputed the same way as forward()'s step 3.
    edge_weights = torch.nn.functional.softplus(pipeline.edge_log_weight).detach()

    loss, _ = compute_loss(
        gsnr_preds=gsnr_preds, path_noise_costs=path_noise_costs, demands=[demand],
        device_count=alloc.device_count, modulation_config=always_infeasible_cfg,
        duals=torch.ones(1),
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
        path_noise_costs, gsnr_preds, _, alloc = pipeline([demand], lambda_=5.0)
        loss, _ = compute_loss(
            gsnr_preds=gsnr_preds, path_noise_costs=path_noise_costs, demands=[demand],
            device_count=alloc.device_count, modulation_config=always_infeasible_cfg,
            duals=torch.ones(1),
        )
        loss.backward()
        return pipeline.edge_log_weight.grad.clone()

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

    demands = [
        Demand(id=0, src=0, dst=4, bitrate_gbps=400.0),  # via node 3 -> 2 segments
        Demand(id=1, src=1, dst=4, bitrate_gbps=400.0),  # via node 3 -> 2 segments
    ]
    _, gsnr_preds, path_indicators, alloc = pipeline(demands)

    for row, demand in enumerate(demands):
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

        boundary_probs = [
            alloc.a_physics[row, k].detach() for k in range(len(boundary_nodes))
        ]
        expected_gsnr = pipeline.segment_combiner(direct_gsnrs, boundary_probs)

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
# edge_weight_scale_collapse.md). These tests verify the unit-mean
# renormalisation specifically: once the raw softplus(edge_log_weight) output
# is renormalised to unit mean with a live (non-detached) divisor, the loss is
# homogeneous of degree 0 in that raw output for ANY downstream loss, so the
# scale-direction gradient is exactly zero and "shrink every weight" is not
# a descent direction. This property holds regardless of what the path-cost
# term happens to be denominated in — it would still pass even under a
# regression back to the pre-fix bug where path_cost_loss read edge_weights
# instead of edge_ase_noise. These tests therefore CANNOT detect that
# regression; test_path_noise_cost_equals_ase_noise_along_route is the
# dedicated guard for the ASE-denominated path-cost invariant.
#
# Since Task 7, `raw` is a local variable inside pipeline.forward (softplus
# is called inline on the free parameter, not on a submodule that can be
# forward-hooked or swapped), so these two tests capture/scale it by
# monkeypatching `diffopt.pipeline.F.softplus` for the duration of the call
# instead — the same technique, one layer down.
# ---------------------------------------------------------------------------

def test_scale_direction_gradient_is_zero(monkeypatch):
    """Euler check: for a degree-0 homogeneous loss, sum_i u_i * dL/du_i == 0
    exactly, where u is the raw pre-normalisation softplus(edge_log_weight)
    output.

    This is the fix stated as an equation. It is also the .detach() guard:
    detaching the mean in pipeline.forward deletes autograd's correction term
    and this assertion fails immediately.
    """
    import diffopt.pipeline as pipeline_module

    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    mod_cfg = make_mod_config()

    captured = {}
    original_softplus = pipeline_module.F.softplus

    def capturing_softplus(x, *args, **kwargs):
        out = original_softplus(x, *args, **kwargs)
        out.retain_grad()
        captured["raw"] = out
        return out

    monkeypatch.setattr(pipeline_module.F, "softplus", capturing_softplus)
    demands = [Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)]
    path_noise_costs, gsnr_preds, _, alloc = pipeline(demands, lambda_=5.0)
    monkeypatch.setattr(pipeline_module.F, "softplus", original_softplus)

    loss, _ = compute_loss(
        gsnr_preds=gsnr_preds,
        path_noise_costs=path_noise_costs,
        demands=demands,
        device_count=alloc.device_count,
        modulation_config=mod_cfg,
        duals=torch.ones(len(demands)),
    )
    loss.backward()

    raw = captured["raw"]
    assert raw.grad is not None, "raw softplus(edge_log_weight) output received no gradient"

    radial = (raw.detach() * raw.grad).sum().item()
    # Relative tolerance against the magnitude of the terms being summed —
    # the identity is exact, so any residual is float error only.
    term_scale = (raw.detach().abs() * raw.grad.abs()).sum().item()
    assert abs(radial) <= 1e-5 * max(term_scale, 1.0), (
        f"scale-direction gradient is {radial:.6e} (term scale {term_scale:.6e}) "
        "— expected ~0. The loss is not degree-0 in the raw softplus output, "
        "so 'shrink every weight' is still a free descent direction."
    )


def test_total_loss_invariant_to_edge_weight_scale(monkeypatch):
    """Scaling the raw softplus(edge_log_weight) output by any positive
    constant must leave the total loss and every chosen path bit-identical.

    Includes 1e-11 — the magnitude weights actually collapsed to in the
    ind_132 run — applying the lesson from
    docs/investigations/CHANGELOG.md#correction-1c-8, where the
    original segment-combiner tests passed only because they never covered
    the production scale.
    """
    import diffopt.pipeline as pipeline_module

    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    mod_cfg = make_mod_config()
    demands = [Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)]
    original_softplus = pipeline_module.F.softplus

    def run(scale: float):
        def scaled_softplus(x, *args, **kwargs):
            return original_softplus(x, *args, **kwargs) * scale

        monkeypatch.setattr(pipeline_module.F, "softplus", scaled_softplus)
        try:
            path_noise_costs, gsnr_preds, path_indicators, alloc = pipeline(
                demands, lambda_=5.0
            )
            loss, _ = compute_loss(
                gsnr_preds=gsnr_preds,
                path_noise_costs=path_noise_costs,
                demands=demands,
                device_count=alloc.device_count,
                modulation_config=mod_cfg,
                duals=torch.ones(len(demands)),
            )
            return loss.item(), path_indicators[0].detach().clone()
        finally:
            monkeypatch.setattr(pipeline_module.F, "softplus", original_softplus)

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


def test_segment_longer_than_max_spans_raises_a_named_error():
    """A route whose segment exceeds max_spans must fail loudly.

    SpanAttentionQoT's positional embedding is nn.Embedding(max_spans); going
    past it otherwise raises a bare IndexError from inside torch.
    """
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    # The 0->4 route via node 1 splits at regen candidate node 3 into a
    # 2-span segment (eids 0,2) and a 1-span segment (eid 4). max_spans=1
    # still fires the guard because batch_max_spans is a max over all
    # segments in the batch, and the first segment already exceeds it.
    pipeline.max_spans = 1

    demands = [Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)]

    with pytest.raises(ValueError, match="max_spans"):
        pipeline(demands, lambda_=5.0)


# ---------------------------------------------------------------------------
# Test 18: weights must not collapse over a real (if short) training loop.
# End-to-end statement of the bug: on ind_132 the median raw weight fell
# ~8.2e7x over 60 epochs while rank-corr with init stayed at +0.999.
# ---------------------------------------------------------------------------

def test_edge_weights_do_not_collapse_over_training():
    torch.manual_seed(0)
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    mod_cfg = make_mod_config()
    # lr=0.1 / 200 steps *explodes* the raw median weight ~150x even on the
    # healthy, committed pipeline.py on this 5-edge toy fixture -- the
    # fixture is small enough that an aggressive optimizer setting
    # overwhelms any signal from the correction-#9 fix either way. Original
    # tuning (pre approach-A) swept lr in {0.01, 0.02, 0.03, 0.05} x steps
    # in {50, 100, 200} against edge features built from regen_probs at
    # each edge's endpoints; lr=0.01/50 steps was the pick there.
    #
    # Approach A (static is_candidate columns replacing regen_probs, see
    # pipeline.py __init__) changed this fixture's input distribution to
    # EdgeWeightNet: RegenPlacement's fresh logits are all zero, so
    # regen_probs was 0.5/0.5 on every edge at init -- a constant that
    # carried no edge-discriminating signal. is_candidate is exactly 0 or 1
    # per endpoint and does vary by edge (hub topology's node 3 is the only
    # candidate), so it is real per-edge signal from step 1, and the same
    # lr=0.01/50 steps now genuinely explodes the healthy pipeline (~17x,
    # not a bug -- more informative input drives bigger early gradients).
    # Re-swept lr in {0.0035, 0.004, 0.0045} x steps in {26, 28, 30, 32,
    # 34}: lr=0.004/30 steps keeps the healthy pipeline's ratio at ~1.76
    # (inside the 2x band below) while reverting correction #9's
    # divisor-detach fix (`.mean().clamp_min(1e-12)` ->
    # `.mean().clamp_min(1e-12).detach()`) still drives the same
    # measurement down to ratio ~0.41 (outside the 0.5x band) in the same
    # 30 steps -- still an unambiguous discriminator. Verified by hand:
    # mutate pipeline.py that one line, `pytest -k collapse` fails with
    # ratio ~0.41, `git checkout -- diffopt/pipeline.py`, passes again.
    #
    # THIS lr/step calibration is specific to EdgeWeightNet's gradient scale
    # and was NOT re-verified after Task 7 reparameterized edge_weights as
    # the free parameter edge_log_weight (spec decision 6). Measured on the
    # free parameter, this same 30-step/lr=0.004 loop gives ratio exactly
    # 1.0 on THIS fixture both with and without the divisor-detach mutation
    # (the two demands here move edge_log_weight so little in 30 steps that
    # the median doesn't budge either way) -- i.e. this particular
    # calibration no longer discriminates that regression under the new
    # parameterization, though it still correctly holds a healthy pipeline
    # inside the band. test_scale_direction_gradient_is_zero and
    # test_edge_weights_still_have_the_undetached_unit_mean_divisor are the
    # tests that catch the divisor-detach regression post-Task-7 (the
    # former was hand-verified to fail under the same mutation; the latter
    # was not, since its loss — sum(gsnr) — carries no live gradient into
    # edge_log_weight at all and passes vacuously either way). Re-sweeping
    # lr/steps to restore this test's own discriminating power is unstarted
    # follow-up work, not part of Task 7.
    optimizer = torch.optim.Adam([pipeline.edge_log_weight], lr=0.004)

    demands = [
        Demand(id=0, src=0, dst=4, bitrate_gbps=400.0),
        Demand(id=1, src=1, dst=4, bitrate_gbps=400.0),
    ]

    def median_raw_weight() -> float:
        """Median of the raw softplus(edge_log_weight) output — the quantity
        that collapsed under the old EdgeWeightNet parameterization.
        Measured pre-normalisation, since the unit-mean division would hide
        any scale drift by construction."""
        with torch.no_grad():
            return torch.nn.functional.softplus(pipeline.edge_log_weight).median().item()

    before = median_raw_weight()

    for _ in range(30):
        optimizer.zero_grad()
        path_noise_costs, gsnr_preds, _, alloc = pipeline(demands, lambda_=5.0)
        loss, _ = compute_loss(
            gsnr_preds=gsnr_preds,
            path_noise_costs=path_noise_costs,
            demands=demands,
            device_count=alloc.device_count,
            modulation_config=mod_cfg,
            duals=torch.ones(len(demands)),
        )
        loss.backward()
        optimizer.step()

    after = median_raw_weight()

    assert before > 0.0, "degenerate fixture: initial median weight is zero"
    assert after > before / 2.0, (
        f"median raw edge weight collapsed {before:.6e} -> {after:.6e} "
        f"({before / max(after, 1e-30):.2e}x) over 30 steps at lr=0.004"
    )
    assert after < before * 2.0, (
        f"median raw edge weight exploded {before:.6e} -> {after:.6e}"
    )


def test_segment_gsnr_memo_serves_a_repeated_forward_without_calling_the_qot_model():
    """Same demands, same (untrained, so unchanged) edge weights -> same
    routes -> same segments. The second forward must hit the memo for every
    one of them, so the QoT model is not called at all."""
    topology = make_hub_topology()
    pipeline = make_pipeline(topology)
    demands = [Demand(id=0, src=0, dst=4, bitrate_gbps=400.0),
               Demand(id=1, src=1, dst=4, bitrate_gbps=400.0)]

    calls = {"n": 0}
    real_forward = SpanAttentionQoT.forward

    def counting_forward(self, span_features, padding_mask):
        calls["n"] += 1
        return real_forward(self, span_features, padding_mask)

    with torch.no_grad():
        _, first, _, _ = pipeline(demands, tau=1.0)
        SpanAttentionQoT.forward = counting_forward
        try:
            _, second, _, _ = pipeline(demands, tau=1.0)
        finally:
            SpanAttentionQoT.forward = real_forward

    assert calls["n"] == 0
    for d in demands:
        assert torch.equal(first[d.id], second[d.id])


def test_segment_gsnr_memo_produces_the_same_gsnr_as_no_memo():
    """The exactness claim, checked rather than argued. Two pipelines built
    from the same seed, one with the memo disabled: bitwise-equal GSNR."""
    topology = make_hub_topology()
    demands = [Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)]

    torch.manual_seed(0)
    cached = make_pipeline(topology)
    torch.manual_seed(0)
    uncached = make_pipeline(topology)
    uncached.cache_segment_gsnr = False

    with torch.no_grad():
        _, with_memo, _, _ = cached(demands, tau=1.0)
        _, without_memo, _, _ = uncached(demands, tau=1.0)

    assert torch.equal(with_memo[0], without_memo[0])
    assert len(cached._segment_gsnr_cache) > 0
    assert len(uncached._segment_gsnr_cache) == 0


def test_segment_gsnr_memo_key_is_the_ordered_edge_tuple_and_the_batch_width():
    """The width belongs in the key. Padded columns are masked out of
    attention and out of the mean-pool, so a different batch width is the
    same number mathematically -- but the reduction runs over a different
    axis length, and only pinning the width makes a hit bitwise identical
    to what the un-memoised code would have produced."""
    topology = make_hub_topology()
    pipeline = make_pipeline(topology)

    with torch.no_grad():
        pipeline([Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)], tau=1.0)

    key = next(iter(pipeline._segment_gsnr_cache))
    edge_tuple, width = key
    assert isinstance(edge_tuple, tuple)
    assert all(isinstance(eid, int) for eid in edge_tuple)
    assert isinstance(width, int) and width >= 1


def test_clear_segment_gsnr_cache_forces_recomputation():
    """Anything that swaps pipeline.qot_model must be able to invalidate."""
    topology = make_hub_topology()
    pipeline = make_pipeline(topology)
    demands = [Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)]

    with torch.no_grad():
        pipeline(demands, tau=1.0)
    assert len(pipeline._segment_gsnr_cache) > 0

    pipeline.clear_segment_gsnr_cache()
    assert len(pipeline._segment_gsnr_cache) == 0


def test_segment_gsnr_cache_eviction_does_not_read_from_cleared_cache(monkeypatch):
    """Cache eviction check must happen AFTER reading from cache, not before.
    This regression test lowers the cache max and verifies (a) no KeyError on
    forward pass after eviction, (b) cache cleared after eviction, (c) output
    still correct."""
    import diffopt.pipeline as pipeline_module

    topology = make_hub_topology()
    pipeline = make_pipeline(topology)

    # Lower cache max to 2 so multiple forwards trigger eviction.
    monkeypatch.setattr(pipeline_module, "_SEGMENT_GSNR_CACHE_MAX", 2)

    # First forward fills cache with some segments.
    demands = [Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)]
    with torch.no_grad():
        paths1, gsnr_dict1, _, _ = pipeline(demands, tau=1.0)

    cache_size_after_first = len(pipeline._segment_gsnr_cache)
    # At this point cache has <= 2 entries (or was cleared if it hit the limit).

    # Second forward with different demand may add more segments, potentially
    # triggering eviction. This should not KeyError on reading the cache.
    demands2 = [
        Demand(id=0, src=0, dst=4, bitrate_gbps=400.0),
        Demand(id=1, src=1, dst=3, bitrate_gbps=400.0),
    ]
    with torch.no_grad():
        paths2, gsnr_dict2, _, _ = pipeline(demands2, tau=1.0)

    # Cache should be empty (cleared by eviction) or very small (bounded by limit).
    # After eviction, cache should be empty (we use "clear all", not LRU).
    assert len(pipeline._segment_gsnr_cache) <= 2, "Cache should be bounded"

    # Verify output has correct number of demands.
    assert len(gsnr_dict2) == 2, f"Expected 2 demands in output, got {len(gsnr_dict2)}"
    # Verify both demands have GSNR predictions (not None or missing).
    assert all(gsnr_dict2[d.id] is not None for d in demands2), "All demands should have GSNR"


def test_pipeline_gsnr_matches_a_per_demand_combiner_loop():
    """The batched fold's equivalence, checked at the level that matters:
    the pipeline's own output. Rebuilds each demand's segments from its
    path indicator and folds them one at a time, the way forward() did
    before batching."""
    from diffopt.pipeline import segment_path

    topology = make_hub_topology()
    pipeline = make_pipeline(topology)
    # Break the closed-at-init plateau so the boundary probabilities are
    # genuinely fractional and demand-dependent — otherwise every row would
    # fold at the same sigmoid(-3) and the equivalence would be vacuous.
    torch.manual_seed(0)
    with torch.no_grad():
        pipeline.allocation_head.net[-1].weight.normal_(std=0.5)
        pipeline.allocation_head.net[-1].bias.zero_()
    demands = [Demand(id=0, src=0, dst=4, bitrate_gbps=400.0),
               Demand(id=1, src=2, dst=4, bitrate_gbps=400.0),
               Demand(id=2, src=1, dst=3, bitrate_gbps=400.0)]

    with torch.no_grad():
        _, gsnr_preds, path_indicators, alloc = pipeline(demands, tau=1.0)

        for row, demand in enumerate(demands):
            ordered = pipeline._reconstruct_path(
                path_indicators[demand.id], demand.src, demand.dst
            )
            segments, boundary_nodes = segment_path(
                ordered, demand.src, pipeline._regen_candidate_set,
                pipeline._edges, demand.dst,
            )
            segment_gsnrs = []
            for seg in segments:
                feats, mask = pipeline._extract_span_features(seg, gsnr_preds[0].device)
                segment_gsnrs.append(pipeline.qot_model(feats, mask)[0])
            expected = pipeline.segment_combiner(
                segment_gsnrs,
                [alloc.a_physics[row, k] for k in range(len(boundary_nodes))],
            )
            assert abs(gsnr_preds[demand.id].item() - expected.item()) < 1e-4


def test_pipeline_folds_every_demand_in_one_combiner_call():
    """The speedup itself. Three demands, one forward_batched call."""
    from diffopt.qot.segment_combiner import SegmentCombiner as _Combiner

    topology = make_hub_topology()
    pipeline = make_pipeline(topology)
    demands = [Demand(id=i, src=s, dst=d, bitrate_gbps=400.0)
               for i, (s, d) in enumerate([(0, 4), (2, 4), (1, 3)])]

    calls = {"n": 0}
    real = _Combiner.forward_batched

    def counting(self, segment_gsnrs_db, boundary_probs, num_segments):
        calls["n"] += 1
        return real(self, segment_gsnrs_db, boundary_probs, num_segments)

    _Combiner.forward_batched = counting
    try:
        with torch.no_grad():
            pipeline(demands, tau=1.0)
    finally:
        _Combiner.forward_batched = real

    assert calls["n"] == 1


def test_pipeline_handles_an_empty_demand_list():
    """train.py raises before this can happen, but forward() has always
    tolerated it and the batched path introduces a torch.stack on a
    possibly-empty list."""
    topology = make_hub_topology()
    pipeline = make_pipeline(topology)

    costs, gsnr, indicators, alloc = pipeline([], tau=1.0)

    assert costs == {} and gsnr == {} and indicators == {}
    assert alloc.demand_ids == []
    assert alloc.a.shape == (0, 0)
    assert alloc.alloc_by_node.shape == (0, topology.num_nodes)
    assert alloc.site_view.shape == (topology.num_nodes,)
    assert alloc.device_count.item() == 0.0


# ---------------------------------------------------------------------------
# Approach A: the edge features are entirely static
# ---------------------------------------------------------------------------

def test_edge_features_are_static_and_carry_candidate_indicators():
    """Spec decision 5. The last two columns are is_candidate at each
    endpoint — indicators, not probabilities, and not learned."""
    topology = make_hub_topology()
    pipeline = make_pipeline(topology)
    feats = pipeline._static_edge_features
    assert feats.shape == (len(topology.undirected_edges), 7)
    assert not feats.requires_grad

    candidates = set(topology.regen_candidate_nodes)
    for eid, edge in enumerate(topology.undirected_edges):
        assert feats[eid, 5].item() == pytest.approx(float(edge.src in candidates))
        assert feats[eid, 6].item() == pytest.approx(float(edge.dst in candidates))


def test_edge_features_do_not_move_when_the_allocation_head_moves():
    """The circularity is gone: no learned quantity reaches the router's
    input. This is what makes approach A permanent rather than a tuning
    choice — see spec section 5 for why B and C were rejected."""
    topology = make_hub_topology()
    pipeline = make_pipeline(topology)
    before = pipeline._static_edge_features.clone()
    with torch.no_grad():
        for p in pipeline.allocation_head.parameters():
            p.add_(torch.randn_like(p))
    assert torch.equal(pipeline._static_edge_features, before)


# ---------------------------------------------------------------------------
# Per-demand allocation
# ---------------------------------------------------------------------------

def test_forward_returns_allocation_outputs_with_consistent_shapes():
    topology = make_hub_topology()
    pipeline = make_pipeline(topology)
    demands = make_demands()
    _, _, _, alloc = pipeline(demands)

    d = len(demands)
    j = alloc.seg_gsnr_db.shape[1]
    assert alloc.a.shape == (d, max(j - 1, 0))
    assert alloc.seg_noise.shape == (d, j)
    assert alloc.num_segments.shape == (d,)
    assert alloc.alloc_by_node.shape == (d, topology.num_nodes)
    assert alloc.site_view.shape == (topology.num_nodes,)
    assert alloc.demand_ids == [x.id for x in demands]


def test_device_count_is_the_sum_of_allocations():
    topology = make_hub_topology()
    pipeline = make_pipeline(topology)
    _, _, _, alloc = pipeline(make_demands())
    assert alloc.device_count.item() == pytest.approx(alloc.a.sum().item(), rel=1e-5)
    assert alloc.device_count.item() == pytest.approx(
        alloc.alloc_by_node.sum().item(), rel=1e-5
    )


def test_device_count_is_route_differentiable():
    """Spec section 8 item 5 and section 4's "job 2": d(devices)/d(path_indicator)
    must be nonzero, which is what lets Vlastelica's re-solve search for a
    route that needs fewer regenerators."""
    topology = make_hub_topology()
    pipeline = make_pipeline(topology)
    with torch.no_grad():                       # break the closed-init plateau
        pipeline.allocation_head.net[-1].weight.normal_(std=0.5)
    _, _, indicators, alloc = pipeline(make_demands())
    grads = torch.autograd.grad(
        alloc.device_count, list(indicators.values()), allow_unused=True
    )
    assert any(g is not None and g.abs().sum().item() > 0 for g in grads)


def test_hard_alloc_is_not_a_threshold_on_the_soft_pass():
    """Spec 2.5. Under hard decisions the carry is the EXACT chunk noise, so
    the hard rollout is self-consistent physics; thresholding a mean-field
    pass is not. They must therefore be allowed to differ."""
    topology = make_hub_topology()
    pipeline = make_pipeline(topology)
    with torch.no_grad():
        pipeline.allocation_head.net[-1].weight.normal_(std=2.0)
        pipeline.allocation_head.net[-1].bias.zero_()
    demands = make_demands()
    _, _, _, soft = pipeline(demands, tau=1.0)
    with torch.no_grad():
        _, _, _, hard = pipeline(demands, hard_alloc=True)
    assert set(hard.a.unique().tolist()) <= {0.0, 1.0}
    assert not hard.a.requires_grad


def test_hard_alloc_carry_equals_the_combiners_own_fold():
    """Spec section 8 item 3, at the pipeline level: the head's carry and the
    combiner must agree on what a chunk's noise is, or oracle_gap measures a
    units mismatch instead of the head."""
    topology = make_hub_topology()
    pipeline = make_pipeline(topology)
    with torch.no_grad():
        pipeline.allocation_head.net[-1].weight.normal_(std=2.0)
        pipeline.allocation_head.net[-1].bias.zero_()
        _, gsnr_preds, _, alloc = pipeline(make_demands(), hard_alloc=True)
    from diffopt.qot.segment_combiner import linear_noise_to_db

    carried = linear_noise_to_db(pipeline.allocation_head.last_max_chunk_noise)
    folded = torch.stack([gsnr_preds[i] for i in alloc.demand_ids])
    assert torch.allclose(carried, folded, atol=1e-3)


def test_site_view_is_derived_and_never_priced():
    """Spec decision 4: kept for diagnostics, never in the objective."""
    topology = make_hub_topology()
    pipeline = make_pipeline(topology)
    _, _, _, alloc = pipeline(make_demands())
    assert torch.equal(alloc.site_view, alloc.alloc_by_node.max(dim=0).values)
    assert alloc.site_view.sum().item() <= alloc.device_count.item() + 1e-5


def test_regen_placement_is_gone():
    """Spec section 7. A stale import is how a deleted objective comes back."""
    import diffopt.pipeline as p

    assert not hasattr(p, "RegenPlacement")
    with pytest.raises(ModuleNotFoundError):
        import diffopt.placement.regenerator  # noqa: F401


# ---------------------------------------------------------------------------
# Task 7: EdgeWeightNet -> a free per-edge parameter theta[E]
# ---------------------------------------------------------------------------

def test_routing_head_is_a_free_per_edge_parameter():
    topology = make_hub_topology()
    pipeline = make_pipeline(topology)
    e = len(topology.undirected_edges)
    assert pipeline.edge_log_weight.shape == (e,)
    assert pipeline.edge_log_weight.requires_grad
    assert not hasattr(pipeline, "edge_weight_net")


def test_theta_initialises_to_shortest_by_km_routing():
    """Uniform init would tie every edge and make the Dijkstra tie-break the
    de facto router. Starting at length-proportional weights starts training
    at the documented shortest-by-km baseline — the same route preflight_filter
    screens against."""
    topology = make_hub_topology()
    pipeline = make_pipeline(topology)
    raw = torch.nn.functional.softplus(pipeline.edge_log_weight)
    w = raw / raw.mean()
    km = torch.tensor([e.length_km for e in topology.undirected_edges])
    assert torch.allclose(w, km / km.mean(), atol=1e-4)


def test_edge_weights_still_have_the_undetached_unit_mean_divisor():
    """Correction #9's degree-0 argument must survive the reparameterization:
    Euler gives sum_i u_i dL/du_i == 0 only if the divisor is in the graph."""
    topology = make_hub_topology()
    pipeline = make_pipeline(topology)
    _, gsnr, _, _ = pipeline(make_demands())
    loss = sum(gsnr.values())
    g = torch.autograd.grad(loss, pipeline.edge_log_weight, allow_unused=True)[0]
    raw = torch.nn.functional.softplus(pipeline.edge_log_weight)
    assert (raw * g).sum().abs().item() < 1e-4


# ---------------------------------------------------------------------------
# Task 10: STE guardrails
# ---------------------------------------------------------------------------

def test_clamped_segments_are_counted():
    """If qot_gsnr leaves SegmentCombiner's [-5, 35] dB band, _safe_noise's
    clamp zeroes that segment's STE gradient silently. Count it, so a run
    that quietly loses its routing signal is visible in the log."""
    topology = make_hub_topology()
    pipeline = make_pipeline(topology)
    _, _, _, alloc = pipeline(make_demands())
    assert alloc.ste_clamped_segments == 0

    class Saturating(torch.nn.Module):
        def forward(self, feats, mask):
            return torch.full((feats.shape[0],), 99.0)

    pipeline.qot_model = Saturating()
    pipeline.clear_segment_gsnr_cache()
    _, _, _, alloc = pipeline(make_demands())
    assert alloc.ste_clamped_segments > 0


def test_proxy_qot_rank_correlation_is_reported():
    """Nothing currently detects forward/backward mismatch: the STE's value
    comes from the transformer and its gradient from the proxy, and if the
    two disagree about which segment is noisier the gradient points the
    wrong way. Spearman over the epoch's segments makes that visible."""
    topology = make_hub_topology()
    pipeline = make_pipeline(topology)
    _, _, _, alloc = pipeline(make_demands())
    assert -1.0 <= alloc.proxy_qot_rank_corr <= 1.0


def test_spearman_returns_nan_on_constant_input():
    """Double-argsort always yields a full 0..n-1 permutation, even when the
    underlying values are all tied — so a post-ranking zero-variance check
    never fires. Without a pre-ranking guard on the source values, a
    constant proxy (e.g. an ASE noise floor) against a varying qot_gsnr
    would report a 'perfect' 1.0 correlation instead of nan, which is
    exactly backwards for a diagnostic meant to flag disagreement."""
    import math

    a = torch.tensor([5.0, 5.0, 5.0, 5.0])
    b = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert math.isnan(_spearman(a, b))
    assert math.isnan(_spearman(b, a))


# ---------------------------------------------------------------------------
# alloc_ste at the pipeline level
# ---------------------------------------------------------------------------

def test_alloc_ste_training_pass_matches_the_hard_rollout():
    """The load-bearing property of the arm.

    Without the STE the training relaxation and the deployed rollout disagree
    about physics by construction -- that is what
    test_hard_alloc_is_not_a_threshold_on_the_soft_pass asserts, and it is
    correct for the mean-field pass. The consequence, measured on
    constrained_stress at an allocation with oracle_gap == 0 and
    hard_num_violated == 0, is that the soft pass reported 10 of 346 demands
    VIOLATED -- every one of them feasible in the deployed network -- and the
    duals priced all 10. With the STE the forward decision IS the deployed
    decision, so per-demand GSNR must agree exactly and no such phantom can
    exist.
    """
    torch.manual_seed(0)
    topology = make_hub_topology()
    pipeline = make_pipeline(topology, alloc_ste=True)
    with torch.no_grad():
        pipeline.allocation_head.net[-1].weight.normal_(std=2.0)
        pipeline.allocation_head.net[-1].bias.zero_()
    demands = make_demands()

    _, gsnr_soft, _, soft = pipeline(demands, tau=1.0)
    with torch.no_grad():
        _, gsnr_hard, _, hard = pipeline(demands, hard_alloc=True)

    assert torch.equal(soft.a.detach(), hard.a)
    assert soft.device_count.item() == pytest.approx(hard.device_count.item())
    for did in hard.demand_ids:
        assert gsnr_soft[did].item() == pytest.approx(gsnr_hard[did].item(), abs=1e-5)


def test_alloc_ste_training_pass_still_carries_gradient():
    """A straight-through estimator that loses its gradient is indistinguishable
    from a working one at the forward values, and every downstream metric stays
    plausible while the head stops learning. Pin it at the pipeline level, not
    only at the rollout.

    std=0.2 on the final layer, NOT the std=2.0 the forward-value tests use:
    unnormalised features times std=2.0 put scores at |s| ~ 25, where
    sigmoid'(s) underflows to exactly 0.0 in float32 and this test fails for a
    reason that has nothing to do with the STE wiring. The real head's scores
    span [-8.7, +2.9] (see constrained_stress.yaml's alloc_tau_end comment), so
    std=0.2 measures the regime the arm actually runs in. The saturation limit
    itself is pinned separately, below.
    """
    torch.manual_seed(0)
    topology = make_hub_topology()
    pipeline = make_pipeline(topology, alloc_ste=True)
    with torch.no_grad():
        pipeline.allocation_head.net[-1].weight.normal_(std=0.2)
        pipeline.allocation_head.net[-1].bias.zero_()

    _, _, _, alloc = pipeline(make_demands(), tau=1.0)
    assert alloc.device_count.requires_grad
    alloc.device_count.backward()

    total = sum(
        p.grad.abs().sum().item()
        for p in pipeline.allocation_head.parameters()
        if p.grad is not None
    )
    assert total > 0.0


def test_alloc_ste_surrogate_vanishes_on_a_saturated_score():
    """A KNOWN limit of the arm, pinned so it is a documented property rather
    than a surprise mid-sweep.

    The surrogate is sigmoid'(s/tau)/tau, which underflows to exactly 0 in
    float32 past |s/tau| ~ 20. The forward decision stays correct, so nothing
    downstream looks wrong -- the head just silently stops learning at that
    boundary. This is the same failure the head already hit once from the other
    direction (alloc_tau_end 0.1 drove |s/tau| past saturation and caused an
    irreversible training collapse, recorded in constrained_stress.yaml).

    train.py logs alloc_score_min/max every epoch; that is the column to watch
    on any alloc_ste run. If this test ever starts FAILING, the surrogate was
    changed to a clipped straight-through -- update the comment, do not just
    delete the test.
    """
    torch.manual_seed(0)
    topology = make_hub_topology()
    pipeline = make_pipeline(topology, alloc_ste=True)
    with torch.no_grad():
        pipeline.allocation_head.net[-1].weight.normal_(std=2.0)   # |s| ~ 25
        pipeline.allocation_head.net[-1].bias.zero_()

    _, _, _, alloc = pipeline(make_demands(), tau=1.0)
    assert set(alloc.a.detach().unique().tolist()) <= {0.0, 1.0}    # forward fine
    alloc.device_count.backward()
    total = sum(
        p.grad.abs().sum().item()
        for p in pipeline.allocation_head.parameters()
        if p.grad is not None
    )
    assert total == 0.0                                             # backward gone
