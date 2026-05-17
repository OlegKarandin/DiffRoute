"""
Tests for the end-to-end DiffONet pipeline (Phase 1c).

Two in-memory topologies:

  Linear: 0—1—2—3—4  (4 edges, all degree ≤ 2, no regen candidates)
          Useful for single-segment identity test.

  Hub:    0—1—3—4   and   0—2—3—4   (5 edges, node 3 has degree 3)
          Node 3 is the sole regen candidate.
          Two paths 0→4 enable routing choice (EdgeWeightNet gradient test)
          and a boundary at node 3 (regen_logits gradient test).
"""
from __future__ import annotations

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
from diffopt.topology import Edge, Topology


# ---------------------------------------------------------------------------
# Topology factories
# ---------------------------------------------------------------------------

def _make_edge(src: int, dst: int) -> Edge:
    return Edge(
        src=src, dst=dst,
        length_km=80.0, num_spans=1,
        span_lengths_km=[80.0],
        fiber_type="SSMF",
        amplifier_nf_db=[5.0],
    )


def make_linear_topology() -> Topology:
    """Chain 0—1—2—3—4. All degrees ≤ 2 → no regen candidates."""
    edges = [_make_edge(i, i + 1) for i in range(4)]
    return Topology(nodes=[{"id": i} for i in range(5)], edges=edges)


def make_hub_topology() -> Topology:
    """
    Edges: 0-1 (eid 0), 0-2 (eid 1), 1-3 (eid 2), 2-3 (eid 3), 3-4 (eid 4)
    Degrees: 0→2, 1→2, 2→2, 3→3, 4→1  →  node 3 is sole regen candidate.
    Two paths 0→4: via node 1 (eids 0,2,4) and via node 2 (eids 1,3,4).
    """
    edges = [
        _make_edge(0, 1),   # eid 0
        _make_edge(0, 2),   # eid 1
        _make_edge(1, 3),   # eid 2
        _make_edge(2, 3),   # eid 3
        _make_edge(3, 4),   # eid 4
    ]
    return Topology(nodes=[{"id": i} for i in range(5)], edges=edges)


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def make_pipeline(topology: Topology, temperature: float = 0.5) -> DiffONetPipeline:
    qot_model = SpanAttentionQoT(max_spans=60)
    segment_combiner = SegmentCombiner(soft_max_temperature=temperature)
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
        assert path_indicators[did].shape == torch.Size([topo.num_edges])

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
    pipeline = make_pipeline(topo, temperature=0.01)

    demand = Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)
    path_costs, gsnr_preds, path_indicators, regen_probs = pipeline([demand])

    # Find which edges are on the path
    indicator = path_indicators[0].detach()
    active_eids = [e for e in range(indicator.shape[0]) if indicator[e].item() > 0.5]

    # Build the same span feature tensor the pipeline would have used
    rows = []
    accum_dist = 0.0
    for eid in active_eids:
        edge = topo.edges[eid]
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
