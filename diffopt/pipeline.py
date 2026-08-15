"""DiffONetPipeline: end-to-end differentiable routing + regenerator placement."""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from diffopt.demands import Demand
from diffopt.placement.regenerator import RegenPlacement
from diffopt.qot.edge_noise import compute_edge_ase_noise
from diffopt.qot.model import SpanAttentionQoT
from diffopt.qot.segment_combiner import SegmentCombiner
from diffopt.routing.edge_weight_net import EdgeWeightNet
from diffopt.routing.surrogate import surrogate_shortest_path
from diffopt.topology import FIBER_TYPE_INDEX, Edge, Topology


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def segment_path(
    ordered_edge_ids: List[int],
    start_node: int,
    regen_candidate_set: Set[int],
    edges: List[Edge],
    demand_dst: int,
) -> Tuple[List[List[int]], List[int]]:
    """Split an ordered edge list into transparent segments at regen candidates.

    Args:
        ordered_edge_ids: Edge IDs in traversal order (src → dst).
        start_node:       First node of the path (demand source).
        regen_candidate_set: Set of node IDs that are regen candidates.
        edges:            List of Edge, indexable by edge id, giving endpoint lookup.
        demand_dst:       Demand destination node — prevents a spurious empty
                          trailing segment when the path ends at a regen candidate.

    Returns:
        segments:         List of segments, each a list of edge IDs.
        boundary_nodes:   Node IDs where segments join (length = len(segments)-1).
    """
    segments: List[List[int]] = []
    boundary_nodes: List[int] = []
    current_segment: List[int] = []
    current_node = start_node

    for eid in ordered_edge_ids:
        u = edges[eid].src
        v = edges[eid].dst
        exit_node = v if current_node == u else u
        current_segment.append(eid)
        current_node = exit_node

        if exit_node in regen_candidate_set and exit_node != demand_dst:
            segments.append(current_segment)
            boundary_nodes.append(exit_node)
            current_segment = []

    segments.append(current_segment)
    return segments, boundary_nodes


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class DiffONetPipeline(nn.Module):
    """End-to-end differentiable pipeline for joint routing and regen placement.

    Forward pass:
      1. Compute regen probabilities from RegenPlacement.
      2. Build per-edge feature vectors (topology + regen probs at endpoints).
      3. Compute edge weights via EdgeWeightNet.
      4. For each demand:
         a. Run surrogate Dijkstra → binary path indicator (differentiable).
         b. Accumulate path_cost = (path_indicator · edge_weights).sum()
            — this is the live autograd path into EdgeWeightNet.
         c. Reconstruct ordered edge list (using detached indicator).
         d. Segment path at regen candidate nodes.
         e. Run frozen QoT model on each segment → scalar GSNR.
         f. Combine segment GSNRs via SegmentCombiner with boundary probs.

    The QoT model is frozen at construction via requires_grad_(False), not via
    torch.no_grad(), to avoid accidentally severing regen_probs from the graph.
    """

    def __init__(
        self,
        topology: Topology,
        qot_model: SpanAttentionQoT,
        segment_combiner: SegmentCombiner,
        edge_weight_net: EdgeWeightNet,
        regen_placement: RegenPlacement,
        channel_loading_fraction: float = 0.5,
        max_spans: int = 60,
        edge_ase_noise: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        # Freeze QoT parameters — they must not be updated by end-to-end gradients
        for p in qot_model.parameters():
            p.requires_grad_(False)

        self.qot_model = qot_model
        self.segment_combiner = segment_combiner
        self.edge_weight_net = edge_weight_net
        self.regen_placement = regen_placement
        self.channel_loading_fraction = channel_loading_fraction
        self.max_spans = max_spans
        self._topology = topology
        self._edges: List[Edge] = list(topology.undirected_edges)
        self._num_nodes = topology.num_nodes
        self._regen_candidate_set: Set[int] = set(topology.regen_candidate_nodes)

        # Register topology-derived tensors as buffers so they move with the model
        topo_edge_features = topology.get_edge_features()          # (E, 5)
        edge_index = topology.edge_index                           # (2, E)
        self.register_buffer("_topo_edge_features", topo_edge_features)
        self.register_buffer("_edge_index", edge_index)
        self.register_buffer("_edge_src_ids", edge_index[0])       # (E,)
        self.register_buffer("_edge_dst_ids", edge_index[1])       # (E,)

        if edge_ase_noise is None:
            edge_ase_noise = compute_edge_ase_noise(topology)
        else:
            assert edge_ase_noise.shape == (len(self._edges),), (
                f"edge_ase_noise shape {edge_ase_noise.shape} != "
                f"expected ({len(self._edges)},)"
            )
        self.register_buffer("_edge_ase_noise", edge_ase_noise)
        self._proxy_eps = 1e-12

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reconstruct_path(
        self,
        path_indicator: torch.Tensor,
        src: int,
        dst: int,
    ) -> List[int]:
        """Convert binary (E,) path indicator to ordered list of edge IDs.

        Uses path_indicator.detach() for the discrete graph walk, leaving
        path_indicator live in the autograd graph for path_cost computation.
        """
        active = [
            e for e in range(path_indicator.shape[0])
            if path_indicator.detach()[e].item() > 0.5
        ]

        # Build undirected adjacency: node → [(neighbour, edge_id)]
        adj: Dict[int, List[Tuple[int, int]]] = {}
        for eid in active:
            u = self._edges[eid].src
            v = self._edges[eid].dst
            adj.setdefault(u, []).append((v, eid))
            adj.setdefault(v, []).append((u, eid))

        # Walk from src to dst
        ordered: List[int] = []
        visited: Set[int] = {src}
        current = src
        while current != dst:
            moved = False
            for neighbour, eid in adj.get(current, []):
                if neighbour not in visited:
                    ordered.append(eid)
                    visited.add(neighbour)
                    current = neighbour
                    moved = True
                    break
            if not moved:
                break  # disconnected (shouldn't happen with valid Dijkstra output)
        return ordered

    def _span_feature_rows(self, segment_edge_ids: List[int]) -> List[List[float]]:
        """Build raw (unpadded) per-span feature rows for one segment."""
        rows: List[List[float]] = []
        accum_dist = 0.0

        for eid in segment_edge_ids:
            edge = self._edges[eid]
            ftype_idx = float(FIBER_TYPE_INDEX.get(edge.fiber_type, 0))
            for span_idx in range(edge.num_spans):
                rows.append([
                    edge.span_lengths_km[span_idx],
                    ftype_idx,
                    edge.amplifier_nf_db[span_idx],
                    self.channel_loading_fraction,
                    accum_dist,
                ])
                accum_dist += edge.span_lengths_km[span_idx]

        return rows

    def _extract_span_features(
        self,
        segment_edge_ids: List[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build (1, max_spans, 5) span feature tensor and (1, max_spans) padding mask.

        Pads to the architectural max_spans (not a batch-local width) — used
        for single-segment direct QoT calls (tests, diagnostics). The
        batched path in forward() below pads to the batch's true max span
        count instead, since attention masking and mean-pooling over
        real spans only make the two paddings numerically equivalent.
        """
        rows = self._span_feature_rows(segment_edge_ids)
        n_spans = len(rows)

        span_feats = torch.zeros(1, self.max_spans, 5, device=device)
        if n_spans > 0:
            span_feats[0, :n_spans] = torch.tensor(rows, dtype=torch.float32, device=device)

        padding_mask = torch.zeros(1, self.max_spans, dtype=torch.bool, device=device)
        padding_mask[0, :n_spans] = True   # True = real span (inverted inside QoT model)

        return span_feats, padding_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        demands: List[Demand],
        tau: float = 1.0,
        lambda_: float = 10.0,
        soft_max_temperature: float = 0.5,
    ) -> Tuple[
        Dict[int, torch.Tensor],   # path_costs
        Dict[int, torch.Tensor],   # gsnr_preds
        Dict[int, torch.Tensor],   # path_indicators (for diagnostics)
        torch.Tensor,              # regen_probs (num_nodes,)
    ]:
        """Run the full differentiable pipeline for a list of demands.

        Args:
            demands:  List of Demand namedtuples (id, src, dst, bitrate_gbps).
            tau:      Regen placement temperature. Passed per-call, never stored.
            lambda_:  Vlastelica perturbation strength. Passed per-call.
            soft_max_temperature: SegmentCombiner's soft-max sharpness for
                this call. Passed per-call, never stored — same reasoning
                as tau/lambda_. Real training should anneal this toward
                0.01 (see SegmentCombiner's docstring); the 0.5 default
                here is only a safety net for callers that don't care
                (e.g. ad-hoc/test calls), not a value real e2e training
                should hold fixed.

        Returns:
            path_costs:      demand_id → scalar tensor, live in autograd graph.
            gsnr_preds:      demand_id → scalar GSNR tensor (dB).
            path_indicators: demand_id → (E,) binary tensor (for logging/debug).
            regen_probs:     (num_nodes,) tensor from RegenPlacement.
        """
        device = self._topo_edge_features.device

        # 1. Regen probabilities — shape (num_nodes,), requires_grad=True
        regen_probs = self.regen_placement.get_regen_probs(tau)

        # 2. Build (E, 7) edge features: topology cols + regen probs at endpoints
        edge_feats = torch.cat([
            self._topo_edge_features,
            regen_probs[self._edge_src_ids].unsqueeze(1),
            regen_probs[self._edge_dst_ids].unsqueeze(1),
        ], dim=1)

        # 3. Edge weights via EdgeWeightNet — (E,), strictly positive via
        # Softplus, then renormalised to unit mean.
        #
        # The divisor is deliberately NOT detached. With w = u / mean(u), the
        # loss becomes homogeneous of degree 0 in the raw output u, since
        # (c*u)/mean(c*u) == u/mean(u) for any c > 0. Euler's theorem then
        # gives sum_i u_i * dL/du_i == 0 identically: the scale component of
        # the gradient is annihilated as an algebraic fact, not merely
        # discouraged. Autograd produces this because differentiating through
        # mean(u) contributes the term that subtracts the radial component.
        #
        # Detaching would delete that subtraction and leave the collapse
        # degeneracy fully intact, merely rescaled — weights fell ~8.2e7x on
        # ind_132 while Spearman rank-corr with init stayed at +0.999. See
        # docs/investigations/edge_weight_scale_collapse.md.
        #
        # NOTE: this is the OPPOSITE choice from soft_max's scale
        # normalisation in qot/segment_combiner.py, which detaches on purpose
        # (there the goal is to rescale an error term without adding a
        # gradient path). The two lines look nearly identical and mean
        # opposite things. Do not "make them consistent".
        raw_edge_weights = self.edge_weight_net(edge_feats).squeeze(-1)
        edge_weights = raw_edge_weights / raw_edge_weights.mean().clamp_min(1e-12)

        # 4. Route and segment every demand first (no QoT calls yet), so all
        # segments across all demands can be sent through the QoT model in
        # one batched call instead of one call per segment (~920 calls/epoch
        # at batch size 1 — see docs/investigations/open_followups.md #1).
        demand_path_costs: Dict[int, torch.Tensor] = {}
        demand_path_indicators: Dict[int, torch.Tensor] = {}
        demand_segments: Dict[int, List[List[int]]] = {}
        demand_boundary_nodes: Dict[int, List[int]] = {}
        all_segments: List[List[int]] = []
        segment_owner_demand_id: List[int] = []

        for demand in demands:
            # 4a. Surrogate Dijkstra → (E,) binary path indicator
            path_indicator = surrogate_shortest_path(
                edge_weights,
                self._edge_index,
                demand.src,
                demand.dst,
                self._num_nodes,
                lambda_=lambda_,
            )

            # 4b. Path cost — live in autograd graph, activates EdgeWeightNet gradient
            path_cost = (path_indicator * edge_weights).sum()

            # 4c. Ordered edge list via detached indicator
            ordered_edges = self._reconstruct_path(path_indicator, demand.src, demand.dst)

            # 4d. Segment at regen candidate nodes
            segments, boundary_nodes = segment_path(
                ordered_edges,
                demand.src,
                self._regen_candidate_set,
                self._edges,
                demand.dst,
            )

            demand_path_costs[demand.id] = path_cost
            demand_path_indicators[demand.id] = path_indicator
            demand_segments[demand.id] = segments
            demand_boundary_nodes[demand.id] = boundary_nodes
            for seg_edge_ids in segments:
                all_segments.append(seg_edge_ids)
                segment_owner_demand_id.append(demand.id)

        # 5. One batched QoT call over every segment from every demand,
        # padded only to this batch's true max span count (not the
        # architectural max_spans=60) — real segments run <=12 spans,
        # median ~7, so padding to 60 wastes ~98% of attention compute on
        # masked positions (docs/investigations/open_followups.md #1).
        # Safe because TransformerEncoder's src_key_padding_mask excludes
        # padded positions from attention and the mean-pool divides only by
        # real-span count, so a narrower shared width changes nothing but
        # the wasted columns.
        all_segment_rows = [self._span_feature_rows(seg) for seg in all_segments]

        if all_segments:
            batch_max_spans = max(1, max(len(rows) for rows in all_segment_rows))
            n_total = len(all_segments)
            batched_span_feats = torch.zeros(n_total, batch_max_spans, 5, device=device)
            batched_padding_mask = torch.zeros(n_total, batch_max_spans, dtype=torch.bool, device=device)
            for i, rows in enumerate(all_segment_rows):
                n_spans = len(rows)
                if n_spans > 0:
                    batched_span_feats[i, :n_spans] = torch.tensor(rows, dtype=torch.float32, device=device)
                    batched_padding_mask[i, :n_spans] = True
            # Forward value only — this has zero live gradient w.r.t. any
            # path_indicator (span_feats comes from static topology data,
            # and seg_edge_ids was derived via path_indicator.detach()).
            batched_qot_gsnr = self.qot_model(batched_span_feats, batched_padding_mask)
        else:
            batched_qot_gsnr = torch.zeros(0, device=device)

        # 6. Scatter the batched QoT output back per demand, blend with the
        # STE proxy per segment, and combine each demand's segments.
        path_costs: Dict[int, torch.Tensor] = {}
        gsnr_preds: Dict[int, torch.Tensor] = {}
        path_indicators: Dict[int, torch.Tensor] = {}
        flat_idx = 0

        for demand in demands:
            path_indicator = demand_path_indicators[demand.id]
            segment_gsnrs: List[torch.Tensor] = []
            for seg_edge_ids in demand_segments[demand.id]:
                qot_gsnr = batched_qot_gsnr[flat_idx]
                flat_idx += 1

                # Analytical proxy — linear in path_indicator, independent of
                # edge_weights. Supplies the backward gradient direction.
                seg_idx = torch.tensor(seg_edge_ids, dtype=torch.long, device=device)
                proxy_noise = (path_indicator[seg_idx] * self._edge_ase_noise[seg_idx]).sum()
                proxy_gsnr = -10.0 * torch.log10(proxy_noise + self._proxy_eps)

                # STE blend: forward value = qot_gsnr exactly; gradient = proxy's.
                # Note: if qot_gsnr falls outside SegmentCombiner's [-5, 35] dB
                # clamp range, the STE gradient for this segment is zeroed by
                # that clamp (segment_combiner.py's _safe_noise has zero
                # gradient outside the clamped band).
                segment_gsnr = qot_gsnr + (proxy_gsnr - proxy_gsnr.detach())
                segment_gsnrs.append(segment_gsnr)

            # Combine segments with soft boundary probabilities
            boundary_probs = [regen_probs[n] for n in demand_boundary_nodes[demand.id]]
            path_gsnr = self.segment_combiner(segment_gsnrs, boundary_probs, temperature=soft_max_temperature)

            path_costs[demand.id] = demand_path_costs[demand.id]
            gsnr_preds[demand.id] = path_gsnr
            path_indicators[demand.id] = path_indicator

        return path_costs, gsnr_preds, path_indicators, regen_probs
