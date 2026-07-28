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
from diffopt.topology import FIBER_TYPE_INDEX, Topology


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def segment_path(
    ordered_edge_ids: List[int],
    start_node: int,
    regen_candidate_set: Set[int],
    topology: Topology,
    demand_dst: int,
) -> Tuple[List[List[int]], List[int]]:
    """Split an ordered edge list into transparent segments at regen candidates.

    Args:
        ordered_edge_ids: Edge IDs in traversal order (src → dst).
        start_node:       First node of the path (demand source).
        regen_candidate_set: Set of node IDs that are regen candidates.
        topology:         Topology object for edge endpoint lookup.
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
        u = topology.edge_src(eid)
        v = topology.edge_dst(eid)
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
            assert edge_ase_noise.shape == (topology.num_edges,), (
                f"edge_ase_noise shape {edge_ase_noise.shape} != "
                f"expected ({topology.num_edges},)"
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
            u = self._topology.edge_src(eid)
            v = self._topology.edge_dst(eid)
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

    def _extract_span_features(
        self,
        segment_edge_ids: List[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build (1, max_spans, 5) span feature tensor and (1, max_spans) padding mask."""
        rows: List[List[float]] = []
        accum_dist = 0.0

        for eid in segment_edge_ids:
            edge = self._topology.edges[eid]
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

        # 3. Edge weights via EdgeWeightNet — (E,), strictly positive via Softplus
        edge_weights = self.edge_weight_net(edge_feats).squeeze(-1)

        path_costs: Dict[int, torch.Tensor] = {}
        gsnr_preds: Dict[int, torch.Tensor] = {}
        path_indicators: Dict[int, torch.Tensor] = {}

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
                self._topology,
                demand.dst,
            )

            # 4e. QoT evaluation per segment, blended with the STE proxy
            segment_gsnrs: List[torch.Tensor] = []
            for seg_edge_ids in segments:
                span_feats, padding_mask = self._extract_span_features(seg_edge_ids, device)
                # QoT returns (batch,); [0] extracts scalar. Forward value only —
                # this has zero live gradient w.r.t. path_indicator (span_feats
                # comes from static topology data, and seg_edge_ids was derived
                # via path_indicator.detach()).
                qot_gsnr = self.qot_model(span_feats, padding_mask)[0]

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

            # 4f. Combine segments with soft boundary probabilities
            boundary_probs = [regen_probs[n] for n in boundary_nodes]
            path_gsnr = self.segment_combiner(segment_gsnrs, boundary_probs)

            path_costs[demand.id] = path_cost
            gsnr_preds[demand.id] = path_gsnr
            path_indicators[demand.id] = path_indicator

        return path_costs, gsnr_preds, path_indicators, regen_probs
