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
from diffopt.qot.span_features import SPAN_FEATURE_DIM, span_feature_rows
from diffopt.routing.edge_weight_net import EdgeWeightNet
from diffopt.routing.surrogate import surrogate_shortest_path
from diffopt.topology import Edge, Topology


# The memo holds one float per distinct (ordered edge-id tuple, batch
# width). Routes move during training, so the set grows; the bound exists
# so a long run cannot turn a speedup into a memory leak. Clearing wholesale
# rather than evicting LRU is deliberate: the working set is "the segments
# of the current routing", which turns over as a block.
_SEGMENT_GSNR_CACHE_MAX = 100_000


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
         b. Accumulate path_noise_cost = (path_indicator · edge_ase_noise).sum()
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
        cache_segment_gsnr: bool = True,
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

        # Standardise the static topology features once, here, rather than
        # inside EdgeWeightNet — the statistics are topology-derived and the
        # pipeline owns the topology, so EdgeWeightNet stays a
        # topology-agnostic MLP on (E, 7) and diagnostics that construct it
        # standalone keep working.
        #
        # Raw features are wildly unscaled: on ind_132 total_length_km spans
        # 19-597 while the two regen_prob columns appended in forward() live
        # in [0, 1], so first-layer pre-activations were dominated by raw
        # kilometres and the random init priced longer edges CHEAPER
        # (corr(w, length_km) = -0.463). See
        # docs/investigations/edge_weight_scale_collapse.md.
        #
        # unbiased=False so a single-edge topology gives std 0 rather than
        # NaN; clamp_min then maps any constant column (e.g. fiber_type_idx
        # and mean_amp_nf_db on a single-fiber-type topology) to exactly 0
        # instead of dividing by ~0. Constant columns are deliberately NOT
        # dropped — they are constant on ind_132, not in general, and
        # dropping them would break mixed-fiber-type topologies.
        raw_topo_edge_features = topology.get_edge_features()       # (E, 5)
        feat_mean = raw_topo_edge_features.mean(dim=0, keepdim=True)             # (1, 5)
        feat_std = raw_topo_edge_features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
        topo_edge_features = (raw_topo_edge_features - feat_mean) / feat_std

        edge_index = topology.edge_index                           # (2, E)
        self.register_buffer("_topo_feat_mean", feat_mean)
        self.register_buffer("_topo_feat_std", feat_std)
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

        # Memo for step 5. A segment's QoT input is a pure function of its
        # ORDERED edge-id tuple: span_feature_rows reads only frozen Edge
        # fields plus the fixed config constant channel_loading_fraction,
        # and accum_dist_km makes the tuple direction-sensitive, so this is
        # a tuple key and never a frozenset. The QoT model is frozen at
        # construction, so the mapping never moves during a run — see
        # docs/architecture/invariants.md, "Physics layer".
        #
        # Set cache_segment_gsnr=False to bypass it: used by
        # tests/test_pipeline.py to check the memo against the direct path,
        # and appropriate for any caller that swaps qot_model without
        # calling clear_segment_gsnr_cache().
        self.cache_segment_gsnr = cache_segment_gsnr
        self._segment_gsnr_cache: Dict[Tuple[Tuple[int, ...], int], float] = {}

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
        path_indicator live in the autograd graph for path_noise_cost computation.
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

    def _extract_span_features(
        self,
        segment_edge_ids: List[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build (1, max_spans, SPAN_FEATURE_DIM) span feature tensor and (1, max_spans) padding mask.

        Pads to the architectural max_spans (not a batch-local width) — used
        for single-segment direct QoT calls (tests, diagnostics). The
        batched path in forward() below pads to the batch's true max span
        count instead, since attention masking and mean-pooling over
        real spans only make the two paddings numerically equivalent.

        forward() itself never calls this (see step 5 of forward() below,
        which needs batch-local padding, not the architectural max_spans) —
        it exists for single-segment callers: scripts/diagnose_*.py and
        tests/test_pipeline.py. Thin wrapper over the shared
        diffopt.qot.span_features.span_feature_rows.
        """
        rows = span_feature_rows(
            self._topology, segment_edge_ids,
            channel_loading_fraction=self.channel_loading_fraction,
        )
        n_spans = len(rows)

        span_feats = torch.zeros(1, self.max_spans, SPAN_FEATURE_DIM, device=device)
        if n_spans > 0:
            span_feats[0, :n_spans] = torch.tensor(rows, dtype=torch.float32, device=device)

        padding_mask = torch.zeros(1, self.max_spans, dtype=torch.bool, device=device)
        padding_mask[0, :n_spans] = True   # True = real span (inverted inside QoT model)

        return span_feats, padding_mask

    def clear_segment_gsnr_cache(self) -> None:
        """Drop the memo. Required after replacing `self.qot_model` — the
        memo's exactness rests on the model being frozen for the lifetime
        of the entries."""
        self._segment_gsnr_cache.clear()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        demands: List[Demand],
        tau: float = 1.0,
        lambda_: float = 10.0,
        regen_probs_override: Optional[torch.Tensor] = None,
        gate_dropout_p: float = 0.0,
    ) -> Tuple[
        Dict[int, torch.Tensor],   # path_noise_costs
        Dict[int, torch.Tensor],   # gsnr_preds
        Dict[int, torch.Tensor],   # path_indicators (for diagnostics)
        torch.Tensor,              # regen_probs (num_nodes,)
    ]:
        """Run the full differentiable pipeline for a list of demands.

        Args:
            demands:  List of Demand namedtuples (id, src, dst, bitrate_gbps).
            tau:      Regen placement temperature. Passed per-call, never stored.
            lambda_:  Vlastelica perturbation strength. Passed per-call.
            regen_probs_override: (num_nodes,) probabilities to use INSTEAD of
                      RegenPlacement's. Lets a caller evaluate a specific
                      placement — e.g. train.py's hard-placement selection
                      pass, or diagnose_regen_ablation.py's leave-one-out
                      sweep — without mutating regen_logits and restoring
                      them afterwards. Flows to both consumers (edge features
                      and boundary probabilities) and is echoed back as the
                      fourth return value, so `regen_probs` always describes
                      what the forward pass actually used.
            gate_dropout_p: Training-only probability of zeroing each node's
                      gate in the PHYSICS path. The returned regen_probs are
                      always undropped, so `lambda_regen`'s penalty is priced
                      on the real probabilities — otherwise the price per
                      regenerator fluctuates with the mask. Ignored under
                      .eval() and whenever regen_probs_override is given.

        Returns:
            path_noise_costs: demand_id → scalar accumulated-ASE-noise tensor, live in autograd graph.
            gsnr_preds:      demand_id → scalar GSNR tensor (dB).
            path_indicators: demand_id → (E,) binary tensor (for logging/debug).
            regen_probs:     (num_nodes,) tensor from RegenPlacement.
        """
        device = self._topo_edge_features.device

        # 1. Regen probabilities — shape (num_nodes,), requires_grad=True
        # unless overridden.
        if regen_probs_override is not None:
            # An explicit placement the caller wants evaluated (hard-eval
            # selection, leave-one-out ablation). Dropout must NOT touch it:
            # masking a placement someone asked to measure would make the
            # measurement random.
            regen_probs = regen_probs_override
            regen_probs_physics = regen_probs_override
        else:
            regen_probs = self.regen_placement.get_regen_probs(tau)
            if self.training and gate_dropout_p > 0.0:
                # Gate dropout. Once every demand clears threshold + margin,
                # relu(bar - gsnr) is flat and d(feasibility)/d(logit) is
                # exactly 0 on every node — the only surviving force is
                # lambda_regen's L1 push, which is identical on every node by
                # construction, and identical pressure cannot sort a
                # load-bearing node from a redundant one. Dropping gates
                # manufactures violations on purpose, putting demands back in
                # the hinge's ACTIVE region, the only region that produces
                # node-discriminating gradient. See
                # docs/investigations/regen_over_provisioning.md Finding 2.
                keep = (torch.rand_like(regen_probs) >= gate_dropout_p).to(
                    regen_probs.dtype
                )
                regen_probs_physics = regen_probs * keep
            else:
                regen_probs_physics = regen_probs

        # 2. Build (E, 7) edge features: topology cols + regen probs at endpoints
        edge_feats = torch.cat([
            self._topo_edge_features,
            regen_probs_physics[self._edge_src_ids].unsqueeze(1),
            regen_probs_physics[self._edge_dst_ids].unsqueeze(1),
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
        # NOTE: this is the OPPOSITE choice from the `soft_max` helper's
        # scale normalisation in qot/segment_combiner.py, which detaches on
        # purpose (there the goal is to rescale an error term without adding
        # a gradient path). The two lines look nearly identical and mean
        # opposite things. Do not "make them consistent".
        raw_edge_weights = self.edge_weight_net(edge_feats).squeeze(-1)
        edge_weights = raw_edge_weights / raw_edge_weights.mean().clamp_min(1e-12)

        # 4. Route and segment every demand first (no QoT calls yet), so all
        # segments across all demands can be sent through the QoT model in
        # one batched call instead of one call per segment (~920 calls/epoch
        # at batch size 1 — see docs/investigations/open_followups.md #1).
        demand_path_noise_costs: Dict[int, torch.Tensor] = {}
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

            # 4b. Physical path cost — accumulated ASE noise along the route.
            #
            # The coefficients are a fixed topology-derived buffer, so this
            # term is degree-0 in edge_weights by construction: weights enter
            # only through Dijkstra's scale-invariant argmin. Using
            # edge_weights here instead made the term degree-1 and created an
            # unopposed shrink direction — see
            # docs/investigations/edge_weight_scale_collapse.md.
            #
            # It also restores a routing signal that survives feasibility:
            # once num_infeasible hits 0 the feasibility term contributes
            # gradient on 0/168 edges, so before this change shrink pressure
            # was the ONLY signal reaching EdgeWeightNet for the back half of
            # training.
            #
            # ASE-only, not ASE+NLI: NLI depends on the spectrum position of
            # the channels, so it is not determined by the route. Charging the
            # router for a quantity it cannot control would add noise, not
            # signal. ASE is the routing-controllable part of the physics.
            path_noise_cost = (path_indicator * self._edge_ase_noise).sum()

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

            demand_path_noise_costs[demand.id] = path_noise_cost
            demand_path_indicators[demand.id] = path_indicator
            demand_segments[demand.id] = segments
            demand_boundary_nodes[demand.id] = boundary_nodes
            for seg_edge_ids in segments:
                all_segments.append(seg_edge_ids)
                segment_owner_demand_id.append(demand.id)

        # 5. One batched QoT call over every segment that is NOT already
        # memoised, padded only to this batch's true max span count (not the
        # architectural max_spans=60) — real segments run <=12 spans, median
        # ~7, so padding to 60 wastes ~98% of attention compute on masked
        # positions (docs/investigations/open_followups.md #1). Safe because
        # TransformerEncoder's src_key_padding_mask excludes padded
        # positions from attention and the mean-pool divides only by
        # real-span count, so a narrower shared width changes nothing but
        # the wasted columns.
        #
        # batch_max_spans is computed over ALL segments, hits included, so
        # the miss batch is padded to exactly the width the un-memoised code
        # would have used. Together with the width being part of the memo
        # key, a hit is bitwise what a recompute would have given.
        seg_keys = [tuple(seg) for seg in all_segments]
        # Span count without building the rows — the rows are only needed
        # for misses, and building them for every segment is the cost this
        # memo exists to avoid.
        seg_n_spans = [
            sum(self._edges[eid].num_spans for eid in seg) for seg in all_segments
        ]

        if all_segments:
            batch_max_spans = max(1, max(seg_n_spans))
            if batch_max_spans > self.max_spans:
                raise ValueError(
                    f"Routed a transparent segment of {batch_max_spans} spans, but "
                    f"max_spans={self.max_spans} is a hard architecture parameter "
                    f"(SpanAttentionQoT's positional embedding is sized to it). "
                    f"Either shorten the route or regenerate datasets and retrain "
                    f"with a larger max_spans."
                )

            cache = self._segment_gsnr_cache if self.cache_segment_gsnr else {}
            miss_positions = [
                i for i, key in enumerate(seg_keys)
                if (key, batch_max_spans) not in cache
            ]

            if miss_positions:
                n_miss = len(miss_positions)
                batched_span_feats = torch.zeros(
                    n_miss, batch_max_spans, SPAN_FEATURE_DIM, device=device
                )
                batched_padding_mask = torch.zeros(
                    n_miss, batch_max_spans, dtype=torch.bool, device=device
                )
                for row, i in enumerate(miss_positions):
                    rows = span_feature_rows(
                        self._topology, all_segments[i],
                        channel_loading_fraction=self.channel_loading_fraction,
                    )
                    n_spans = len(rows)
                    if n_spans > 0:
                        batched_span_feats[row, :n_spans] = torch.tensor(
                            rows, dtype=torch.float32, device=device
                        )
                        batched_padding_mask[row, :n_spans] = True
                # Forward value only — this has zero live gradient w.r.t. any
                # path_indicator (span_feats comes from static topology data,
                # and seg_edge_ids was derived via path_indicator.detach()).
                # That is also what makes storing plain floats safe: there is
                # no graph to sever.
                miss_gsnr = self.qot_model(
                    batched_span_feats, batched_padding_mask
                ).tolist()
                for row, i in enumerate(miss_positions):
                    cache[(seg_keys[i], batch_max_spans)] = miss_gsnr[row]

            batched_qot_gsnr = torch.tensor(
                [cache[(key, batch_max_spans)] for key in seg_keys],
                dtype=torch.float32, device=device,
            )

            # Cache eviction: check size after building batched_qot_gsnr (not
            # before) to avoid clearing entries we just read. The read at
            # batched_qot_gsnr construction uses cache[(key, batch_max_spans)]
            # for every key in seg_keys; only after that read completes is it
            # safe to evict.
            if len(self._segment_gsnr_cache) > _SEGMENT_GSNR_CACHE_MAX:
                self._segment_gsnr_cache.clear()
        else:
            batched_qot_gsnr = torch.zeros(0, device=device)

        # 6. Scatter the batched QoT output back per demand, blend with the
        # STE proxy per segment, then fold EVERY demand in one call.
        #
        # Pre-batching this was one SegmentCombiner call per demand, each
        # running a Python loop over that demand's boundaries — about 2 400
        # interpreter steps per epoch on constrained_stress, doubled by
        # train.py's hard-evaluation pass. Batched it is J_max - 1.
        path_noise_costs: Dict[int, torch.Tensor] = {}
        gsnr_preds: Dict[int, torch.Tensor] = {}
        path_indicators: Dict[int, torch.Tensor] = {}
        flat_idx = 0

        segment_gsnr_flat: List[torch.Tensor] = []
        seg_rows: List[int] = []
        seg_cols: List[int] = []
        boundary_node_ids: List[int] = []
        bnd_rows: List[int] = []
        bnd_cols: List[int] = []
        seg_counts: List[int] = []

        for row, demand in enumerate(demands):
            path_indicator = demand_path_indicators[demand.id]
            segments = demand_segments[demand.id]
            seg_counts.append(len(segments))

            for col, seg_edge_ids in enumerate(segments):
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
                segment_gsnr_flat.append(segment_gsnr)
                seg_rows.append(row)
                seg_cols.append(col)

            for col, node in enumerate(demand_boundary_nodes[demand.id]):
                boundary_node_ids.append(node)
                bnd_rows.append(row)
                bnd_cols.append(col)

            path_noise_costs[demand.id] = demand_path_noise_costs[demand.id]
            path_indicators[demand.id] = path_indicator

        if demands:
            num_demands = len(demands)
            j_max = max(seg_counts)
            long_ = dict(dtype=torch.long, device=device)

            # index_put on a zeros tensor rather than in-place assignment:
            # out-of-place keeps the autograd path to segment_gsnr_flat and
            # to regen_probs_physics explicit, and the (row, col) pairs are
            # unique so accumulate=False is right.
            gsnr_matrix = torch.zeros(num_demands, j_max, device=device).index_put(
                (torch.tensor(seg_rows, **long_), torch.tensor(seg_cols, **long_)),
                torch.stack(segment_gsnr_flat),
            )

            prob_matrix = torch.zeros(num_demands, max(j_max - 1, 0), device=device)
            if boundary_node_ids:
                # ONE gather into regen_probs_physics, not one per boundary:
                # the per-boundary list comprehension this replaces built
                # sum_d (N_d - 1) separate 0-dim views every forward.
                boundary_values = regen_probs_physics[
                    torch.tensor(boundary_node_ids, **long_)
                ]
                prob_matrix = prob_matrix.index_put(
                    (torch.tensor(bnd_rows, **long_), torch.tensor(bnd_cols, **long_)),
                    boundary_values,
                )

            path_gsnrs = self.segment_combiner.forward_batched(
                gsnr_matrix,
                prob_matrix,
                torch.tensor(seg_counts, **long_),
            )
            for row, demand in enumerate(demands):
                gsnr_preds[demand.id] = path_gsnrs[row]

        return path_noise_costs, gsnr_preds, path_indicators, regen_probs
