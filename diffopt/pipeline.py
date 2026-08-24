"""DiffONetPipeline: end-to-end differentiable routing + regenerator placement."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffopt.demands import Demand
from diffopt.modulation import ModulationConfig, bar_db_for_demands
from diffopt.placement.allocation import AllocationHead, site_view, total_device_cost
from diffopt.qot.edge_noise import compute_edge_ase_noise
from diffopt.qot.model import SpanAttentionQoT
from diffopt.qot.segment_combiner import (
    GSNR_MAX,
    GSNR_MIN,
    SegmentCombiner,
    db_to_linear_noise,
)
from diffopt.qot.span_features import SPAN_FEATURE_DIM, span_feature_rows
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


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation between two 1-D tensors.

    Ranks each input by double argsort (argsort(argsort(x)) gives each
    element's rank), then computes the Pearson correlation coefficient on
    the two rank vectors. Ties are broken by original order — the same
    simplification plain argsort makes — rather than proper tie-corrected
    (averaged) ranking; acceptable here since this is a diagnostic, not a
    statistic anything downstream reads.

    Degenerate input (a or b constant) is checked on the SOURCE values,
    before ranking: double-argsort always yields a full 0..n-1 permutation
    even when the underlying values are all tied, so a post-ranking
    zero-variance check would never fire — it would silently report a
    "perfect" correlation for a constant input instead of nan.
    """
    if a.numel() > 0 and (a.max() == a.min()).item():
        return float("nan")
    if b.numel() > 0 and (b.max() == b.min()).item():
        return float("nan")
    a_rank = torch.argsort(torch.argsort(a)).float()
    b_rank = torch.argsort(torch.argsort(b)).float()
    a_c = a_rank - a_rank.mean()
    b_c = b_rank - b_rank.mean()
    denom = torch.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
    if denom.item() == 0.0:
        return float("nan")
    return (a_c * b_c).sum().item() / denom.item()


# ---------------------------------------------------------------------------
# Allocation record
# ---------------------------------------------------------------------------

@dataclass
class AllocationOutputs:
    """Everything the allocation half of a forward pass produced.

    Returned as forward()'s fourth value, replacing the bare (num_nodes,)
    regen_probs tensor. It is a record rather than a tuple because the oracle
    and the deployment repair are PURE POST-PROCESSING on these fields — they
    never re-run the pipeline, so every quantity they need has to come back
    from one call.
    """

    a: torch.Tensor                   # (D, J-1) priced allocations
    a_physics: torch.Tensor           # (D, J-1) what the fold actually saw
    alloc_by_node: torch.Tensor       # (D, N) a scattered onto boundary nodes
    device_count: torch.Tensor        # scalar, sum_n sum_d
    site_view: torch.Tensor           # (N,) max_d, diagnostics only
    seg_gsnr_db: torch.Tensor         # (D, J) STE-blended per-segment GSNR
    seg_noise: torch.Tensor           # (D, J) the same, as linear noise
    num_segments: torch.Tensor        # (D,) long
    boundary_node_ids: torch.Tensor   # (D, J-1) long, -1 where padded
    demand_ids: List[int]             # row index -> Demand.id
    ste_clamped_segments: int         # count of segments whose qot_gsnr fell
                                       # outside SegmentCombiner's [-5, 35] dB
                                       # clamp band this forward call
    proxy_qot_rank_corr: float        # Spearman(proxy, qot) over this call's
                                       # segments; drift toward 0 means the
                                       # STE gradient disagrees with the
                                       # forward value about segment ordering


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class DiffONetPipeline(nn.Module):
    """End-to-end differentiable pipeline for joint routing and regen placement.

    Forward pass:
      2. Read the static per-edge feature buffer (topology + is_candidate
         indicators at each endpoint; built once in __init__). Kept for
         diagnostics; no longer consumed by step 3 (spec decision 6).
      3. Compute edge weights from the free per-edge parameter
         edge_log_weight (Softplus, then renormalised to unit mean).
      4. For each demand:
         a. Run surrogate Dijkstra → binary path indicator (differentiable).
         b. Accumulate path_noise_cost = (path_indicator · edge_ase_noise).sum()
            — this is the live autograd path into edge_log_weight.
         c. Reconstruct ordered edge list (using detached indicator).
         d. Segment path at regen candidate nodes.
      5. Run the frozen QoT model on every segment in one batched call.
      6. Roll the AllocationHead along each demand's boundaries → a per-
         (demand, boundary) allocation, then fold every demand in one
         SegmentCombiner call using those allocations as boundary
         probabilities.

    Step numbering starts at 2 because step 1 — "compute regen probabilities
    from RegenPlacement" — is gone: there is no per-node probability vector
    any more, only per-(demand, boundary) allocations produced in step 6.

    The QoT model is frozen at construction via requires_grad_(False), not via
    torch.no_grad(), to avoid accidentally severing the allocation head from
    the graph.
    Note requires_grad_(False) only guarantees non-differentiability; the
    per-segment GSNR memo below additionally requires the model to be
    deterministic and train/eval-mode-invariant, which holds today only
    because SpanAttentionQoT hardcodes dropout=0.0 with no other
    stochastic/mode-dependent layer.
    """

    def __init__(
        self,
        topology: Topology,
        qot_model: SpanAttentionQoT,
        segment_combiner: SegmentCombiner,
        allocation_head: AllocationHead,
        modulation_config: ModulationConfig,
        margin_db: float,
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
        self.allocation_head = allocation_head
        # Run constants, stored the same way channel_loading_fraction and
        # max_spans already are: the bar is a property of the config, not of
        # a call, and computing it inside forward keeps every caller's
        # signature short. bar_db_for_demands is the single definition.
        self._modulation_config = modulation_config
        self._margin_db = margin_db
        self.channel_loading_fraction = channel_loading_fraction
        self.max_spans = max_spans
        self._topology = topology
        self._edges: List[Edge] = list(topology.undirected_edges)
        self._num_nodes = topology.num_nodes
        self._regen_candidate_set: Set[int] = set(topology.regen_candidate_nodes)

        # Spec decision 6. Under approach A the edge features are constant,
        # so EdgeWeightNet(constant) is a fixed function of its own weights —
        # a reparameterization of E numbers with 5 000-odd parameters and a
        # curvature landscape nobody chose. The class is retained in
        # diffopt/routing/edge_weight_net.py so this commit can be reverted
        # on its own.
        #
        # Init at length-proportional weights, i.e. shortest-by-km routing:
        # the documented baseline, and what preflight_filter screens against.
        # Uniform init (theta = 0) would tie every edge and hand routing to
        # Dijkstra's tie-break order — see the adjacency-ordering invariant.
        km = torch.tensor(
            [e.length_km for e in self._edges], dtype=torch.float32
        )
        target = km / km.mean()
        self.edge_log_weight = nn.Parameter(torch.log(torch.expm1(target)))

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
        edge_index = topology.edge_index                           # (2, E)

        raw_topo_edge_features = topology.get_edge_features()       # (E, 5)
        feat_mean = raw_topo_edge_features.mean(dim=0, keepdim=True)             # (1, 5)
        feat_std = raw_topo_edge_features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
        topo_edge_features = (raw_topo_edge_features - feat_mean) / feat_std

        self.register_buffer("_topo_feat_mean", feat_mean)
        self.register_buffer("_topo_feat_std", feat_std)
        self.register_buffer("_topo_edge_features", topo_edge_features)
        self.register_buffer("_edge_index", edge_index)
        self.register_buffer("_edge_src_ids", edge_index[0])       # (E,)
        self.register_buffer("_edge_dst_ids", edge_index[1])       # (E,)

        # Spec decision 5, "approach A": the two trailing columns are STATIC
        # `is_candidate` indicators, not the learned regenerator
        # probabilities they replace.
        #
        # Before this, regen_probs fed EdgeWeightNet at both endpoints of
        # every edge, so the router's input moved whenever the placement head
        # moved and vice versa. That loop is why a non-candidate logit could
        # pick up feasibility signal at all (train.py's num_regen_noncand
        # warning), and under per-demand allocation there is no per-node
        # probability to feed it anyway.
        #
        # The structural signal approaches B and C were meant to supply
        # already exists and is computed exactly: Vlastelica's backward
        # perturbs the weights by lambda * grad_output and RE-SOLVES with
        # SPFA (surrogate.py:56-88). Since grad_output now carries the device
        # term, that re-solve is literally searching for a route that needs
        # fewer regenerators, including one through a different candidate
        # set. See spec section 5.
        is_candidate = torch.zeros(topology.num_nodes)
        is_candidate[list(topology.regen_candidate_nodes)] = 1.0
        static_edge_features = torch.cat(
            [
                topo_edge_features,
                is_candidate[edge_index[0]].unsqueeze(1),
                is_candidate[edge_index[1]].unsqueeze(1),
            ],
            dim=1,
        )
        self.register_buffer("_static_edge_features", static_edge_features)

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
        # docs/architecture/invariants.md, "Physics layer". Being frozen
        # (requires_grad_(False)) only rules out gradient updates; the memo's
        # exactness also needs the model to be deterministic and
        # train/eval-mode-invariant, which holds today only because
        # SpanAttentionQoT hardcodes dropout=0.0 and has no other
        # stochastic/mode-dependent layer.
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
        of the entries. "Frozen" here means both non-differentiable
        (requires_grad_(False)) and deterministic/mode-invariant (no
        stochastic or train/eval-dependent layers) — SpanAttentionQoT's
        hardcoded dropout=0.0 is what currently makes the latter true."""
        self._segment_gsnr_cache.clear()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        demands: List[Demand],
        tau: float = 1.0,
        lambda_: float = 10.0,
        *,
        hard_alloc: bool = False,
        alloc_dropout_p: float = 0.0,
    ) -> Tuple[
        Dict[int, torch.Tensor],   # path_noise_costs
        Dict[int, torch.Tensor],   # gsnr_preds
        Dict[int, torch.Tensor],   # path_indicators (for diagnostics)
        "AllocationOutputs",       # the allocation half of this pass
    ]:
        """Run the full differentiable pipeline for a list of demands.

        Args:
            demands:  List of Demand namedtuples (id, src, dst, bitrate_gbps).
            tau:      Allocation decision temperature — sharpness of
                      sigmoid(score / tau). Passed per-call, never stored.
            lambda_:  Vlastelica perturbation strength. Passed per-call.
            hard_alloc: Take deterministic decisions a_k = 1 if score_k > 0
                      and run the whole rollout under torch.no_grad(). NOT a
                      threshold applied to the soft pass: under hard
                      decisions the head's carry is the EXACT chunk noise, so
                      the rollout is self-consistent physics, and the two
                      passes are allowed to disagree. Spec 2.5.
            alloc_dropout_p: Training-only probability of zeroing a PHYSICS
                      allocation. `AllocationOutputs.a` (what lambda_dev
                      prices) is always undropped; only `a_physics` (what the
                      fold sees and what resets the carry) is masked, so the
                      price per device does not fluctuate with the mask.
                      Ignored under .eval() and whenever hard_alloc is set.

        Returns:
            path_noise_costs: demand_id → scalar accumulated-ASE-noise tensor, live in autograd graph.
            gsnr_preds:      demand_id → scalar GSNR tensor (dB).
            path_indicators: demand_id → (E,) binary tensor (for logging/debug).
            alloc_outputs:   AllocationOutputs — see that dataclass.
        """
        device = self._topo_edge_features.device

        # 2. Edge features — static, built once in __init__ (approach A).
        # `_static_edge_features` itself is no longer read by routing (see
        # spec decision 6: edge_log_weight is a free per-edge parameter, not
        # a function of these features any more); the buffer is kept for
        # diagnostics and the standardisation invariant tests.

        # 3. Edge weights from the free per-edge parameter edge_log_weight —
        # (E,), strictly positive via Softplus, then renormalised to unit mean.
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
        raw_edge_weights = F.softplus(self.edge_log_weight)
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

        # Count segments whose raw QoT prediction falls outside
        # SegmentCombiner's [GSNR_MIN, GSNR_MAX] clamp band: _safe_noise's
        # clamp zeroes the STE gradient for exactly those segments, silently.
        clamped = int(
            ((batched_qot_gsnr < GSNR_MIN) | (batched_qot_gsnr > GSNR_MAX))
            .sum().item()
        )

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
        proxy_flat: List[torch.Tensor] = []
        seg_rows: List[int] = []
        seg_cols: List[int] = []
        seg_km_flat: List[float] = []
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
                proxy_flat.append(proxy_gsnr)
                seg_rows.append(row)
                seg_cols.append(col)
                seg_km_flat.append(
                    sum(self._edges[eid].length_km for eid in seg_edge_ids)
                )

            for col, node in enumerate(demand_boundary_nodes[demand.id]):
                boundary_node_ids.append(node)
                bnd_rows.append(row)
                bnd_cols.append(col)

            path_noise_costs[demand.id] = demand_path_noise_costs[demand.id]
            path_indicators[demand.id] = path_indicator

        # Spearman: rank-correlate the proxy's segment ordering against
        # the QoT model's. The STE is only a legitimate gradient
        # substitute to the extent the two agree on which segment is
        # worse; a correlation drifting toward 0 means the backward pass
        # is pointing somewhere the forward pass does not go.
        rank_corr = _spearman(
            torch.stack(proxy_flat).detach(), batched_qot_gsnr
        ) if len(proxy_flat) > 1 else float("nan")

        if demands:
            num_demands = len(demands)
            j_max = max(seg_counts)
            long_ = dict(dtype=torch.long, device=device)

            # index_put on a zeros tensor rather than in-place assignment:
            # out-of-place keeps the autograd path to segment_gsnr_flat
            # explicit, and the (row, col) pairs are unique so
            # accumulate=False is right.
            gsnr_matrix = torch.zeros(num_demands, j_max, device=device).index_put(
                (torch.tensor(seg_rows, **long_), torch.tensor(seg_cols, **long_)),
                torch.stack(segment_gsnr_flat),
            )

            # Per-segment linear noise for the allocation carry.
            #
            # NOTE this is db_to_linear_noise of the STE-BLENDED segment
            # GSNR, not the raw ASE proxy the spec's section 2 pseudocode
            # writes. The blended value is already "QoT forward value, proxy
            # gradient", so the carry's forward value is exactly the quantity
            # SegmentCombiner folds while its backward is still the proxy's —
            # which is what keeps d(devices)/d(path_indicator) nonzero
            # (spec section 4, "job 2"). With the raw proxy the carry would be
            # ASE-only and median-normalised, and neither
            # "carry equals true chunk noise" nor the oracle representability
            # test could hold. See this plan's Deviations section 2.
            seg_noise = db_to_linear_noise(
                gsnr_matrix.clamp(GSNR_MIN, GSNR_MAX)
            )
            seg_km_matrix = torch.zeros(num_demands, j_max, device=device).index_put(
                (torch.tensor(seg_rows, **long_), torch.tensor(seg_cols, **long_)),
                torch.tensor(seg_km_flat, dtype=torch.float32, device=device),
            )
            num_segments = torch.tensor(seg_counts, **long_)
            bar_db = bar_db_for_demands(
                demands, self._modulation_config, self._margin_db
            ).to(device)

            # ONE call site, wrapped in a selected context rather than
            # duplicated across an if/else. Deterministic decisions carry no
            # useful gradient, so hard_alloc runs under no_grad and building
            # the graph would only retain it (spec 2.5) — but that is the
            # ONLY difference between the two modes, and copying the argument
            # list to say so invites a kwarg added to one branch and not the
            # other, i.e. a soft/hard divergence no test would catch.
            ctx = torch.no_grad() if hard_alloc else contextlib.nullcontext()
            with ctx:
                a, a_physics = self.allocation_head.rollout(
                    seg_noise,
                    seg_km_matrix,
                    bar_db,
                    num_segments,
                    tau=tau,
                    hard=hard_alloc,
                    dropout_p=alloc_dropout_p,
                )

            # Scatter a onto (D, N) so the cost is written sum_n sum_d and
            # site_view falls out for free. -1 marks a padded column.
            bnd_matrix = torch.full(
                (num_demands, max(j_max - 1, 0)), -1, **long_
            )
            if boundary_node_ids:
                bnd_matrix = bnd_matrix.index_put(
                    (torch.tensor(bnd_rows, **long_), torch.tensor(bnd_cols, **long_)),
                    torch.tensor(boundary_node_ids, **long_),
                )
            alloc_by_node = torch.zeros(num_demands, self._num_nodes, device=device)
            real = bnd_matrix >= 0
            if real.any():
                rows_idx = torch.arange(num_demands, device=device).unsqueeze(1)
                alloc_by_node = alloc_by_node.index_put(
                    (rows_idx.expand_as(bnd_matrix)[real], bnd_matrix[real]),
                    a[real],
                    accumulate=True,      # one demand can cut twice at one node
                )

            path_gsnrs = self.segment_combiner.forward_batched(
                gsnr_matrix, a_physics, num_segments
            )
            for row, demand in enumerate(demands):
                gsnr_preds[demand.id] = path_gsnrs[row]

            alloc_outputs = AllocationOutputs(
                a=a,
                a_physics=a_physics,
                alloc_by_node=alloc_by_node,
                device_count=total_device_cost(alloc_by_node),
                site_view=site_view(alloc_by_node),
                seg_gsnr_db=gsnr_matrix,
                seg_noise=seg_noise,
                num_segments=num_segments,
                boundary_node_ids=bnd_matrix,
                demand_ids=[d.id for d in demands],
                ste_clamped_segments=clamped,
                proxy_qot_rank_corr=rank_corr,
            )
        else:
            # No demands: every per-demand tensor is empty in its row
            # dimension. site_view is written out rather than derived,
            # because max over a zero-length dim is an error, not 0.
            empty_dd = torch.zeros(0, 0, device=device)
            alloc_outputs = AllocationOutputs(
                a=empty_dd,
                a_physics=empty_dd,
                alloc_by_node=torch.zeros(0, self._num_nodes, device=device),
                device_count=torch.zeros((), device=device),
                site_view=torch.zeros(self._num_nodes, device=device),
                seg_gsnr_db=empty_dd,
                seg_noise=empty_dd,
                num_segments=torch.zeros(0, dtype=torch.long, device=device),
                boundary_node_ids=torch.zeros(0, 0, dtype=torch.long, device=device),
                demand_ids=[],
                ste_clamped_segments=clamped,
                proxy_qot_rank_corr=rank_corr,
            )

        return path_noise_costs, gsnr_preds, path_indicators, alloc_outputs
