"""Fixed traffic matrix — the constraint set the duals in diffopt/loss.py act on.

Before this module, diffopt/train.py called `generate_demands(..., seed=epoch)`
and drew 100 fresh random demands every epoch. "All demands feasible" cannot be
stated, let alone enforced, against a set that is replaced each epoch, and a
per-demand dual is meaningless without demand identity persisting across
epochs (spec §1 finding #4).

The matrix is NOT committed as a file. It regenerates deterministically from
(topology, seed, scale, alpha) and `test_traffic.py` pins a checksum of the
result. This catches silent upstream drift without a committed artifact that
can go stale — a deliberate choice given this project's history of a dependency
pin that became uninstallable (invariants.md, "Pin a tag, never a branch
commit").
"""
from __future__ import annotations

import hashlib
import heapq
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from multilayer_optical_network.model.traffic import (
    generate_demands as _gravity_demands,
)

from diffopt.demands import Demand
from diffopt.pipeline import segment_path
from diffopt.qot.span_features import SPAN_FEATURE_DIM, span_feature_rows
from diffopt.topology import Topology


# `alpha` is the distance exponent in the gravity kernel
# (weight ~ mass(u)*mass(v) / dist(u,v)^alpha). It is derived from a *named*
# scenario rather than set directly in configs, so the two settings stay
# reportable things rather than free-floating numbers.
#
#   realistic (alpha=1.0): gravity falls off with distance; big pipes land on
#       short hops.
#   stress    (alpha=0.0): pure mass-product, no distance falloff; changes
#       which pairs clear the volume floor and their volumes, not just their
#       bitrate assignment. Spec design doc §8.4's "-0.008 corr, 20/865
#       infeasible" evidence for the decoupled-bitrate case was measured by
#       taking the alpha=1 pair set and shuffling bitrates at random, NOT by
#       running this module's own alpha=0.0 kernel — the two are different
#       constructions and this module's alpha=0.0 output should not be
#       assumed to reproduce that number.
#
# Both are kept deliberately (spec §3): reporting only `realistic` would
# demonstrate the dual machinery on a problem where the constraint never binds;
# reporting only `stress` would overstate difficulty relative to real traffic.
SCENARIO_ALPHA: Dict[str, float] = {"realistic": 1.0, "stress": 0.0}

# A pair offering less than the smallest lightpath minus half a bitrate step
# (300 - 25 = 275 G) carries no lightpath at all and is dropped rather than
# rounded up to 300 G.
_MIN_VOLUME_MARGIN_GBPS = 25.0


def scenario_alpha(scenario: str) -> float:
    """Map a named traffic scenario to its gravity distance exponent."""
    if scenario not in SCENARIO_ALPHA:
        raise ValueError(
            f"unknown traffic scenario {scenario!r}; "
            f"valid values: {sorted(SCENARIO_ALPHA)}"
        )
    return SCENARIO_ALPHA[scenario]


def build_traffic_matrix(
    topology: Topology,
    *,
    seed: int,
    scale: float,
    alpha: float,
    bitrate_options: List[float],
) -> List[Demand]:
    """Build the fixed traffic matrix for a topology.

    Wraps upstream `generate_demands` with `aggregate=True` (one record per
    pair carrying its raw unquantized offered Gbps, an OD-matrix shape rather
    than 100 G grooming units), `undirected=True` (one record per unordered
    pair — gravity weight is symmetric, so both directions are exact
    duplicates), and `protected_fraction=0.0` (this project has no protection
    concept; a `protected` flag would be dead weight on every record).

    Each pair's offered volume is mapped to the NEAREST member of
    `bitrate_options`; pairs below `min(bitrate_options) - 25` are dropped.

    Args:
        topology:        Loaded Topology (a bare OpticalNetworkModel subclass —
                         upstream derives node ids from OMS endpoints when
                         `list_routers` is absent, which is why the v0.1.2 pin
                         is required).
        seed:            Matrix identity. Enters upstream only through a
                         deterministic per-node mass jitter, so a fixed seed is
                         byte-stable and a different seed gives a distinct
                         held-out matrix.
        scale:           Total offered load in Gbps, spread across pairs by
                         gravity weight. Calibrated per scenario to land in
                         roughly 500-900 demands.
        alpha:           Gravity distance exponent — use `scenario_alpha()`.
        bitrate_options: The 11 valid bitrates. Every emitted `bitrate_gbps` is
                         an exact member, because `ModulationConfig.
                         required_snr_threshold` is an exact-float dict lookup
                         with no interpolation (invariants.md).

    Returns:
        List of `Demand` with ids contiguous `0..N-1`. Contiguity is
        load-bearing: `Demand.id` indexes the per-demand dual vector in
        `diffopt.loss.compute_loss`.
    """
    options = sorted(float(b) for b in bitrate_options)
    volume_floor = options[0] - _MIN_VOLUME_MARGIN_GBPS

    raw = _gravity_demands(
        topology,
        seed=seed,
        scale=scale,
        alpha=alpha,
        aggregate=True,
        undirected=True,
        protected_fraction=0.0,
    )

    demands: List[Demand] = []
    for record in raw:
        volume = float(record["demand_gbps"])
        if volume < volume_floor:
            continue
        # Nearest option; ties broken toward the lower bitrate so the mapping
        # is a deterministic function of the volume alone.
        bitrate = min(options, key=lambda b: (abs(b - volume), b))
        demands.append(Demand(
            id=len(demands),
            # Upstream node ids are strings ("0", "1", ... "131"); nodes are
            # integer IDs only downstream (invariants.md).
            src=int(record["src"]),
            dst=int(record["dst"]),
            bitrate_gbps=bitrate,
        ))
    return demands


def traffic_matrix_checksum(demands: List[Demand]) -> str:
    """Stable 16-hex-char digest of a matrix's (src, dst, bitrate) content.

    Deliberately excludes `id`, which is a positional artifact — this way the
    digest is unchanged by a renumbering that preserves content, and changes
    when the actual demand set does.
    """
    payload = ";".join(
        f"{d.src}-{d.dst}-{d.bitrate_gbps:.6f}" for d in demands
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Preflight: exclude the impossible
# ---------------------------------------------------------------------------

def shortest_path_edges_by_km(
    topology: Topology, src: int, dst: int
) -> Optional[List[int]]:
    """Edge ids of the minimum-kilometre path, in traversal order src -> dst.

    Deliberately NOT routed by `EdgeWeightNet`: its routing changes during
    training, which would make the traffic matrix depend on whichever
    checkpoint happened to build it. Shortest-by-km is a fixed property of the
    topology.

    Returns `None` if `dst` is unreachable from `src`, and `[]` when
    `src == dst`.

    `diffopt.routing.shortest_path.dijkstra` is not reused here because it
    returns an unordered `(E,)` binary indicator, and `segment_path` needs
    traversal order — recovering the order from the indicator would mean a
    second graph walk over the same data.
    """
    edges = topology.undirected_edges
    adj: Dict[int, List[Tuple[int, int, float]]] = {}
    for eid, e in enumerate(edges):
        adj.setdefault(e.src, []).append((e.dst, eid, e.length_km))
        adj.setdefault(e.dst, []).append((e.src, eid, e.length_km))

    dist: Dict[int, float] = {src: 0.0}
    prev: Dict[int, Tuple[int, int]] = {}
    settled: set = set()
    heap: List[Tuple[float, int]] = [(0.0, src)]

    while heap:
        d, u = heapq.heappop(heap)
        if u in settled:
            continue
        settled.add(u)
        if u == dst:
            break
        for v, eid, km in adj.get(u, []):
            nd = d + km
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = (u, eid)
                heapq.heappush(heap, (nd, v))

    if dst not in dist:
        return None

    ordered: List[int] = []
    node = dst
    while node != src:
        parent, eid = prev[node]
        ordered.append(eid)
        node = parent
    ordered.reverse()
    return ordered


def preflight_filter(
    topology: Topology,
    demands: Sequence[Demand],
    *,
    qot_model: torch.nn.Module,
    segment_combiner: torch.nn.Module,
    modulation_config,
    margin_db: float,
    channel_loading_fraction: float = 0.5,
    max_spans: int = 60,
) -> Tuple[List[Demand], List[Tuple[Demand, float]]]:
    """Drop demands that are infeasible under the most favourable conditions.

    Every demand is evaluated with **all** regen candidates active and routed
    **shortest-by-km**. That route/placement pair is the most favourable one
    available, so a demand infeasible under it is infeasible under every
    placement the model could learn.

    This is a **necessary, not sufficient** screen: a demand that survives it
    may still be unreachable under the routing the model actually learns.
    Those surface later as duals pinned at `dual_max` in
    `diffopt/train.py`'s end-of-run report, not here.

    The screen is non-conservative by construction. `SegmentCombiner` folds
    chunks with an exact max (invariants.md, "Segment combiner"), so the GSNR
    it reports here is exactly the one production arithmetic computes for this
    route and placement — the screen can neither over- nor under-state a
    demand's best case by an approximation margin. It used to over-estimate
    noise by the soft-max's relative `t*ln2` (~0.03 dB at t=0.01), which could
    exclude a demand sitting that close to the bar; the exact fold removes
    that margin entirely.

    Args:
        margin_db: The same delta the constrained loss adds inside the hinge.
            Included here on purpose — a demand that cannot reach
            `threshold + margin` even ideally can never satisfy the
            constraint, so excluding it now is the difference between a
            reported exclusion and a silently non-converging dual.

    Returns:
        `(kept, excluded)`. `kept` is renumbered with contiguous ids `0..N-1`
        in input order — `Demand.id` indexes the dual vector, so a gap would
        attach every later dual to the wrong demand. `excluded` is a list of
        `(demand, shortfall_db)` with `shortfall_db = threshold + margin -
        best_case_gsnr`, or `inf` when there is no route at all.
    """
    device = next(qot_model.parameters()).device
    candidate_set = set(topology.regen_candidate_nodes)
    edges = list(topology.undirected_edges)

    # Pass 1 — route and segment everything, collecting segments for one
    # batched QoT call (the same batching pipeline.forward step 5 does; a
    # per-segment call over ~900 demands is ~6x slower for no benefit).
    routed: List[Optional[Tuple[List[List[int]], List[int]]]] = []
    all_segments: List[List[int]] = []
    for demand in demands:
        ordered = shortest_path_edges_by_km(topology, demand.src, demand.dst)
        if ordered is None:
            routed.append(None)
            continue
        segments, boundary_nodes = segment_path(
            ordered, demand.src, candidate_set, edges, demand.dst
        )
        routed.append((segments, boundary_nodes))
        all_segments.extend(segments)

    with torch.no_grad():
        if all_segments:
            all_rows = [
                span_feature_rows(
                    topology, seg,
                    channel_loading_fraction=channel_loading_fraction,
                )
                for seg in all_segments
            ]
            batch_max_spans = max(1, max(len(rows) for rows in all_rows))
            if batch_max_spans > max_spans:
                raise ValueError(
                    f"Shortest-by-km routing produced a transparent segment of "
                    f"{batch_max_spans} spans, but max_spans={max_spans} is a "
                    f"hard architecture parameter (SpanAttentionQoT's positional "
                    f"embedding is sized to it)."
                )
            n_total = len(all_segments)
            span_feats = torch.zeros(
                n_total, batch_max_spans, SPAN_FEATURE_DIM, device=device
            )
            padding_mask = torch.zeros(
                n_total, batch_max_spans, dtype=torch.bool, device=device
            )
            for i, rows in enumerate(all_rows):
                if rows:
                    span_feats[i, :len(rows)] = torch.tensor(
                        rows, dtype=torch.float32, device=device
                    )
                    padding_mask[i, :len(rows)] = True   # True = real span
            batched_gsnr = qot_model(span_feats, padding_mask)
        else:
            batched_gsnr = torch.zeros(0, device=device)

        # Pass 2 — combine each demand's segments with every boundary
        # regenerator fully active (p = 1.0) and test against the bar.
        kept: List[Demand] = []
        excluded: List[Tuple[Demand, float]] = []
        flat_idx = 0
        one = torch.ones((), device=device)

        for demand, entry in zip(demands, routed):
            threshold = modulation_config.required_snr_threshold(demand.bitrate_gbps)
            if entry is None:
                excluded.append((demand, float("inf")))
                continue
            segments, boundary_nodes = entry
            segment_gsnrs = [
                batched_gsnr[flat_idx + i] for i in range(len(segments))
            ]
            flat_idx += len(segments)
            path_gsnr = segment_combiner(segment_gsnrs, [one for _ in boundary_nodes])
            shortfall = threshold + margin_db - float(path_gsnr.item())
            if shortfall > 0.0:
                excluded.append((demand, shortfall))
            else:
                kept.append(demand)

    return [d._replace(id=i) for i, d in enumerate(kept)], excluded
