"""Spec 2.6's trigger metric: is routing choosing badly-regenerable corridors?

REPORTED, NOT GATED. A persistent gap does not mean this stage failed — it
means the routing head is picking corridors whose regen candidates sit in
the wrong places, which is a different problem from allocation. If it
appears, spec 2.6 says the fix is approach C (an unpriced node propensity
feature), NOT approach B — and section 5 records why B was rejected
(it makes forward() non-idempotent, and every diagnose_*.py, evaluate_matrix
and the hard-eval pass assume forward is pure).

Zero gap is the good case and the expected one: under approach A the
Vlastelica backward already re-solves with the device term in grad_output,
so the solver is searching over candidate sets directly.

What is compared, per demand d:

    oracle_on_learned_route[d]   the oracle's minimum device count on the
                                 route the trained pipeline actually picks
    best_candidate_route[d]      the smallest of those minima over the
                                 demand's k=5 shortest-by-km candidate paths

and the reported aggregate is `sum_d (learned - best_candidate)`, summed
over the demands where BOTH sides are feasible. Positive means routing is
leaving devices on the table.

Only the ORACLE is used on both sides — never the head's own allocation.
The question here is about the corridor, not about the head's competence
on it, and mixing the two would make a routing finding unreadable whenever
the head happens to be over-buying (which is exactly `oracle_gap`'s job to
report separately).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml

# data/ is not a package (no __init__.py, not installed editable), so
# `from generate_qot_dataset import ...` needs it on sys.path explicitly —
# the same insert tests/conftest.py already does for the same reason.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from generate_qot_dataset import build_nx_graph, get_k_shortest_paths  # noqa: E402

from diffopt.modulation import bar_db_for_demands  # noqa: E402
from diffopt.pipeline import segment_path  # noqa: E402
from diffopt.placement.oracle import oracle_allocation  # noqa: E402
from diffopt.routing.shortest_path import dijkstra  # noqa: E402
from diffopt.qot.span_features import SPAN_FEATURE_DIM, span_feature_rows  # noqa: E402

from _common import (  # noqa: E402
    DiagContext,
    add_common_args,
    build_context,
    fixed_traffic_demands,
    schedule_at,
)


def edge_ids_along(nx_graph, node_path: List[int]) -> List[int]:
    """Ordered edge ids for a node path.

    Reads `edge_idx` off the graph `build_nx_graph` built, which is
    `enumerate(topology.undirected_edges)` — the SAME indexing
    `DiagContext.edges`, `segment_path` and the pipeline's `_edges` use, so
    the ids are directly interchangeable with the pipeline's own.
    """
    return [
        nx_graph[u][v]["edge_idx"] for u, v in zip(node_path, node_path[1:])
    ]


def segment_gsnr_db(ctx: DiagContext, segments: List[List[int]]) -> Optional[torch.Tensor]:
    """(len(segments),) QoT GSNR in dB, mirroring the pipeline's batched call.

    Same construction as `DiffONetPipeline.forward` step 5: `span_feature_rows`
    per segment, padded to THIS batch's true max span count (not the
    architectural `max_spans`), with `src_key_padding_mask` marking the real
    spans. Returns None when any segment is longer than `max_spans` — the
    pipeline raises in that case, and a candidate path the architecture
    cannot evaluate is not a candidate.
    """
    n_spans = [
        sum(ctx.edges[eid].num_spans for eid in seg) for seg in segments
    ]
    max_spans = ctx.cfg.get("max_spans_per_segment", 60)
    if max(n_spans) > max_spans:
        return None

    width = max(1, max(n_spans))
    feats = torch.zeros(len(segments), width, SPAN_FEATURE_DIM, device=ctx.device)
    mask = torch.zeros(len(segments), width, dtype=torch.bool, device=ctx.device)
    for row, seg in enumerate(segments):
        rows = span_feature_rows(
            ctx.topology, seg,
            channel_loading_fraction=ctx.cfg["pipeline"]["channel_loading_fraction"],
        )
        if rows:
            feats[row, : len(rows)] = torch.tensor(
                rows, dtype=torch.float32, device=ctx.device
            )
            mask[row, : len(rows)] = True

    with torch.no_grad():
        return ctx.qot_model(feats, mask)


def oracle_devices_on_path(
    ctx: DiagContext, node_path: List[int], demand, bar: torch.Tensor, nx_graph
) -> Optional[int]:
    """Oracle minimum device count for one demand on one concrete node path.

    None when the path is unevaluable (a segment longer than `max_spans`) or
    when no allocation on it is feasible (`oracle.feasible` False — some
    single segment busts the bar on its own, so cutting cannot help).
    """
    eids = edge_ids_along(nx_graph, node_path)
    segments, _ = segment_path(
        eids, demand.src, ctx.regen_candidates, ctx.edges, demand.dst
    )
    seg_db = segment_gsnr_db(ctx, segments)
    if seg_db is None:
        return None
    res = oracle_allocation(
        seg_db.unsqueeze(0),
        bar.reshape(1),
        torch.tensor([len(segments)]),
    )
    if not bool(res.feasible.item()):
        return None
    return int(res.count.item())


def candidate_oracle_devices(
    ctx: DiagContext, demand, bar: torch.Tensor, nx_graph, k: int = 5
) -> Optional[int]:
    """Minimum devices over the demand's k shortest-by-km candidate paths.

    Reuses data/generate_qot_dataset.py's get_k_shortest_paths — the SAME
    candidate generator Phase 1a's dataset used and Phase 1d's greedy
    baseline will use, so this comparison is against the baseline's actual
    option set rather than a second implementation of "k shortest".
    """
    best = None
    for node_path in get_k_shortest_paths(nx_graph, demand.src, demand.dst, k):
        n = oracle_devices_on_path(ctx, node_path, demand, bar, nx_graph)
        if n is None:
            continue
        best = n if best is None else min(best, n)
    return best


def learned_node_path(ctx: DiagContext, demand) -> List[int]:
    """Node path the trained pipeline routes this demand over.

    Routing depends only on edge_log_weight, not on hard vs. soft
    decisions, so a fresh Dijkstra call at the pipeline's current (unit-
    mean-renormalised) weights IS the routing the hard rollout evaluated,
    not a re-derivation of it (docs/investigations/
    pipeline_profile_and_restoration_scaling.md, Finding 2 /
    open_followups.md #7b).
    """
    raw_edge_weights = F.softplus(ctx.pipeline.edge_log_weight)
    edge_weights = (
        raw_edge_weights / raw_edge_weights.mean().clamp_min(1e-12)
    ).detach().numpy()
    ei = ctx.pipeline._edge_index.numpy()
    _, eids = dijkstra(
        edge_weights, ei, demand.src, demand.dst, ctx.pipeline._num_nodes,
        return_order=True,
    )
    nodes = [demand.src]
    for eid in eids:
        e = ctx.edges[eid]
        nodes.append(e.dst if nodes[-1] == e.src else e.src)
    return nodes


def main() -> None:
    ap = add_common_args(
        argparse.ArgumentParser(description=__doc__),
        default_config="configs/experiment/constrained_stress.yaml",
        with_demands=False,
    )
    ap.add_argument("--k", type=int, default=5, help="Candidate paths per demand")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ctx = build_context(cfg, checkpoint_path=args.checkpoint)
    nx_graph = build_nx_graph(ctx.topology)

    demands, _excluded = fixed_traffic_demands(ctx)
    _, vlastelica_lambda = schedule_at(cfg, ctx.cfg["training"]["epochs_e2e"])
    if ctx.ckpt is not None and "vlastelica_lambda" in ctx.ckpt:
        vlastelica_lambda = ctx.ckpt["vlastelica_lambda"]

    margin_db = cfg["constraint"]["margin_db"]
    bar_db = bar_db_for_demands(demands, ctx.mod_cfg, margin_db)

    # One hard pass gives BOTH the learned routes and the canonical
    # per-demand oracle counts on them — the same quantities train.py's
    # hard_rollout logs as `oracle_devices`, so the learned side of this
    # comparison is the number the training log already reported.
    with torch.no_grad():
        _, _, path_indicators, alloc = ctx.pipeline(
            demands, lambda_=vlastelica_lambda, hard_alloc=True
        )
        oracle = oracle_allocation(alloc.seg_gsnr_db, bar_db, alloc.num_segments)

    learned_by_id: Dict[int, Optional[int]] = {}
    for row, did in enumerate(alloc.demand_ids):
        learned_by_id[did] = (
            int(oracle.count[row].item()) if bool(oracle.feasible[row].item()) else None
        )

    bar_by_id = {d.id: bar_db[i] for i, d in enumerate(demands)}

    total_learned = 0
    total_best = 0
    gaps: List[Tuple[int, int, int, int]] = []   # (gap, demand_id, learned, best)
    n_compared = 0
    n_learned_infeasible = 0
    n_no_candidate = 0
    n_negative = 0

    for demand in demands:
        learned = learned_by_id.get(demand.id)
        if learned is None:
            n_learned_infeasible += 1
            continue
        best = candidate_oracle_devices(
            ctx, demand, bar_by_id[demand.id], nx_graph, k=args.k
        )
        if best is None:
            n_no_candidate += 1
            continue
        n_compared += 1
        total_learned += learned
        total_best += best
        gap = learned - best
        if gap != 0:
            gaps.append((gap, demand.id, learned, best))
        if gap < 0:
            n_negative += 1

    print(f"config          : {args.config}")
    print(f"checkpoint      : {args.checkpoint or cfg.get('checkpoint_dir') + '/best_e2e.pt'}")
    print(f"demands         : {len(demands)} in the constraint set, k={args.k} candidates each")
    print(f"compared        : {n_compared}")
    print(f"skipped         : {n_learned_infeasible} learned route infeasible for the oracle, "
          f"{n_no_candidate} with no feasible candidate path")
    print()
    print(f"oracle_on_learned_route  total = {total_learned}")
    print(f"best_candidate_route     total = {total_best}")
    print()
    print(f"sum_d (oracle_on_learned_route[d] - best_candidate_route[d]) = "
          f"{total_learned - total_best}")
    print(f"demands with a nonzero gap: {len(gaps)} "
          f"({sum(1 for g in gaps if g[0] > 0)} positive, {n_negative} negative)")

    if n_negative:
        # The learned route beating every k-shortest-by-km candidate is not a
        # bug: the candidate set is shortest by KM, and the router optimises
        # ASE noise, so it is allowed to find a longer-but-cleaner corridor.
        print("  (negative = the learned route beats every km-shortest candidate; "
              "the candidate set is shortest-by-km, not shortest-by-noise)")

    if gaps:
        print()
        print("largest gaps (gap, demand_id, learned, best_candidate):")
        for g in sorted(gaps, key=lambda t: -t[0])[:20]:
            print(f"  {g[0]:+4d}  demand {g[1]:4d}  learned {g[2]:3d}  best {g[3]:3d}")


if __name__ == "__main__":
    main()
