"""Spec §1.2: how much headroom does the head leave on the table at every
boundary it decides, versus the oracle's own greedy rule?

Feature 4 is `-10 log10(c + n_next) - bar_d` — the headroom the chunk would
still have if this boundary were NOT cut and the chunk swallowed one more
segment. `oracle_allocation` cuts iff `c + n_k > bar_noise`, i.e. iff
feature 4 < 0 (diffopt/placement/allocation.py's feature-layout comment).
So a head decision with feature 4 > 0 is a cut the greedy-optimal rule
would NOT have taken there — it is a place the head is spending a device
it did not need to, not (necessarily) a place the constraint required it.

This script replays feature 4 for every valid boundary of every demand
under the DEPLOYED head's own hard decisions' carry: re-walk each demand's
segments using `alloc.a` (the deployed cuts) to reset the carry exactly the
way `AllocationHead.rollout`'s loop does (`c = c*(1-a_phys) + n_next`, but
since these are hard 0/1 decisions here, `c` resets to 0 then accumulates,
exactly matching `rollout`'s own physics for a hard pass).

THE SANITY ROW. An earlier, throwaway version of this script printed an
"ORACLE cuts (sanity: must all be < 0)" row, but computed it by replaying
feature 4 under the HEAD's own carry trajectory — the same replay as above
— rather than under the ORACLE's own carry, which is a different state
sequence whenever the head and the oracle disagree about where to cut. That
made the "must be < 0" claim measure the wrong quantity. Fixed here (option
(a) of the plan's fix, chosen because `oracle_allocation`'s return value —
`OracleResult.a`, a (D, J-1) 0/1 tensor of the oracle's OWN cut decisions —
makes replaying its own carry exactly as easy as replaying the head's): the
sanity row below replays feature 4 a SECOND time, walking the SAME boundary
sequence but resetting the carry on `oracle.a` instead of `alloc.a`, in
float64 to match `oracle_allocation`'s own accumulation precision. Every
value in that row is over an oracle CUT, so by construction it is exactly
the quantity `oracle_allocation` itself thresholds at 0 — the row should
therefore read all negative, restoring the sanity check's actual meaning.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

# This project's `diffopt` is pip-installed editable FROM THE SHARED CHECKOUT
# (see pyproject.toml), not from any one worktree. A plain `python
# scripts/foo.py` invocation does not put the cwd on sys.path (only the
# script's own directory is added), so without this insert `import diffopt`
# would silently resolve to the shared checkout's copy instead of this
# worktree's — the same class of bug `diagnose_route_candidate_gap.py`
# already works around for `data/`, just one directory up. Must run before
# any `diffopt`/`_common` import below.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffopt.modulation import bar_db_for_demands  # noqa: E402
from diffopt.placement.allocation import _EPS  # noqa: E402
from diffopt.placement.oracle import oracle_allocation  # noqa: E402
from diffopt.qot.segment_combiner import GSNR_MAX, GSNR_MIN, db_to_linear_noise  # noqa: E402

from _common import add_common_args, build_context, fixed_traffic_demands, schedule_at  # noqa: E402


# ---------------------------------------------------------------------------
# The carry replay — pure tensor arithmetic, no pipeline/checkpoint state.
# Driven by an externally supplied 0/1 `cuts` tensor so the SAME function
# replays either the head's own hard decisions or the oracle's.
# ---------------------------------------------------------------------------

def replay_boundary_feature4(
    seg_noise: torch.Tensor,
    num_segments: torch.Tensor,
    cuts: torch.Tensor,
    bar_db: torch.Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(f4, valid), both (D, J-1).

    Mirrors `AllocationHead.rollout`'s carry walk exactly: `c = c + n[:, k]`
    then feature 4 against `n[:, k+1]`, then `c = c * (1 - cuts[:, k])`.
    `valid` follows the SAME `num_segments`-derived mask `rollout` uses for
    `cut_valid` (a boundary is real only if its following segment exists) —
    not `boundary_node_ids`, so this needs no pipeline-internal indexing to
    stay correct.

    `dtype` defaults to `seg_noise`'s own dtype (float32, matching the real
    deployed rollout) — pass `torch.float64` to replay the ORACLE's own
    carry, which `oracle_allocation` accumulates in float64.
    """
    d, j = seg_noise.shape
    device = seg_noise.device
    dtype = dtype or seg_noise.dtype

    positions = torch.arange(j, device=device)
    seg_valid = positions.unsqueeze(0) < num_segments.unsqueeze(1)
    valid = seg_valid[:, 1:] if j > 1 else seg_valid[:, :0]

    n = seg_noise.to(dtype) * seg_valid.to(dtype)
    bar = bar_db.to(dtype)
    cuts = cuts.to(dtype)

    c = torch.zeros(d, dtype=dtype, device=device)
    cols: List[torch.Tensor] = []
    for k in range(j - 1):
        c = c + n[:, k]
        n_next = n[:, k + 1]
        cols.append(-10.0 * torch.log10(c + n_next + _EPS) - bar)
        c = c * (1.0 - cuts[:, k])

    if not cols:
        empty = torch.zeros(d, 0, dtype=dtype, device=device)
        return empty, valid
    return torch.stack(cols, dim=1), valid


# ---------------------------------------------------------------------------
# Pure aggregation/formatting helpers — unit-tested in
# tests/test_diagnose_scripts.py with hand-built lists, never a real replay.
# ---------------------------------------------------------------------------

def percentile_summary(values: Sequence[float]) -> Dict[str, float]:
    """count/median/p10/p90 of a list of floats. Empty -> count 0, NaNs."""
    if not values:
        return {"count": 0, "median": math.nan, "p10": math.nan, "p90": math.nan}
    arr = np.asarray(list(values), dtype=float)
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def cut_rate_above_zero(f4_at_cuts: Sequence[float]) -> Tuple[int, int]:
    """(count of cuts with f4 > 0, total cuts). f4 > 0 at a cut means the
    greedy-optimal rule (cut iff f4 < 0) would not have cut there."""
    values = list(f4_at_cuts)
    return sum(1 for v in values if v > 0), len(values)


def demand_oracle_vs_head(
    oracle_counts: Sequence[int], head_counts: Sequence[int]
) -> Dict[str, int]:
    """Per-demand accounting matching the spec table's
    'demands with oracle count 0 : N -> head buys M' rows: split demands by
    whether the oracle needed any device at all, and total what the head
    actually bought in each bucket (plus what the oracle wanted, in the
    oracle>0 bucket)."""
    if len(oracle_counts) != len(head_counts):
        raise ValueError("oracle_counts and head_counts must be the same length")
    zero_n = zero_head = pos_n = pos_head = pos_oracle = 0
    for oc, hc in zip(oracle_counts, head_counts):
        if oc == 0:
            zero_n += 1
            zero_head += hc
        else:
            pos_n += 1
            pos_head += hc
            pos_oracle += oc
    return {
        "oracle_zero_demands": zero_n,
        "oracle_zero_head_buys": zero_head,
        "oracle_pos_demands": pos_n,
        "oracle_pos_head_buys": pos_head,
        "oracle_pos_oracle_wants": pos_oracle,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = add_common_args(
        argparse.ArgumentParser(description=__doc__),
        default_config="configs/experiment/constrained_stress.yaml",
        with_demands=False,
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ctx = build_context(cfg, checkpoint_path=args.checkpoint)
    demands, _excluded = fixed_traffic_demands(ctx)

    _, vlastelica_lambda = schedule_at(cfg, cfg["training"]["epochs_e2e"])
    if ctx.ckpt is not None and "vlastelica_lambda" in ctx.ckpt:
        vlastelica_lambda = ctx.ckpt["vlastelica_lambda"]

    margin_db = cfg["constraint"]["margin_db"]

    with torch.no_grad():
        _, _gsnr_preds, _path_indicators, alloc = ctx.pipeline(
            demands, lambda_=vlastelica_lambda, hard_alloc=True,
        )
        bar_db = bar_db_for_demands(demands, ctx.mod_cfg, margin_db).to(alloc.a.device)
        oracle = oracle_allocation(alloc.seg_gsnr_db, bar_db, alloc.num_segments)

    # Head's own carry, head's own decisions, real deployed precision.
    f4_head, valid = replay_boundary_feature4(
        alloc.seg_noise, alloc.num_segments, alloc.a, bar_db,
    )

    # Oracle's own carry, oracle's own decisions, float64 — matching
    # oracle_allocation's own accumulation order exactly (clamp -> .double()
    # -> db_to_linear_noise), see the module docstring's "sanity row" note.
    seg_noise_oracle = db_to_linear_noise(
        alloc.seg_gsnr_db.clamp(GSNR_MIN, GSNR_MAX).double()
    )
    f4_oracle, valid_oracle = replay_boundary_feature4(
        seg_noise_oracle, alloc.num_segments, oracle.a, bar_db.double(),
        dtype=torch.float64,
    )

    valid_mask = valid.bool()
    head_cuts = (alloc.a > 0.5) & valid_mask
    head_nocuts = valid_mask & ~(alloc.a > 0.5)

    f4_cut_values = f4_head[head_cuts].tolist()
    f4_nocut_values = f4_head[head_nocuts].tolist()

    n_demands = len(demands)
    n_valid_boundaries = int(valid_mask.sum().item())
    n_cuts = int(head_cuts.sum().item())

    per_demand_head = alloc.a.sum(dim=1).round().long().tolist()
    per_demand_oracle = oracle.count.tolist()
    dem_summary = demand_oracle_vs_head(per_demand_oracle, per_demand_head)

    summary_cut = percentile_summary(f4_cut_values)
    summary_nocut = percentile_summary(f4_nocut_values)
    above, total = cut_rate_above_zero(f4_cut_values)
    pct_above = 100.0 * above / total if total else math.nan

    oracle_valid_mask = valid_oracle.bool()
    oracle_cuts = (oracle.a > 0.5) & oracle_valid_mask
    f4_oracle_cut_values = f4_oracle[oracle_cuts].tolist()
    oracle_summary = percentile_summary(f4_oracle_cut_values)
    n_oracle_sanity_violations = sum(1 for v in f4_oracle_cut_values if v >= 0)

    print(f"config          : {args.config}")
    print(f"checkpoint      : {args.checkpoint or cfg.get('checkpoint_dir') + '/best_e2e.pt'}")
    print()
    print(f"{n_demands} demands | {n_valid_boundaries} valid boundaries | {n_cuts} cuts")
    print(
        f"demands with oracle count 0 : {dem_summary['oracle_zero_demands']:3d} "
        f"-> head buys {dem_summary['oracle_zero_head_buys']}"
    )
    print(
        f"demands with oracle count >0: {dem_summary['oracle_pos_demands']:3d} "
        f"-> head buys {dem_summary['oracle_pos_head_buys']} "
        f"(oracle wants {dem_summary['oracle_pos_oracle_wants']})"
    )
    print(
        f"f4 where head CUTS      : median {summary_cut['median']:+.2f} dB   "
        f"p10 {summary_cut['p10']:+.2f}   p90 {summary_cut['p90']:+.2f}"
    )
    print(
        f"f4 where head does NOT  : median {summary_nocut['median']:+.2f} dB   "
        f"p10 {summary_nocut['p10']:+.2f}   p90 {summary_nocut['p90']:+.2f}"
    )
    print(f"cuts at f4 > 0          : {above} of {total}  ({pct_above:.1f}%)")
    print()
    print(
        "ORACLE cuts, replayed under the ORACLE's OWN carry "
        "(sanity: must all be < 0):"
    )
    print(
        f"  median {oracle_summary['median']:+.2f} dB   "
        f"p10 {oracle_summary['p10']:+.2f}   p90 {oracle_summary['p90']:+.2f}   "
        f"violations (f4 >= 0): {n_oracle_sanity_violations} of {oracle_summary['count']}"
    )


if __name__ == "__main__":
    main()
