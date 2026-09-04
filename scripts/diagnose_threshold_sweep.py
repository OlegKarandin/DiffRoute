"""Spec §7's decision-threshold sweep: shift `net[-1].bias` by `delta`,
re-run `diffopt.train.hard_rollout`, and read off where hard_num_violated
first hits 0 without buying more than 1.25x the oracle's device floor.

Score is monotone in the AllocationHead's final-layer bias (a bigger bias
makes every boundary's score bigger, i.e. more likely to cut), so adding a
constant `delta` to it sweeps the DECISION THRESHOLD with the head's own
*ordering* of boundaries held fixed — it is not retraining, and it is not
the same axis as the score itself once route_context/lookahead differ
between arms. Routes are untouched (`edge_log_weight` is never touched),
which is why `oracle_devices` is the same number at every delta below: the
oracle is a property of the routes and the per-segment GSNRs, neither of
which this sweep changes.

PLAN DEVIATION 5 (binding — see
.superpowers/sdd/plan-docs-superpowers-specs-2026-08-25-a-cozy-whistle/task-3-brief.md):
a fixed literal delta grid is not comparable across arms whose score scale
differs (e.g. arm 3's `-alpha*f4` vs a plain MLP head). The default grid is
therefore derived from THIS checkpoint's own observed score range
(`diffopt.train.alloc_score_stats`, a soft pass at the checkpoint's own
tau) rather than a fixed table of deltas — pass `--delta-grid` to override
it with an explicit comma-separated list when you need one.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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

from diffopt.train import alloc_score_stats, hard_rollout  # noqa: E402

from _common import add_common_args, build_context, fixed_traffic_demands, schedule_at  # noqa: E402

# One row of the sweep table: (delta, hard_num_devices, hard_num_sites,
# hard_num_violated, hard_worst_margin_db, oracle_devices, oracle_gap).
SweepRow = Tuple[float, int, int, int, float, int, int]


# ---------------------------------------------------------------------------
# Pure helpers — unit-tested in tests/test_diagnose_scripts.py without a
# checkpoint, a pipeline, or a real sweep.
# ---------------------------------------------------------------------------

def parse_delta_grid(text: str) -> List[float]:
    """'-2.0,-1.0,0.0' -> [-2.0, -1.0, 0.0]. Whitespace around commas is fine."""
    return [float(tok.strip()) for tok in text.split(",") if tok.strip()]


def build_score_delta_grid(
    score_min: float,
    score_max: float,
    *,
    num_points: int = 10,
    margin_frac: float = 0.1,
) -> List[float]:
    """~num_points evenly spaced deltas spanning [score_min, score_max], padded
    by margin_frac of the span on each side.

    Degenerate case (span == 0, e.g. a head whose final layer is still
    zero-weight so every boundary scores identically, or there were no
    boundaries at all so alloc_score_stats returned NaNs): falls
    back to a fixed +-1.0 window around score_min (or around 0.0 if score_min
    itself is not finite) rather than returning a single-point or NaN grid,
    since a delta sweep with one point cannot locate a threshold crossing.
    """
    if not (math.isfinite(score_min) and math.isfinite(score_max)):
        score_min = score_max = 0.0
    span = score_max - score_min
    if span <= 0.0:
        lo, hi = score_min - 1.0, score_min + 1.0
    else:
        pad = span * margin_frac
        lo, hi = score_min - pad, score_max + pad
    n = max(num_points, 2)
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def first_gate1_delta(
    rows: Sequence[Tuple[float, int, int, int]],
) -> Optional[float]:
    """First `delta` (in the order given — the sweep order) where spec §7
    gate 1 holds: hard_num_violated == 0 AND hard_num_devices <=
    1.25 * oracle_devices. `rows` is (delta, hard_num_devices,
    hard_num_violated, oracle_devices). None if no row qualifies.
    """
    for delta, hard_num_devices, hard_num_violated, oracle_devices in rows:
        if hard_num_violated == 0 and hard_num_devices <= 1.25 * oracle_devices:
            return delta
    return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = add_common_args(
        argparse.ArgumentParser(description=__doc__),
        default_config="configs/experiment/constrained_stress.yaml",
        with_demands=False,
    )
    ap.add_argument(
        "--delta-grid", type=str, default=None,
        help="Comma-separated deltas (score units) OVERRIDING the default "
             "score-range-derived grid, e.g. -2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0",
    )
    ap.add_argument(
        "--grid-points", type=int, default=10,
        help="Number of points in the default derived grid (ignored with --delta-grid)",
    )
    ap.add_argument(
        "--grid-margin-frac", type=float, default=0.1,
        help="Fractional margin padded onto each side of the observed score "
             "range for the default derived grid (ignored with --delta-grid)",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ctx = build_context(cfg, checkpoint_path=args.checkpoint)
    demands, _excluded = fixed_traffic_demands(ctx)

    # Canonical lambda/tau recovery (mirrors diagnose_route_candidate_gap.py's
    # main()): prefer the loaded checkpoint's OWN saved values over a
    # recomputation from schedule_at at a guessed epoch, falling back to
    # schedule_at only when there is no checkpoint (or it predates a field).
    schedule_epoch = (
        ctx.ckpt["epoch"] if ctx.ckpt is not None and "epoch" in ctx.ckpt
        else cfg["training"]["epochs_e2e"]
    )
    tau, vlastelica_lambda = schedule_at(cfg, schedule_epoch)
    if ctx.ckpt is not None and "vlastelica_lambda" in ctx.ckpt:
        vlastelica_lambda = ctx.ckpt["vlastelica_lambda"]

    margin_db = cfg["constraint"]["margin_db"]

    if args.delta_grid is not None:
        delta_grid = parse_delta_grid(args.delta_grid)
    else:
        # A soft pass at the checkpoint's own tau, purely to recover the raw
        # score range the deployed head produces on REAL boundaries.
        #
        # The grid this feeds is a sweep of the final-layer BIAS, so it has to
        # span the true score range. alloc_score_stats used to invert
        # a = sigmoid(s/tau) to get there, which is exact only while the head
        # is unsaturated: on an alloc_ste checkpoint `a` is exactly 0/1 and
        # every boundary read back as a float32 clamp, so the grid spanned
        # +-87.3*tau regardless of where the head actually sat. It now reads
        # alloc.score directly and the range is the real one.
        with torch.no_grad():
            _, _, _, soft_alloc = ctx.pipeline(
                demands, tau=tau, lambda_=vlastelica_lambda, hard_alloc=False,
            )
        _score_mean, score_min, score_max = alloc_score_stats(soft_alloc)
        delta_grid = build_score_delta_grid(
            score_min, score_max,
            num_points=args.grid_points, margin_frac=args.grid_margin_frac,
        )

    net = ctx.allocation_head.net
    original_bias = net[-1].bias.detach().clone()

    rows: List[SweepRow] = []
    try:
        for delta in delta_grid:
            with torch.no_grad():
                net[-1].bias.copy_(original_bias + delta)
            hard = hard_rollout(
                ctx.pipeline, demands, ctx.mod_cfg,
                lambda_=vlastelica_lambda, margin_db=margin_db,
            )
            rows.append((
                delta,
                hard["hard_num_devices"],
                hard["hard_num_sites"],
                hard["hard_num_violated"],
                hard["hard_worst_margin_db"],
                hard["oracle_devices"],
                hard["oracle_gap"],
            ))
    finally:
        # Restore the head's bias to its original value even if a delta
        # raised — a script that leaves the loaded head mutated on exit is a
        # footgun for any code that reuses the same process.
        with torch.no_grad():
            net[-1].bias.copy_(original_bias)

    gate_delta = first_gate1_delta(
        [(d, hd, hv, od) for d, hd, _hs, hv, _wm, od, _og in rows]
    )

    print(f"config          : {args.config}")
    print(f"checkpoint      : {args.checkpoint or cfg.get('checkpoint_dir') + '/best_e2e.pt'}")
    print(f"demands         : {len(demands)} in the constraint set")
    print(f"tau={tau:.4f}  vlastelica_lambda={vlastelica_lambda:.4f}  margin_db={margin_db}")
    print(f"delta grid      : {['%.3f' % d for d in delta_grid]}")
    print()
    if gate_delta is None:
        print(
            "gate 1 (hard_num_violated == 0 AND hard_num_devices <= "
            "1.25 * oracle_devices) NOT satisfied by any delta in the grid"
        )
    else:
        print(
            f"gate 1 (hard_num_violated == 0 AND hard_num_devices <= "
            f"1.25 * oracle_devices) FIRST satisfied at delta={gate_delta:+.3f}"
        )
    print()

    header = (
        f"{'delta':>8} {'devices':>8} {'sites':>6} {'violated':>9} "
        f"{'worst_margin_db':>16} {'oracle_devices':>15} {'oracle_gap':>11}"
    )
    print(header)
    for delta, num_devices, num_sites, num_violated, worst_margin, oracle_devices, oracle_gap in rows:
        print(
            f"{delta:>+8.3f} {num_devices:>8d} {num_sites:>6d} {num_violated:>9d} "
            f"{worst_margin:>16.4f} {oracle_devices:>15d} {oracle_gap:>11d}"
        )


if __name__ == "__main__":
    main()
