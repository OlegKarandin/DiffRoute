"""Calibrate `rho` by MEASURING the fixed point, not by sweeping.

The calibration script whose measurement `calibrate_lambda_dev.py` cannot
make. `calibrate_lambda_dev.py` denominates its band in the HINGE's
gradient — and the hinge's gradient is exactly zero at the state this design
is aimed at, an allocation where every demand is already feasible. A
quantity that is identically zero where it matters cannot calibrate
anything.

The augmented Lagrangian's fixed point (design spec section 2.3) supplies an
anchor the hinge has no analogue for. At rest, on the marginal cut:

    primal:  (lambda_d + rho*g_d) * s = lambda_dev
    dual:    lambda_d + rho*g_d = lambda_d      =>  g_d = 0

so the demand sits exactly on its bar and its price settles where defending
it costs the same as the regenerator doing the defending:

    lambda* = lambda_dev / s ,      s = d gsnr_d / d a[d,k] > 0

The penalty stays active over `lambda*/rho` dB of headroom INSIDE the
feasible region, so fixing that width fixes rho:

    rho = lambda* / BAND_TARGET_DB

`s` is not measured directly. As in both companion scripts the practical
estimator is a ratio of L1 gradient norms over the allocation head's
parameters, which stands in for `lambda_dev / s`.

The calibration (check 1, fixed point): separate backward passes, one per
loss term, recording ||dL/dw||_1 on the allocation head. The feasibility
pass differentiates the LINEAR signed constraint sum, sum_d (bar_d -
gsnr_d), with NO relu and NO duals. That is the deliberate departure from
the companion scripts and the reason this measurement is possible at all:
the relu'd hinge is unmeasurable at the state that matters, while the
signed constraint has a gradient everywhere. It also means the denominator
sums over ALL demands rather than only the violated ones, so expect
lambda* to come out substantially smaller than any figure derived from a
hinge-mode measurement.

A ceiling applies on top of check 1: `rho <= CEILING_MULTIPLE * lambda*`, so
one epoch's dual step on a 1 dB violation cannot move a dual more than ~10x
its own resting value. At the default band target it is not binding
(rho = 2.857 * lambda*); a run that hits it is reporting something unusual.

A confirmatory trial (previously "check 2") then runs the candidate rho for
real epochs under the augmented penalty and reports whether
`hard_num_violated == 0` holds over the final `TAIL_EPOCHS`. This USED TO be
a comparison against hinge mode; the hinge penalty was removed 2026-09
(augmented was measured winning past ~150 epochs at the shipped 300-epoch
budget), so there is no longer a second
arm to compare against or a tiebreak to win — the trial is a confirmation of
the candidate, not a gate between two systems.

Usage (module form — `python scripts/calibrate_rho.py` does NOT work,
because that puts scripts/ on sys.path but not the project root, and the
`scripts._common` import below needs the root):
    python -m scripts.calibrate_rho --config configs/experiment/constrained_stress.yaml

Writes nothing. Its output is hand-copied into the arm's `constraint.rho`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.optim as optim
import yaml

from diffopt.demands import Demand
from diffopt.loss import compute_loss, update_duals
from diffopt.modulation import bar_db_for_demands
from diffopt.train import hard_rollout

from scripts._common import (
    DiagContext,
    add_common_args,
    build_context,
    fixed_traffic_demands,
    schedule_at,
)

# The band the penalty stays active over at the resting dual, in dB of
# headroom INSIDE the feasible region. ONE value, not a range (spec section
# 4), so rho is uniquely determined by the measurement rather than chosen
# from an interval. 0.35 dB sits inside the 0.5 dB margin, which is the
# point: the margin's surrogate-error allowance is not AL's to spend, and the
# band stacks inside it. If the confirmatory trial below shows the candidate
# is not holding hard_num_violated == 0, re-run with a smaller value and
# hand-pick what ships.
BAND_TARGET_DB = 0.35

# rho is also the dual step, so a 1 dB violation moves a dual by rho in one
# epoch. Cap that at 10x the dual's own resting value.
CEILING_MULTIPLE = 10.0

# The confirmatory trial is read over the final 10 epochs.
TAIL_EPOCHS = 10


def implied_rho(
    *,
    dev_push: float,
    unit_feasibility_push: float,
    band_target_db: float = BAND_TARGET_DB,
) -> Tuple[float, float, float]:
    """The (lambda_star, rho, ceiling) implied by one force measurement.

    `lambda_star` is the fixed point's resting dual, `lambda_dev / s`,
    estimated as the ratio of the device push to the UNIT feasibility push
    (the signed constraint sum differentiated with no dual weighting at all).
    `rho` places the active band at `band_target_db`. `ceiling` is the largest
    rho keeping one epoch's dual step on a 1 dB violation under
    CEILING_MULTIPLE resting duals.
    """
    if unit_feasibility_push <= 0.0:
        raise ValueError(
            "unit feasibility push is zero — the signed constraint sum has no "
            "gradient on the allocation head. Unlike the hinge this cannot "
            "mean 'every demand is feasible': the signed form is live "
            "everywhere. Check that the head is not saturated at a = 0 or "
            "a = 1 on every variable and that gsnr_preds still require grad."
        )
    if band_target_db <= 0.0:
        raise ValueError(
            f"band_target_db must be > 0 (it is a width in dB, and it divides "
            f"lambda*), got {band_target_db}"
        )
    lambda_star = dev_push / unit_feasibility_push
    return lambda_star, lambda_star / band_target_db, CEILING_MULTIPLE * lambda_star


def _trial_run(
    cfg: dict,
    ctx: DiagContext,
    demands: List[Demand],
    *,
    rho: float,
    max_epochs: int,
) -> List[Dict[str, float]]:
    """`diffopt.train.main`'s per-epoch step in miniature, at a trial mode,
    under the augmented penalty at a cold-start dual (every lambda = 0 —
    spec section 3).

    Same construction and the same deliberate omissions as `diffopt.train`'s
    own loop — both optimizers, zero_grad before the forward, the
    tau/vlastelica schedules, hard_rollout PRE-step, then backward/step, then
    the dual update on what the loss consumed. Drops the CSV, the trajectory
    file, the checkpoint and its selection key. Writes nothing.

    Returns one row per epoch so the caller can read the TRAJECTORY rather
    than an endpoint: the confirmation is a property of the final ten
    epochs, and a single final number cannot tell "held" from "broke and
    recovered".

    MUTATES `ctx.pipeline` — rebuild the context before a second call.
    """
    c_cfg, p_cfg, t_cfg = cfg["constraint"], cfg["pipeline"], cfg["training"]

    ctx.pipeline.train()
    duals = torch.zeros(len(demands), device=ctx.device)
    opt_edge = optim.Adam([ctx.pipeline.edge_log_weight], lr=t_cfg["lr_edge_net"])
    opt_alloc = optim.SGD(
        ctx.pipeline.allocation_head.parameters(), lr=t_cfg["lr_alloc"],
        momentum=0.0,
    )

    rows: List[Dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        tau, vlastelica_lambda = schedule_at(cfg, epoch)

        opt_edge.zero_grad()
        opt_alloc.zero_grad()

        path_noise_costs, gsnr_preds, _, alloc = ctx.pipeline(
            demands, tau=tau, lambda_=vlastelica_lambda,
        )
        loss, metrics = compute_loss(
            gsnr_preds=gsnr_preds,
            path_noise_costs=path_noise_costs,
            demands=demands,
            device_count=alloc.device_count,
            modulation_config=ctx.mod_cfg,
            duals=duals,
            margin_db=c_cfg["margin_db"],
            lambda_dev=p_cfg["lambda_dev"],
            lambda_cost=p_cfg["lambda_cost"],
            rho=rho,
        )

        hard = hard_rollout(
            ctx.pipeline, demands, alloc, ctx.mod_cfg,
            margin_db=c_cfg["margin_db"],
        )

        loss.backward()
        opt_edge.step()
        opt_alloc.step()

        duals = update_duals(
            duals, metrics["constraint_g"], eta=rho,
            dual_max=c_cfg["dual_max"],
        )

        rows.append({
            "epoch": epoch,
            "violated": hard["hard_num_violated"],
            "devices": hard["hard_num_devices"],
            "oracle_devices": hard["oracle_devices"],
            "gap": hard["oracle_gap"],
            "dual_max": float(duals.max().item()),
        })
    return rows


def _tail_max_violated(rows: List[Dict[str, float]]) -> float:
    """Read over the final TAIL_EPOCHS epochs of a trial."""
    return max(r["violated"] for r in rows[-TAIL_EPOCHS:])


def _summarise(rows: List[Dict[str, float]]) -> str:
    tail = rows[-TAIL_EPOCHS:]
    viol = [r["violated"] for r in rows]
    tail_viol = [r["violated"] for r in tail]
    gaps = [r["gap"] for r in rows]
    devs = [r["devices"] for r in rows]
    return (
        f"violated max={max(viol):.0f} final={viol[-1]:.0f} | "
        f"last-{len(tail)} violated max={max(tail_viol):.0f} "
        f"zero_in={sum(1 for v in tail_viol if v == 0)}/{len(tail_viol)} | "
        f"gap min={min(gaps):.0f} final={gaps[-1]:.0f} | "
        f"devices final={devs[-1]:.0f} oracle={rows[-1]['oracle_devices']:.0f} | "
        f"lambda_max={rows[-1]['dual_max']:.3f}"
    )


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(), with_checkpoint=False,
                         with_demands=False)
    ap.add_argument("--trial-epochs", type=int, default=20,
                    help="How many epochs check 2 simulates per arm")
    ap.add_argument("--band-target-db", type=float, default=BAND_TARGET_DB,
                    help="Headroom, in dB inside the feasible region, over "
                         "which the penalty stays active at the resting dual")
    ap.add_argument("--no-ste", dest="ste", action="store_false",
                    help="Measure the config exactly as written, instead of "
                         "the alloc_ste arm this rho is for")
    ap.set_defaults(ste=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    # The defect this rho is calibrated against does not exist without the
    # STE. Under the un-STE'd relaxation the soft pass disagrees with the
    # deployed rollout about who is feasible: measured at an allocation with
    # oracle_gap == 0 and hard_num_violated == 0, the soft pass called 10 of
    # 346 demands violated, and those phantoms carried 8.58e4 of 8.58e4 of
    # the hinge force (hinge itself has since been removed, but the STE's
    # own defect is penalty-agnostic). Applied here rather than required of
    # the config file, so
    # the committed configs stay unmodified.
    if args.ste:
        cfg.setdefault("placement", {}).update({"alloc_ste": True})
        cfg["training"]["alloc_tau_end"] = cfg["training"]["alloc_tau_start"]
        print(f"  measuring the alloc_ste arm (tau pinned "
              f"at {cfg['training']['alloc_tau_start']}); --no-ste "
              f"measures the config as written")

    c_cfg, p_cfg = cfg["constraint"], cfg["pipeline"]

    ctx = build_context(cfg, load_e2e_checkpoint=False)
    demands, _ = fixed_traffic_demands(ctx)
    head_params = list(ctx.pipeline.allocation_head.parameters())
    _, vlastelica_lambda = schedule_at(cfg, 1)

    _, gsnr, _, alloc = ctx.pipeline(
        demands, tau=cfg["training"]["alloc_tau_start"],
        lambda_=vlastelica_lambda,
    )

    # Separate backward passes, not one decomposition of a summed loss — the
    # terms share subexpressions and a single backward attributes shared
    # gradient arbitrarily. Same reasoning as both companion scripts.
    bar = bar_db_for_demands(demands, ctx.mod_cfg, c_cfg["margin_db"])
    # LINEAR and unweighted: no relu, no duals. This is the departure that
    # makes the measurement possible — see the module docstring.
    unit_feasibility = sum(bar[i] - gsnr[d.id] for i, d in enumerate(demands))

    def l1(term):
        g = torch.autograd.grad(term, head_params, retain_graph=True,
                                allow_unused=True)
        return sum(x.abs().sum().item() for x in g if x is not None)

    unit_feas_l1 = l1(unit_feasibility)
    dev_l1_unit = l1(alloc.device_count)
    lambda_dev = float(p_cfg["lambda_dev"])
    dev_push = lambda_dev * dev_l1_unit

    print(f"  demands={len(demands)}  lambda_dev={lambda_dev}  "
          f"margin_db={c_cfg['margin_db']}  "
          f"current rho={c_cfg.get('rho')}")
    print(f"  device_count={alloc.device_count.item():.2f}")
    print(f"\n  device push           ||dL/dw||_1 = {dev_push:.4e}   "
          f"(at lambda_dev = {lambda_dev})")
    print(f"  unit feasibility push ||dL/dw||_1 = {unit_feas_l1:.4e}   "
          f"(LINEAR, no relu, at a dual of 1.0)")

    lambda_star, rho, ceiling = implied_rho(
        dev_push=dev_push,
        unit_feasibility_push=unit_feas_l1,
        band_target_db=args.band_target_db,
    )
    print(f"\n  check 1: lambda* = lambda_dev / s = {lambda_star:.6f}   "
          f"(the resting dual at the fixed point)")
    print(f"  rho at a {args.band_target_db:g} dB band = {rho:.6f}")
    print(f"  ceiling ({CEILING_MULTIPLE:g}x lambda*)  = {ceiling:.6f}")

    candidate = min(rho, ceiling)
    if rho > ceiling:
        print(f"  !! the band target's rho ({rho:.6f}) is ABOVE the ceiling. "
              f"Using the ceiling, which widens the band to "
              f"{lambda_star / ceiling:.4f} dB.")

    print(f"\n  confirmatory trial: {args.trial_epochs} epochs under the "
          f"augmented penalty at rho={candidate:.6f} "
          f"(the last-{TAIL_EPOCHS} violated count is what's read)")
    aug_rows = _trial_run(
        cfg, ctx, demands,
        rho=candidate,
        max_epochs=args.trial_epochs,
    )
    print(f"    augmented rho={candidate:<8.4f} {_summarise(aug_rows)}")

    aug_tail = _tail_max_violated(aug_rows)
    if aug_tail > 0:
        print(f"\n  !! the confirmatory trial is UNHAPPY: the last-"
              f"{TAIL_EPOCHS} violated count peaked at {aug_tail:.0f}, not "
              f"0. A WIDER band (SMALLER rho) gives a satisfied demand more "
              f"reach inside the feasible region; try {candidate / 3:.6f} "
              f"and re-run.")
    else:
        print(f"\n  RECOMMEND constraint.rho: {candidate:.6f}   "
              f"(with constraint.dual_init: 0.0)")


if __name__ == "__main__":
    main()
