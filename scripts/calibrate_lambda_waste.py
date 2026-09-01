"""Calibrate `lambda_waste` by MEASURING the forces, not by sweeping.

Companion to `scripts/calibrate_lambda_dev.py`, answering a different question.
`lambda_dev` prices every device equally, so once feasibility is slack the only
thing it can do is shed devices — load-bearing and redundant alike, at the same
rate. That undiscriminating shed is the mechanism behind
`docs/investigations/open_followups.md` item #6's limit cycle: the allocation
shrinks until some demand breaks, its dual fires, the head re-buys, repeat.

`lambda_waste` prices `sum_{d,k} a[d,k] * relu(f4[d,k])`, where f4 is the
headroom remaining AFTER the next segment. `relu(f4) > 0` exactly when the cut
was not needed, so the term charges for premature cuts specifically and
contributes exactly zero on a load-bearing one. Its job is to make the shed
DISCRIMINATE, and the force it has to out-pull is the device push, not the
feasibility push. Hence the band below is denominated in the device push,
unlike calibrate_lambda_dev.py's.

Two checks, and if they conflict CHECK 2 WINS:

  check 1 (discrimination)   separate backward passes, one per loss term,
                             recording ||dL/dw||_1 on the allocation head. Set
                             lambda_waste so the waste push lands at 2-5x the
                             device push, i.e. so an unneeded cut sheds several
                             times faster than a needed one.

  check 2 (feasibility cost) the waste term is EXTRA downward force, so raising
                             it breaks more demands, not fewer. Run real epochs
                             at the candidate and at 0.0 and check that
                             hard_num_violated does not degrade. Discrimination
                             bought with feasibility is not a win.

A ceiling applies on top of check 1: the COMBINED shed push (device + waste)
must stay under `CEILING_FRACTION` of the feasibility push, reusing
calibrate_lambda_dev.py's recorded finding that Adam saturates past a ~20:1
force ratio — beyond that the weaker force stops influencing direction at all.

Usage (module form — `python scripts/calibrate_lambda_waste.py` does NOT work,
because that puts scripts/ on sys.path but not the project root, and the
`scripts._common` import below needs the root):
    python -m scripts.calibrate_lambda_waste --config configs/experiment/constrained_stress.yaml

Writes nothing. Its output is hand-copied into the configs' pipeline.lambda_waste.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

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

# Waste push as a multiple of the device push. Below 2x the waste term is a
# rounding correction on an undiscriminating shed; above 5x it is effectively
# the only price on allocation, which converges at |S| = 1 by encoding the
# greedy rule into the objective and does not transfer to Stage IV's
# shared-restoration cost `sum_n max_s sum_d a[d,s,n]`.
TARGET_DISCRIMINATION = (2.0, 5.0)

# Combined shed push (device + waste) as a fraction of the feasibility push.
# Same 20% ceiling and same rationale as calibrate_lambda_dev.py.
CEILING_FRACTION = 0.20


def implied_lambda_waste(
    *, dev_push: float, waste_push_at_unit_lambda: float, feas_l1: float
):
    """The (low, high, ceiling) lambda_waste for this measurement.

    `low`/`high` put the waste push inside TARGET_DISCRIMINATION x the device
    push. `ceiling` is the largest lambda_waste keeping the combined shed push
    under CEILING_FRACTION of the feasibility push; it is 0.0 when the device
    push alone already spends the whole allowance, which is a real finding
    about `lambda_dev` rather than a reason to return a negative weight.
    """
    if waste_push_at_unit_lambda <= 0.0:
        raise ValueError(
            "waste push is zero at lambda_waste=1 — no allocated cut has "
            "positive headroom. Either the head buys nothing (check "
            "device_count > 0) or every cut it buys is load-bearing, in "
            "which case there is no waste to price."
        )
    lo, hi = TARGET_DISCRIMINATION
    ceiling = (CEILING_FRACTION * feas_l1 - dev_push) / waste_push_at_unit_lambda
    return (
        lo * dev_push / waste_push_at_unit_lambda,
        hi * dev_push / waste_push_at_unit_lambda,
        max(0.0, ceiling),
    )


def _trial_run(
    cfg: dict,
    ctx: DiagContext,
    demands: List[Demand],
    *,
    lambda_waste: float,
    max_epochs: int,
) -> List[Dict[str, float]]:
    """`diffopt.train.main`'s per-epoch step in miniature, at a trial weight.

    Same construction and the same deliberate omissions as
    calibrate_lambda_dev.py's `_first_recruitment_epoch` — both optimizers,
    zero_grad before the forward, the tau/vlastelica schedules, hard_rollout
    PRE-step, then backward/step, then dual ascent on the shortfalls the loss
    consumed. Drops the CSV, the trajectory file, the checkpoint and its
    selection key. Writes nothing.

    Returns one row per epoch so the caller can compare TRAJECTORIES rather
    than endpoints: check 2 asks whether the extra shed force costs
    feasibility, and a single final number cannot tell "held" from "broke and
    recovered".

    MUTATES `ctx.pipeline` — rebuild the context before a second call.
    """
    c_cfg, p_cfg, t_cfg = cfg["constraint"], cfg["pipeline"], cfg["training"]
    pl_cfg = cfg.get("placement", {})
    alloc_dropout_p: float = pl_cfg.get("alloc_dropout_p", 0.0)

    ctx.pipeline.train()
    duals = torch.full(
        (len(demands),), float(c_cfg["dual_init"]), device=ctx.device
    )
    opt_edge = optim.Adam([ctx.pipeline.edge_log_weight], lr=t_cfg["lr_edge_net"])
    opt_alloc = optim.Adam(
        ctx.pipeline.allocation_head.parameters(), lr=t_cfg["lr_alloc"]
    )

    rows: List[Dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        tau, vlastelica_lambda = schedule_at(cfg, epoch)

        opt_edge.zero_grad()
        opt_alloc.zero_grad()

        path_noise_costs, gsnr_preds, _, alloc = ctx.pipeline(
            demands, tau=tau, lambda_=vlastelica_lambda,
            alloc_dropout_p=alloc_dropout_p,
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
            waste_cost=alloc.waste_cost,
            lambda_waste=lambda_waste,
        )

        hard = hard_rollout(
            ctx.pipeline, demands, ctx.mod_cfg,
            lambda_=vlastelica_lambda, margin_db=c_cfg["margin_db"],
        )

        loss.backward()
        opt_edge.step()
        opt_alloc.step()

        duals = update_duals(
            duals, metrics["shortfalls"],
            eta=c_cfg["dual_lr"], dual_max=c_cfg["dual_max"],
            decay=c_cfg.get("dual_decay", 0.0),
        )

        rows.append({
            "epoch": epoch,
            "violated": hard["hard_num_violated"],
            "devices": hard["hard_num_devices"],
            "oracle_devices": hard["oracle_devices"],
            "gap": hard["oracle_gap"],
        })
    return rows


def _summarise(rows: List[Dict[str, float]]) -> str:
    viol = [r["violated"] for r in rows]
    gaps = [r["gap"] for r in rows]
    devs = [r["devices"] for r in rows]
    return (
        f"violated max={max(viol):.0f} final={viol[-1]:.0f} "
        f"zero_in={sum(1 for v in viol if v == 0)}/{len(viol)} | "
        f"gap min={min(gaps):.0f} final={gaps[-1]:.0f} | "
        f"devices final={devs[-1]:.0f} oracle={rows[-1]['oracle_devices']:.0f}"
    )


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(), with_checkpoint=False,
                         with_demands=False)
    ap.add_argument("--trial-epochs", type=int, default=20,
                    help="How many epochs check 2 simulates per arm")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    c_cfg, p_cfg = cfg["constraint"], cfg["pipeline"]

    if not cfg.get("placement", {}).get("greedy_residual", False):
        print("NOTE: placement.greedy_residual is false in this config. The "
              "waste term keys off feature 4 and so does the residual; "
              "calibrating one without the other measures a configuration "
              "nobody runs.\n")

    ctx = build_context(cfg, load_e2e_checkpoint=False)
    demands, _ = fixed_traffic_demands(ctx)
    head_params = list(ctx.pipeline.allocation_head.parameters())
    duals = torch.full((len(demands),), float(c_cfg["dual_init"]))
    _, vlastelica_lambda = schedule_at(cfg, 1)

    path_noise, gsnr, _, alloc = ctx.pipeline(
        demands, tau=cfg["training"]["alloc_tau_start"],
        lambda_=vlastelica_lambda,
    )

    # Separate backward passes, not one decomposition of a summed loss — the
    # terms share subexpressions and a single backward attributes shared
    # gradient arbitrarily. Same reasoning as calibrate_lambda_dev.py.
    bar = bar_db_for_demands(demands, ctx.mod_cfg, c_cfg["margin_db"])
    feas = sum(
        duals[d.id] * torch.relu(bar[i] - gsnr[d.id])
        for i, d in enumerate(demands)
    )

    def l1(term):
        g = torch.autograd.grad(term, head_params, retain_graph=True,
                                allow_unused=True)
        return sum(x.abs().sum().item() for x in g if x is not None)

    feas_l1 = l1(feas)
    dev_l1_unit = l1(alloc.device_count)          # at lambda_dev = 1.0
    waste_l1_unit = l1(alloc.waste_cost)          # at lambda_waste = 1.0
    lambda_dev = float(p_cfg["lambda_dev"])
    dev_push = lambda_dev * dev_l1_unit

    print(f"  demands={len(demands)}  lambda_dev={lambda_dev}  "
          f"dual_init={c_cfg['dual_init']}  "
          f"current lambda_waste={p_cfg.get('lambda_waste', 0.0)}")
    print(f"  device_count={alloc.device_count.item():.2f}   "
          f"waste_cost={alloc.waste_cost.item():.2f}   "
          f"mean relu(f4) per allocated device="
          f"{alloc.waste_cost.item() / max(alloc.device_count.item(), 1e-12):.3f} dB")
    print(f"\n  feasibility push ||dL/dw||_1 = {feas_l1:.2e}   "
          f"(at dual_init = {c_cfg['dual_init']})")
    print(f"  device push      ||dL/dw||_1 = {dev_push:.2e}   "
          f"(at lambda_dev = {lambda_dev})")
    print(f"  waste push       ||dL/dw||_1 = {waste_l1_unit:.2e}   "
          f"(at lambda_waste = 1.0)")

    lo, hi, ceiling = implied_lambda_waste(
        dev_push=dev_push,
        waste_push_at_unit_lambda=waste_l1_unit,
        feas_l1=feas_l1,
    )
    print(f"\n  check 1: implied lambda_waste band "
          f"({TARGET_DISCRIMINATION[0]:g}-{TARGET_DISCRIMINATION[1]:g}x the "
          f"device push): [{lo:.3f}, {hi:.3f}]")
    print(f"  ceiling (combined shed push <= "
          f"{int(CEILING_FRACTION * 100)}% of feasibility): {ceiling:.3f}")
    candidate = min(lo, ceiling) if ceiling > 0 else lo
    if ceiling <= 0.0:
        print("  !! the device push ALONE already exceeds the ceiling — "
              "lambda_dev is too high for any waste weight to sit under it.")
    elif ceiling < lo:
        print(f"  !! ceiling is BELOW the band: no lambda_waste both "
              f"discriminates and stays under the cap. Using the ceiling "
              f"({ceiling:.3f}) and reporting the conflict.")

    print(f"\n  check 2: {args.trial_epochs} epochs at lambda_waste=0.0 vs "
          f"{candidate:.3f} (feasibility must not degrade)")
    base_rows = _trial_run(cfg, ctx, demands, lambda_waste=0.0,
                           max_epochs=args.trial_epochs)
    print(f"    lambda_waste=0.000      {_summarise(base_rows)}")

    # _trial_run trains in place, so the second arm needs a fresh context.
    ctx2 = build_context(cfg, load_e2e_checkpoint=False)
    demands2, _ = fixed_traffic_demands(ctx2)
    cand_rows = _trial_run(cfg, ctx2, demands2, lambda_waste=candidate,
                           max_epochs=args.trial_epochs)
    print(f"    lambda_waste={candidate:.3f}      {_summarise(cand_rows)}")

    base_worst = max(r["violated"] for r in base_rows)
    cand_worst = max(r["violated"] for r in cand_rows)
    if cand_worst > base_worst:
        print(f"\n  !! check 2 is the tiebreak and it is UNHAPPY: worst "
              f"violated rose {base_worst:.0f} -> {cand_worst:.0f}. "
              f"RECOMMEND a value below the band; try {candidate / 3:.3f}.")
    else:
        print(f"\n  RECOMMEND pipeline.lambda_waste: {candidate:.3f}")


if __name__ == "__main__":
    main()
