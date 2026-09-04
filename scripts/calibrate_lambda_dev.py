"""Calibrate `lambda_dev` by MEASURING the forces, not by sweeping.

Spec 2.4. `lambda_regen` was 1.0 against a quantity of order 1e1 (sites);
`lambda_dev` multiplies a quantity of order 1e2-1e3 (devices), so the old
value is not a starting point — it is a different problem.

Two independent checks, and if they conflict CHECK 2 WINS:

  check 1 (force ratio)  three separate backward passes, one per loss term,
                         recording ||dL/dw||_1 on the allocation head. Set
                         lambda_dev so the device push lands at 10-20% of
                         the feasibility push at dual_init. The 20% ceiling
                         comes from loss.py's recorded finding that Adam
                         saturates past a ~20:1 force ratio — beyond that
                         the weaker force stops influencing direction at
                         all, so a larger lambda buys nothing but a worse
                         conditioned problem.

  check 2 (flip time)    a violated demand's dual rises by rho * g per
                         epoch under the augmented penalty (`--rho`; see
                         `python -m scripts.calibrate_rho`). Verify that
                         this flips one a[d,k] from closed to open within a
                         handful of epochs. If a needed cut takes 40 epochs
                         at a 60-epoch budget, lambda_dev is too high no
                         matter what the force ratio said: feasibility that
                         arrives after training ends is not feasibility.

Check 1's own force-ratio measurement below still uses the ORIGINAL
one-sided hinge-shaped quantity `sum_d dual_d * relu(bar_d - gsnr_d)` as its
proxy for "the feasibility push" — a quantity `compute_loss` itself no
longer computes (open_followups.md item #8 removed the hinge penalty). This
is the SAME known limitation `calibrate_rho.py`'s docstring already flags:
that quantity is unmeasurable at the state that actually matters, an
allocation where every demand is already feasible. Only check 2, which
drives a real training loop, needs `compute_loss` and therefore `--rho`.

Usage:
    python scripts/calibrate_lambda_dev.py --config configs/experiment/constrained_stress.yaml --rho 0.310338

Writes nothing. Its output is hand-copied into the configs' pipeline.lambda_dev.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

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

TARGET_BAND = (0.10, 0.20)


def implied_lambda_dev(*, feas_l1: float, dev_l1_at_unit_lambda: float):
    """The (low, high) lambda_dev putting the device push in TARGET_BAND."""
    if dev_l1_at_unit_lambda <= 0.0:
        raise ValueError(
            "device push is zero at lambda_dev=1 — the allocation head is "
            "saturated or disconnected from the graph. Check that "
            "AllocationOutputs.device_count requires grad and that the head "
            "is not sitting at a = 0 or a = 1 on every variable."
        )
    lo, hi = TARGET_BAND
    return (lo * feas_l1 / dev_l1_at_unit_lambda,
            hi * feas_l1 / dev_l1_at_unit_lambda)


def _first_recruitment_epoch(
    cfg: dict,
    ctx: DiagContext,
    demands: List[Demand],
    *,
    lambda_dev: float,
    rho: float,
    max_epochs: int,
) -> Optional[int]:
    """Epoch at which the DEPLOYED allocation first buys a device, or None.

    This is `diffopt.train.main`'s per-epoch step in miniature, and it has to
    be the real one: check 2 asks how long dual ascent takes to flip a cut
    open, and that is a property of the actual interaction between the primal
    step, the dual step and the tau/lambda schedules. An approximation that
    skipped any of them would answer a different question.

    What is replicated from `main()`, in `main()`'s order:

      * both optimizers, Adam over `[pipeline.edge_log_weight]` at
        `lr_edge_net` and SGD over `allocation_head.parameters()` at
        `lr_alloc`;
      * `zero_grad()` on both BEFORE the forward pass;
      * `tau` and `vlastelica_lambda` from the epoch's schedule
        (`schedule_at` replays `main()`'s anneal and decay exactly);
      * the soft forward pass;
      * `compute_loss` with the trial `lambda_dev` and the live duals, under
        the augmented penalty at `rho`;
      * `hard_rollout` PRE-STEP — before `backward()`/`step()`, exactly where
        `main()` measures it, so the reported epoch describes the parameters
        the epoch's forward pass actually ran on rather than next epoch's;
      * `backward()`, then both `step()`s;
      * `update_duals` AFTER the primal step, on the SIGNED constraint_g the
        loss consumed, at `eta=rho`.

    What is deliberately dropped: the CSV logs, the trajectory file, the
    checkpoint and its lexicographic selection key, and the printed
    per-epoch summary. This function writes nothing.

    It DOES mutate `ctx.pipeline` (that is what training is), so a caller
    that wants the pristine epoch-0 state afterwards must rebuild the
    context.
    """
    c_cfg, p_cfg, t_cfg = cfg["constraint"], cfg["pipeline"], cfg["training"]

    # build_context defaults to eval_mode=True; train.py never calls .eval().
    # Nothing in this pipeline is actually mode-dependent today
    # (SpanAttentionQoT hardcodes dropout=0.0 and the head has no stochastic
    # layer), but the trial run should be in the mode training runs in, not
    # one step removed from it.
    ctx.pipeline.train()

    # Cold start: every lambda = 0, so slack demands exert nothing and a
    # violated demand feels the pure quadratic penalty rho*g (spec section
    # 3) — the same cold start diffopt.train.main uses under the augmented
    # penalty.
    duals = torch.zeros(len(demands), device=ctx.device)

    opt_edge = optim.Adam(
        [ctx.pipeline.edge_log_weight], lr=t_cfg["lr_edge_net"]
    )
    opt_alloc = optim.SGD(
        ctx.pipeline.allocation_head.parameters(), lr=t_cfg["lr_alloc"],
        momentum=0.0,
    )

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
            lambda_dev=lambda_dev,
            lambda_cost=p_cfg["lambda_cost"],
            rho=rho,
        )

        # Pre-step, like train.py: this describes the parameters the forward
        # pass above ran on, not the post-step ones.
        hard = hard_rollout(
            ctx.pipeline, demands, ctx.mod_cfg,
            lambda_=vlastelica_lambda,
            margin_db=c_cfg["margin_db"],
        )

        loss.backward()
        opt_edge.step()
        opt_alloc.step()

        # Dual ascent AFTER the primal step, on the SIGNED constraint_g the
        # loss actually consumed this epoch, at the same rho the penalty
        # uses.
        duals = update_duals(
            duals,
            metrics["constraint_g"],
            eta=rho,
            dual_max=c_cfg["dual_max"],
        )

        if hard["hard_num_devices"] > 0:
            return epoch

    return None


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(), with_checkpoint=False,
                         with_demands=False)
    ap.add_argument("--flip-epochs", type=int, default=20,
                    help="How many epochs check 2 simulates before giving up")
    ap.add_argument("--rho", type=float, required=True,
                    help="Augmented-penalty coefficient check 2's real "
                         "trial loop runs under (compute_loss now requires "
                         "one unconditionally). Get a value from "
                         "`python -m scripts.calibrate_rho`; an exact "
                         "value is not needed here since check 1's force "
                         "ratio does not depend on it — check 2 only asks "
                         "whether recruitment happens soon enough.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ctx = build_context(cfg, load_e2e_checkpoint=False)
    demands, _ = fixed_traffic_demands(ctx)
    c_cfg, p_cfg = cfg["constraint"], cfg["pipeline"]

    head_params = list(ctx.pipeline.allocation_head.parameters())
    duals = torch.full((len(demands),), float(c_cfg["dual_init"]))
    _, vlastelica_lambda = schedule_at(cfg, 1)

    path_noise, gsnr, _, alloc = ctx.pipeline(
        demands, tau=cfg["training"]["alloc_tau_start"],
        lambda_=vlastelica_lambda,
    )

    # Three SEPARATE backward passes, not one decomposition of a summed
    # loss: the terms share subexpressions, and reading a decomposition off
    # a single backward attributes shared gradient arbitrarily.
    bar = bar_db_for_demands(demands, ctx.mod_cfg, c_cfg["margin_db"])
    feas = sum(
        duals[d.id] * torch.relu(bar[i] - gsnr[d.id])
        for i, d in enumerate(demands)
    )
    dev = alloc.device_count                      # at lambda_dev = 1.0
    cost = p_cfg["lambda_cost"] * sum(path_noise.values())

    def l1(term):
        g = torch.autograd.grad(term, head_params, retain_graph=True,
                                allow_unused=True)
        return sum(x.abs().sum().item() for x in g if x is not None)

    feas_l1, dev_l1, cost_l1 = l1(feas), l1(dev), l1(cost)
    print(f"  feasibility push  ||dL/dw||_1 = {feas_l1:.2e}   "
          f"(at dual_init = {c_cfg['dual_init']})")
    print(f"  device push       ||dL/dw||_1 = {dev_l1:.2e}   (at lambda_dev = 1.0)")
    print(f"  path-noise push   ||dL/dw||_1 = {cost_l1:.2e}")

    lo, hi = implied_lambda_dev(feas_l1=feas_l1, dev_l1_at_unit_lambda=dev_l1)
    print(f"\n  implied lambda_dev band "
          f"({int(TARGET_BAND[0]*100)}-{int(TARGET_BAND[1]*100)}% of "
          f"feasibility): [{lo:.2f}, {hi:.2f}]")

    flip = _first_recruitment_epoch(cfg, ctx, demands, lambda_dev=lo,
                                    rho=args.rho, max_epochs=args.flip_epochs)
    print(f"\n  check 2: at lambda_dev = {lo:.2f}, first device recruited at "
          f"epoch {flip if flip else '>' + str(args.flip_epochs)}.")
    if flip is None or flip > args.flip_epochs // 4:
        print("  !! check 2 is the tiebreak and it is UNHAPPY: feasibility "
              "that arrives after training ends is not feasibility. "
              f"RECOMMEND a value below the band; try {lo / 3:.2f}.")
    else:
        print(f"  RECOMMEND pipeline.lambda_dev: {lo:.2f}")


if __name__ == "__main__":
    main()
