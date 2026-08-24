"""Diagnostic: where does gradient into the AllocationHead actually point?

Decomposes d(total_loss)/d(allocation_head parameters) into its three loss-
term contributions at several points on the training annealing schedule.
The pre-Stage-II version of this script decomposed the same three terms into
a (num_nodes,) `regen_logits` vector and asked "which NODES want more
regen"; there is no such vector any more, so the same decomposition is
reported per PARAMETER TENSOR of the head, plus the one scalar direction
that still has the old question's meaning:

    descent moves theta by -grad, so a negative total gradient on the
    output layer's bias means the head wants to OPEN (buy more devices).

`L_regen = lambda_regen * regen_probs.sum()` is replaced by
`L_dev = lambda_dev * alloc.device_count` — the sum over (demand, boundary)
allocations, which is what the deployed network actually pays for.

This script is the qualitative companion to scripts/calibrate_lambda_dev.py
and shares its measurement: the ratio L1(grad L_dev) / L1(grad L_feas) is
the force balance that calibration sets numerically.

Note on reading the numbers at init. `AllocationHead` starts CLOSED
deterministically (spec 2.2): its output layer has zero weights and bias
-3.0. A zero output-layer weight means d(score)/d(hidden) is exactly 0, so
WITHOUT --checkpoint every hidden layer reads a gradient of exactly 0 and
only the output layer's own weight and bias move on the first step. That is
the real epoch-0 state, not an instrumentation artefact. Pass --checkpoint
to see the decomposition at a trained operating point, where every layer is
live.
"""
from __future__ import annotations

import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
import yaml

from _common import add_common_args, build_context, demands_for, schedule_at


def grads_of(term: torch.Tensor, params: List[torch.Tensor]) -> List[torch.Tensor]:
    gs = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(params, gs)]


def l1(gs: List[torch.Tensor]) -> float:
    return float(sum(g.abs().sum().item() for g in gs))


def maxabs(gs: List[torch.Tensor]) -> float:
    return max((float(g.abs().max().item()) for g in gs if g.numel()), default=0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap, with_demands=False)
    ap.add_argument("--epochs", default="1,5,10,15,18,20")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # --checkpoint is optional here, same convention as diagnose_surrogate.py:
    # no checkpoint means "report on the closed init train.py actually starts
    # from" rather than falling back to <checkpoint_dir>/best_e2e.pt.
    ctx = build_context(
        cfg,
        load_e2e_checkpoint=bool(args.checkpoint),
        checkpoint_path=args.checkpoint,
    )
    pipeline = ctx.pipeline
    topology = ctx.topology
    mod_cfg = ctx.mod_cfg
    head = ctx.allocation_head

    p_cfg = cfg["pipeline"]
    c_cfg = cfg["constraint"]
    t_cfg = cfg["training"]

    named: List[Tuple[str, torch.Tensor]] = [
        (n, p) for n, p in head.named_parameters() if p.requires_grad
    ]
    params = [p for _, p in named]
    # The output layer's bias by identity, not by a hardcoded "net.4.bias":
    # AllocationHead's hidden width is a constructor argument and the module
    # indices would move if a layer were ever added.
    out_bias = head.net[-1].bias
    out_bias_name = next(n for n, p in named if p is out_bias)

    n_cand = len(ctx.regen_candidates)
    print(f"nodes={topology.num_nodes}  edges={len(ctx.edges)}  "
          f"regen_candidates(deg>=3)={n_cand}")
    print(f"allocation head: lookahead={head.lookahead}  "
          f"params={sum(p.numel() for p in params)} in {len(params)} tensors")
    if args.checkpoint:
        print(f"loaded e2e checkpoint: epoch {ctx.ckpt['epoch']}, "
              f"loss {ctx.ckpt['total_loss']:.4f}")
    else:
        print("no --checkpoint: reporting on the CLOSED init (spec 2.2) — "
              "the output layer's zero weight zeroes every hidden gradient")
    print(f"lambda_dev={p_cfg['lambda_dev']}  dual_init={c_cfg['dual_init']} "
          f"margin_db={c_cfg['margin_db']}  "
          f"lambda_cost={p_cfg['lambda_cost']}  lr_alloc={t_cfg['lr_alloc']}\n")

    # No per-epoch reset of the head. The old script zeroed `regen_logits`
    # each iteration to isolate the tau/lambda effect; here nothing mutates
    # the head at all (this script takes no optimizer step), so every
    # iteration already reports the same parameters under a different
    # schedule point.
    for epoch in [int(x) for x in args.epochs.split(",")]:
        # vlastelica_lambda here is the DECAYED per-epoch value schedule_at
        # replays from train.py's loop. It enters the Vlastelica backward
        # directly (diffopt/routing/surrogate.py's c_target = w +
        # lambda_*grad_output), and the head's gradient reaches routing
        # through the STE-blended per-segment GSNR that feeds the rollout's
        # carry, so this is a SECOND source of cross-epoch gradient-magnitude
        # change here, independent of the tau anneal. Printed alongside it
        # for exactly that reason.
        tau, vlastelica_lambda = schedule_at(cfg, epoch=epoch)

        # seed=epoch: this script's own per-epoch demand draw for gradient
        # decomposition; train.py no longer reseeds demands per epoch (Task 4
        # of the feasibility-constraint migration built a fixed matrix once,
        # before the loop).
        demands = demands_for(ctx, seed=epoch)

        path_noise_costs, gsnr_preds, _, alloc = pipeline(
            demands, tau=tau, lambda_=vlastelica_lambda)

        feas = torch.zeros(1)
        infeasible_ids = []
        for d in demands:
            thr = torch.tensor(mod_cfg.required_snr_threshold(d.bitrate_gbps))
            sf = F.relu(thr + c_cfg["margin_db"] - gsnr_preds[d.id])
            feas = feas + sf
            if sf.item() > 0:
                infeasible_ids.append(d.id)

        # At epoch 0 every per-demand dual equals dual_init (they diverge
        # only after update_duals starts adjusting them per demand), so this
        # single-scalar substitution is only valid at epoch 0.
        L_feas = c_cfg["dual_init"] * feas.squeeze()
        L_dev = p_cfg["lambda_dev"] * alloc.device_count
        L_cost = p_cfg["lambda_cost"] * sum(path_noise_costs.values())

        g_feas = grads_of(L_feas, params)
        g_dev = grads_of(L_dev, params)
        g_cost = grads_of(L_cost, params)
        g_tot = [a + b + c for a, b, c in zip(g_feas, g_dev, g_cost)]

        names = [n for n, _ in named]
        bias_grad_feas = dict(zip(names, g_feas))[out_bias_name]
        bias_grad_dev = dict(zip(names, g_dev))[out_bias_name]
        bias_grad_tot = dict(zip(names, g_tot))[out_bias_name]

        n_live = sum(1 for g in g_tot if float(g.abs().max().item() if g.numel() else 0.0) > 0)

        print(f"--- epoch {epoch:2d}  tau={tau:.3f}  "
              f"lambda={vlastelica_lambda:.3f} ---")
        print(f"  infeasible demands: {len(infeasible_ids)}/{len(demands)}   "
              f"feasibility_loss={feas.item():.2f}   "
              f"device_count={alloc.device_count.item():.2f}   "
              f"sites={int((alloc.site_view > 0.5).sum())}")
        print(f"  parameter tensors receiving nonzero gradient: "
              f"{n_live}/{len(params)}")
        print(f"  grad_feasibility : L1={l1(g_feas):.4e}  max|g|={maxabs(g_feas):.4e}")
        print(f"  grad_device_count: L1={l1(g_dev):.4e}  max|g|={maxabs(g_dev):.4e}")
        print(f"  grad_path_noise  : L1={l1(g_cost):.4e}  max|g|={maxabs(g_cost):.4e}")
        print(f"  grad_TOTAL       : L1={l1(g_tot):.4e}  max|g|={maxabs(g_tot):.4e}")
        lf, ld = l1(g_feas), l1(g_dev)
        print(f"  force balance    : L1(device)/L1(feasibility) = "
              f"{(ld / lf) if lf > 0 else float('inf'):.4f}   "
              f"(what calibrate_lambda_dev.py sets)")

        # The one direction that still carries the old per-node question's
        # meaning: descent moves the output bias by -grad, so grad < 0 means
        # the head is being pushed OPEN (towards buying more devices).
        b_tot = float(bias_grad_tot.item())
        print(f"  output bias ({out_bias_name}, currently {float(out_bias.item()):+.3f}): "
              f"feas={float(bias_grad_feas.item()):+.4e}  "
              f"dev={float(bias_grad_dev.item()):+.4e}  "
              f"total={b_tot:+.4e}  -> head wants "
              f"{'MORE' if b_tot < 0 else 'FEWER'} devices")

        print("  per-tensor TOTAL gradient L1:")
        for name, g in zip(names, g_tot):
            print(f"    {name:<16} L1={g.abs().sum().item():.4e}  "
                  f"max|g|={(g.abs().max().item() if g.numel() else 0.0):.4e}")
        print()


if __name__ == "__main__":
    main()
