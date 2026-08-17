"""Diagnostic: where does gradient into regen_logits actually point?

Decomposes d(total_loss)/d(regen_logits) into its three loss-term
contributions at several points on the training annealing schedule,
and reports the sign/magnitude on the nodes that actually sit on
infeasible paths (the ones that *should* be climbing).
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
import yaml

from _common import add_common_args, build_context, demands_for, schedule_at


def grad_of(term: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    g = torch.autograd.grad(term, param, retain_graph=True, allow_unused=True)[0]
    return torch.zeros_like(param) if g is None else g


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap, with_checkpoint=False, with_demands=False)
    ap.add_argument("--epochs", default="1,5,10,15,18,20")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ctx = build_context(cfg, load_e2e_checkpoint=False)
    pipeline = ctx.pipeline
    topology = ctx.topology
    mod_cfg = ctx.mod_cfg
    regen = ctx.regen_placement

    p_cfg = cfg["pipeline"]
    t_cfg = cfg["training"]

    n_cand = len(ctx.regen_candidates)
    print(f"nodes={topology.num_nodes}  edges={len(ctx.edges)}  "
          f"regen_candidates(deg>=3)={n_cand}")
    print(f"lambda_regen={p_cfg['lambda_regen']}  lambda_infeasible={p_cfg['lambda_infeasible']}  "
          f"lambda_cost={p_cfg['lambda_cost']}  lr_regen={t_cfg['lr_regen']}\n")

    for epoch in [int(x) for x in args.epochs.split(",")]:
        tau, tsm, vlastelica_lambda = schedule_at(cfg, epoch=epoch)

        # seed=epoch: matches train.py's per-epoch demand reseeding exactly.
        demands = demands_for(ctx, seed=epoch)

        # fresh logits at 0 each time: isolates the temperature effect
        with torch.no_grad():
            regen.regen_logits.zero_()

        path_noise_costs, gsnr_preds, _, regen_probs = pipeline(
            demands, tau=tau, lambda_=vlastelica_lambda, soft_max_temperature=tsm)

        feas = torch.zeros(1)
        infeasible_ids = []
        for d in demands:
            thr = torch.tensor(mod_cfg.required_snr_threshold(d.bitrate_gbps))
            sf = F.relu(thr - gsnr_preds[d.id])
            feas = feas + sf
            if sf.item() > 0:
                infeasible_ids.append(d.id)

        L_feas = p_cfg["lambda_infeasible"] * feas.squeeze()
        L_regen = p_cfg["lambda_regen"] * regen_probs.sum()
        L_cost = p_cfg["lambda_cost"] * sum(path_noise_costs.values())

        g_feas = grad_of(L_feas, regen.regen_logits)
        g_regen = grad_of(L_regen, regen.regen_logits)
        g_cost = grad_of(L_cost, regen.regen_logits)
        g_tot = g_feas + g_regen + g_cost

        # which nodes are boundaries on infeasible paths?
        touched = (g_feas.abs() > 0)
        n_touched = int(touched.sum())

        # descent moves logit by -grad; negative total grad == prob climbs
        climbing = int((g_tot < 0).sum())

        print(f"--- epoch {epoch:2d}  tau={tau:.3f}  t_softmax={tsm:.4f} ---")
        print(f"  infeasible demands: {len(infeasible_ids)}/{len(demands)}   "
              f"feasibility_loss={feas.item():.2f}")
        print(f"  nodes with nonzero feasibility grad: {n_touched}/{topology.num_nodes}")
        print(f"  grad_feasibility : sum={g_feas.sum():+.4f}  min={g_feas.min():+.4f}  "
              f"max={g_feas.max():+.4f}")
        print(f"  grad_regen_count : sum={g_regen.sum():+.4f}  per-node={g_regen.max():+.4f}")
        print(f"  grad_path_cost   : sum={g_cost.sum():+.4f}  |max|={g_cost.abs().max():+.4f}")
        print(f"  grad_TOTAL       : min={g_tot.min():+.4f}  max={g_tot.max():+.4f}  "
              f"nodes wanting MORE regen (grad<0): {climbing}/{topology.num_nodes}")
        if n_touched:
            gf = g_feas[touched]
            print(f"    on touched nodes: feas grad mean={gf.mean():+.4f} "
                  f"min={gf.min():+.4f} max={gf.max():+.4f}")
            print(f"    on touched nodes: TOTAL grad mean={g_tot[touched].mean():+.4f}  "
                  f"wanting more regen: {int((g_tot[touched] < 0).sum())}/{n_touched}")
        print()


if __name__ == "__main__":
    main()
