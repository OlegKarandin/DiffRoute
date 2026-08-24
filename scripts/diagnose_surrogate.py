"""Diagnostic: Vlastelica surrogate health check.

Checks two things the smoke-run output does not reveal:

1. Allocation distribution
   The training log prints device_count = sum_n sum_d a[d, n], the SOFT
   (mean-field) device count. It is not a count of deployed regenerators —
   it is the sum of all per-(demand, boundary) allocation probabilities, and
   it moves with tau even on frozen scores. This script shows the full
   distribution behind that one number, plus the site_view (max over
   demands) that is logged and never priced.

2. Hamming distance in the Vlastelica backward
   For the surrogate to provide routing-change signal, the perturbed solve
   must find a DIFFERENT path. Since correction #6, grad_output =
   ∂L/∂path_indicator is dominated by the straight-through per-edge
   ASE-noise proxy (threshold-gated), not by edge_weights;
   and since correction #9, the path_noise_cost term it is added to is
   denominated in edge_ase_noise rather than edge_weights. Neither term is
   proportional to edge_weights, so a zero Hamming distance here no longer
   reflects the old uniform-scaling degeneracy — it would mean the
   ASE-noise-driven perturbation happens to leave the shortest path
   unchanged for every demand, which is worth investigating on its own
   (see the report this script prints when that happens).

Usage:
    conda activate diffopt
    python scripts/diagnose_surrogate.py --config configs/experiment/small_test_ind132.yaml

--config now defaults to configs/experiment/small_test_ind132.yaml (via
add_common_args, matching every other migrated script) rather than being
required with no default — pass --config explicitly to use a different one.
"""
from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import torch
import yaml

from diffopt.loss import compute_loss
from diffopt.routing.shortest_path import spfa
from _common import add_common_args, build_context, demands_for, edge_weights_of, schedule_at


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    # This script's --checkpoint default is deliberately different from
    # every other script's: None means "report on random-init routing heads
    # (train.py's epoch-0 state)", not "fall back to <checkpoint_dir>/
    # best_e2e.pt". That distinction is the point of the script (compare
    # untrained vs trained surrogate health), so it's kept and re-documented
    # here rather than unified away. add_common_args' --seed default (1)
    # already matches this script's own historical default.
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ctx = build_context(cfg, load_e2e_checkpoint=bool(args.checkpoint), checkpoint_path=args.checkpoint)
    pipeline = ctx.pipeline
    topology = ctx.topology
    mod_cfg = ctx.mod_cfg

    if args.checkpoint:
        print(f"Loaded e2e checkpoint: epoch {ctx.ckpt['epoch']}, loss {ctx.ckpt['total_loss']:.4f}")
    else:
        print("No --checkpoint given: reporting on randomly-initialised routing "
              "heads (seeded, matches train.py's epoch-0 state).")

    lambda_cost = cfg["pipeline"]["lambda_cost"]
    # Start-of-training values (epoch 1): matches this script's own
    # historical tau=alloc_tau_start / undecayed vlastelica_lambda when no
    # --checkpoint overrides them.
    tau, lambda_ = schedule_at(cfg, epoch=1)

    demands = demands_for(ctx, seed=args.seed)

    # ------------------------------------------------ intercept grad_output
    # edge_weights_of independently recomputes exactly what pipeline.forward
    # will compute internally for edge_weights (the same live
    # edge_log_weight, same normalisation) -- no monkeypatch needed to
    # capture it. The hook below is only for grad_output on each
    # path_indicator, which edge_weights_of has no access to.
    captured_edge_weights = edge_weights_of(ctx, tau).detach().clone()
    captured_grad_output: Dict[int, torch.Tensor] = {}

    original_forward = pipeline.forward

    def instrumented_forward(demands, *fwd_args, **fwd_kwargs):
        # *args/**kwargs, not a copied signature: pipeline.forward grew
        # hard_alloc/alloc_dropout_p, and a shim that pins the old parameter
        # list would silently drop any kwarg added after it.
        path_noise_costs_out, gsnr_preds_out, path_inds_out, alloc_out = \
            original_forward(demands, *fwd_args, **fwd_kwargs)

        for did, pi in path_inds_out.items():
            def make_hook(demand_id):
                def hook(g):
                    captured_grad_output[demand_id] = g.detach().clone()
                return hook
            pi.register_hook(make_hook(did))

        return path_noise_costs_out, gsnr_preds_out, path_inds_out, alloc_out

    pipeline.forward = instrumented_forward

    # --------------------------------------------------------- forward + backward
    path_noise_costs, gsnr_preds, path_indicators, alloc = pipeline(
        demands, tau=tau, lambda_=lambda_
    )
    loss, metrics = compute_loss(
        gsnr_preds=gsnr_preds,
        path_noise_costs=path_noise_costs,
        demands=demands,
        device_count=alloc.device_count,
        modulation_config=mod_cfg,
        lambda_dev=cfg["pipeline"]["lambda_dev"],
        duals=torch.full((len(demands),), cfg["constraint"]["dual_init"]),
        margin_db=cfg["constraint"]["margin_db"],
        lambda_cost=lambda_cost,
    )
    loss.backward()

    edge_weights_np = captured_edge_weights.numpy()
    edge_index_np = pipeline._edge_index.numpy()
    num_nodes = topology.num_nodes

    # ================================================================ REPORT 1
    print("=" * 65)
    print("1. ALLOCATION DISTRIBUTION")
    print("=" * 65)
    # Only the REAL (demand, boundary) variables: padded columns are forced
    # to exactly 0 by the rollout and would drag every statistic below
    # toward zero in proportion to how ragged the routes happen to be.
    real = alloc.boundary_node_ids >= 0
    a = alloc.a.detach()[real].numpy()
    sv = alloc.site_view.detach().numpy()
    print(f"  demands             : {alloc.a.shape[0]}   "
          f"(demand, boundary) variables: {len(a)}")
    print(f"  device_count        : {alloc.device_count.item():.4f}  "
          f"(printed as 'device_count' in the training log)")
    print(f"  num a > 0.5         : {int((a > 0.5).sum())}")
    print(f"  num a > 0.9         : {int((a > 0.9).sum())}")
    if len(a):
        print(f"  min / mean / max    : {a.min():.3f} / {a.mean():.3f} / {a.max():.3f}")
        top5 = np.sort(a)[::-1][:5]
        print(f"  top-5 allocations   : {' '.join(f'{v:.3f}' for v in top5)}")
    print(f"  site_view > 0.5     : {int((sv > 0.5).sum())} node(s)  "
          f"(DIAGNOSTIC ONLY — never priced, never in the selection key)")
    touched = [n for n in sorted(ctx.regen_candidates) if sv[n] > 0.5]
    print(f"  regen candidates    : {len(ctx.regen_candidates)} nodes, "
          f"{len(touched)} of them above 0.5: {touched}")

    # ================================================================ REPORT 2
    print()
    print("=" * 65)
    print("2. VLASTELICA BACKWARD: HAMMING DISTANCE PER DEMAND")
    print("=" * 65)
    print(f"  λ_vlastelica = {lambda_:.1f},  λ_cost = {lambda_cost}")
    print()

    total_hamming = 0
    for demand in demands:
        did = demand.id
        pi = path_indicators[did].detach().numpy()
        active_original = set(np.where(pi > 0.5)[0])

        if did not in captured_grad_output:
            print(f"  demand {did}: grad not captured (path_indicator hook missed)")
            continue

        g = captured_grad_output[did].numpy()
        c_target = edge_weights_np + lambda_ * g

        # Check if perturbation is proportional to edge_weights
        # If g ∝ edge_weights, then c_target ∝ edge_weights → same path
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(
                np.abs(edge_weights_np) > 1e-8,
                g / edge_weights_np,
                np.nan,
            )
        valid_ratio = ratio[~np.isnan(ratio)]
        ratio_std = float(np.std(valid_ratio)) if len(valid_ratio) > 1 else float('nan')

        # Perturbed solve
        path_perturbed = spfa(c_target, edge_index_np, demand.src, demand.dst, num_nodes)
        if path_perturbed is None:
            print(f"  demand {did} ({demand.src}→{demand.dst}): SPFA returned None on perturbed graph")
            continue

        active_perturbed = set(np.where(path_perturbed > 0.5)[0])
        hamming = len(active_original.symmetric_difference(active_perturbed))
        total_hamming += hamming

        print(f"  demand {did} ({demand.src:2d}→{demand.dst:2d}): "
              f"path_len={len(active_original):2d} edges | "
              f"Hamming={hamming} | "
              f"g/w ratio std={ratio_std:.2e} | "
              f"g norm={np.linalg.norm(g):.2e} | "
              f"perturbation norm={np.linalg.norm(c_target - edge_weights_np):.2e}")

    print()
    print(f"  Total Hamming across {len(demands)} demands: {total_hamming}")
    if total_hamming == 0:
        print()
        print("  WARNING: Hamming distance is 0 for ALL demands.")
        print("  The Vlastelica backward is finding the same path after perturbation.")
        print()
        print("  This is no longer explained by grad_output being proportional to")
        print("  edge_weights: since correction #6, grad_output is dominated by the")
        print("  straight-through per-edge ASE-noise proxy, and since correction #9")
        print("  path_noise_cost is denominated in edge_ase_noise, not edge_weights.")
        print("  Both terms already differentiate per edge, so a zero Hamming")
        print("  distance here means the ASE-noise-driven perturbation happens to")
        print("  leave every demand's shortest path unchanged on this topology --")
        print("  check whether edge_ase_noise actually varies across its edges")
        print("  (docs/architecture/invariants.md: 'on a topology whose edges are physically identical,")
        print("  the STE contributes no within-segment routing signal').")

    # ================================================================ REPORT 3
    print()
    print("=" * 65)
    print("3. GRADIENT BREAKDOWN (routing parameter and allocation head)")
    print("=" * 65)
    theta = pipeline.edge_log_weight
    theta_grad = theta.grad
    theta_norm = theta_grad.norm().item() if theta_grad is not None else 0.0
    print(f"  edge_log_weight grad norm: {theta_norm:.4e}   "
          f"(E={theta.numel()} free per-edge parameters, spec decision 6)")
    if theta_grad is not None:
        print(f"    nonzero on {int((theta_grad != 0).sum())}/{theta.numel()} edges  "
              f"max|g|={theta_grad.abs().max().item():.4e}")

    head_grad_norm = sum(
        p.grad.norm().item() for p in pipeline.allocation_head.parameters()
        if p.grad is not None
    )
    print(f"  allocation head grad norm (all params): {head_grad_norm:.4e}")
    print()
    print("  per-parameter grad norms (AllocationHead):")
    for name, p in pipeline.allocation_head.named_parameters():
        g_norm = p.grad.norm().item() if p.grad is not None else 0.0
        print(f"    {name:30s} grad_norm={g_norm:.4e}")
    print("    (a zero on every HIDDEN layer is the closed init, not a bug: "
          "the output\n     layer starts at zero weight, so d(score)/d(hidden) "
          "is exactly 0 — spec 2.2)")


if __name__ == "__main__":
    main()
