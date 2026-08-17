"""Diagnostic: Vlastelica surrogate health check.

Checks two things the smoke-run output does not reveal:

1. Regen probability distribution
   The training log prints regen_loss = regen_probs.sum(), which starts at
   num_nodes * sigmoid(0) = num_nodes * 0.5 and decreases. The number
   "2.54 regens" is NOT a count of placed regenerators — it is the sum of
   all per-node probabilities. This script shows the full distribution.

2. Hamming distance in the Vlastelica backward
   For the surrogate to provide routing-change signal, the perturbed solve
   must find a DIFFERENT path. Since correction #6, grad_output =
   ∂L/∂path_indicator is dominated by the straight-through per-edge
   ASE-noise proxy (threshold-gated, regen-modulated), not by edge_weights;
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
    # historical tau=regen_tau_start / soft_max_temperature start /
    # undecayed vlastelica_lambda when no --checkpoint overrides them.
    tau, soft_max_temp, lambda_ = schedule_at(cfg, epoch=1)

    demands = demands_for(ctx, seed=args.seed)

    # ------------------------------------------------ intercept grad_output
    # edge_weights_of independently recomputes exactly what pipeline.forward
    # will compute internally for edge_weights (same live edge_weight_net /
    # regen_placement, same tau) -- no monkeypatch needed to capture it.
    # The hook below is only for grad_output on each path_indicator, which
    # edge_weights_of has no access to.
    captured_edge_weights = edge_weights_of(ctx, tau).detach().clone()
    captured_grad_output: Dict[int, torch.Tensor] = {}

    original_forward = pipeline.forward

    def instrumented_forward(demands, tau=1.0, lambda_=10.0, soft_max_temperature=0.5):
        path_noise_costs_out, gsnr_preds_out, path_inds_out, regen_probs_out = \
            original_forward(demands, tau=tau, lambda_=lambda_, soft_max_temperature=soft_max_temperature)

        for did, pi in path_inds_out.items():
            def make_hook(demand_id):
                def hook(g):
                    captured_grad_output[demand_id] = g.detach().clone()
                return hook
            pi.register_hook(make_hook(did))

        return path_noise_costs_out, gsnr_preds_out, path_inds_out, regen_probs_out

    pipeline.forward = instrumented_forward

    # --------------------------------------------------------- forward + backward
    path_noise_costs, gsnr_preds, path_indicators, regen_probs = pipeline(
        demands, tau=tau, lambda_=lambda_, soft_max_temperature=soft_max_temp
    )
    loss, metrics = compute_loss(
        gsnr_preds=gsnr_preds,
        path_noise_costs=path_noise_costs,
        demands=demands,
        regen_probs=regen_probs,
        modulation_config=mod_cfg,
        lambda_regen=cfg["pipeline"]["lambda_regen"],
        lambda_infeasible=cfg["pipeline"]["lambda_infeasible"],
        lambda_cost=lambda_cost,
    )
    loss.backward()

    edge_weights_np = captured_edge_weights.numpy()
    edge_index_np = pipeline._edge_index.numpy()
    num_nodes = topology.num_nodes

    # ================================================================ REPORT 1
    print("=" * 65)
    print("1. REGENERATOR PROBABILITY DISTRIBUTION")
    print("=" * 65)
    rp = regen_probs.detach().numpy()
    print(f"  num_nodes         : {len(rp)}")
    print(f"  regen_probs.sum() : {rp.sum():.4f}  (printed as 'regen=' in training log)")
    print(f"  num nodes > 0.5   : {int((rp > 0.5).sum())}  (num_regen_soft metric)")
    print(f"  num nodes > 0.9   : {int((rp > 0.9).sum())}")
    print(f"  min / mean / max  : {rp.min():.3f} / {rp.mean():.3f} / {rp.max():.3f}")
    sorted_probs = np.sort(rp)[::-1]
    top5 = sorted_probs[:5]
    print(f"  top-5 probs       : {' '.join(f'{v:.3f}' for v in top5)}")
    regen_candidates = sorted(ctx.regen_candidates)
    print(f"  regen candidates  : nodes {regen_candidates}")
    print(f"  their probs       : {' '.join(f'{rp[n]:.3f}' for n in regen_candidates)}")

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
        print("  (CLAUDE.md: 'on a topology whose edges are physically identical,")
        print("  the STE contributes no within-segment routing signal').")

    # ================================================================ REPORT 3
    print()
    print("=" * 65)
    print("3. GRADIENT BREAKDOWN (EdgeWeightNet)")
    print("=" * 65)
    total_grad_norm = sum(
        p.grad.norm().item() for p in pipeline.edge_weight_net.parameters()
        if p.grad is not None
    )
    print(f"  Total grad norm (EdgeWeightNet params): {total_grad_norm:.4e}")
    print(f"  regen_logits grad norm: "
          f"{pipeline.regen_placement.regen_logits.grad.norm().item():.4e}")
    print()
    print("  edge_weights.grad (from EdgeWeightNet.parameters().grad):")
    ew_grad_norms = []
    for name, p in pipeline.edge_weight_net.named_parameters():
        g_norm = p.grad.norm().item() if p.grad is not None else 0.0
        ew_grad_norms.append((name, g_norm))
        print(f"    {name:30s} grad_norm={g_norm:.4e}")


if __name__ == "__main__":
    main()
