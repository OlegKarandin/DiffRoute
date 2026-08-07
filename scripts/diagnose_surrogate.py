"""Diagnostic: Vlastelica surrogate health check.

Checks two things the smoke-run output does not reveal:

1. Regen probability distribution
   The training log prints regen_loss = regen_probs.sum(), which starts at
   num_nodes * sigmoid(0) = num_nodes * 0.5 and decreases. The number
   "2.54 regens" is NOT a count of placed regenerators — it is the sum of
   all per-node probabilities. This script shows the full distribution.

2. Hamming distance in the Vlastelica backward
   For the surrogate to provide routing-change signal, the perturbed solve
   must find a DIFFERENT path. If grad_output = ∂L/∂path_indicator is
   proportional to edge_weights (e.g., only from path_cost), then
   c_target = w + λ * g = w * (1 + λ * λ_cost), which is a uniform scaling
   that preserves the shortest-path ordering — Hamming distance is zero and
   the surrogate gradient is zero.

Usage:
    conda activate diffopt
    python scripts/diagnose_surrogate.py --config configs/experiment/small_test.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

from diffopt.demands import generate_demands
from diffopt.loss import compute_loss
from diffopt.modulation import ModulationConfig
from diffopt.pipeline import DiffONetPipeline
from diffopt.placement.regenerator import RegenPlacement
from diffopt.qot.model import SpanAttentionQoT
from diffopt.qot.segment_combiner import SegmentCombiner
from diffopt.routing.edge_weight_net import EdgeWeightNet
from diffopt.routing.shortest_path import spfa
from diffopt.topology import load_topology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="E2E checkpoint; omit to use random weights")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cpu")

    topology = load_topology(cfg["topology"], cfg["modulation_formats"])
    mod_cfg = ModulationConfig.from_yaml(cfg["modulation_formats"])

    # ------------------------------------------------------------------ model
    qot_ckpt_path = args.checkpoint or cfg.get("qot_checkpoint", "checkpoints/best_qot.pt")

    from diffopt.train import load_qot_model
    qot_model = load_qot_model(qot_ckpt_path, cfg, device)

    edge_weight_net = EdgeWeightNet().to(device)
    regen_placement = RegenPlacement(topology.num_nodes).to(device)

    pipeline = DiffONetPipeline(
        topology=topology,
        qot_model=qot_model,
        segment_combiner=SegmentCombiner(0.5),
        edge_weight_net=edge_weight_net,
        regen_placement=regen_placement,
        channel_loading_fraction=cfg["pipeline"]["channel_loading_fraction"],
    ).to(device)

    lambda_ = cfg["training"]["vlastelica_lambda"]
    lambda_cost = cfg["pipeline"]["lambda_cost"]
    tau = cfg["training"]["regen_tau_start"]

    demands = generate_demands(
        topology, cfg["num_demands"], cfg["bitrate_options"], seed=args.seed
    )

    # ------------------------------------------------ intercept grad_output
    # Register hooks on each path_indicator to capture ∂L/∂path_indicator
    captured_grad_output: Dict[int, torch.Tensor] = {}
    captured_edge_weights: List[torch.Tensor] = []

    # We patch the forward to capture edge_weights and attach hooks to
    # path_indicators before they enter the surrogate backward.
    original_forward = pipeline.forward

    def instrumented_forward(demands, tau=1.0, lambda_=10.0):
        regen_probs = pipeline.regen_placement.get_regen_probs(tau)
        edge_feats = torch.cat([
            pipeline._topo_edge_features,
            regen_probs[pipeline._edge_src_ids].unsqueeze(1),
            regen_probs[pipeline._edge_dst_ids].unsqueeze(1),
        ], dim=1)
        ew = pipeline.edge_weight_net(edge_feats).squeeze(-1)
        captured_edge_weights.clear()
        captured_edge_weights.append(ew.detach().clone())

        path_costs_out, gsnr_preds_out, path_inds_out, regen_probs_out = \
            original_forward(demands, tau=tau, lambda_=lambda_)

        for did, pi in path_inds_out.items():
            def make_hook(demand_id):
                def hook(g):
                    captured_grad_output[demand_id] = g.detach().clone()
                return hook
            pi.register_hook(make_hook(did))

        return path_costs_out, gsnr_preds_out, path_inds_out, regen_probs_out

    pipeline.forward = instrumented_forward

    # --------------------------------------------------------- forward + backward
    path_costs, gsnr_preds, path_indicators, regen_probs = pipeline(
        demands, tau=tau, lambda_=lambda_
    )
    loss, metrics = compute_loss(
        gsnr_preds=gsnr_preds,
        path_costs=path_costs,
        demands=demands,
        regen_probs=regen_probs,
        modulation_config=mod_cfg,
        lambda_regen=cfg["pipeline"]["lambda_regen"],
        lambda_infeasible=cfg["pipeline"]["lambda_infeasible"],
        lambda_cost=lambda_cost,
    )
    loss.backward()

    edge_weights_np = captured_edge_weights[0].numpy()
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
    regen_candidates = topology.regen_candidate_nodes
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
        print("  Root cause: ∂L/∂path_indicator = λ_cost * edge_weights.")
        print("  The perturbation c_target = w + λ*(λ_cost*w) = w*(1 + λ*λ_cost)")
        print("  is a uniform scaling of all edge costs, which preserves the")
        print("  shortest-path ordering. The surrogate gradient to EdgeWeightNet")
        print("  is therefore zero — only the direct ∂(p·w)/∂w = p term is active.")
        print()
        print("  EdgeWeightNet still receives gradient (from the direct path term),")
        print("  but it only learns 'reduce weights of currently-selected edges',")
        print("  NOT 'change routing to improve feasibility'.")
        print()
        print("  Fix: ∂L/∂path_indicator needs per-edge differentiation.")
        print("  Option A: per-edge GSNR proxy (requires E QoT calls per demand).")
        print("  Option B: segment-level differentiation by making boundary_nodes")
        print("            a soft function of path_indicator instead of a hard detach.")

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
