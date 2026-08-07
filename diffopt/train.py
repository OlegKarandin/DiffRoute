"""End-to-end training script for DiffONet (Phase 1c).

Entry point: python -m diffopt.train --config configs/experiment/base.yaml
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List

import torch
import torch.optim as optim
import yaml

from diffopt.demands import generate_demands
from diffopt.loss import compute_loss
from diffopt.modulation import ModulationConfig
from diffopt.pipeline import DiffONetPipeline
from diffopt.placement.regenerator import RegenPlacement
from diffopt.qot.model import SpanAttentionQoT
from diffopt.qot.segment_combiner import SegmentCombiner
from diffopt.routing.edge_weight_net import EdgeWeightNet
from diffopt.topology import load_topology


def load_qot_model(checkpoint_path: str, cfg: dict, device: torch.device) -> SpanAttentionQoT:
    """Load a pretrained SpanAttentionQoT from a Phase-1a checkpoint."""
    model = SpanAttentionQoT(
        feature_dim=cfg.get("feature_dim", 5),
        model_dim=cfg.get("model_dim", 64),
        num_heads=cfg.get("num_heads", 4),
        num_layers=cfg.get("num_layers", 2),
        max_spans=cfg.get("max_spans_per_segment", 60),
        ff_dim=cfg.get("ff_dim", 128),
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    # Freeze all parameters — done again in DiffONetPipeline.__init__,
    # but doing it here makes the intent explicit at load time.
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def linear_anneal(
    epoch: int,
    start: float,
    end: float,
    anneal_start: int,
    anneal_end: int,
) -> float:
    """Linearly anneal a scalar from `start` to `end` over [anneal_start, anneal_end].

    Generic — used for both RegenPlacement's `tau` (decision sharpness) and
    SegmentCombiner's `soft_max_temperature` (noise-combination accuracy).
    Both represent "how sharp should this continuous relaxation be," and
    annealing them on the same epoch window is deliberate: as regen
    decisions sharpen, the physics evaluation backing them sharpens too, at
    the same pace (see SegmentCombiner's docstring for why an un-annealed
    soft_max_temperature silently broke the "regen helps" invariant).
    """
    if epoch <= anneal_start:
        return start
    if epoch >= anneal_end:
        return end
    frac = (epoch - anneal_start) / (anneal_end - anneal_start)
    return start + frac * (end - start)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    topology = load_topology(cfg["topology"], cfg["modulation_formats"])
    mod_cfg = ModulationConfig.from_yaml(cfg["modulation_formats"])

    qot_model = load_qot_model(cfg["qot_checkpoint"], cfg, device)

    segment_combiner = SegmentCombiner()
    edge_weight_net = EdgeWeightNet().to(device)
    regen_placement = RegenPlacement(topology.num_nodes).to(device)

    pipeline = DiffONetPipeline(
        topology=topology,
        qot_model=qot_model,
        segment_combiner=segment_combiner,
        edge_weight_net=edge_weight_net,
        regen_placement=regen_placement,
        channel_loading_fraction=cfg["pipeline"]["channel_loading_fraction"],
        max_spans=cfg.get("max_spans_per_segment", 60),
    ).to(device)

    opt_edge = optim.Adam(
        edge_weight_net.parameters(),
        lr=cfg["training"]["lr_edge_net"],
    )
    opt_regen = optim.Adam(
        [regen_placement.regen_logits],
        lr=cfg["training"]["lr_regen"],
    )

    t_cfg = cfg["training"]
    p_cfg = cfg["pipeline"]
    sc_cfg = cfg.get("segment_combiner", {})
    soft_max_temp_start: float = sc_cfg.get("soft_max_temperature", 0.5)
    soft_max_temp_end: float = sc_cfg.get("soft_max_temperature_min", 0.01)

    vlastelica_lambda: float = t_cfg["vlastelica_lambda"]
    lambda_min: float = t_cfg["vlastelica_lambda_min"]
    lambda_decay: float = t_cfg["vlastelica_lambda_decay"]
    epochs: int = t_cfg["epochs_e2e"]
    checkpoint_interval: int = t_cfg.get("checkpoint_interval", 10)

    log_dir = Path(cfg.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "e2e_train_log.csv"
    best_loss = math.inf

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "total_loss", "feasibility_loss", "regen_loss",
            "path_cost_loss", "num_regen_soft", "num_infeasible",
            "tau", "soft_max_temperature", "vlastelica_lambda",
        ])

        for epoch in range(1, epochs + 1):
            tau = linear_anneal(
                epoch,
                t_cfg["regen_tau_start"],
                t_cfg["regen_tau_end"],
                t_cfg["regen_tau_anneal_start_epoch"],
                t_cfg["regen_tau_anneal_end_epoch"],
            )
            # Annealed on the same epoch window as tau — see linear_anneal's
            # docstring for why. Un-annealed soft_max_temperature (fixed at
            # 0.5) previously made the "regen helps" invariant backwards;
            # see SegmentCombiner's docstring and CLAUDE.md's Phase 1c
            # corrections for the diagnosis.
            soft_max_temp = linear_anneal(
                epoch,
                soft_max_temp_start,
                soft_max_temp_end,
                t_cfg["regen_tau_anneal_start_epoch"],
                t_cfg["regen_tau_anneal_end_epoch"],
            )

            # seed=epoch: different demand set each epoch, reproducible
            demands = generate_demands(
                topology,
                cfg["num_demands"],
                cfg["bitrate_options"],
                seed=epoch,
            )

            opt_edge.zero_grad()
            opt_regen.zero_grad()

            path_costs, gsnr_preds, _, regen_probs = pipeline(
                demands, tau=tau, lambda_=vlastelica_lambda, soft_max_temperature=soft_max_temp
            )
            loss, metrics = compute_loss(
                gsnr_preds=gsnr_preds,
                path_costs=path_costs,
                demands=demands,
                regen_probs=regen_probs,
                modulation_config=mod_cfg,
                lambda_regen=p_cfg["lambda_regen"],
                lambda_infeasible=p_cfg["lambda_infeasible"],
                lambda_cost=p_cfg["lambda_cost"],
            )

            loss.backward()
            opt_edge.step()
            opt_regen.step()

            vlastelica_lambda = max(lambda_min, vlastelica_lambda * lambda_decay)

            writer.writerow([
                epoch,
                f"{loss.item():.6f}",
                f"{metrics['feasibility_loss']:.6f}",
                f"{metrics['regen_loss']:.6f}",
                f"{metrics['path_cost_loss']:.6f}",
                metrics["num_regen_soft"],
                metrics["num_infeasible"],
                f"{tau:.4f}",
                f"{soft_max_temp:.4f}",
                f"{vlastelica_lambda:.4f}",
            ])
            f.flush()

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:4d}/{epochs} | loss={loss.item():.4f} "
                    f"| feasibility={metrics['feasibility_loss']:.4f} "
                    f"| regen={metrics['regen_loss']:.4f} "
                    f"| infeasible={metrics['num_infeasible']} "
                    f"| tau={tau:.3f} | t_sm={soft_max_temp:.3f} | λ={vlastelica_lambda:.3f}"
                )

            if epoch % checkpoint_interval == 0 and loss.item() < best_loss:
                best_loss = loss.item()
                ckpt_path = checkpoint_dir / "best_e2e.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "edge_weight_net_state": edge_weight_net.state_dict(),
                        "regen_logits": regen_placement.regen_logits.detach().cpu(),
                        "opt_edge_state": opt_edge.state_dict(),
                        "opt_regen_state": opt_regen.state_dict(),
                        "vlastelica_lambda": vlastelica_lambda,
                        "total_loss": loss.item(),
                    },
                    ckpt_path,
                )
                print(f"  -> Checkpoint saved (loss={best_loss:.4f})")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
