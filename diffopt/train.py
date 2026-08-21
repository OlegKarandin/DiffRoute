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

from diffopt.loss import compute_loss, update_duals
from diffopt.modulation import ModulationConfig
from diffopt.pipeline import DiffONetPipeline
from diffopt.placement.regenerator import RegenPlacement
from diffopt.qot.model import SpanAttentionQoT
from diffopt.qot.segment_combiner import SegmentCombiner
from diffopt.routing.edge_weight_net import EdgeWeightNet
from diffopt.topology import load_topology
from diffopt.traffic import build_traffic_matrix, preflight_filter, scenario_alpha


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

    Generic — currently drives RegenPlacement's `tau` (decision sharpness).
    It also used to drive SegmentCombiner's `soft_max_temperature` on the
    same epoch window, so that the physics evaluation sharpened at the same
    pace as the regen decisions it backed; SegmentCombiner's fold is exact
    now and has no such knob, but the helper stays generic — nothing about
    it is tau-specific.
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

    # Seed before any module construction — EdgeWeightNet's init otherwise
    # varies run to run and changes routing enough to move num_infeasible by
    # an order of magnitude (measured 5/100 vs 98/100 at epoch 1 on
    # ind_132), making runs incomparable.
    torch.manual_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    topology = load_topology(cfg["topology"], cfg["modulation_formats"])
    mod_cfg = ModulationConfig.from_yaml(cfg["modulation_formats"])

    qot_model = load_qot_model(cfg["qot_checkpoint"], cfg, device)

    segment_combiner = SegmentCombiner()

    # --- Fixed traffic matrix -------------------------------------------
    # Built ONCE, before the loop. train.py previously called
    # generate_demands(..., seed=epoch) and drew 100 fresh random demands
    # every epoch: "all demands feasible" cannot be enforced against a set
    # that is replaced each epoch, and a per-demand dual is meaningless
    # without demand identity persisting across epochs.
    tr_cfg = cfg["traffic"]
    c_cfg = cfg["constraint"]

    raw_matrix = build_traffic_matrix(
        topology,
        seed=tr_cfg["seed"],
        scale=tr_cfg["scale"],
        alpha=scenario_alpha(tr_cfg["scenario"]),
        bitrate_options=cfg["bitrate_options"],
    )
    demands, excluded = preflight_filter(
        topology,
        raw_matrix,
        qot_model=qot_model,
        segment_combiner=segment_combiner,
        modulation_config=mod_cfg,
        margin_db=c_cfg["margin_db"],
        channel_loading_fraction=cfg["pipeline"]["channel_loading_fraction"],
        max_spans=cfg.get("max_spans_per_segment", 60),
    )
    print(
        f"Traffic matrix ({tr_cfg['scenario']}, seed={tr_cfg['seed']}, "
        f"scale={tr_cfg['scale']:.3g}): {len(raw_matrix)} pairs, "
        f"{len(demands)} in the constraint set, {len(excluded)} excluded by preflight"
    )
    for demand, shortfall in excluded:
        print(
            f"  excluded d{demand.id}: {demand.src}->{demand.dst} @ "
            f"{demand.bitrate_gbps:.0f}G, shortfall {shortfall:.2f} dB"
        )
    if not demands:
        raise ValueError(
            "Preflight excluded every demand — check traffic.scale and the "
            "qot_checkpoint before training."
        )

    # One dual per demand, persisted across epochs, uniform at lambda_0.
    # Not an nn.Parameter and not on any optimizer: these are Lagrange
    # multipliers driven by an explicit ascent rule, not gradient descent.
    duals = torch.full((len(demands),), float(c_cfg["dual_init"]), device=device)

    # Mask of nodes where a placed regenerator can actually do something.
    # `lambda_regen * regen_probs.sum()` prices ALL num_nodes nodes, but
    # `pipeline.segment_path` only splits a route at regen candidates
    # (undirected degree >= 3 — 48 of 132 on ind_132), so probability mass
    # anywhere else is physically inert while still costing loss. Tracked as
    # its own log column rather than fixed here: restricting the penalty to
    # candidates would shrink it ~2.75x at fixed lambda_regen, which IS a
    # lambda_regen retune by the back door and is an explicit non-goal of the
    # design spec (§2).
    regen_candidate_mask = torch.zeros(
        topology.num_nodes, dtype=torch.bool, device=device
    )
    regen_candidate_mask[
        torch.tensor(topology.regen_candidate_nodes, dtype=torch.long, device=device)
    ] = True
    warned_inert = False

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
    vlastelica_lambda: float = t_cfg["vlastelica_lambda"]
    lambda_min: float = t_cfg["vlastelica_lambda_min"]
    lambda_decay: float = t_cfg["vlastelica_lambda_decay"]
    epochs: int = t_cfg["epochs_e2e"]

    log_dir = Path(cfg.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "e2e_train_log.csv"
    # Lexicographic selection: fewest violated demands, then fewest
    # regenerators, then lowest total loss.
    #
    # `loss.item() < best_loss` was already wrong under the old objective —
    # the shipped checkpoints/e2e_ind132/best_e2e.pt is epoch 40 while its own
    # run log records three later improvements ending at epoch 55, because
    # total loss is dominated by the tau anneal rather than by solution
    # quality (the epoch-40 and epoch-55 placements differ in 3 of 8 nodes and
    # in held-out infeasibility, 9/400 vs 7/400). Under a constrained
    # formulation `lambda` is deliberately non-stationary, so total loss
    # becomes still less comparable across epochs.
    #
    # num_regen_soft is (regen_probs > 0.5).sum(), which is exactly
    # (regen_logits > 0).sum() for any tau > 0 — tau-invariant, so it is a
    # legitimate cross-epoch comparison during annealing.
    best_key = (math.inf, math.inf, math.inf)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        # regen_logit_* are the RAW learned parameters. regen_loss alone is
        # misleading: it reports sum(sigmoid(logit/tau)) while tau is being
        # annealed, so it moves dramatically even when the logits are static
        # (measured: 87% of the observed 66 -> 18 fall came from tau, not
        # learning). Log the parameter itself so drift is visible directly.
        writer.writerow([
            "epoch", "total_loss", "feasibility_loss", "weighted_feasibility_loss",
            "regen_loss", "path_noise_loss", "num_regen_soft", "num_regen_noncand",
            "num_infeasible", "num_violated", "worst_margin_db",
            "lambda_max_observed", "num_at_cap",
            "tau", "vlastelica_lambda",
            "regen_logit_mean", "regen_logit_min", "regen_logit_max",
            "regen_prob_max",
        ])

        for epoch in range(1, epochs + 1):
            tau = linear_anneal(
                epoch,
                t_cfg["regen_tau_start"],
                t_cfg["regen_tau_end"],
                t_cfg["regen_tau_anneal_start_epoch"],
                t_cfg["regen_tau_anneal_end_epoch"],
            )

            opt_edge.zero_grad()
            opt_regen.zero_grad()

            path_noise_costs, gsnr_preds, _, regen_probs = pipeline(
                demands, tau=tau, lambda_=vlastelica_lambda
            )
            loss, metrics = compute_loss(
                gsnr_preds=gsnr_preds,
                path_noise_costs=path_noise_costs,
                demands=demands,
                regen_probs=regen_probs,
                modulation_config=mod_cfg,
                duals=duals,
                margin_db=c_cfg["margin_db"],
                lambda_regen=p_cfg["lambda_regen"],
                lambda_cost=p_cfg["lambda_cost"],
            )

            loss.backward()
            opt_edge.step()
            opt_regen.step()

            # Dual ascent, using the same shortfalls the loss consumed this
            # epoch. Runs AFTER the primal step so the duals price the
            # constraint violation the step was actually taken against.
            duals = update_duals(
                duals,
                metrics["shortfalls"],
                eta=c_cfg["dual_lr"],
                dual_max=c_cfg["dual_max"],
            )
            lambda_max_observed = duals.max().item()
            num_at_cap = int((duals >= c_cfg["dual_max"]).sum().item())

            # Placements that can never split a path. Expected to stay 0: a
            # non-candidate logit receives only down-pressure from the regen
            # penalty, and on the shipped epoch-40 checkpoint all 84
            # non-candidates sit pinned in a 0.0069-wide band at ~-0.817
            # while 8 of 8 placements land on candidates.
            #
            # Watched anyway because the duals now scale to dual_max, and they
            # amplify the one path by which feasibility DOES reach a
            # non-candidate logit: regen_probs feed EdgeWeightNet's edge
            # features at both endpoints of every edge. That path is worth
            # ~0.007 of logit at a fixed weight of 10; at 100x it is ~0.7,
            # comparable to Adam's entire +/-1.2 reachable excursion.
            placed_nodes = regen_probs.detach() > 0.5
            num_regen_noncand = int(
                (placed_nodes & ~regen_candidate_mask).sum().item()
            )
            if num_regen_noncand > 0 and not warned_inert:
                warned_inert = True
                inert = (placed_nodes & ~regen_candidate_mask)
                inert_ids = inert.nonzero(as_tuple=True)[0].tolist()
                print(
                    f"  !! epoch {epoch}: {num_regen_noncand} placement(s) on "
                    f"NON-candidate node(s) {inert_ids} — physically inert "
                    f"(segment_path never splits there) but counted by "
                    f"num_regen_soft and priced by lambda_regen. "
                    f"lambda_max_observed={lambda_max_observed:.1f}"
                )

            vlastelica_lambda = max(lambda_min, vlastelica_lambda * lambda_decay)

            logits = regen_placement.regen_logits.detach()
            writer.writerow([
                epoch,
                f"{loss.item():.6f}",
                f"{metrics['feasibility_loss']:.6f}",
                f"{metrics['weighted_feasibility_loss']:.6f}",
                f"{metrics['regen_loss']:.6f}",
                f"{metrics['path_noise_loss']:.6f}",
                metrics["num_regen_soft"],
                num_regen_noncand,
                metrics["num_infeasible"],
                metrics["num_violated"],
                f"{metrics['worst_margin_db']:.4f}",
                f"{lambda_max_observed:.4f}",
                num_at_cap,
                f"{tau:.4f}",
                f"{vlastelica_lambda:.4f}",
                f"{logits.mean().item():.6f}",
                f"{logits.min().item():.6f}",
                f"{logits.max().item():.6f}",
                f"{regen_probs.max().item():.6f}",
            ])
            f.flush()

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:4d}/{epochs} | loss={loss.item():.4f} "
                    f"| violated={metrics['num_violated']}/{len(demands)} "
                    f"| infeasible={metrics['num_infeasible']} "
                    f"| worst_margin={metrics['worst_margin_db']:+.2f}dB "
                    f"| regen_soft={metrics['num_regen_soft']}"
                    f"({num_regen_noncand} inert) "
                    f"| lambda_max={lambda_max_observed:.1f} at_cap={num_at_cap} "
                    f"| logit[{logits.min().item():+.3f},{logits.max().item():+.3f}] "
                    f"| tau={tau:.3f} | λ={vlastelica_lambda:.3f}"
                )

            selection_key = (
                metrics["num_violated"],
                metrics["num_regen_soft"],
                loss.item(),
            )
            if selection_key < best_key:
                best_key = selection_key
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
                        "num_violated": metrics["num_violated"],
                        "num_regen_soft": metrics["num_regen_soft"],
                        "duals": duals.detach().cpu(),
                    },
                    ckpt_path,
                )
                print(
                    f"  -> Checkpoint saved (violated={selection_key[0]}, "
                    f"regens={selection_key[1]}, loss={selection_key[2]:.4f})"
                )

    print(
        f"\nTraining complete. Best: violated={best_key[0]}, "
        f"regens={best_key[1]}, loss={best_key[2]:.4f}"
    )
    at_cap = (duals >= c_cfg["dual_max"]).nonzero(as_tuple=True)[0].tolist()
    if at_cap:
        # A demand pinned at the cap is one the preflight could not exclude
        # (it IS reachable under shortest-by-km with full regeneration) but
        # that the learned routing never reaches. Naming them turns a
        # non-converging constraint into a report instead of silent
        # oscillation.
        print(f"{len(at_cap)} demand(s) pinned at dual_max={c_cfg['dual_max']}:")
        for i in at_cap:
            d = demands[i]
            print(f"  d{d.id}: {d.src}->{d.dst} @ {d.bitrate_gbps:.0f}G")
    else:
        print(f"No demands pinned at dual_max (max observed {duals.max().item():.2f})")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
