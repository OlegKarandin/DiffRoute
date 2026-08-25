"""End-to-end training script for DiffONet (Phase 1c).

Entry point: python -m diffopt.train --config configs/experiment/base.yaml
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml

from diffopt.demands import Demand
from diffopt.loss import compute_loss, update_duals
from diffopt.modulation import ModulationConfig, bar_db_for_demands
from diffopt.pipeline import DiffONetPipeline
from diffopt.placement.allocation import AllocationHead
from diffopt.placement.oracle import oracle_allocation, oracle_gap
from diffopt.qot.model import SpanAttentionQoT
from diffopt.qot.segment_combiner import SegmentCombiner
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

    Generic — currently drives the AllocationHead's `tau` (decision
    sharpness). It also used to drive SegmentCombiner's
    `soft_max_temperature` on the same epoch window, so that the physics
    evaluation sharpened at the same pace as the allocation decisions it
    backed; SegmentCombiner's fold is exact now and has no such knob, but the
    helper stays generic — nothing about it is tau-specific.
    """
    if epoch <= anneal_start:
        return start
    if epoch >= anneal_end:
        return end
    frac = (epoch - anneal_start) / (anneal_end - anneal_start)
    return start + frac * (end - start)


def alloc_score_stats(alloc, tau: float) -> Tuple[float, float, float]:
    """(mean, min, max) of the head's raw scores on REAL boundaries.

    The scores themselves are not returned by the forward pass, but they are
    exactly recoverable from the priced allocations that are: on a valid
    boundary `a = sigmoid(score / tau)`, so `score = tau * logit(a)`. Padded
    columns (`boundary_node_ids < 0`) hold exactly 0 and are excluded — a
    padded cut is no cut, not a score of -inf.

    `a` is clamped to float32's own representable range before the logit
    (~[-87.3, +15.9] * tau in score units, asymmetric because a sigmoid
    approaches 1 far sooner than it underflows to 0). The clamp is therefore
    the point where the information is genuinely gone from `a`, not an
    arbitrary readout window: a column pinned at either cap IS saturation,
    which is what this diagnostic exists to make visible. Read a
    score_min == score_max at a cap as "every boundary saturated", not as
    "the head stopped discriminating between boundaries".

    Returns (nan, nan, nan) when the routing produced no boundary at all.
    """
    a = alloc.a.detach()
    if a.numel() == 0:
        return math.nan, math.nan, math.nan
    valid = alloc.boundary_node_ids >= 0
    if not bool(valid.any()):
        return math.nan, math.nan, math.nan
    finfo = torch.finfo(torch.float32)
    p = a[valid].to(dtype=torch.float64).clamp(finfo.tiny, 1.0 - finfo.eps)
    scores = tau * (torch.log(p) - torch.log1p(-p))
    return (
        float(scores.mean().item()),
        float(scores.min().item()),
        float(scores.max().item()),
    )


def hard_rollout(
    pipeline,
    demands: List[Demand],
    modulation_config: ModulationConfig,
    *,
    lambda_: float,
    margin_db: float,
) -> dict:
    """Evaluate the DEPLOYED allocation — the one that would actually ship.

    Deterministic hard decisions (a_k = 1 if score_k > 0), NOT a threshold
    applied to the soft pass. The distinction is physical, not cosmetic:
    under hard decisions the head's carry `c` is the EXACT noise of the
    current chunk, so the rollout is self-consistent. The soft pass is a
    mean-field relaxation whose carry `c * (1 - a)` is an expectation over
    partitions; thresholding it reports a number no single allocation
    produces. Spec 2.5.

    Also computes the oracle's minimum on the SAME routes and segment GSNRs,
    so `oracle_gap` costs no extra forward pass. The gap is the acceptance
    metric for the whole stage: 0 means the head allocates optimally given
    the routes, and therefore that anything still wrong is a ROUTING
    problem. Note the oracle is ground truth only — its allocation is never
    substituted here. Repair is a deployment step
    (`diffopt.placement.oracle.repair_with_oracle`), and running it during
    training would destroy this signal.

    Runs under no_grad, so it neither disturbs the live graph nor mutates a
    parameter. `tau` is irrelevant to a hard decision and is not passed.

    WHICH DEMANDS THE GAP IS MEASURED OVER. `oracle.count[d]` is a lower
    bound on device count only among allocations that make demand d
    FEASIBLE. Two situations break that:

      the head under-buys   epoch 1 has the head initialised CLOSED (spec
                            2.2), so it buys 0 devices while the oracle
                            needs many. `0 - many` is negative, and it is
                            not a competence measurement — the shortfall is
                            already reported, exactly, as
                            `hard_num_violated`.

      nothing works at all  when a single segment alone busts the bar
                            (`oracle.feasible[d]` False) no allocation on
                            this route is feasible, and `oracle.count[d]`
                            is only the oracle's best-effort attempt — not
                            a minimum of anything.

    So the gap sums over the demands the head actually made feasible, where
    minimality really does apply and `oracle_gap`'s negative-gap assertion
    really does mean "the fold and the oracle disagree about chunk noise",
    which is the bug it exists to catch. `oracle_devices` stays the total
    floor over ALL demands, because "this routing needs at least N devices"
    is the useful reading of it. The two therefore satisfy
    `oracle_gap == hard_num_devices - oracle_devices` exactly when
    `hard_num_violated == 0` — which is the only regime the stage's
    acceptance test (`oracle_gap == 0` at the selected epoch, an epoch
    selected on zero violations first) ever reads.
    """
    with torch.no_grad():
        _, gsnr_preds, _, alloc = pipeline(demands, lambda_=lambda_, hard_alloc=True)

        bar_db = bar_db_for_demands(demands, modulation_config, margin_db).to(
            alloc.a.device
        )
        oracle = oracle_allocation(alloc.seg_gsnr_db, bar_db, alloc.num_segments)

    num_violated = 0
    worst_margin_db = math.inf
    bar_by_id = {}
    for demand in demands:
        threshold = modulation_config.required_snr_threshold(demand.bitrate_gbps)
        bar_by_id[demand.id] = threshold + margin_db
        gsnr = gsnr_preds[demand.id].item()
        if gsnr < threshold + margin_db:
            num_violated += 1
        worst_margin_db = min(worst_margin_db, gsnr - threshold)

    # Row order follows `alloc.demand_ids`, not `demands` — the oracle's
    # rows are indexed the same way.
    hard_feasible = torch.tensor(
        [gsnr_preds[did].item() >= bar_by_id[did] for did in alloc.demand_ids],
        dtype=torch.bool,
    )

    site_mask = alloc.site_view > 0.5
    hard_devices = int(alloc.a.sum().item())

    return {
        "hard_num_violated": num_violated,
        "hard_num_devices": hard_devices,
        "hard_num_sites": int(site_mask.sum().item()),
        "hard_worst_margin_db": worst_margin_db if demands else math.nan,
        "oracle_devices": int(oracle.count.sum().item()),
        "oracle_gap": oracle_gap(
            alloc.a[hard_feasible], oracle.count[hard_feasible]
        ),
        "oracle_infeasible": int((~oracle.feasible).sum().item()),
        "site_mask": site_mask.cpu(),
        "alloc_by_node": alloc.alloc_by_node.cpu(),
        # The per-demand path GSNRs this rollout measured. Returned so a
        # report can quote margins from the SAME deployed pass that produced
        # the device/site counts above: a second forward would be a soft one
        # and would describe a different allocation, which is precisely the
        # mismatch scripts/evaluate_matrix.py used to have.
        "gsnr_preds": {did: g.detach().cpu() for did, g in gsnr_preds.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    # Seed before any module construction. edge_log_weight's init is now
    # deterministic (length-proportional, spec decision 6), but
    # AllocationHead's init is not, and an unseeded init otherwise varies
    # run to run and changes routing/placement enough to move num_infeasible
    # by an order of magnitude (measured 5/100 vs 98/100 at epoch 1 on
    # ind_132 under the old EdgeWeightNet), making runs incomparable.
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

    # There is no inert-placement column any more: the old `num_regen_noncand`
    # watched for probability mass landing on a node `segment_path` never
    # splits at, which a per-(demand, boundary) variable cannot express —
    # every allocation variable IS a real boundary of a real route.

    pl_cfg = cfg.get("placement", {})
    lookahead: bool = pl_cfg.get("lookahead", True)
    route_context: bool = pl_cfg.get("route_context", True)
    greedy_residual: bool = pl_cfg.get("greedy_residual", False)
    allocation_head = AllocationHead(
        lookahead=lookahead, route_context=route_context, greedy_residual=greedy_residual
    ).to(device)

    pipeline = DiffONetPipeline(
        topology=topology,
        qot_model=qot_model,
        segment_combiner=segment_combiner,
        allocation_head=allocation_head,
        modulation_config=mod_cfg,
        margin_db=c_cfg["margin_db"],
        channel_loading_fraction=cfg["pipeline"]["channel_loading_fraction"],
        max_spans=cfg.get("max_spans_per_segment", 60),
    ).to(device)

    opt_edge = optim.Adam(
        [pipeline.edge_log_weight],
        lr=cfg["training"]["lr_edge_net"],
    )
    opt_alloc = optim.Adam(
        allocation_head.parameters(),
        lr=cfg["training"]["lr_alloc"],
    )

    t_cfg = cfg["training"]
    p_cfg = cfg["pipeline"]

    vlastelica_lambda: float = t_cfg["vlastelica_lambda"]
    lambda_min: float = t_cfg["vlastelica_lambda_min"]
    lambda_decay: float = t_cfg["vlastelica_lambda_decay"]
    epochs: int = t_cfg["epochs_e2e"]

    # Training-only masking of the PHYSICS allocation decisions. See
    # DiffONetPipeline.forward's alloc_dropout_p docstring. OFF by default:
    # the old gate_dropout manufactured node-discriminating gradient that a
    # per-node mask could not otherwise get, and a per-demand variable
    # already has a sharp, demand-specific signal.
    alloc_dropout_p: float = pl_cfg.get("alloc_dropout_p", 0.0)

    log_dir = Path(cfg.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "e2e_train_log.csv"
    # One row per epoch of the DEPLOYED allocation. A checkpoint is one
    # sample of a process that may not be converging at all; this is the
    # process. Two things it makes measurable that no snapshot can:
    # churn (mean Hamming distance between consecutive epochs' sets over the
    # last N epochs — a direct read on the limit cycle described in
    # open_followups.md item #6) and whether the final epoch agrees with the
    # selected one. It also allows re-running selection under a different
    # key without retraining.
    trajectory_path = log_dir / "placement_trajectory.csv"
    # Lexicographic selection on the DEPLOYED allocation: fewest violated
    # demands, then fewest DEVICES, then the most headroom.
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
    # The middle slot is a DEVICE count, not a site count. A site count let
    # an epoch win by touching fewer nodes while buying more hardware —
    # 40 demands regenerating at node 7 need 40 devices, not 1.
    best_key = (math.inf, math.inf, math.inf)

    with open(log_path, "w", newline="") as f, \
         open(trajectory_path, "w", newline="") as tf:
        writer = csv.writer(f)
        traj_writer = csv.writer(tf)
        traj_writer.writerow(["epoch", "num_devices", "num_sites", "site_nodes"])
        # alloc_score_* are the head's RAW scores, recovered from the priced
        # allocations (see alloc_score_stats). device_count alone is
        # misleading while tau is annealed: it reports sum(sigmoid(score/tau))
        # and so moves dramatically even when the scores are static. Log the
        # scores themselves so saturation is visible directly.
        writer.writerow([
            "epoch", "total_loss", "feasibility_loss", "weighted_feasibility_loss",
            "device_loss", "path_noise_loss", "device_count",
            "num_infeasible", "num_violated",
            "hard_num_violated", "hard_num_devices", "hard_num_sites",
            "hard_worst_margin_db", "oracle_devices", "oracle_gap",
            "oracle_infeasible", "worst_margin_db",
            "lambda_max_observed", "num_at_cap",
            "tau", "vlastelica_lambda",
            "alloc_score_mean", "alloc_score_min", "alloc_score_max",
            "alloc_dropout_p", "lookahead",
            "route_context", "greedy_residual", "lambda_waste", "waste_loss", "alloc_alpha",
            "ste_clamped_segments", "proxy_qot_rank_corr",
        ])

        for epoch in range(1, epochs + 1):
            tau = linear_anneal(
                epoch,
                t_cfg["alloc_tau_start"],
                t_cfg["alloc_tau_end"],
                t_cfg["alloc_tau_anneal_start_epoch"],
                t_cfg["alloc_tau_anneal_end_epoch"],
            )

            opt_edge.zero_grad()
            opt_alloc.zero_grad()

            path_noise_costs, gsnr_preds, _, alloc = pipeline(
                demands, tau=tau, lambda_=vlastelica_lambda,
                alloc_dropout_p=alloc_dropout_p,
            )
            loss, metrics = compute_loss(
                gsnr_preds=gsnr_preds,
                path_noise_costs=path_noise_costs,
                demands=demands,
                device_count=alloc.device_count,
                modulation_config=mod_cfg,
                duals=duals,
                margin_db=c_cfg["margin_db"],
                lambda_dev=p_cfg["lambda_dev"],
                lambda_cost=p_cfg["lambda_cost"],
                waste_cost=alloc.waste_cost,
                lambda_waste=p_cfg.get("lambda_waste", 0.0),
            )

            # Pre-step, like the state snapshots below — same row, same
            # parameters, not next epoch's already-updated ones.
            score_mean, score_min, score_max = alloc_score_stats(alloc, tau)
            device_loss = p_cfg["lambda_dev"] * metrics["device_count"]
            waste_loss = p_cfg.get("lambda_waste", 0.0) * metrics["waste_cost"]

            # Snapshot the state the forward pass above (and therefore
            # `metrics` -- num_violated, device_count, worst_margin_db,
            # loss.item()) actually describes, BEFORE the optimizer step
            # mutates it. Checkpoint selection compares `metrics` across
            # epochs, so the checkpoint must save the parameters those
            # metrics were measured on, not next epoch's already-updated
            # ones. Confirmed as a real (if usually small) discrepancy on
            # the dual_decay hypothesis-test run in open_followups.md item
            # #3: the log recorded 9 regens for the saved epoch, but the
            # previously-saved (post-step) regen_logits had 10 nodes above
            # threshold once reloaded.
            edge_log_weight_pre_step = pipeline.edge_log_weight.detach().clone()
            alloc_head_state_pre_step = {
                k: v.clone() for k, v in allocation_head.state_dict().items()
            }
            # Read straight from the pre-step state dict, not the live
            # module's `.alpha` accessor — by the time the CSV row is
            # written, opt_alloc.step() has already mutated alpha_raw, and
            # this row must describe the SAME parameters as the snapshot
            # above (comment there explains why).
            alpha_pre_step = (
                float(F.softplus(alloc_head_state_pre_step["alpha_raw"]))
                if greedy_residual
                else float("nan")
            )

            # Selection metrics, measured on the deployed allocation rather
            # than on the relaxation the gradient step is taken through.
            # Pre-step, like the two snapshots above: the checkpoint must
            # save the parameters its own key describes.
            hard = hard_rollout(
                pipeline, demands, mod_cfg,
                lambda_=vlastelica_lambda,
                margin_db=c_cfg["margin_db"],
            )

            loss.backward()
            opt_edge.step()
            opt_alloc.step()

            # Dual ascent, using the same shortfalls the loss consumed this
            # epoch. Runs AFTER the primal step so the duals price the
            # constraint violation the step was actually taken against.
            duals = update_duals(
                duals,
                metrics["shortfalls"],
                eta=c_cfg["dual_lr"],
                dual_max=c_cfg["dual_max"],
                decay=c_cfg.get("dual_decay", 0.0),
            )
            lambda_max_observed = duals.max().item()
            num_at_cap = int((duals >= c_cfg["dual_max"]).sum().item())

            vlastelica_lambda = max(lambda_min, vlastelica_lambda * lambda_decay)

            writer.writerow([
                epoch,
                f"{loss.item():.6f}",
                f"{metrics['feasibility_loss']:.6f}",
                f"{metrics['weighted_feasibility_loss']:.6f}",
                f"{device_loss:.6f}",
                f"{metrics['path_noise_loss']:.6f}",
                f"{metrics['device_count']:.6f}",
                metrics["num_infeasible"],
                metrics["num_violated"],
                hard["hard_num_violated"],
                hard["hard_num_devices"],
                hard["hard_num_sites"],
                f"{hard['hard_worst_margin_db']:.4f}",
                hard["oracle_devices"],
                hard["oracle_gap"],
                hard["oracle_infeasible"],
                f"{metrics['worst_margin_db']:.4f}",
                f"{lambda_max_observed:.4f}",
                num_at_cap,
                f"{tau:.4f}",
                f"{vlastelica_lambda:.4f}",
                f"{score_mean:.6f}",
                f"{score_min:.6f}",
                f"{score_max:.6f}",
                f"{alloc_dropout_p:.3f}",
                lookahead,
                route_context,
                greedy_residual,
                f"{p_cfg.get('lambda_waste', 0.0):.4f}",
                f"{waste_loss:.6f}",
                f"{alpha_pre_step:.6f}",
                alloc.ste_clamped_segments,
                f"{alloc.proxy_qot_rank_corr:.4f}",
            ])
            f.flush()

            site_nodes = hard["site_mask"].nonzero(as_tuple=True)[0].tolist()
            traj_writer.writerow([
                epoch,
                hard["hard_num_devices"],
                hard["hard_num_sites"],
                " ".join(str(n) for n in site_nodes),
            ])
            tf.flush()

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:4d}/{epochs} | loss={loss.item():.4f} "
                    f"| violated={metrics['num_violated']}/{len(demands)} "
                    f"| infeasible={metrics['num_infeasible']} "
                    f"| worst_margin={metrics['worst_margin_db']:+.2f}dB "
                    f"| devices={hard['hard_num_devices']}"
                    f"(oracle {hard['oracle_devices']}, gap {hard['oracle_gap']}) "
                    f"| sites={hard['hard_num_sites']} "
                    f"| lambda_max={lambda_max_observed:.1f} at_cap={num_at_cap} "
                    f"| score[{score_min:+.3f},{score_max:+.3f}] "
                    f"| tau={tau:.3f} | λ={vlastelica_lambda:.3f}"
                )

            # Lexicographic on the DEPLOYED allocation: fewest violated, then
            # fewest DEVICES, then the most headroom. The third slot used to
            # be loss.item(), which invariants.md itself calls non-comparable
            # across epochs — the tau anneal dominates it and the duals are
            # deliberately non-stationary. worst_margin_db is physical,
            # tau-invariant and already computed. Negated because the key is
            # minimised and MORE headroom is better.
            selection_key = (
                hard["hard_num_violated"],
                hard["hard_num_devices"],
                -hard["hard_worst_margin_db"],
            )
            if selection_key < best_key:
                best_key = selection_key
                ckpt_path = checkpoint_dir / "best_e2e.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "edge_log_weight": edge_log_weight_pre_step.cpu(),
                        "alloc_head_state": alloc_head_state_pre_step,
                        "opt_edge_state": opt_edge.state_dict(),
                        "opt_alloc_state": opt_alloc.state_dict(),
                        "vlastelica_lambda": vlastelica_lambda,
                        "total_loss": loss.item(),
                        "num_violated": metrics["num_violated"],
                        "device_count": metrics["device_count"],
                        "hard_num_violated": hard["hard_num_violated"],
                        "hard_num_devices": hard["hard_num_devices"],
                        "hard_num_sites": hard["hard_num_sites"],
                        "hard_worst_margin_db": hard["hard_worst_margin_db"],
                        "oracle_gap": hard["oracle_gap"],
                        "site_mask": hard["site_mask"],
                        "duals": duals.detach().cpu(),
                    },
                    ckpt_path,
                )
                print(
                    f"  -> Checkpoint saved (violated={selection_key[0]}, "
                    f"devices={selection_key[1]}, "
                    f"worst_margin={-selection_key[2]:+.4f}dB)"
                )

    print(
        f"\nTraining complete. Best: violated={best_key[0]}, "
        f"devices={best_key[1]}, worst_margin={-best_key[2]:+.4f}dB"
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
