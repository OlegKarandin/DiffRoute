"""End-to-end training script for DiffONet (Phase 1c).

Entry point: python -m diffopt.train --config configs/experiment/base.yaml
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch
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


def cosine_anneal(
    epoch: int,
    start: float,
    end: float,
    anneal_start: int,
    anneal_end: int,
) -> float:
    """Cosine-decay a scalar from `start` to `end` over [anneal_start, anneal_end].

    Same interface as `linear_anneal`. Drives `training.lr_alloc` when
    `lr_alloc_schedule: cosine` (open_followups.md #7a): the binding
    constraint per optimizer_normalization_and_the_score_runaway.md's
    Finding 8 is the feasible-EPOCH rate under checkpoint selection, not the
    trajectory mean, and decaying lr is the standard way to turn an
    oscillating SGD trajectory into a converging one — which is what would
    make that rate stop mattering. Cosine (not linear) spends more of the
    budget near both endpoints: mostly-`start` early (still exploring),
    mostly-`end` late (settling), with the fastest descent through the
    middle.
    """
    if epoch <= anneal_start:
        return start
    if epoch >= anneal_end:
        return end
    frac = (epoch - anneal_start) / (anneal_end - anneal_start)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * frac))


def step_decay(epoch: int, start: float, step_size: int, gamma: float) -> float:
    """Multiply `start` by `gamma` every `step_size` epochs (1-indexed, so
    the value is still exactly `start` for epochs [1, step_size]) —
    PyTorch's own StepLR rule, reimplemented here (rather than driving
    opt_alloc through a torch.optim.lr_scheduler) so the schedule train.py
    logs and the lr opt_alloc actually steps with can never drift apart.
    `step_size <= 0` disables decay (start is returned unchanged)."""
    if step_size <= 0:
        return start
    return start * (gamma ** ((epoch - 1) // step_size))


def alloc_score_stats(alloc) -> Tuple[float, float, float]:
    """(mean, min, max) of the head's raw scores on REAL boundaries.

    Read straight off `alloc.score`, which the head publishes from the walk.
    Padded columns (`boundary_node_ids < 0`) are excluded — a padded cut is
    no cut, not a score of -inf.

    This used to invert the forward value instead: on a valid boundary
    `a = sigmoid(score / tau)`, so `score = tau * logit(a)`, with `a` clamped
    to float32's representable range first. That is exact while the head is
    unsaturated, and reports nothing at all once it is not — which inverts
    the diagnostic's whole purpose. Under `alloc_ste` the forward value is
    EXACTLY 0 or 1, so every boundary pinned to a clamp on every epoch:
    the augmented-Lagrangian gate's three `al_ste_greedy` seeds each logged
    score_min == -87.336545 and score_max == +15.942385 for 60/60 epochs
    while their true scores ran to -47, and score_mean was an exact affine
    restatement of device_count. Worse, because the readout was `tau *`
    clamp, a pinned column TRACKED THE TAU ANNEAL: `al_baseline`'s logged
    "recovery" from -60.165 to -26.201 over epochs 30-55 is
    `-87.336545 * tau` at tau = 0.6889 and 0.3000, not the head moving.
    See docs/investigations/augmented_lagrangian_gate.md.

    Takes no `tau`: a raw score does not depend on one.

    Returns (nan, nan, nan) when the routing produced no boundary at all.
    """
    scores = alloc.score
    if scores.numel() == 0:
        return math.nan, math.nan, math.nan
    valid = alloc.boundary_node_ids >= 0
    if not bool(valid.any()):
        return math.nan, math.nan, math.nan
    scores = scores[valid].to(dtype=torch.float64)
    return (
        float(scores.mean().item()),
        float(scores.min().item()),
        float(scores.max().item()),
    )


# Below this, sigmoid'(s/tau)/tau contributes nothing a shared parameter can
# feel. At tau=1 it is |s| > ~14.5; the head's gradient is a SUM over
# boundaries, and the live ones sit at up to 0.25, so a 1e-6 boundary is
# eight orders of magnitude down on its own neighbours. Not a cliff — the
# underflow is gradual — but a readable line on one side of which a boundary
# has stopped participating.
ALLOC_DEAD_SLOPE = 1e-6


def alloc_dead_fraction(alloc, tau: float, threshold: float = ALLOC_DEAD_SLOPE) -> float:
    """Fraction of REAL boundaries whose backward surrogate slope has vanished.

    The slope is `sigmoid'(s/tau)/tau`, which is what the STE propagates and
    also what the plain relaxation's chain rule carries. A boundary below
    `threshold` cannot move the head's parameters however wrong its decision
    is, and no dual-side quantity — not `lambda`, not `rho`, not a dual
    floor — reaches it, because the gradient has already underflowed by the
    time the price arrives.

    This is the column `alloc_score_*` cannot be: saturation is not the same
    as failure, and the two are not even correlated in the direction one
    would guess. On the augmented-Lagrangian gate the seed with the CLEANEST
    tail (al_ste_greedy 42, ten violation-free epochs) was the most saturated
    run measured, at 71% of boundaries dead and true scores reaching -47,
    while `al_baseline` at its selected epoch had 0% dead. What separates
    them is where the s=0 crossing sits relative to the dead mass, which is
    what `greedy_residual` pins and what a run without it has to find on its
    own. See docs/investigations/augmented_lagrangian_gate.md.

    Returns nan when the routing produced no boundary at all, matching
    `alloc_score_stats`.
    """
    scores = alloc.score
    if scores.numel() == 0:
        return math.nan
    valid = alloc.boundary_node_ids >= 0
    if not bool(valid.any()):
        return math.nan
    s = scores[valid].to(dtype=torch.float64)
    a = torch.sigmoid(s / tau)
    slope = a * (1.0 - a) / tau
    return float((slope < threshold).to(dtype=torch.float64).mean().item())


def hard_rollout(
    pipeline,
    demands: List[Demand],
    soft_alloc,
    modulation_config: ModulationConfig,
    *,
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

    `soft_alloc` must be THIS pipeline's forward() output for these SAME
    `demands`, at the SAME edge_log_weight this call would otherwise
    re-derive — i.e. call this before opt_edge.step() moves the weights.
    Reuses its routes/segments/GSNRs via
    `DiffONetPipeline.hard_rollout_from_soft` instead of running a second
    full forward pass: routing, segmentation and QoT never depend on
    hard_alloc, so this is exact, not approximate (open_followups.md #7b /
    docs/investigations/pipeline_profile_and_restoration_scaling.md,
    Finding 2).

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
        gsnr_preds, alloc = pipeline.hard_rollout_from_soft(demands, soft_alloc)

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
        # The hard AllocationOutputs this rollout evaluated. Already
        # constructed above, so this is a reference, not a copy. The viz
        # frame writer needs the deployed per-(demand, boundary) cuts in
        # PATH ORDER; alloc_by_node (D, N) is accumulated and loses
        # position along the path.
        "alloc": alloc,
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
    # ind_132 under an earlier routing parameterization), making runs
    # incomparable.
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
    alloc_ste: bool = pl_cfg.get("alloc_ste", False)
    # Under the STE, tau has no forward job left — the forward decision is the
    # deployed one — so the anneal only sharpens the backward surrogate, and
    # sharpening it starves every boundary whose score is not already near 0.
    # The arm therefore pins alloc_tau_end to alloc_tau_start. Warn rather than
    # override: a config is the record of what a run actually did, and silently
    # rewriting one is how an arm stops meaning what its name says.
    # cfg["training"] rather than t_cfg: that alias is not bound until later
    # in this function, and binding it early here would leave two names for
    # one dict in the same scope.
    _tau_start = cfg["training"]["alloc_tau_start"]
    _tau_end = cfg["training"]["alloc_tau_end"]
    if alloc_ste and _tau_end != _tau_start:
        print(
            f"WARNING: placement.alloc_ste is on but alloc_tau_start "
            f"({_tau_start}) != alloc_tau_end ({_tau_end}). "
            f"Under the STE tau only scales the "
            f"backward surrogate sigmoid'(s/tau)/tau, so annealing it "
            f"concentrates gradient onto near-zero scores and starves the "
            f"rest of the head. Pin alloc_tau_end to alloc_tau_start unless "
            f"you are deliberately measuring that."
        )

    # Augmented Lagrangian (method of multipliers): the primal derivative
    # max(0, lambda + rho*g) stays nonzero for a band of width lambda/rho
    # INSIDE the feasible region — the only reason a satisfied demand can
    # defend the cut that satisfies it. rho has no default: a guessed value
    # sets the band width. Validation lives in compute_loss, so a diagnostic
    # script calling compute_loss directly gets the same named error main()
    # does. MEASURE rho with `python -m scripts.calibrate_rho`.
    rho = c_cfg["rho"]

    allocation_head = AllocationHead(
        lookahead=lookahead, route_context=route_context,
        alloc_ste=alloc_ste,
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
    # `lambda_dev * device_count` is the only loss term that touches every
    # boundary unconditionally, and d/ds of it is `lambda_dev *
    # sigmoid'(s/tau)/tau` — strictly positive, so it pushes every score down
    # every step and never changes sign. The only opposing force is
    # `weighted_feasibility`, which under the augmented penalty reaches into
    # the feasible region across a band of width `lambda/rho` — ZERO once the
    # dual decays to 0, which complementary slackness is designed to make
    # happen. Constant force, no restoring force: the head drifts until
    # saturation stops it. Measured on constrained_stress with
    # greedy_residual off: scores reach -329 (frozen closed, 0 devices) or
    # +229 (frozen open, 1415 devices vs an oracle of 22), with
    # alloc_dead_frac == 1.0000 for the last 50 of 150 epochs on all three
    # seeds. Raising alloc_tau only moves the wall — the score range scales
    # with tau, so |s|/tau lands at 71-110 whatever tau is.
    # See docs/investigations/augmented_lagrangian_gate.md.
    #
    # SGD, not Adam: Adam's step is a RATIO, m/(sqrt(v)+eps), so a
    # consistently-signed gradient gives m ~ sqrt(v) and therefore a step of
    # ~lr HOWEVER SMALL that gradient has become. Two measured consequences,
    # both in docs/investigations/score_runaway_and_dual_windup.md:
    #
    #   1. It removes the brake the objective already has. The device force
    #      per boundary is lambda_dev * sigmoid'(s/tau)/tau, which decays
    #      5.0e-3 (s=-3) -> 5.0e-6 (s=-10) -> 3.0e-11 (s=-22). Under SGD the
    #      step decays with it and the score asymptotes near the sigmoid's
    #      own death; under Adam the drift runs at constant velocity to -131
    #      (al_baseline) and -329 (alloc_ste at tau=3).
    #   2. It strips the dual of authority over the head. A dual enters only
    #      as a gradient SCALE, and Adam divides scale back out, so a dual
    #      winding 0 -> 21.2 moves the head no faster than a dual of 0. That
    #      is a rate-limited actuator driven by an integral controller, the
    #      textbook wind-up setup this file's Finding 7 records.
    #
    # No weight decay: Finding 9 found no viable sizing under SGD (acting
    # within a 300-step run needs wd ~ 6.7, but by wd = 3 the decay force is
    # already 103% of the loss gradient). No momentum: Adam's beta1 = 0.9 is
    # a momentum term, and it is what makes the plant second-order — a
    # second-order plant under the augmented dual's PI-shaped force is what
    # oscillates in the first place. Removed 2026-09 (open_followups.md
    # item #8); recoverable from git history if Adam is needed again.
    opt_alloc = optim.SGD(
        allocation_head.parameters(),
        lr=cfg["training"]["lr_alloc"],
        momentum=0.0,
    )

    # Max total grad-norm on the allocation head, 0.0 = off (the shipped
    # default, and what every recorded run used). This exists because with a
    # proportional (SGD) step, one transient gradient at the epoch 7-9
    # whipsaw would force lr down far enough to make the remaining 50 epochs
    # useless. Clipping caps the transient so a healthy lr survives the
    # whole run.
    alloc_grad_clip: float = float(cfg["training"].get("alloc_grad_clip", 0.0))
    if alloc_grad_clip < 0.0:
        raise ValueError(
            f"training.alloc_grad_clip must be >= 0 (0.0 disables it), "
            f"got {alloc_grad_clip}"
        )

    t_cfg = cfg["training"]
    p_cfg = cfg["pipeline"]

    vlastelica_lambda: float = t_cfg["vlastelica_lambda"]
    lambda_min: float = t_cfg["vlastelica_lambda_min"]
    lambda_decay: float = t_cfg["vlastelica_lambda_decay"]
    epochs: int = t_cfg["epochs_e2e"]

    # lr_alloc schedule (open_followups.md #7a). "none" (default) reproduces
    # today's flat lr exactly — opt_alloc's own constructor lr, never
    # touched again — so every existing config's trajectory is unchanged
    # unless it opts in.
    lr_alloc_start: float = t_cfg["lr_alloc"]
    lr_alloc_schedule: str = t_cfg.get("lr_alloc_schedule", "none")
    if lr_alloc_schedule not in ("none", "cosine", "step"):
        raise ValueError(
            f"training.lr_alloc_schedule must be one of 'none', 'cosine', "
            f"'step', got {lr_alloc_schedule!r}"
        )
    lr_alloc_end: float = float(t_cfg.get("lr_alloc_end", lr_alloc_start))
    lr_alloc_anneal_start_epoch: int = int(t_cfg.get("lr_alloc_anneal_start_epoch", 1))
    lr_alloc_anneal_end_epoch: int = int(t_cfg.get("lr_alloc_anneal_end_epoch", epochs))
    lr_alloc_step_size: int = int(t_cfg.get("lr_alloc_step_size", 0))
    lr_alloc_step_gamma: float = float(t_cfg.get("lr_alloc_step_gamma", 1.0))

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

    # Trajectory frame dump (Phase 1d demo). Off by default: a run that does
    # not opt in must produce a byte-identical e2e_train_log.csv to one built
    # before this existed.
    v_cfg = cfg.get("viz", {})
    frame_writer = None
    if v_cfg.get("dump_frames", False):
        from diffopt.viz import FrameWriter

        frame_writer = FrameWriter(
            log_dir / "frames.json",
            topology=topology,
            demands=demands,
            cfg={**cfg, "_config_path": str(args.config)},
            modulation_config=mod_cfg,
            every=int(v_cfg.get("every", 1)),
            keyframe_every=int(v_cfg.get("keyframe_every", 50)),
        )

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
    # The selected epoch is otherwise recoverable only from inside the saved
    # checkpoint dict; frames.json's `run.selected_epoch` needs it here.
    best_epoch: Optional[int] = None

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
            "tau", "vlastelica_lambda", "lr_alloc",
            "alloc_score_mean", "alloc_score_min", "alloc_score_max",
            "lookahead",
            "route_context", "waste_cost",
            "ste_clamped_segments", "proxy_qot_rank_corr",
            "alloc_dead_frac", "alloc_grad_norm",
        ])

        for epoch in range(1, epochs + 1):
            tau = linear_anneal(
                epoch,
                t_cfg["alloc_tau_start"],
                t_cfg["alloc_tau_end"],
                t_cfg["alloc_tau_anneal_start_epoch"],
                t_cfg["alloc_tau_anneal_end_epoch"],
            )

            if lr_alloc_schedule == "cosine":
                lr_alloc = cosine_anneal(
                    epoch, lr_alloc_start, lr_alloc_end,
                    lr_alloc_anneal_start_epoch, lr_alloc_anneal_end_epoch,
                )
            elif lr_alloc_schedule == "step":
                lr_alloc = step_decay(
                    epoch, lr_alloc_start, lr_alloc_step_size, lr_alloc_step_gamma,
                )
            else:
                lr_alloc = lr_alloc_start
            opt_alloc.param_groups[0]["lr"] = lr_alloc

            opt_edge.zero_grad()
            opt_alloc.zero_grad()

            path_noise_costs, gsnr_preds, _, alloc = pipeline(
                demands, tau=tau, lambda_=vlastelica_lambda,
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
                rho=rho,
            )

            # Pre-step, like the state snapshots below — same row, same
            # parameters, not next epoch's already-updated ones.
            score_mean, score_min, score_max = alloc_score_stats(alloc)
            dead_frac = alloc_dead_fraction(alloc, tau)
            device_loss = p_cfg["lambda_dev"] * metrics["device_count"]

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
            # Selection metrics, measured on the deployed allocation rather
            # than on the relaxation the gradient step is taken through.
            # Pre-step, like the two snapshots above: the checkpoint must
            # save the parameters its own key describes.
            hard = hard_rollout(
                pipeline, demands, alloc, mod_cfg,
                margin_db=c_cfg["margin_db"],
            )

            loss.backward()
            # Returned norm is measured BEFORE clipping, so the logged column
            # reports what the loss actually produced rather than what
            # survived the cap. max_norm=inf makes this a pure measurement
            # when clipping is off, which is the default.
            alloc_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    allocation_head.parameters(),
                    alloc_grad_clip if alloc_grad_clip > 0.0 else float("inf"),
                )
            )
            opt_edge.step()
            opt_alloc.step()

            # Dual ascent, using the same constraint values the loss consumed
            # this epoch. Runs AFTER the primal step so the duals price the
            # constraint violation the step was actually taken against. The
            # augmented dual is gradient ascent on the dual function, so it
            # takes the SIGNED g at the SAME rho the penalty uses: that
            # shared coefficient is what makes the step well-scaled against
            # the penalty's own curvature, and what makes the pair's fixed
            # point g* = 0, lambda* = lambda_dev / s.
            duals = update_duals(
                duals,
                metrics["constraint_g"],
                eta=rho,
                dual_max=c_cfg["dual_max"],
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
                f"{lr_alloc:.6e}",
                f"{score_mean:.6f}",
                f"{score_min:.6f}",
                f"{score_max:.6f}",
                lookahead,
                route_context,
                f"{metrics['waste_cost']:.6f}",
                alloc.ste_clamped_segments,
                f"{alloc.proxy_qot_rank_corr:.4f}",
                f"{dead_frac:.6f}",
                f"{alloc_grad_norm:.6e}",
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

            if frame_writer is not None:
                # `hard` only — routes and cuts from hard["alloc"], margins
                # from hard["gsnr_preds"], so every number in a frame is
                # measured on the DEPLOYED allocation. Never the soft pass.
                frame_writer.append(epoch, hard)

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
                best_epoch = epoch
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

    if frame_writer is not None:
        frame_writer.close(best_epoch, log_path)
        print(f"Frames: {frame_writer.path}")

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
