"""Follow-up #3 (open_followups.md): are all of the trained checkpoint's
placed regenerators actually load-bearing?

Loads a trained e2e checkpoint, thresholds the learned probabilities into a
hard placement set R = {n : sigmoid(logit_n / tau_end) > 0.5}, then on the
SAME fixed traffic matrix `train.py` actually trained/constrained this
checkpoint against (`_common.fixed_traffic_demands` — `build_traffic_matrix`
+ `preflight_filter` over `cfg["traffic"]`/`cfg["constraint"]`, not an ad
hoc `generate_demands` draw unrelated to the checkpoint's training set):

  1. Confirms the baseline: with all of R active (hard p=1 on R, p=0
     elsewhere), num_infeasible over the fixed set.
  2. Leave-one-out: for each node n in R, evaluates R \\ {n} and records how
     many demands become infeasible that weren't under the full R baseline
     (node is "redundant" if 0, "load-bearing for k" if k > 0).
  3. Converse (under-provisioning check): for each regen-candidate node NOT
     in R, evaluates R + {n} and checks whether it rescues any demand that
     was infeasible under R.
  4. Greedy minimisation: repeatedly drops the redundant node with the
     lowest regen_prob (permanently, if the set stays feasible) until every
     remaining node is load-bearing, reporting the minimal set size vs |R|.

All evaluations reuse the pipeline's own routing (trained EdgeWeightNet) —
this is deliberately about the deployed system, not a physical-baseline
question. "Hard" placement is implemented via DiffONetPipeline.forward's
regen_probs_override — an explicit 0/1 tensor passed per call — rather than
saturating regen_logits to +/-30 in place and restoring them afterwards.

Usage:
    conda activate diffopt
    python scripts/diagnose_regen_ablation.py --config configs/experiment/small_test_ind132.yaml
"""
import argparse

import torch
import yaml

from _common import add_common_args, build_context, fixed_traffic_demands


def ablate(pipeline, demands, thresholds, placed, candidates, *,
           tau, lambda_, num_nodes, order_key):
    """Leave-one-out + converse + greedy minimisation over a hard placement.

    Extracted from the script body so the arm-comparison harness
    (scripts/run_placement_ablation.py) uses this exact implementation rather
    than a second copy. Two copies of a sweep like this drift — see
    scripts/_common.py's docstring for the same thing happening to
    pipeline-construction boilerplate across 11 scripts.

    Greedy leave-one-out UNDER-REPORTS redundancy: it cannot see that
    removing several nodes jointly stays feasible. On the toy replica it
    returned 6 where brute force found 3. `minimal_set` is therefore an
    UPPER BOUND on the true minimum, not a floor. Say so wherever it is
    reported.
    """
    def infeasible_set(active):
        override = torch.zeros(num_nodes)
        if active:
            override[list(active)] = 1.0
        with torch.no_grad():
            _, gsnr_preds, _, _ = pipeline(
                demands, tau=tau, lambda_=lambda_, regen_probs_override=override
            )
        return {d.id for d in demands if gsnr_preds[d.id].item() < thresholds[d.id]}

    baseline_infeasible = infeasible_set(placed)

    loo_broken = {n: infeasible_set(placed - {n}) - baseline_infeasible
                  for n in sorted(placed)}
    redundant = [n for n, broken in loo_broken.items() if not broken]

    rescues = {}
    for n in sorted(candidates - placed):
        rescued = baseline_infeasible - infeasible_set(placed | {n})
        if rescued:
            rescues[n] = rescued

    kept, dropped = set(placed), []
    for n in sorted(placed, key=order_key):
        if n not in kept:
            continue
        if infeasible_set(kept - {n}) <= baseline_infeasible:
            kept -= {n}
            dropped.append(n)

    return {
        "baseline_infeasible": baseline_infeasible,
        "loo_broken": loo_broken,
        "redundant": redundant,
        "rescues": rescues,
        "minimal_set": sorted(kept),
        "dropped": dropped,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # No --num-demands/--seed: the fixed traffic matrix is fully determined by
    # cfg["traffic"]["seed"], not a CLI-chosen draw.
    add_common_args(ap, with_demands=False)
    args = ap.parse_args()

    with open(args.config) as f:
        _cfg_for_ctx = yaml.safe_load(f)
    ctx = build_context(
        _cfg_for_ctx,
        load_e2e_checkpoint=True,
        checkpoint_path=args.checkpoint,
    )
    cfg = ctx.cfg
    topo = ctx.topology
    pipe = ctx.pipeline
    regen_placement = ctx.regen_placement
    t_cfg = cfg["training"]

    tau_end = t_cfg["regen_tau_end"]
    vlastelica_lambda = ctx.ckpt["vlastelica_lambda"]

    learned_logits = regen_placement.regen_logits.detach().clone()
    learned_probs = torch.sigmoid(learned_logits / tau_end)
    # The deployed placement, from the one gate-agnostic definition. Prefer the
    # checkpoint's own recorded mask when present (train.py writes it since
    # 2026-08-21); fall back to recomputing for older checkpoints.
    if ctx.ckpt is not None and "placement_mask" in ctx.ckpt:
        R = set(ctx.ckpt["placement_mask"].nonzero(as_tuple=True)[0].tolist())
    else:
        R = set(regen_placement.hard_placement_mask().nonzero(as_tuple=True)[0].tolist())
    cands = ctx.regen_candidates
    ckpt_label = args.checkpoint or f"{cfg['checkpoint_dir']}/best_e2e.pt"
    print(f"Checkpoint {ckpt_label} "
          f"(epoch {ctx.ckpt['epoch']}): |R|={len(R)} nodes "
          f"({sorted(R)}), all regen candidates: {R <= cands}")

    demands, excluded = fixed_traffic_demands(ctx)
    thresholds = {d.id: ctx.mod_cfg.required_snr_threshold(d.bitrate_gbps) for d in demands}
    tr_cfg = cfg["traffic"]
    print(
        f"{len(demands)} demands ({tr_cfg['scenario']}, seed={tr_cfg['seed']}, "
        f"scale={tr_cfg['scale']:.3g}) — train.py's own fixed matrix, "
        f"{len(excluded)} excluded by preflight, fixed across all evaluations\n"
    )

    result = ablate(
        pipe, demands, thresholds, R, cands,
        tau=tau_end, lambda_=vlastelica_lambda, num_nodes=topo.num_nodes,
        order_key=lambda n: learned_probs[n].item(),
    )
    baseline_infeasible = result["baseline_infeasible"]
    loo_broken = result["loo_broken"]
    redundant = result["redundant"]
    rescues = result["rescues"]
    kept = set(result["minimal_set"])
    dropped = result["dropped"]

    print(f"Baseline (R active, {len(R)} regens): "
          f"num_infeasible = {len(baseline_infeasible)}/{len(demands)}")
    if baseline_infeasible:
        print(f"  infeasible demand_ids: {sorted(baseline_infeasible)}")

    print("\n--- Leave-one-out ablation ---")
    for n in sorted(R):
        newly_broken = loo_broken[n]
        tag = "REDUNDANT" if not newly_broken else f"load-bearing for {len(newly_broken)}"
        print(f"  drop node {n:>4} (p={learned_probs[n]:.4f}): {tag}"
              + (f" ({sorted(newly_broken)})" if newly_broken else ""))

    print(f"\n{len(redundant)}/{len(R)} nodes redundant under single-node removal: {sorted(redundant)}")

    print("\n--- Converse: under-provisioning check (unplaced candidates) ---")
    unplaced = sorted(cands - R)
    for n in unplaced:
        if n in rescues:
            print(f"  add node {n:>4}: rescues {sorted(rescues[n])}")
    if not rescues:
        print(f"  none of {len(unplaced)} unplaced candidates rescue any demand "
              f"infeasible under R (baseline already has "
              f"{len(baseline_infeasible)} infeasible demands)")

    print("\n--- Greedy minimisation (drop lowest-prob redundant node first) ---")
    for i, n in enumerate(dropped, start=1):
        running_kept = len(R) - i
        print(f"  dropped node {n:>4} (p={learned_probs[n]:.4f}) — set stays feasible, |kept|={running_kept}")
    print(f"\nMinimal load-bearing set: {len(kept)}/{len(R)} nodes ({sorted(kept)})")
    print(f"Dropped as redundant: {sorted(dropped)}")
