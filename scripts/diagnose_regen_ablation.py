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


def infeasible_set(active):
    """Forward on an explicit hard placement and return the set of demand_ids
    below threshold.

    Previously this saturated regen_logits to +/-30 in place and restored
    them at the end of the script. pipeline.forward's regen_probs_override
    does the same thing without mutating a parameter, so an exception
    mid-sweep can no longer leave the module holding a saturated placement
    that was never in any checkpoint.
    """
    override = torch.zeros(topo.num_nodes)
    override[list(active)] = 1.0
    with torch.no_grad():
        _, gsnr_preds, _, _ = pipe(
            demands, tau=tau_end, lambda_=vlastelica_lambda,
            regen_probs_override=override,
        )
    return {d.id for d in demands if gsnr_preds[d.id].item() < thresholds[d.id]}


baseline_infeasible = infeasible_set(R)
print(f"Baseline (R active, {len(R)} regens): "
      f"num_infeasible = {len(baseline_infeasible)}/{len(demands)}")
if baseline_infeasible:
    print(f"  infeasible demand_ids: {sorted(baseline_infeasible)}")

print("\n--- Leave-one-out ablation ---")
loo_broken = {}  # node -> set of newly-infeasible demand ids
for n in sorted(R):
    infeas = infeasible_set(R - {n})
    newly_broken = infeas - baseline_infeasible
    loo_broken[n] = newly_broken
    tag = "REDUNDANT" if not newly_broken else f"load-bearing for {len(newly_broken)}"
    print(f"  drop node {n:>4} (p={learned_probs[n]:.4f}): {tag}"
          + (f" ({sorted(newly_broken)})" if newly_broken else ""))

redundant = [n for n, broken in loo_broken.items() if not broken]
print(f"\n{len(redundant)}/{len(R)} nodes redundant under single-node removal: {sorted(redundant)}")

print("\n--- Converse: under-provisioning check (unplaced candidates) ---")
unplaced = sorted(cands - R)
rescues = {}
for n in unplaced:
    infeas = infeasible_set(R | {n})
    rescued = baseline_infeasible - infeas
    if rescued:
        rescues[n] = rescued
        print(f"  add node {n:>4}: rescues {sorted(rescued)}")
if not rescues:
    print(f"  none of {len(unplaced)} unplaced candidates rescue any demand "
          f"infeasible under R (baseline already has "
          f"{len(baseline_infeasible)} infeasible demands)")

print("\n--- Greedy minimisation (drop lowest-prob redundant node first) ---")
kept = set(R)
order = sorted(R, key=lambda n: learned_probs[n].item())
dropped = []
for n in order:
    if n not in kept:
        continue
    trial = kept - {n}
    infeas = infeasible_set(trial)
    if infeas <= baseline_infeasible:
        kept = trial
        dropped.append(n)
        print(f"  dropped node {n:>4} (p={learned_probs[n]:.4f}) — set stays feasible, |kept|={len(kept)}")
print(f"\nMinimal load-bearing set: {len(kept)}/{len(R)} nodes ({sorted(kept)})")
print(f"Dropped as redundant: {sorted(dropped)}")
