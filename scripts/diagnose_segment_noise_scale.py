"""Diagnostic: per-segment noise scale vs the soft_max approximation error.

SegmentCombiner does its max in LINEAR NOISE space, where a good segment is
a very small number (~0.0025 for the ~26 dB segments the pipeline actually
produces). Regen "helps" only if soft_max(a, b) < a + b; for a ~= b = n that
requires the soft_max overshoot to stay below n.

The historical bug (see docs/investigations/regen_placement_not_concentrating.md)
was that the naive form `t * logsumexp([a/t, b/t])` overshoots by `t*ln2` —
ABSOLUTE, so it does not shrink with the operands. At the schedule's sharpest
temperature (0.01) that floor is 0.0069, larger than the segment noise itself,
which inverted the sign of every gradient reaching regen_logits.

soft_max is now scale-normalised, so the overshoot is `max(a,b)*t*ln2` and the
invariant holds at any magnitude for t < 1/ln2 ~= 1.44. This script keeps
measuring the segment noise distribution (it is the quantity the whole
combiner operates on, and worth watching per topology) and reports both the
current relative headroom and what the old absolute floor would have been.

Usage:
    conda activate diffopt
    python scripts/diagnose_segment_noise_scale.py --config configs/experiment/base.yaml
"""
from __future__ import annotations

import argparse
import math

import torch
import yaml

from diffopt.qot.segment_combiner import SegmentCombiner
from diffopt.pipeline import segment_path
from diffopt.routing.surrogate import surrogate_shortest_path
from _common import add_common_args, build_context, demands_for, edge_weights_of, schedule_at

ap = argparse.ArgumentParser()
add_common_args(ap, with_checkpoint=False, with_demands=False)
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
ctx = build_context(cfg, load_e2e_checkpoint=False)
pipe = ctx.pipeline

demands = demands_for(ctx, seed=1)

seg_gsnrs, nsegs = [], []
with torch.no_grad():
    ew = edge_weights_of(ctx, tau=1.0)
    vlastelica_lambda = cfg["training"]["vlastelica_lambda"]
    for d in demands:
        pi = surrogate_shortest_path(ew, pipe._edge_index, d.src, d.dst,
                                     pipe._num_nodes, lambda_=vlastelica_lambda)
        ordered = pipe._reconstruct_path(pi, d.src, d.dst)
        segs, _ = segment_path(ordered, d.src, pipe._regen_candidate_set,
                               pipe._edges, d.dst)
        nsegs.append(len(segs))
        for s in segs:
            sf, pm = pipe._extract_span_features(s, ctx.device)
            seg_gsnrs.append(pipe.qot_model(sf, pm)[0].item())

g = torch.tensor(seg_gsnrs)
noise = torch.pow(10.0, -g.clamp(-5.0, 35.0) / 10.0)
ns = torch.tensor(nsegs, dtype=torch.float)

print(f"segments observed: {len(g)}  over {len(demands)} demands")
print(f"segments/path: mean={ns.mean():.1f} median={ns.median():.0f} max={ns.max():.0f}")
qs = [0.05, 0.25, 0.5, 0.75, 0.95]
print("per-segment GSNR  dB quantiles: " +
      "  ".join(f"p{int(q*100)}={torch.quantile(g, q):.2f}" for q in qs))
print("per-segment noise    quantiles: " +
      "  ".join(f"p{int(q*100)}={torch.quantile(noise, q):.5f}" for q in qs))

print("\nWhat the OLD absolute floor (t*ln2) would have done on this workload:")
print("epoch  t_sm     floor=t*ln2   frac of segments with noise < floor")
print("                              (= fraction where 'regen helps' was INVERTED)")
t_cfg, sc = cfg["training"], cfg["segment_combiner"]
n_epochs = t_cfg["epochs_e2e"]
step = max(1, n_epochs // 20)
for ep in range(1, n_epochs + 1, step):
    _, t, _ = schedule_at(cfg, epoch=ep)
    floor = t * math.log(2.0)
    frac = (noise < floor).float().mean().item()
    gsnr_at_floor = -10.0 * math.log10(floor)
    print(f"{ep:5d}  {t:.4f}   {floor:.5f}      {frac*100:5.1f}%   "
          f"(would cap regenerated path at {gsnr_at_floor:.1f} dB)")

print("\nWith the CURRENT scale-normalised soft_max the overshoot is")
print("max(a,b)*t*ln2, i.e. a fixed FRACTION of the operands, so the")
print("'regen helps' invariant holds for any t < 1/ln2 = 1.443 at every")
print("noise magnitude above. Schedule max t = "
      f"{sc['soft_max_temperature']} -> headroom factor "
      f"{1.0 / math.log(2.0) / sc['soft_max_temperature']:.2f}x.")

# Live regression check against the real segment scale: every row must say
# HELPS. Any INVERTED row means the combiner is telling the optimiser that
# regenerators degrade the path, and regen_logits will be driven to zero.
print("\nLIVE CHECK: 2-segment path, both segments at the median measured GSNR")
med = float(torch.quantile(g, 0.5))
comb = SegmentCombiner()
for t in [0.5, 0.3115, 0.1231, 0.0477, 0.01, 0.001]:
    p = torch.tensor(0.5, requires_grad=True)
    out = comb([torch.tensor(med), torch.tensor(med)], [p], temperature=t)
    out.backward()
    verdict = "regen HELPS" if p.grad.item() > 0 else "regen HURTS (INVERTED)"
    print(f"  t={t:<7.4f} path_gsnr={out.item():7.2f} dB   d(gsnr)/dp={p.grad.item():+8.3f}"
          f"   -> {verdict}")
