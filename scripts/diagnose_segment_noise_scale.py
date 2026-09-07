"""Diagnostic: per-segment noise scale, and the live "regen helps" check.

SegmentCombiner does its max in LINEAR NOISE space, where a good segment is
a very small number (~0.0025 for the ~26 dB segments the pipeline actually
produces). Regen "helps" only if the combined noise of a cut path is below
that of an uncut one; for two equal segments of noise n that means the fold
must return something strictly below 2n.

The historical bug was that the fold approximated the max with
`t * logsumexp([a/t, b/t])`, which
overshoots by `t*ln2` — ABSOLUTE, so it does not shrink with the operands. At
the schedule's sharpest temperature (0.01) that floor is 0.0069, larger than
the segment noise itself, which inverted the sign of every gradient reaching
regen_logits. Scale-normalising the soft max made the overshoot relative
(`max(a,b)*t*ln2`) and bounded the damage; the fold is now an exact max over
chunks, with no overshoot at all and no temperature to schedule.

So this script has two jobs:

  1. Measure the per-segment noise distribution the combiner actually
     operates on. That is worth watching per topology regardless of which
     fold is shipped, and it sets the scale for everything below.
  2. Assert, live against the shipped code at that measured scale, that
     regenerating never hurts: value-wise (a fully regenerated path is
     never worse than a transparent one) and gradient-wise
     (`d(path_gsnr)/dp > 0`). Under an exact max both are provable — cutting
     a boundary splits a chunk into two no-larger pieces — but this is the
     cheapest standing check that the shipped code still has the property,
     which is exactly the invariant the whole investigation exists to hold.

Usage:
    conda activate diffopt
    python scripts/diagnose_segment_noise_scale.py --config configs/experiment/base.yaml
"""
from __future__ import annotations

import argparse
import math
import sys

import torch
import yaml

from diffopt.qot.segment_combiner import SegmentCombiner
from diffopt.pipeline import segment_path
from diffopt.routing.surrogate import surrogate_shortest_path
from _common import add_common_args, build_context, demands_for, edge_weights_of

ap = argparse.ArgumentParser()
add_common_args(ap, with_checkpoint=False, with_demands=False)
args = ap.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)
ctx = build_context(cfg, load_e2e_checkpoint=False)
pipe = ctx.pipeline

demands = demands_for(ctx, seed=1)

seg_gsnrs, nsegs = [], []
with torch.no_grad():
    ew = edge_weights_of(ctx, tau=1.0)
    vlastelica_lambda = cfg["training"]["vlastelica_lambda"]
    for d in demands:
        pi, ordered = surrogate_shortest_path(ew, pipe._edge_index, d.src, d.dst,
                                              pipe._num_nodes, lambda_=vlastelica_lambda)
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

# The historical schedule's endpoints, pinned as literals: they are gone from
# the config and from the code, and this table is about what the OLD fold
# would have done on THIS workload, so it must not track a current value.
print("\nWhat the OLD absolute soft_max floor (t*ln2) would have done here:")
print("t_sm     floor=t*ln2   frac of segments with noise < floor")
print("                       (= fraction where 'regen helps' was INVERTED)")
for t in (0.5, 0.1, 0.01):
    floor = t * math.log(2.0)
    frac = (noise < floor).float().mean().item()
    gsnr_at_floor = -10.0 * math.log10(floor)
    print(f"{t:.4f}   {floor:.5f}      {frac*100:5.1f}%   "
          f"(would cap regenerated path at {gsnr_at_floor:.1f} dB)")
print("The shipped fold takes an EXACT max, so its overshoot is 0 at every")
print("noise magnitude and there is no temperature left to get wrong.")

# Live regression check against the real segment scale: every row must say
# HELPS. Any INVERTED row means the combiner is telling the optimiser that
# regenerators degrade the path, and regen_logits will be driven to zero.
comb = SegmentCombiner()
failures = []

print("\nLIVE CHECK A: value + gradient on a 2-segment path, both segments at")
print("the measured GSNR quantile. Exact fold => path noise is")
print("(1-p)*(n1+n2) + p*max(n1,n2), so d(gsnr)/dp > 0 whenever max < sum.")
print("  quantile  seg_gsnr   path_gsnr@p=0.5   d(gsnr)/dp   |value err|   verdict")
for q in qs:
    seg_db = float(torch.quantile(g, q))
    p = torch.tensor(0.5, requires_grad=True)
    out = comb([torch.tensor(seg_db), torch.tensor(seg_db)], [p])
    out.backward()

    n_lin = 10.0 ** (-min(max(seg_db, -5.0), 35.0) / 10.0)
    expected = -10.0 * math.log10(0.5 * 2 * n_lin + 0.5 * n_lin)
    err = abs(out.item() - expected)

    ok = p.grad.item() > 0.0 and err < 1e-3
    verdict = "regen HELPS" if ok else "FAIL"
    if not ok:
        failures.append(f"2-segment path at p{int(q * 100)} ({seg_db:.2f} dB): "
                        f"d(gsnr)/dp={p.grad.item():+.4f}, value err={err:.2e}")
    print(f"  p{int(q*100):<8} {seg_db:7.2f}   {out.item():13.2f}   "
          f"{p.grad.item():+10.3f}   {err:11.2e}   -> {verdict}")

print("\nLIVE CHECK B: a fully regenerated path is never worse than a")
print("transparent one, over path lengths this topology actually produces.")
med = float(torch.quantile(g, 0.5))
print("  n_segs   transparent   regenerated   delta")
for k in range(2, int(ns.max().item()) + 1):
    gsnrs = [torch.tensor(med) for _ in range(k)]
    with torch.no_grad():
        transparent = comb(gsnrs, [torch.tensor(0.0)] * (k - 1)).item()
        regenerated = comb(gsnrs, [torch.tensor(1.0)] * (k - 1)).item()
    delta = regenerated - transparent
    if delta < 0.0:
        failures.append(f"{k}-segment path at the median GSNR: regenerating "
                        f"costs {delta:.4f} dB")
    print(f"  {k:>6}   {transparent:11.4f}   {regenerated:11.4f}   {delta:+7.4f}")

if failures:
    print("\nFAILED — the 'regen helps' invariant is broken:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("\nOK — 'regen helps' holds in value and in gradient at every point checked.")
