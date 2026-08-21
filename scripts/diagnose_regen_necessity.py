"""Does this topology actually need regenerators?

Routes every demand on the shortest-by-km path (physical baseline, no
dependence on EdgeWeightNet init — this deliberately does NOT use
edge_weights_of/EdgeWeightNet at all), then evaluates each path in three
regimes and compares against the demand's real SNR threshold:

  p=0  fully TRANSPARENT   (no regenerator anywhere)  <- the honest question
  p=0.5                    (the pipeline's init state)
  p=1  fully REGENERATED   (regen at every candidate)

Note that p=0.5 -- the pipeline's initialisation -- is *half a regenerator at
every candidate node*, i.e. a heavily regenerated network. Feasibility
measured at init is therefore not evidence that regenerators are unnecessary;
p=0 is the honest baseline.

Usage:
    conda activate diffopt
    python scripts/diagnose_regen_necessity.py --config configs/experiment/base.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from diffopt.pipeline import segment_path
from diffopt.qot.segment_combiner import SegmentCombiner
from diffopt.routing.shortest_path import dijkstra
from _common import add_common_args, build_context, demands_for

ap = argparse.ArgumentParser()
add_common_args(ap, with_checkpoint=False)
# Overrides for this script's own historical defaults: 400 demands (not
# cfg["num_demands"]=100) at a fixed seed=7 -- diagnose_regen_ablation.py's
# own --seed default explicitly documents "matches diagnose_regen_necessity.py",
# so this default must stay 7, not add_common_args' generic default of 1.
ap.set_defaults(num_demands=400, seed=7)
args = ap.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)
ctx = build_context(cfg, load_e2e_checkpoint=False)
pipe = ctx.pipeline
edges = ctx.edges
cands = ctx.regen_candidates
ei = pipe._edge_index.numpy()
lens = np.array([e.length_km for e in edges], dtype=float)
comb = SegmentCombiner()

demands = demands_for(ctx, seed=args.seed, num_demands=args.num_demands)
rows = []
with torch.no_grad():
    for d in demands:
        pi = dijkstra(lens, ei, d.src, d.dst, ctx.topology.num_nodes)
        if pi is None:
            continue
        pit = torch.tensor(pi, dtype=torch.float32)
        ordered = pipe._reconstruct_path(pit, d.src, d.dst)
        if not ordered:
            continue
        km = sum(edges[e].length_km for e in ordered)
        nsp = sum(edges[e].num_spans for e in ordered)
        segs, bnodes = segment_path(ordered, d.src, cands, edges, d.dst)
        gs = [pipe.qot_model(*pipe._extract_span_features(s, ctx.device))[0] for s in segs]
        thr = ctx.mod_cfg.required_snr_threshold(d.bitrate_gbps)
        out = {}
        for name, p in [("p0", 0.0), ("p05", 0.5), ("p1", 1.0)]:
            bp = [torch.tensor(p)] * (len(segs) - 1)
            out[name] = comb(gs, bp).item()
        rows.append((km, nsp, len(segs), d.bitrate_gbps, thr,
                     out["p0"], out["p05"], out["p1"]))

topo_name = Path(cfg["topology"]).stem
print(f"{len(rows)} demands (seed={args.seed}), shortest-by-km routing, {topo_name}\n")
buckets = [(0, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 10000)]
print(f"{'path km':>13} {'n':>4} {'segs':>5} {'spans':>6} | "
      f"{'GSNR p=0':>9} {'p=0.5':>7} {'p=1':>7} | {'infeasible @ p=0':>17} {'@ p=1':>7}")
for lo, hi in buckets:
    b = [r for r in rows if lo <= r[0] < hi]
    if not b:
        continue
    inf0 = sum(1 for r in b if r[5] < r[4])
    inf1 = sum(1 for r in b if r[7] < r[4])
    print(f"{lo:5d}-{hi:<7d} {len(b):>4} {np.mean([r[2] for r in b]):>5.1f} "
          f"{np.mean([r[1] for r in b]):>6.1f} | "
          f"{np.mean([r[5] for r in b]):>9.2f} {np.mean([r[6] for r in b]):>7.2f} "
          f"{np.mean([r[7] for r in b]):>7.2f} | "
          f"{inf0:>8}/{len(b):<8} {inf1:>3}/{len(b)}")

tot0 = sum(1 for r in rows if r[5] < r[4])
tot1 = sum(1 for r in rows if r[7] < r[4])
rescued = sum(1 for r in rows if r[5] < r[4] <= r[7])
print(f"\nTOTAL infeasible fully-transparent (p=0): {tot0}/{len(rows)}")
print(f"TOTAL infeasible fully-regenerated (p=1): {tot1}/{len(rows)}")
print(f"demands RESCUED by regeneration:          {rescued}/{len(rows)}")
print(f"\nlongest path: {max(r[0] for r in rows):.0f} km, "
      f"{max(r[1] for r in rows)} spans (max_spans={pipe.max_spans})")
print(f"paths exceeding max_spans={pipe.max_spans} as ONE segment: "
      f"{sum(1 for r in rows if r[1] > pipe.max_spans)}/{len(rows)}")

print("\nGSNR gain from regeneration (p=1 minus p=0), by path length:")
for lo, hi in buckets:
    b = [r for r in rows if lo <= r[0] < hi]
    if b:
        print(f"  {lo:5d}-{hi:<7d} mean gain = {np.mean([r[7]-r[5] for r in b]):+6.2f} dB")
