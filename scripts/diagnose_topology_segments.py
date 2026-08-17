"""Compare the per-segment noise scale across topologies.

Answers: was t=0.01 ever right, and if so for which topology?

Every other config key (QoT checkpoint, pipeline params, seed, ...) comes
from --config; only `topology` is overridden per iteration, since comparing
topologies is this script's entire purpose.
"""
import argparse
import math

import torch
import yaml

from diffopt.pipeline import segment_path
from diffopt.routing.surrogate import surrogate_shortest_path
from _common import add_common_args, build_context, demands_for, edge_weights_of

ap = argparse.ArgumentParser()
add_common_args(ap, with_checkpoint=False, with_demands=False)
args = ap.parse_args()

base_cfg = yaml.safe_load(open(args.config))

for topo_name in ["german_17", "ind_132"]:
    cfg = dict(base_cfg)
    cfg["topology"] = f"configs/topology/{topo_name}.json"
    ctx = build_context(cfg, load_e2e_checkpoint=False)
    pipe = ctx.pipeline
    topology = ctx.topology
    edges = ctx.edges
    cands = ctx.regen_candidates
    demands = demands_for(ctx, seed=1)
    vlastelica_lambda = cfg["training"]["vlastelica_lambda"]

    seg_g, nseg, seg_len, path_len = [], [], [], []
    with torch.no_grad():
        ew = edge_weights_of(ctx, tau=1.0)
        for d in demands:
            pi = surrogate_shortest_path(ew, pipe._edge_index, d.src, d.dst,
                                         pipe._num_nodes, lambda_=vlastelica_lambda)
            ordered = pipe._reconstruct_path(pi, d.src, d.dst)
            if not ordered:
                continue
            path_len.append(sum(edges[e].length_km for e in ordered))
            segs, _ = segment_path(ordered, d.src, cands, edges, d.dst)
            nseg.append(len(segs))
            for s in segs:
                seg_len.append(sum(edges[e].length_km for e in s))
                sf, pm = pipe._extract_span_features(s, ctx.device)
                seg_g.append(pipe.qot_model(sf, pm)[0].item())

    g = torch.tensor(seg_g)
    noise = torch.pow(10.0, -g.clamp(-5.0, 35.0) / 10.0)
    med_noise = float(noise.median())

    print(f"=== {topo_name} ===")
    print(f"  nodes={topology.num_nodes} edges={len(edges)} regen_candidates(deg>=3)={len(cands)}"
          f"  ({100*len(cands)/topology.num_nodes:.0f}% of nodes)")
    print(f"  path length km : mean={sum(path_len)/len(path_len):8.0f}")
    print(f"  segments/path  : mean={sum(nseg)/len(nseg):6.1f}  max={max(nseg)}")
    print(f"  segment len km : mean={sum(seg_len)/len(seg_len):8.0f}  "
          f"median={float(torch.tensor(seg_len).median()):.0f}")
    print(f"  segment GSNR dB: median={float(g.median()):6.2f}  p5={float(torch.quantile(g,0.05)):.2f}")
    print(f"  segment NOISE  : median={med_noise:.5f}")
    for t in [0.5, 0.01]:
        floor = t * math.log(2)
        frac = float((noise < floor).float().mean())
        print(f"    t={t:<5}: floor={floor:.5f} = {floor/med_noise:7.2f}x median noise "
              f"-> {frac*100:5.1f}% of segments INVERTED")
    print()
