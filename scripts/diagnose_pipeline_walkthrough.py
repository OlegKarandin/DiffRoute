"""Walk one real demand through the pipeline, printing each stage."""
import argparse

import torch
import yaml

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
topology = ctx.topology
edges = ctx.edges
cands = ctx.regen_candidates
vlastelica_lambda = cfg["training"]["vlastelica_lambda"]
demands = demands_for(ctx, seed=1)

with torch.no_grad():
    # NORMALISED — this is the unit-mean-renormalised weight pipeline.forward
    # actually routes on (correction #9), not edge_log_weight's raw Softplus
    # output. Before this script was migrated onto scripts/_common.py it
    # printed the raw output labelled "edge_weights", which described a
    # tensor the pipeline never used — if you're diffing against an older
    # run's output, that's why the numbers below moved.
    ew = edge_weights_of(ctx, tau=1.0, normalised=True)

    # pick a demand with a few segments
    pick = None
    for d in demands:
        pi, o = surrogate_shortest_path(ew, pipe._edge_index, d.src, d.dst,
                                        pipe._num_nodes, lambda_=vlastelica_lambda)
        s, b = segment_path(o, d.src, cands, edges, d.dst)
        if 3 <= len(s) <= 4 and len(o) >= 5:
            pick = (d, pi, o, s, b)
            break
    d, pi, ordered, segs, bnodes = pick

    print(f"DEMAND {d.id}: node {d.src} -> node {d.dst}, {d.bitrate_gbps} Gbps")
    print(f"  edge_index shape {tuple(pipe._edge_index.shape)}, "
          f"edge_weights (normalised, unit-mean) shape {tuple(ew.shape)}")
    print(f"\nSTAGE 4a  path_indicator: (E,) binary, {int(pi.sum())} of {len(edges)} edges = 1")
    print(f"STAGE 4a  ordered edge IDs (from Dijkstra's own prev-chain): {ordered}")

    node = d.src
    print("\n  node walk (regen candidates = degree>=3, marked *):")
    print(f"    start  node {node}{'*' if node in cands else ''}")
    for eid in ordered:
        e = edges[eid]
        nxt = e.dst if node == e.src else e.src
        cut = nxt in cands and nxt != d.dst
        print(f"    edge {eid:3d}  {e.length_km:6.1f} km / {e.num_spans} spans "
              f"-> node {nxt}{'*' if nxt in cands else ''}"
              f"{'   <<< CUT' if cut else ''}")
        node = nxt

    print(f"\nSTAGE 4d  segment_path -> {len(segs)} segments, "
          f"boundary nodes {bnodes}")
    tot = 0
    for i, s in enumerate(segs):
        km = sum(edges[e].length_km for e in s)
        ns = sum(edges[e].num_spans for e in s)
        tot += ns
        sf, pm = pipe._extract_span_features(s, ctx.device)
        g = pipe.qot_model(sf, pm)[0].item()
        n = 10 ** (-g / 10)
        print(f"    seg {i}: edges {s}  {km:7.1f} km  {ns:2d} spans "
              f"-> STAGE 4e QoT = {g:6.2f} dB  (noise {n:.5f})")
    print(f"    (path total {tot} spans; max_spans={pipe.max_spans})")

    # STAGE 4f/5: the allocation is per (demand, boundary) now, so there is
    # no per-node probability vector to index by boundary node. Run the real
    # forward for THIS demand and read its row out of AllocationOutputs —
    # `a[0, k]` is the allocation on this route's k-th boundary, in the same
    # order `segment_path` returned `bnodes`.
    _, gsnr_preds, _, alloc = pipe([d], tau=1.0, lambda_=vlastelica_lambda)
    n_bnd = len(bnodes)
    bp = [alloc.a[0, k] for k in range(n_bnd)]
    print("\nSTAGE 4e/6  AllocationHead.rollout -> a[demand, boundary]")
    print(f"    boundary nodes {bnodes}")
    print(f"    a             {[f'{p.item():.3f}' for p in bp]}")
    print(f"    device_count (sum over demands and boundaries) = "
          f"{alloc.device_count.item():.3f}")
    # site_view = max_d a[d, n]. DIAGNOSTIC ONLY — never priced, never in
    # the selection key. Thresholded at 0.5 to read it as "a site would be
    # built here"; at the closed init every boundary sits at sigmoid(-3)
    # ~ 0.047, so an unthresholded count would report every boundary node.
    sv = alloc.site_view
    touched = (sv > 0.5).nonzero(as_tuple=True)[0].tolist()
    print(f"    site_view (diagnostic only, never priced): "
          f"{len(touched)} node(s) above 0.5 {touched}   "
          f"max={sv.max().item():.3f}")
    print(f"    path GSNR from the real forward = {gsnr_preds[d.id].item():6.2f} dB")

    gs = [pipe.qot_model(*pipe._extract_span_features(s, ctx.device))[0] for s in segs]
    print(f"\nSTAGE 4f  SegmentCombiner(segment_gsnrs, boundary_probs={[f'{p:.2f}' for p in bp]})")
    out = pipe.segment_combiner(gs, bp)
    print(f"    -> path GSNR = {out.item():6.2f} dB")
    print("\n  span features fed to QoT are 5 cols: "
          "[span_len, fiber_idx, amp_nf, load_frac, accum_dist]")
    sf, pm = pipe._extract_span_features(segs[-1], ctx.device)
    ns = int(pm.sum())
    print(f"  e.g. last segment's {ns} real spans (of {pipe.max_spans} padded rows):")
    for r in range(ns):
        print("     ", "  ".join(f"{v:8.2f}" for v in sf[0, r].tolist()))
