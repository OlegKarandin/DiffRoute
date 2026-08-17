"""Does unconditional segmentation vanish at p=0?

Option C's correctness rests on:  cut + blend at p=0  ==  never cut.
Test it directly: for real paths short enough to fit in one QoT call
(<= max_spans), compare
    (i)  QoT(whole path as ONE segment)                  <- ground truth
    (ii) SegmentCombiner(sub-segments, all p=0)          <- what the pipeline does
Also isolates the accum_dist_km reset as the suspected cause by
recomputing (ii) with accum_dist carried across segment boundaries.
"""
import argparse

import torch
import yaml

from diffopt.pipeline import segment_path
from diffopt.qot.segment_combiner import SegmentCombiner
from diffopt.qot.span_features import SPAN_FEATURE_DIM, span_feature_rows
from diffopt.routing.surrogate import surrogate_shortest_path
from _common import add_common_args, build_context, demands_for, edge_weights_of

ap = argparse.ArgumentParser()
add_common_args(ap, with_checkpoint=False, with_demands=False)
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
ctx = build_context(cfg, load_e2e_checkpoint=False)
pipe = ctx.pipeline
topology = ctx.topology
edges = ctx.edges
cands = ctx.regen_candidates
MAXS = pipe.max_spans


def feats(edge_ids, accum_start=0.0):
    """Padded span features + mask for an edge list, optionally continuing
    accumulated distance -- via the shared span_feature_rows (this used to
    be a private copy of the feature-ordering logic; accum_start is exactly
    the parameter that let it be dropped in favour of the shared function)."""
    rows = span_feature_rows(
        topology, edge_ids,
        channel_loading_fraction=pipe.channel_loading_fraction,
        accum_start=accum_start,
    )
    n = len(rows)
    accum = accum_start + sum(row[0] for row in rows)  # running total after all spans
    sf = torch.zeros(1, MAXS, SPAN_FEATURE_DIM)
    if n:
        sf[0, :n] = torch.tensor(rows, dtype=torch.float32)
    pm = torch.zeros(1, MAXS, dtype=torch.bool)
    pm[0, :n] = True
    return sf, pm, accum


def nspans(edge_ids):
    return sum(edges[e].num_spans for e in edge_ids)


demands = demands_for(ctx, seed=1)
comb = SegmentCombiner()
zero = torch.tensor(0.0)
vlastelica_lambda = cfg["training"]["vlastelica_lambda"]

rows_reset, rows_carry, allspans, segspans = [], [], [], []
with torch.no_grad():
    ew = edge_weights_of(ctx, tau=1.0)
    for d in demands:
        pi = surrogate_shortest_path(ew, pipe._edge_index, d.src, d.dst,
                                     pipe._num_nodes, lambda_=vlastelica_lambda)
        ordered = pipe._reconstruct_path(pi, d.src, d.dst)
        if not ordered:
            continue
        allspans.append(nspans(ordered))
        segs, _ = segment_path(ordered, d.src, cands, edges, d.dst)
        segspans.extend(nspans(s) for s in segs)
        if len(segs) < 2 or nspans(ordered) > MAXS:
            continue                      # need whole path in ONE QoT call

        sf, pm, _ = feats(ordered)
        truth = pipe.qot_model(sf, pm)[0]            # ground truth: uncut

        # (ii) pipeline behaviour: accum_dist resets each segment
        gs = [pipe.qot_model(*feats(s)[:2])[0] for s in segs]
        got_reset = comb(gs, [zero] * (len(segs) - 1), temperature=0.01)

        # (iii) same, but accum_dist carried across boundaries
        gs2, acc = [], 0.0
        for s in segs:
            sf2, pm2, acc = feats(s, acc)
            gs2.append(pipe.qot_model(sf2, pm2)[0])
        got_carry = comb(gs2, [zero] * (len(segs) - 1), temperature=0.01)

        rows_reset.append((truth.item(), got_reset.item()))
        rows_carry.append((truth.item(), got_carry.item()))

print(f"spans per WHOLE path : max={max(allspans)}  "
      f"> max_spans({MAXS}): {sum(1 for x in allspans if x > MAXS)}/{len(allspans)} paths")
print(f"spans per SEGMENT    : max={max(segspans)}  "
      f"> max_spans({MAXS}): {sum(1 for x in segspans if x > MAXS)}/{len(segspans)} segments")
print(f"\ncomparable paths (multi-segment AND <= {MAXS} spans whole): {len(rows_reset)}\n")

for name, rows in [("accum_dist RESET (current pipeline)", rows_reset),
                   ("accum_dist CARRIED across boundaries", rows_carry)]:
    err = torch.tensor([g - t for t, g in rows])
    print(f"{name}")
    print(f"   error (cut@p=0 − uncut), dB : mean={err.mean():+.4f}  "
          f"median={err.median():+.4f}  min={err.min():+.4f}  max={err.max():+.4f}")
    print(f"   mean |error| = {err.abs().mean():.4f} dB\n")

print("sample rows (uncut dB -> cut@p=0 dB, reset / carried):")
for i in range(min(8, len(rows_reset))):
    t, g1 = rows_reset[i]
    _, g2 = rows_carry[i]
    print(f"   {t:7.3f} -> {g1:7.3f} ({g1-t:+.3f})   |   {g2:7.3f} ({g2-t:+.3f})")
