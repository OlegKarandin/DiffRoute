"""Does unconditional segmentation vanish at p=0?

Option C's correctness rests on:  cut + blend at p=0  ==  never cut.
Test it directly: for real paths short enough to fit in one QoT call
(<= max_spans), compare
    (i)  QoT(whole path as ONE segment)                  <- ground truth
    (ii) SegmentCombiner(sub-segments, all p=0)          <- what the pipeline does
Also isolates the accum_dist_km reset as the suspected cause by
recomputing (ii) with accum_dist carried across segment boundaries.
"""
import math
from pathlib import Path
import torch, yaml

from diffopt.demands import generate_demands
from diffopt.pipeline import DiffONetPipeline, segment_path
from diffopt.placement.regenerator import RegenPlacement
from diffopt.qot.segment_combiner import SegmentCombiner
from diffopt.routing.edge_weight_net import EdgeWeightNet
from diffopt.routing.surrogate import surrogate_shortest_path
from diffopt.topology import FIBER_TYPE_INDEX, load_topology
from diffopt.train import load_qot_model

cfg = yaml.safe_load(Path("configs/experiment/small_test_ind132.yaml").read_text())
device = torch.device("cpu")
torch.manual_seed(0)

topology = load_topology(cfg["topology"], cfg["modulation_formats"])
qot = load_qot_model(cfg["qot_checkpoint"], cfg, device)
regen = RegenPlacement(topology.num_nodes)
pipe = DiffONetPipeline(
    topology=topology, qot_model=qot, segment_combiner=SegmentCombiner(),
    edge_weight_net=EdgeWeightNet(), regen_placement=regen,
    channel_loading_fraction=cfg["pipeline"]["channel_loading_fraction"],
    max_spans=cfg.get("max_spans_per_segment", 60))
edges = list(topology.undirected_edges)
cands = set(topology.regen_candidate_nodes)
MAXS = cfg.get("max_spans_per_segment", 60)


def feats(edge_ids, accum_start=0.0):
    """Span features for an edge list, optionally continuing accumulated distance."""
    rows, accum = [], accum_start
    for eid in edge_ids:
        e = edges[eid]
        fi = float(FIBER_TYPE_INDEX.get(e.fiber_type, 0))
        for k in range(e.num_spans):
            rows.append([e.span_lengths_km[k], fi, e.amplifier_nf_db[k],
                         pipe.channel_loading_fraction, accum])
            accum += e.span_lengths_km[k]
    n = len(rows)
    sf = torch.zeros(1, MAXS, 5)
    if n:
        sf[0, :n] = torch.tensor(rows, dtype=torch.float32)
    pm = torch.zeros(1, MAXS, dtype=torch.bool)
    pm[0, :n] = True
    return sf, pm, n, accum


def nspans(edge_ids):
    return sum(edges[e].num_spans for e in edge_ids)


demands = generate_demands(topology, 100, cfg["bitrate_options"], seed=1)
comb = SegmentCombiner()
zero = torch.tensor(0.0)

rows_reset, rows_carry, allspans, segspans = [], [], [], []
with torch.no_grad():
    rp = regen.get_regen_probs(1.0)
    ef = torch.cat([pipe._topo_edge_features,
                    rp[pipe._edge_src_ids].unsqueeze(1),
                    rp[pipe._edge_dst_ids].unsqueeze(1)], dim=1)
    ew = pipe.edge_weight_net(ef).squeeze(-1)
    for d in demands:
        pi = surrogate_shortest_path(ew, pipe._edge_index, d.src, d.dst,
                                     pipe._num_nodes, lambda_=10.0)
        ordered = pipe._reconstruct_path(pi, d.src, d.dst)
        if not ordered:
            continue
        allspans.append(nspans(ordered))
        segs, _ = segment_path(ordered, d.src, cands, edges, d.dst)
        segspans.extend(nspans(s) for s in segs)
        if len(segs) < 2 or nspans(ordered) > MAXS:
            continue                      # need whole path in ONE QoT call

        sf, pm, _, _ = feats(ordered)
        truth = qot(sf, pm)[0]            # ground truth: uncut

        # (ii) pipeline behaviour: accum_dist resets each segment
        gs = [qot(*feats(s)[:2])[0] for s in segs]
        got_reset = comb(gs, [zero] * (len(segs) - 1), temperature=0.01)

        # (iii) same, but accum_dist carried across boundaries
        gs2, acc = [], 0.0
        for s in segs:
            sf2, pm2, _, acc = feats(s, acc)
            gs2.append(qot(sf2, pm2)[0])
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
