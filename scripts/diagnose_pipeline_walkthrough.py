"""Walk one real demand through the pipeline, printing each stage."""
from pathlib import Path
import torch, yaml

from diffopt.demands import generate_demands
from diffopt.pipeline import DiffONetPipeline, segment_path
from diffopt.placement.regenerator import RegenPlacement
from diffopt.qot.segment_combiner import SegmentCombiner
from diffopt.routing.edge_weight_net import EdgeWeightNet
from diffopt.routing.surrogate import surrogate_shortest_path
from diffopt.topology import load_topology
from diffopt.train import load_qot_model

cfg = yaml.safe_load(Path("configs/experiment/small_test_ind132.yaml").read_text())
torch.manual_seed(0)
topology = load_topology(cfg["topology"], cfg["modulation_formats"])
qot = load_qot_model(cfg["qot_checkpoint"], cfg, torch.device("cpu"))
regen = RegenPlacement(topology.num_nodes)
pipe = DiffONetPipeline(
    topology=topology, qot_model=qot, segment_combiner=SegmentCombiner(),
    edge_weight_net=EdgeWeightNet(), regen_placement=regen,
    channel_loading_fraction=cfg["pipeline"]["channel_loading_fraction"],
    max_spans=60)
edges = list(topology.undirected_edges)
cands = set(topology.regen_candidate_nodes)
demands = generate_demands(topology, 100, cfg["bitrate_options"], seed=1)

with torch.no_grad():
    rp = regen.get_regen_probs(1.0)
    ef = torch.cat([pipe._topo_edge_features,
                    rp[pipe._edge_src_ids].unsqueeze(1),
                    rp[pipe._edge_dst_ids].unsqueeze(1)], dim=1)
    ew = pipe.edge_weight_net(ef).squeeze(-1)

    # pick a demand with a few segments
    pick = None
    for d in demands:
        pi = surrogate_shortest_path(ew, pipe._edge_index, d.src, d.dst,
                                     pipe._num_nodes, lambda_=10.0)
        o = pipe._reconstruct_path(pi, d.src, d.dst)
        s, b = segment_path(o, d.src, cands, edges, d.dst)
        if 3 <= len(s) <= 4 and len(o) >= 5:
            pick = (d, pi, o, s, b); break
    d, pi, ordered, segs, bnodes = pick

    print(f"DEMAND {d.id}: node {d.src} -> node {d.dst}, {d.bitrate_gbps} Gbps")
    print(f"  edge_index shape {tuple(pipe._edge_index.shape)}, "
          f"edge_weights shape {tuple(ew.shape)}")
    print(f"\nSTAGE 4a  path_indicator: (E,) binary, {int(pi.sum())} of {len(edges)} edges = 1")
    print(f"STAGE 4c  _reconstruct_path -> ordered edge IDs: {ordered}")

    node = d.src
    print(f"\n  node walk (regen candidates = degree>=3, marked *):")
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
        sf, pm = pipe._extract_span_features(s, torch.device("cpu"))
        g = qot(sf, pm)[0].item()
        n = 10 ** (-g / 10)
        print(f"    seg {i}: edges {s}  {km:7.1f} km  {ns:2d} spans "
              f"-> STAGE 4e QoT = {g:6.2f} dB  (noise {n:.5f})")
    print(f"    (path total {tot} spans; max_spans={pipe.max_spans})")

    gs = [qot(*pipe._extract_span_features(s, torch.device('cpu')))[0] for s in segs]
    bp = [rp[n] for n in bnodes]
    print(f"\nSTAGE 4f  SegmentCombiner(segment_gsnrs, boundary_probs={[f'{p:.2f}' for p in bp]})")
    for t in [0.5, 0.01]:
        out = pipe.segment_combiner(gs, bp, temperature=t)
        print(f"    temperature={t:<5} -> path GSNR = {out.item():6.2f} dB")
    print(f"\n  span features fed to QoT are 5 cols: "
          f"[span_len, fiber_idx, amp_nf, load_frac, accum_dist]")
    sf, pm = pipe._extract_span_features(segs[-1], torch.device("cpu"))
    ns = int(pm.sum())
    print(f"  e.g. last segment's {ns} real spans (of {pipe.max_spans} padded rows):")
    for r in range(ns):
        print("     ", "  ".join(f"{v:8.2f}" for v in sf[0, r].tolist()))
