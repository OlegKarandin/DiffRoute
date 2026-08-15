"""Compare the per-segment noise scale across topologies.

Answers: was t=0.01 ever right, and if so for which topology?
"""
import math, sys
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

base = yaml.safe_load(Path("configs/experiment/small_test_ind132.yaml").read_text())
device = torch.device("cpu")

for topo_name in ["german_17", "ind_132"]:
    torch.manual_seed(0)
    topology = load_topology(f"configs/topology/{topo_name}.json", base["modulation_formats"])
    qot = load_qot_model(base["qot_checkpoint"], base, device)
    regen = RegenPlacement(topology.num_nodes)
    pipe = DiffONetPipeline(
        topology=topology, qot_model=qot, segment_combiner=SegmentCombiner(),
        edge_weight_net=EdgeWeightNet(), regen_placement=regen,
        channel_loading_fraction=base["pipeline"]["channel_loading_fraction"],
        max_spans=base.get("max_spans_per_segment", 60))

    edges = list(topology.undirected_edges)
    cands = set(topology.regen_candidate_nodes)
    demands = generate_demands(topology, 100, base["bitrate_options"], seed=1)

    seg_g, nseg, seg_len, path_len = [], [], [], []
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
            path_len.append(sum(edges[e].length_km for e in ordered))
            segs, _ = segment_path(ordered, d.src, cands, edges, d.dst)
            nseg.append(len(segs))
            for s in segs:
                seg_len.append(sum(edges[e].length_km for e in s))
                sf, pm = pipe._extract_span_features(s, device)
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
