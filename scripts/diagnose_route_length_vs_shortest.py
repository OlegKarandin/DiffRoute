"""Regression check: does EdgeWeightNet route close to the shortest-by-km
path?

Pre-correction-#9, the edge-weight scale collapse made trained routes run
~2.1x longer than shortest-by-km (docs/investigations/edge_weight_scale_collapse.md).
The fix (unit-mean weight renormalisation + ASE-denominated path-noise loss)
brought that down to ~1.03x on a 60-epoch small_test_ind132 retrain. This
script re-measures the ratio against whatever checkpoint is passed, so it
doubles as the regression check for a repeat of that collapse.

Loads a trained e2e checkpoint (default: checkpoints/e2e_ind132/best_e2e.pt)
and, on the exact demand set the training run used for its final epoch
(seed=epochs_e2e), compares:

  - the path EdgeWeightNet + the trained routing head actually choose, vs.
  - the shortest-by-km path for the same (src, dst) pair.

Also (re)writes logs/<log_dir>/demand_path_lengths_final_epoch.csv in the
same schema the original ad-hoc investigation snippet used
(demand_id, src, dst, bitrate_gbps, path_length_km, hop_count), so it stays
diffable against logs/e2e_ind132_before_fix/demand_path_lengths_final_epoch.csv.

Usage:
    conda activate diffopt
    python scripts/diagnose_route_length_vs_shortest.py --config configs/experiment/small_test_ind132.yaml
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml

from diffopt.routing.shortest_path import dijkstra
from _common import add_common_args, build_context, demands_for

ap = argparse.ArgumentParser()
add_common_args(ap, with_demands=False)
args = ap.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)
ctx = build_context(cfg, load_e2e_checkpoint=True, checkpoint_path=args.checkpoint)
pipe = ctx.pipeline
topo = ctx.topology
edges = ctx.edges
t_cfg = cfg["training"]

ckpt_label = args.checkpoint or f"{cfg['checkpoint_dir']}/best_e2e.pt"
print(f"Loaded checkpoint {ckpt_label} (epoch {ctx.ckpt['epoch']}, "
      f"total_loss={ctx.ckpt['total_loss']:.4f})")

lens = np.array([e.length_km for e in edges], dtype=float)
ei = pipe._edge_index.numpy()

# train.py no longer reseeds demands per epoch (it builds one fixed traffic
# matrix before its epoch loop and reuses it throughout). This script's
# seed=final_epoch below is purely this diagnostic's own convention for
# picking a demand set to inspect -- not a mirror of train.py's behavior --
# chosen as epochs_e2e (the last of the 1..epochs_e2e epoch range) simply to
# have a stable, reproducible seed tied to the run's config.
final_epoch = t_cfg["epochs_e2e"]
demands = demands_for(ctx, seed=final_epoch)
print(f"{len(demands)} demands (seed={final_epoch}, matching training's final epoch)\n")

tau_end = t_cfg["alloc_tau_end"]
vlastelica_lambda = ctx.ckpt["vlastelica_lambda"]

with torch.no_grad():
    _, _, path_indicators, _ = pipe(
        demands, tau=tau_end, lambda_=vlastelica_lambda
    )

rows = []
for d in demands:
    trained_edges = pipe._reconstruct_path(path_indicators[d.id], d.src, d.dst)
    trained_km = sum(edges[e].length_km for e in trained_edges)
    trained_hops = len(trained_edges)

    pi = dijkstra(lens, ei, d.src, d.dst, topo.num_nodes)
    shortest_edges = pipe._reconstruct_path(torch.tensor(pi, dtype=torch.float32), d.src, d.dst)
    shortest_km = sum(edges[e].length_km for e in shortest_edges)

    rows.append((d.id, d.src, d.dst, d.bitrate_gbps, trained_km, trained_hops, shortest_km))

log_dir = Path(cfg["log_dir"])
log_dir.mkdir(parents=True, exist_ok=True)
out_path = log_dir / "demand_path_lengths_final_epoch.csv"
with out_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["demand_id", "src", "dst", "bitrate_gbps", "path_length_km", "hop_count"])
    for r in sorted(rows, key=lambda r: -r[4]):
        w.writerow([r[0], r[1], r[2], r[3], r[4], r[5]])
print(f"Wrote {out_path}")

trained = np.array([r[4] for r in rows])
shortest = np.array([r[6] for r in rows])
ratio = trained / np.maximum(shortest, 1e-9)

print(f"\n{'':>20} {'trained routing':>16} {'shortest-by-km':>16}")
print(f"{'mean km':>20} {trained.mean():>16.1f} {shortest.mean():>16.1f}")
print(f"{'max km':>20} {trained.max():>16.1f} {shortest.max():>16.1f}")
print(f"{'min km':>20} {trained.min():>16.1f} {shortest.min():>16.1f}")
print(f"\nmean per-demand ratio (trained/shortest): {ratio.mean():.3f}x")
print(f"ratio of means: {trained.mean() / shortest.mean():.3f}x")
print(f"demands where trained path == shortest path (km match): "
      f"{(np.abs(trained - shortest) < 1e-6).sum()}/{len(rows)}")
