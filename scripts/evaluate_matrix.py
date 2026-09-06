"""Evaluate a trained checkpoint against a fixed traffic matrix.

Answers the three questions the validation plan asks (spec §6):

  step 2 — what does the matrix look like? (size, bitrate mix, preflight
           exclusions, route-length distribution)
  step 3 — how does the PRE-change checkpoint score on it? (the baseline the
           constrained runs are compared against)
  step 6 — how does a trained checkpoint score on the HELD-OUT matrix?
           (--holdout, which swaps traffic.seed for traffic.holdout_seed)

Placement is evaluated by `train.hard_rollout` — the same deterministic
rollout checkpoint selection uses, so a report from this script and a
training log row describe the same object.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

# `scripts/` is not a package (no __init__.py) — every diagnose_*.py uses this
# same bare import, which resolves because Python puts a script's own
# directory on sys.path. Do not "fix" this to `from scripts._common import`.
from _common import add_common_args, build_context, schedule_at
from diffopt.qot.segment_combiner import SegmentCombiner
from diffopt.train import hard_rollout
from diffopt.traffic import (
    build_traffic_matrix,
    preflight_filter,
    scenario_alpha,
    shortest_path_edges_by_km,
    traffic_matrix_checksum,
)


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(), with_demands=False)
    ap.add_argument("--holdout", action="store_true",
                    help="Use traffic.holdout_seed instead of traffic.seed")
    ap.add_argument("--scenario", default=None,
                    help="Override traffic.scenario (stress | realistic)")
    ap.add_argument("--no-preflight", action="store_true",
                    help="Report against the raw matrix, before the preflight screen")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ctx = build_context(cfg, checkpoint_path=args.checkpoint)

    tr_cfg = cfg["traffic"]
    c_cfg = cfg["constraint"]
    scenario = args.scenario or tr_cfg["scenario"]
    seed = tr_cfg["holdout_seed"] if args.holdout else tr_cfg["seed"]

    raw = build_traffic_matrix(
        ctx.topology,
        seed=seed,
        scale=tr_cfg["scale"],
        alpha=scenario_alpha(scenario),
        bitrate_options=cfg["bitrate_options"],
    )
    print(f"Matrix: scenario={scenario} seed={seed} scale={tr_cfg['scale']:.3g}")
    print(f"  {len(raw)} pairs, checksum {traffic_matrix_checksum(raw)}")

    bitrate_mix: dict = {}
    for d in raw:
        bitrate_mix[d.bitrate_gbps] = bitrate_mix.get(d.bitrate_gbps, 0) + 1
    print("  bitrate mix: " + ", ".join(
        f"{int(b)}G x{bitrate_mix[b]}" for b in sorted(bitrate_mix)
    ))

    route_kms = []
    for d in raw:
        eids = shortest_path_edges_by_km(ctx.topology, d.src, d.dst)
        if eids is not None:
            route_kms.append(sum(ctx.edges[e].length_km for e in eids))
    if route_kms:
        route_kms.sort()
        mean_km = sum(route_kms) / len(route_kms)
        p95_km = route_kms[int(0.95 * (len(route_kms) - 1))]
        print(f"  shortest-by-km route length: mean {mean_km:.0f} km, p95 {p95_km:.0f} km")

    demands = raw
    if not args.no_preflight:
        demands, excluded = preflight_filter(
            ctx.topology, raw,
            qot_model=ctx.qot_model,
            segment_combiner=SegmentCombiner(),
            modulation_config=ctx.mod_cfg,
            margin_db=c_cfg["margin_db"],
            channel_loading_fraction=cfg["pipeline"]["channel_loading_fraction"],
            max_spans=cfg.get("max_spans_per_segment", 60),
        )
        print(f"  preflight: {len(demands)} kept, {len(excluded)} excluded")
        for d, shortfall in excluded:
            print(f"    d{d.id}: {d.src}->{d.dst} @ {d.bitrate_gbps:.0f}G, "
                  f"shortfall {shortfall:.2f} dB")

    # The deployed allocation, from the same deterministic rollout
    # checkpoint selection uses. `gsnr_preds` below comes out of the hard
    # rollout itself — the soft pass below is only run for its
    # routes/segments/GSNRs (hard_rollout reuses them rather than
    # re-routing; open_followups.md #7b), and its own (soft) allocation is
    # discarded, never mixed into the counts printed here.
    _, vlastelica_lambda = schedule_at(cfg, cfg["training"]["epochs_e2e"])
    with torch.no_grad():
        _, _, _, soft_alloc = ctx.pipeline(demands, lambda_=vlastelica_lambda)
    hard = hard_rollout(
        ctx.pipeline, demands, soft_alloc, ctx.mod_cfg,
        margin_db=c_cfg["margin_db"],
    )
    sites = sorted(hard["site_mask"].nonzero(as_tuple=True)[0].tolist())
    print("\nDeployed allocation (hard rollout):")
    print(f"  devices: {hard['hard_num_devices']}   "
          f"oracle minimum: {hard['oracle_devices']}   "
          f"gap: {hard['oracle_gap']}")
    print(f"  sites touched: {hard['hard_num_sites']} {sites}")
    print(f"  violated: {hard['hard_num_violated']}   "
          f"worst margin: {hard['hard_worst_margin_db']:+.2f} dB")
    if hard["oracle_infeasible"]:
        print(f"  !! {hard['oracle_infeasible']} demand(s) have NO feasible "
              f"allocation on their learned route — a routing failure, not "
              f"an allocation one")

    gsnr_preds = hard["gsnr_preds"]

    margins = []
    violated = []
    infeasible = []
    for d in demands:
        thr = ctx.mod_cfg.required_snr_threshold(d.bitrate_gbps)
        margin = float(gsnr_preds[d.id].item()) - thr
        margins.append(margin)
        if margin < c_cfg["margin_db"]:
            violated.append((d, margin))
        if margin < 0:
            infeasible.append((d, margin))

    margins.sort()
    p5 = margins[int(0.05 * (len(margins) - 1))] if margins else float("nan")
    print(f"\nAgainst {len(demands)} demands:")
    print(f"  violated (margin < {c_cfg['margin_db']} dB): {len(violated)}")
    print(f"  infeasible (margin < 0 dB):     {len(infeasible)}")
    print(f"  worst margin: {margins[0]:+.2f} dB   p5 margin: {p5:+.2f} dB")
    for d, margin in sorted(infeasible, key=lambda t: t[1])[:20]:
        print(f"    d{d.id}: {d.src}->{d.dst} @ {d.bitrate_gbps:.0f}G, "
              f"margin {margin:+.2f} dB")

    by_bitrate: dict = {}
    for d in demands:
        thr = ctx.mod_cfg.required_snr_threshold(d.bitrate_gbps)
        margin = float(gsnr_preds[d.id].item()) - thr
        n, bad = by_bitrate.get(d.bitrate_gbps, (0, 0))
        by_bitrate[d.bitrate_gbps] = (n + 1, bad + (1 if margin < 0 else 0))
    print("\n  infeasible by bitrate:")
    for b in sorted(by_bitrate):
        n, bad = by_bitrate[b]
        print(f"    {int(b)}G: {bad}/{n}")


if __name__ == "__main__":
    main()
