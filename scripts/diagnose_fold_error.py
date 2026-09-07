"""Diagnostic: quantify SegmentCombiner's fold error on the real network.

Permanent guard against the fold-correctness bug class behind the
regenerator over-provisioning, and the script that reproduces that
investigation's headline table. Loads a trained e2e checkpoint
and runs ONE forward pass over that checkpoint's own fixed traffic matrix
(scripts/_common.py's fixed_traffic_demands, at
schedule_at(cfg, cfg["training"]["epochs_e2e"])), then re-folds every
demand's (segment GSNRs, boundary allocations) three different ways and
compares them.

There is no placement to hard-saturate any more. The pre-Stage-II version
of this script pinned a per-node logit vector to +/-30 so "placed" was
unambiguous; a per-(demand, boundary) head has no such vector, and its soft
allocations `alloc.a` are read straight off `AllocationOutputs` — which is
strictly better here, because a genuinely FRACTIONAL boundary probability
is exactly the regime the shipped fold's exactness claim is about. Under
the old saturation every probability was within 1e-8 of 0 or 1 and the
shipped-vs-exact delta was ~0 by construction.

The three folds:

  1. `_legacy_single_accumulator_fold` — a local, clearly-labelled copy of
     the pre-fix recurrence (a single accumulator that takes a soft_max at
     a regenerated boundary but keeps ADDING to it afterwards, instead of
     starting a fresh chunk). Kept here as the exact historical form, so
     this script both reproduces the write-up's numbers now and would
     catch a future regression back to it. It needs a soft_max temperature,
     which the shipped code no longer has anywhere; `_LEGACY_TEMPERATURE`
     below pins it to the value the historical anneal ended at.
  2. The shipped fold — SegmentCombiner.forward as it exists today. Now an
     exact expectation of the max chunk noise over the fractional boundary
     probabilities, so its delta below is no longer an approximation error:
     it is the (real, physical) gap between averaging over soft placements
     and committing to the hard-rounded one.
  3. `_exact_hard_chunk_fold` — the true -10*log10(max over chunks) at
     hard-rounded p (p >= 0.5 -> cut), i.e. the reference both of the above
     are measured against.

Only demands with at least one hard-rounded cut (a >= 0.5) on their route
are scored: when no boundary on a route ever cuts, chunking is
a no-op and the legacy and shipped folds trivially agree with the exact
fold by construction (plain noise addition) — including those demands
would dilute the reported error toward zero without saying anything about
the fold itself.

Usage:
    conda activate diffopt
    python scripts/diagnose_fold_error.py --config configs/experiment/constrained_stress.yaml
"""
from __future__ import annotations

import argparse
import math
import statistics
from typing import List, Tuple

import torch
import yaml

from diffopt.qot.segment_combiner import (
    db_to_linear_noise,
    linear_noise_to_db,
    soft_max,
)
from _common import add_common_args, build_context, fixed_traffic_demands, schedule_at

ap = argparse.ArgumentParser()
add_common_args(
    ap,
    default_config="configs/experiment/constrained_stress.yaml",
    with_demands=False,
)
args = ap.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)
ctx = build_context(cfg, load_e2e_checkpoint=True, checkpoint_path=args.checkpoint)
pipe = ctx.pipeline
t_cfg = cfg["training"]

tau, vlastelica_lambda = schedule_at(cfg, t_cfg["epochs_e2e"])

# The soft_max temperature the historical anneal ended at. Pinned as a
# literal because it no longer exists in the config or the schedule -- the
# shipped fold is exact and takes no temperature. Only
# `_legacy_single_accumulator_fold` below needs it, and it needs the
# historical value, not a current one.
_LEGACY_TEMPERATURE = 0.01

ckpt_label = args.checkpoint or f"{cfg.get('checkpoint_dir', 'checkpoints')}/best_e2e.pt"

demands, excluded = fixed_traffic_demands(ctx)
tr_cfg = cfg["traffic"]
print(f"Checkpoint {ckpt_label} (epoch {ctx.ckpt['epoch']})")
print(
    f"{len(demands)} demands ({tr_cfg['scenario']}, seed={tr_cfg['seed']}), "
    f"{len(excluded)} excluded by preflight, schedule epoch={t_cfg['epochs_e2e']} "
    f"(tau={tau:.4f}, legacy fold replayed at t={_LEGACY_TEMPERATURE})\n"
)


# ---------------------------------------------------------------------------
# One forward pass; the fold inputs come straight off AllocationOutputs
# ---------------------------------------------------------------------------
#
# No recording shim any more. Before the allocation head existed, the only
# way to see what SegmentCombiner had been handed was to wrap it; now
# `AllocationOutputs` returns exactly those tensors -- `seg_gsnr_db` is the
# padded (D, J) matrix pipeline.forward folds, `a` is the (D, J-1)
# allocation matrix it folds with, and `num_segments` says where each row's
# padding starts. Reading them directly removes a whole class of
# shim-vs-reality drift.
#
# `a` (the PRICED allocation), not `a_physics`: they differ only under
# alloc_dropout, which is a training-time perturbation this script does not
# and must not enable.

with torch.no_grad():
    _, _, _, alloc = pipe(demands, tau=tau, lambda_=vlastelica_lambda)

calls: List[Tuple[List[torch.Tensor], List[torch.Tensor]]] = []
for row in range(len(demands)):
    n_seg = int(alloc.num_segments[row].item())
    calls.append((
        [alloc.seg_gsnr_db[row, k].detach().clone() for k in range(n_seg)],
        [alloc.a[row, k].detach().clone() for k in range(n_seg - 1)],
    ))

# Row order follows `alloc.demand_ids`, which pipeline.forward builds by
# enumerating `demands` in order -- assert it rather than trust it, since
# every delta below is attributed to a demand by this pairing.
assert alloc.demand_ids == [d.id for d in demands], (
    "AllocationOutputs row order does not match the demand list"
)


# ---------------------------------------------------------------------------
# The three folds
# ---------------------------------------------------------------------------

def _legacy_single_accumulator_fold(segment_gsnrs_db, regen_probs_at_boundaries, temperature):
    """Historical form, pre-fix:
    a single accumulator takes a soft_max at a regenerated boundary, then
    keeps ADDING subsequent segments to that max instead of starting a
    fresh chunk. Correct only when every post-regenerator chunk happens to
    be a single segment. Kept verbatim (not imported) so it survives even
    if SegmentCombiner.forward changes again."""
    GSNR_MIN, GSNR_MAX = -5.0, 35.0

    def _safe_noise(g_db: torch.Tensor) -> torch.Tensor:
        return db_to_linear_noise(g_db.clamp(GSNR_MIN, GSNR_MAX).double())

    accumulated = _safe_noise(segment_gsnrs_db[0])
    for i in range(1, len(segment_gsnrs_db)):
        p = regen_probs_at_boundaries[i - 1].double()
        next_noise = _safe_noise(segment_gsnrs_db[i])
        noise_no_regen = accumulated + next_noise
        noise_regen = soft_max(accumulated, next_noise, temperature=temperature)
        accumulated = (1.0 - p) * noise_no_regen + p * noise_regen
    return linear_noise_to_db(accumulated).float().item()


def _exact_hard_chunk_fold(segment_gsnrs_db, hard_cuts):
    """-10*log10(max over chunks), chunking at hard_cuts (already rounded
    to 0/1 booleans) -- the ground truth every soft relaxation above is
    approximating."""
    GSNR_MIN, GSNR_MAX = -5.0, 35.0
    noises = [
        10 ** (-min(max(g.item(), GSNR_MIN), GSNR_MAX) / 10.0) for g in segment_gsnrs_db
    ]
    chunks = []
    current = noises[0]
    for i, cut in enumerate(hard_cuts):
        if cut:
            chunks.append(current)
            current = noises[i + 1]
        else:
            current += noises[i + 1]
    chunks.append(current)
    return -10.0 * math.log10(max(chunks))


# ---------------------------------------------------------------------------
# Re-fold every recorded call three ways, scored over routes that cross a
# placed regenerator
# ---------------------------------------------------------------------------

legacy_deltas: List[float] = []
shipped_deltas: List[float] = []
rows = []  # (demand_id, n_segments, n_cuts, max_chunk_len, legacy_delta, shipped_delta)

for demand, (gsnrs, probs) in zip(demands, calls):
    hard_cuts = [p.item() >= 0.5 for p in probs]
    if not any(hard_cuts):
        continue  # no cut on this route; folds agree trivially

    exact_db = _exact_hard_chunk_fold(gsnrs, hard_cuts)
    legacy_db = _legacy_single_accumulator_fold(gsnrs, probs, _LEGACY_TEMPERATURE)
    shipped_db = pipe.segment_combiner(gsnrs, probs).item()

    legacy_delta = legacy_db - exact_db
    shipped_delta = shipped_db - exact_db
    legacy_deltas.append(legacy_delta)
    shipped_deltas.append(shipped_delta)

    # max chunk length under hard rounding, for the chunk-length breakdown
    lengths, cur_len = [], 1
    for cut in hard_cuts:
        if cut:
            lengths.append(cur_len)
            cur_len = 1
        else:
            cur_len += 1
    lengths.append(cur_len)

    rows.append((
        demand.id, len(gsnrs), sum(hard_cuts), max(lengths), legacy_delta, shipped_delta,
    ))

n_cross = len(rows)
print(f"allocation: {alloc.device_count.item():.2f} soft devices, "
      f"{int((alloc.a >= 0.5).sum())} hard-rounded cuts over "
      f"{int(alloc.a.numel())} (demand, boundary) variables")
print(f"{n_cross}/{len(demands)} demands have at least one hard-rounded cut "
      f"(scored below); {len(demands) - n_cross} have none and are excluded "
      f"(folds agree trivially there)\n")


def _report(name: str, deltas: List[float]) -> None:
    if not deltas:
        print(f"{name}: no scored demands")
        return
    worst = max(deltas, key=abs)
    print(
        f"{name}: n={len(deltas)}  mean={statistics.mean(deltas):+.3f} dB  "
        f"median={statistics.median(deltas):+.3f} dB  worst={worst:+.3f} dB"
    )


_report("legacy vs exact  ", legacy_deltas)
_report("shipped vs exact ", shipped_deltas)

print("\nbreakdown by max chunk length (segments in the largest chunk under hard rounding):")
by_len = {}
for _, _, _, max_len, legacy_delta, shipped_delta in rows:
    by_len.setdefault(max_len, []).append((legacy_delta, shipped_delta))
for max_len in sorted(by_len):
    legacy_group = [ld for ld, _ in by_len[max_len]]
    shipped_group = [sd for _, sd in by_len[max_len]]
    print(
        f"  max_chunk_len={max_len:>2}  n={len(legacy_group):>3}  "
        f"legacy mean={statistics.mean(legacy_group):+.3f} dB  "
        f"shipped mean={statistics.mean(shipped_group):+.3f} dB"
    )

print("\nworst 10 demands by |legacy vs exact| delta "
      "(demand_id, n_segments, n_cuts, max_chunk_len, legacy_delta, shipped_delta):")
for row in sorted(rows, key=lambda r: -abs(r[4]))[:10]:
    demand_id, n_segments, n_cuts, max_len, legacy_delta, shipped_delta = row
    print(
        f"  demand {demand_id:>4}  segs={n_segments:>2}  cuts={n_cuts:>2}  "
        f"max_chunk_len={max_len:>2}  legacy={legacy_delta:+.3f} dB  shipped={shipped_delta:+.3f} dB"
    )

# Nothing to restore: this script never mutates a parameter. The
# save/restore that used to live here existed only to undo the +/-30
# saturation of the per-node logit vector, which no longer exists.
