"""The oracle: exact minimum-device allocation on a FIXED route.

A SHIPPED component, not a test fixture (spec section 6 hook 3). Three uses:

  ground truth      oracle_gap = devices_hard - devices_oracle, >= 0 by
                    construction. Acceptance for the whole stage: gap == 0,
                    which isolates allocation competence from routing
                    quality — any remaining badness is then a ROUTING
                    problem, provably not an allocation one.

  deployment repair for any demand still violated under hard_rollout,
                    substitute the oracle allocation on its final route.
                    Legitimate ONLY because site cost is zero (decision 1),
                    which makes the demands independent. NEVER runs during
                    training: it would be a crutch that destroys the very
                    gap signal above.

  trigger metric    devices under the learned route vs the best of the
                    demand's k=5 shortest-by-km candidates. A persistent gap
                    means routing picks corridors whose candidates are badly
                    placed; the fix would be spec approach C, not B.

Optimality is an exchange argument: take any feasible allocation, and push
its first cut rightwards to the greedy position. The chunk before the cut
only shrinks in noise (it loses segments) so it stays feasible, and the
chunk after only grows — but it grows by exactly the segments greedy already
proved fit. Induct. Hence greedy is minimal, so `count` is a true floor and
`gap >= 0` always. tests/test_oracle.py checks this against brute force
rather than trusting the argument.

CHUNK SCORING CONVENTION. A chunk's noise is the SUM of its segments' QoT
linear noises, each segment scored with accum_dist_km restarting at 0 —
identical to what SegmentCombiner folds and what hard_rollout evaluates.
The spec's text asks for one QoT call on the concatenated chunk instead;
that convention differs by up to 1.16 dB (regen_placement_not_concentrating.md
lines 385-410) because NLI depends on accumulated dispersion while ASE does
not. Using it here would make oracle_gap measure a physics disagreement
between certifier and evaluator rather than the head's competence. The
deviation and the follow-up that resolves it are recorded in
docs/investigations/open_followups.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from diffopt.qot.segment_combiner import GSNR_MAX, GSNR_MIN, db_to_linear_noise


@dataclass
class OracleResult:
    """a: (D, J-1) 0/1 allocations. count: (D,) long. feasible: (D,) bool."""

    a: torch.Tensor
    count: torch.Tensor
    feasible: torch.Tensor


def oracle_allocation(
    seg_gsnr_db: torch.Tensor,
    bar_db: torch.Tensor,
    num_segments: torch.Tensor,
) -> OracleResult:
    """Fewest cuts making every chunk clear the bar, per demand.

    Args:
        seg_gsnr_db:  (D, J) per-segment QoT GSNR in dB. Columns at or past
                      num_segments[d] are ignored. Clamped to the same
                      [GSNR_MIN, GSNR_MAX] band the fold uses — if the oracle
                      and the fold clamped differently, gap would measure the
                      clamp.
        bar_db:       (D,) threshold(bitrate_d) + margin_db.
        num_segments: (D,) long, each in [1, J].

    Returns:
        OracleResult. `feasible[d]` is False exactly when some single segment
        alone busts the bar, in which case no allocation on this route works
        and `a[d]` is the best available (still-infeasible) attempt.
    """
    if seg_gsnr_db.dim() != 2:
        raise ValueError(
            f"seg_gsnr_db must be (D, J), got {tuple(seg_gsnr_db.shape)}"
        )
    d, j = seg_gsnr_db.shape
    if num_segments.shape != (d,):
        raise ValueError(
            f"Expected num_segments of shape {(d,)}, got {tuple(num_segments.shape)}"
        )
    if j == 0 or int(num_segments.min()) < 1 or int(num_segments.max()) > j:
        raise ValueError(
            f"num_segments must lie in [1, {j}], got {num_segments.tolist()}"
        )

    device = seg_gsnr_db.device
    positions = torch.arange(j, device=device)
    seg_valid = positions.unsqueeze(0) < num_segments.unsqueeze(1)

    # float64 for the accumulation, matching forward_batched. Masking AFTER
    # the dB->linear conversion puts padded columns on exactly 0 noise
    # whatever dB value they hold; masking the dB value would need +inf.
    n = db_to_linear_noise(
        seg_gsnr_db.clamp(GSNR_MIN, GSNR_MAX).double()
    ) * seg_valid.double()
    bar_noise = db_to_linear_noise(bar_db.double())

    a = torch.zeros(d, max(j - 1, 0), dtype=torch.float32, device=device)
    feasible = torch.ones(d, dtype=torch.bool, device=device)
    c = torch.zeros(d, dtype=torch.float64, device=device)

    for k in range(j):
        valid = seg_valid[:, k]
        n_k = n[:, k]

        # A segment that busts the bar on its own can never be rescued: it is
        # its own chunk even with cuts on both sides.
        feasible = feasible & ~(valid & (n_k > bar_noise))

        if k == 0:
            c = torch.where(valid, n_k, c)
            continue

        # Check if adding the next segment would make the current chunk violate
        over = valid & ((c + n_k) > bar_noise)
        a[:, k - 1] = over.to(a.dtype)
        c = torch.where(valid, torch.where(over, n_k, c + n_k), c)

    return OracleResult(a=a, count=a.sum(dim=1).long(), feasible=feasible)


def oracle_gap(hard_a: torch.Tensor, oracle_count: torch.Tensor) -> int:
    """devices_hard - devices_oracle, summed over demands. Never negative."""
    gap = int(hard_a.sum().item()) - int(oracle_count.sum().item())
    if gap < 0:
        raise AssertionError(
            f"oracle_gap = {gap} < 0. The oracle is the minimum by an "
            f"exchange argument, so a negative gap means the two sides are "
            f"scoring chunks differently — check that hard_rollout and "
            f"oracle_allocation use the same clamp band and the same "
            f"per-segment GSNRs."
        )
    return gap


def repair_with_oracle(
    seg_gsnr_db: torch.Tensor,
    bar_db: torch.Tensor,
    num_segments: torch.Tensor,
    hard_a: torch.Tensor,
    combiner,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Substitute the oracle allocation on every demand the head left violated.

    Optimal (the oracle is the minimum), local (nothing is shared between
    demands once site cost is zero), and monotone (a repaired demand cannot
    make another one worse). Together these give spec 2.6's guarantee:

        the deployed solution is feasible iff the learned routes admit any
        feasible allocation.

    NEVER call this during training.

    Returns (a_repaired, gsnr_repaired, feasible).
    """
    res = oracle_allocation(seg_gsnr_db, bar_db, num_segments)
    with torch.no_grad():
        gsnr_hard = combiner.forward_batched(seg_gsnr_db, hard_a, num_segments)
        violated = gsnr_hard < bar_db
        a_repaired = torch.where(violated.unsqueeze(1), res.a, hard_a)
        gsnr_repaired = combiner.forward_batched(
            seg_gsnr_db, a_repaired, num_segments
        )
    return a_repaired, gsnr_repaired, res.feasible
