"""
Differentiable segment combiner for end-to-end GSNR estimation.

Physics: noise accumulates additively along a transparent path.
A regenerator at a boundary resets accumulated noise (only the worse
segment's noise propagates). The regenerator probability p ∈ (0,1)
interpolates between the two regimes with a soft-max approximation.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def db_to_linear_noise(gsnr_db: torch.Tensor) -> torch.Tensor:
    """Convert GSNR in dB to normalised noise power: 10^(-gsnr_db / 10)."""
    return torch.pow(torch.tensor(10.0, dtype=gsnr_db.dtype, device=gsnr_db.device),
                     -gsnr_db / 10.0)


def linear_noise_to_db(noise_linear: torch.Tensor) -> torch.Tensor:
    """Convert normalised noise power back to GSNR in dB: -10 * log10(noise)."""
    return -10.0 * torch.log10(noise_linear)


def soft_max(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Smooth approximation to max(a, b) via log-sum-exp, normalised by scale.

    For noise values, 'max noise' corresponds to 'worst segment'.

    The naive form `t * logsumexp([a/t, b/t])` equals `max(a, b) + t*delta`
    with `0 < delta <= ln2` — an **absolute** overshoot that does not shrink
    as the operands do. That is fatal here, because these operands are
    linear noise powers: real segments run ~180-210 km, hence ~26 dB, hence
    noise ~0.0025, while even the sharpest scheduled temperature (0.01)
    overshoots by `0.01 * ln2 = 0.0069`. The error then exceeds the signal,
    `soft_max(a, b)` climbs above `a + b`, and the combiner reports that
    regenerating makes a path *worse* — inverting the sign of every gradient
    reaching `regen_logits`. See
    docs/investigations/regen_placement_not_concentrating.md.

    Normalising by `m = max(a, b)` (detached, so it only rescales and never
    contributes gradient) makes the overshoot `m * t * delta` — proportional
    to the operands instead of absolute. The invariant `soft_max(a,b) < a+b`
    then holds for any `temperature < 1/ln2 ~= 1.44` at ANY noise magnitude,
    so the 0.5 -> 0.01 annealing schedule is sign-correct end to end and no
    longer encodes a hidden assumption about topology-dependent noise scale.
    """
    t = temperature
    m = torch.maximum(a, b).detach().clamp_min(1e-30)
    stacked = torch.stack([a / m / t, b / m / t], dim=0)
    return m * t * torch.logsumexp(stacked, dim=0)


# ---------------------------------------------------------------------------
# SegmentCombiner
# ---------------------------------------------------------------------------

class SegmentCombiner(nn.Module):
    """
    Combine per-segment GSNR values into an end-to-end GSNR.

    Stateless: `temperature` is a required `forward()` argument, not a
    constructor parameter — this class holds no annealing state, matching
    `DiffONetPipeline.forward()`'s own `tau`/`lambda_` pattern (passed
    per-call, never stored as a module attribute, to prevent stale
    annealing state across epochs). See CLAUDE.md's "Pipeline"
    architectural constraints.

    Historical note: an earlier version took `soft_max_temperature` at
    construction, defaulting to 0.5 and never annealed anywhere — this
    silently broke the "regenerating helps" physics invariant during real
    training (at t=0.5 the soft-max approximation's floor, `t*ln(2)`,
    swamps real per-segment noise values, making every multi-segment path
    look worse than not regenerating at all). Fixed by making the caller
    supply a real, annealed temperature every call — see CLAUDE.md's
    Phase 1c corrections for the measured numbers.
    """

    def forward(
        self,
        segment_gsnrs_db: List[torch.Tensor],
        regen_probs_at_boundaries: List[torch.Tensor],
        temperature: float,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        segment_gsnrs_db:
            List of N scalar tensors (GSNR in dB for each transparent segment).
        regen_probs_at_boundaries:
            List of N-1 scalar tensors with regenerator probabilities p ∈ (0,1)
            at each inter-segment boundary.
        temperature:
            Soft-max sharpness for this call. Lower → closer to true max
            (more physically accurate); higher → looser approximation.
            CLAUDE.md's Phase 1b correction #3 established 0.01 as the
            value at which the "regen helps" invariant reliably holds —
            callers doing real training should anneal toward that value,
            not hold a fixed loose one.

        Returns
        -------
        Scalar tensor: end-to-end GSNR in dB.
        """
        if len(segment_gsnrs_db) == 0:
            raise ValueError("segment_gsnrs_db must contain at least one segment")
        if len(regen_probs_at_boundaries) != len(segment_gsnrs_db) - 1:
            raise ValueError(
                f"Expected {len(segment_gsnrs_db) - 1} boundary probabilities, "
                f"got {len(regen_probs_at_boundaries)}"
            )

        # Clamp inputs to prevent float32 overflow on extreme values
        GSNR_MIN, GSNR_MAX = -5.0, 35.0

        def _safe_noise(g_db: torch.Tensor) -> torch.Tensor:
            g_clamped = g_db.clamp(GSNR_MIN, GSNR_MAX)
            # Use float64 for accumulation precision
            return db_to_linear_noise(g_clamped.double())

        accumulated_noise = _safe_noise(segment_gsnrs_db[0])

        for i in range(1, len(segment_gsnrs_db)):
            p = regen_probs_at_boundaries[i - 1].double()
            next_noise = _safe_noise(segment_gsnrs_db[i])

            # Passthrough (no regen): noises add
            noise_no_regen = accumulated_noise + next_noise

            # Regenerator: only the worse (larger noise) segment matters
            noise_regen = soft_max(accumulated_noise, next_noise, temperature=temperature)

            # Soft interpolation
            accumulated_noise = (1.0 - p) * noise_no_regen + p * noise_regen

        result_db = linear_noise_to_db(accumulated_noise)
        return result_db.float()
