"""
Differentiable segment combiner for end-to-end GSNR estimation.

Physics: noise accumulates additively along a transparent path. A
regenerator at a boundary splits the path into independent chunks — it
rebuilds the signal, so noise does not carry across it. The end-to-end
noise is the max over these chunks (the whole path must clear its
threshold at its worst chunk, not its sum). Regenerator boundaries are
probabilistic (p ∈ (0,1) per boundary): the fold generalizes the max over
chunks to a probability-weighted soft max over *every* possible chunking of
the path into contiguous segments, where each chunking's weight is its
exact probability under independent Bernoulli(p_i) boundary decisions. This
is exact at every p ∈ {0,1} boundary configuration (every chunking other
than the one realized by that hard assignment gets weight exactly 0) and
reduces to the existing 2-way `soft_max` helper below for a single
boundary. See docs/investigations/regen_over_provisioning.md.
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
    annealing state across epochs). See docs/architecture/invariants.md's
    "Pipeline" architectural constraints.

    Historical note: an earlier version took `soft_max_temperature` at
    construction, defaulting to 0.5 and never annealed anywhere — this
    silently broke the "regenerating helps" physics invariant during real
    training (at t=0.5 the soft-max approximation's floor, `t*ln(2)`,
    swamps real per-segment noise values, making every multi-segment path
    look worse than not regenerating at all). Fixed by making the caller
    supply a real, annealed temperature every call — see
    docs/investigations/CHANGELOG.md's Phase 1c corrections for the measured
    numbers.
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
            docs/investigations/CHANGELOG.md's Phase 1b correction #3 established 0.01 as the
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

        n = torch.stack([_safe_noise(g) for g in segment_gsnrs_db])  # (N,) float64
        num_segments = n.shape[0]
        device = n.device
        dtype = n.dtype

        if num_segments == 1:
            return linear_noise_to_db(n[0]).float()

        p = torch.stack([pr.double() for pr in regen_probs_at_boundaries])  # (N-1,)

        # Every contiguous chunk [s, e] (0 <= s <= e < N) is a candidate
        # realized chunk. chunk_noise(s, e) is deterministic (segments
        # inside a chunk always sum); w(s, e) is the exact probability that
        # [s, e] is the chunk actually realized under independent
        # Bernoulli(p_i) boundary decisions: the boundary immediately
        # before s and immediately after e must both cut (or be the path's
        # own start/end), and every boundary strictly inside must not cut.
        prefix = torch.cat([torch.zeros(1, dtype=dtype, device=device), torch.cumsum(n, dim=0)])

        # log(1-p) prefix sums for inner(s,e) = prod_{k=s}^{e-1} (1-p[k])
        log1mp = torch.log((1.0 - p).clamp_min(1e-300))
        log1mp_prefix = torch.cat(
            [torch.zeros(1, dtype=dtype, device=device), torch.cumsum(log1mp, dim=0)]
        )
        logp = torch.log(p.clamp_min(1e-300))

        # L(s) = logp[s-1] if s>0 else 0 (log 1); R(e) = logp[e] if e<N-1 else 0
        logL = torch.cat([torch.zeros(1, dtype=dtype, device=device), logp])
        logR = torch.cat([logp, torch.zeros(1, dtype=dtype, device=device)])

        idx = torch.arange(num_segments, device=device)
        s_idx = idx.unsqueeze(1).expand(num_segments, num_segments)
        e_idx = idx.unsqueeze(0).expand(num_segments, num_segments)
        valid = e_idx >= s_idx  # only s <= e are valid chunks

        chunk_noise = prefix[e_idx + 1] - prefix[s_idx]
        log_inner = log1mp_prefix[e_idx] - log1mp_prefix[s_idx]
        log_w = logL.unsqueeze(1) + logR.unsqueeze(0) + log_inner

        neg_inf = torch.tensor(float("-inf"), dtype=dtype, device=device)
        log_w = torch.where(valid, log_w, neg_inf)
        chunk_noise = torch.where(valid, chunk_noise, torch.zeros_like(chunk_noise))

        flat_noise = chunk_noise.reshape(-1)
        flat_logw = log_w.reshape(-1)

        # Scale normalizer `m` (same role as soft_max's own detached `m`)
        # must only consider candidates with non-negligible weight. A
        # candidate that is essentially impossible (log-weight far below
        # any float64-representable probability) can still have a LARGE raw
        # chunk_noise -- e.g. the full-path chunk when interior boundaries
        # are near-certainly cut. Including it in an unconditional
        # max(chunk_noise) dilutes the scale for every genuinely relevant
        # candidate and reintroduces the absolute-overshoot problem
        # m-normalization exists to avoid (see
        # docs/investigations/regen_over_provisioning.md). -30 matches this
        # codebase's existing regen_logits saturation convention
        # (scripts/diagnose_regen_ablation.py's +/-30 clamp); exp(-30) ~
        # 9e-14, well below anything that could matter at float64 precision.
        active = flat_logw > -30.0
        active_noise = torch.where(active, flat_noise, torch.zeros_like(flat_noise))
        m = active_noise.max().detach().clamp_min(1e-30)

        terms = flat_logw + flat_noise / m / temperature
        effective_noise = m * temperature * torch.logsumexp(terms, dim=0)

        return linear_noise_to_db(effective_noise).float()
