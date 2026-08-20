"""
Differentiable segment combiner for end-to-end GSNR estimation.

Physics: noise accumulates additively along a transparent path. A
regenerator at a boundary splits the path into independent chunks — it
rebuilds the signal, so noise does not carry across it. The end-to-end
noise is the max over these chunks (the whole path must clear its
threshold at its worst chunk, not its sum). Regenerator boundaries are
probabilistic (p ∈ (0,1) per boundary): the fold is the exact expectation,
over every hard partition of the path into chunks (2^(N-1) of them, one per
subset of boundaries that cuts), of that partition's own soft-max-over-its-
chunks noise, weighted *linearly* by the partition's true realization
probability under independent Bernoulli(p_i) boundary decisions — never
blended through a shared-temperature exponential across partitions, only
smoothed *within* one partition's own fixed chunk set. This is exact at
every p ∈ {0,1} boundary configuration and reduces exactly to the existing
2-way `soft_max` helper below for a single boundary (2 partitions: "no
cut" weighted (1-p), "cut" weighted p). A first attempt at generalizing
`soft_max` put probability *inside* the shared exponential instead; that
swamped the probability term against the noise term's much larger dynamic
range at production temperature and silently reintroduced a shaped version
of the bug this fold exists to fix. See
docs/investigations/regen_over_provisioning.md.
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


def _soft_max_over(noises: torch.Tensor, temperature: float) -> torch.Tensor:
    """N-way generalization of `soft_max` above: scale-normalized log-sum-exp
    over an arbitrary number of already-deterministic (not probability-
    weighted) chunk noises. Used only *within* one hard partition's own
    fixed chunk set — never across partitions, which is precisely the
    distinction the module docstring's "first attempt" paragraph is about.
    """
    if noises.shape[0] == 1:
        return noises[0]
    m = noises.max().detach().clamp_min(1e-30)
    return m * temperature * torch.logsumexp(noises / m / temperature, dim=0)


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
        num_boundaries = num_segments - 1

        is_hard = bool(((p == 0.0) | (p == 1.0)).all().item())

        if is_hard:
            # No real randomness: exactly one partition has nonzero
            # probability. Cut deterministically at p==1 boundaries and
            # soft-max the resulting REAL chunks -- O(N), the only tractable
            # path for very long chains (this file's own float64-precision
            # test uses thousands of segments, all p=0). A genuinely
            # separate code path from the fractional case below, not a
            # special-cased tolerance on the same formula -- see
            # docs/investigations/regen_over_provisioning.md.
            #
            # Reading each p via .item() to make this Python control-flow
            # decision gives this branch an exact-zero gradient w.r.t. p,
            # rather than the fractional branch's own (already vanishingly
            # small, but technically nonzero) limiting gradient as p->{0,1}.
            # Deliberate and practically inert: production probabilities
            # come from sigmoid(logit) with logits saturating around +/-30
            # (this codebase's convention), and sigmoid(30) != 1.0 exactly
            # in float64, so real training never actually lands in this
            # branch -- only this synthetic test and explicit literal-0/1
            # diagnostic calls do.
            chunk_sums = []
            cur = n[0]
            for i in range(1, num_segments):
                if p[i - 1].item() == 1.0:
                    chunk_sums.append(cur)
                    cur = n[i]
                else:
                    cur = cur + n[i]
            chunk_sums.append(cur)
            effective_noise = _soft_max_over(torch.stack(chunk_sums), temperature)
        else:
            # Genuinely fractional boundary probabilities: exact expectation
            # over every hard partition of the path (2^(N-1) of them), each
            # weighted by its TRUE realization probability -- a plain linear
            # combination, never blended through a shared-temperature
            # nonlinear op (that structural mistake is what the module
            # docstring's "first attempt" paragraph describes). Real
            # topology paths measured up to 14 segments
            # (scripts/diagnose_fold_error.py); this is O(2^(N-1)),
            # intractable well before N=8000 -- fail loudly rather than hang
            # if that assumption is ever violated.
            if num_boundaries > 20:
                raise ValueError(
                    f"SegmentCombiner's exact fractional-probability fold is "
                    f"O(2^{num_boundaries}) and intractable at this length "
                    f"(real topology paths measured <=14 segments, see "
                    f"docs/investigations/regen_over_provisioning.md). If "
                    f"this is a real path, something upstream changed; if "
                    f"intentional, this fold needs a different algorithm "
                    f"for this regime."
                )
            num_partitions = 2 ** num_boundaries
            k_idx = torch.arange(num_partitions, device=device, dtype=torch.int64)
            shifts = torch.arange(num_boundaries, device=device, dtype=torch.int64)
            # bit b of partition k: whether boundary b is cut in that partition
            bits = (k_idx.unsqueeze(1) >> shifts.unsqueeze(0)) & 1  # (K, N-1)

            # chunk_id[k, i] = index (within partition k) of the chunk
            # segment i belongs to -- one more than the number of cuts
            # strictly before segment i.
            cum_bits = torch.cumsum(bits, dim=1)
            chunk_id = torch.cat(
                [torch.zeros(num_partitions, 1, device=device, dtype=torch.int64), cum_bits],
                dim=1,
            )  # (K, N)

            n_expand = n.unsqueeze(0).expand(num_partitions, num_segments).to(dtype)
            chunk_sum = torch.zeros(num_partitions, num_segments, device=device, dtype=dtype)
            chunk_sum.scatter_add_(1, chunk_id, n_expand)

            num_chunks = cum_bits[:, -1] + 1
            idx_range = torch.arange(num_segments, device=device, dtype=torch.int64).unsqueeze(0)
            valid = idx_range < num_chunks.unsqueeze(1)  # (K, N): real chunk slots per partition

            neg_inf = torch.tensor(float("-inf"), dtype=dtype, device=device)
            masked_sum = torch.where(valid, chunk_sum, torch.zeros_like(chunk_sum))
            m = masked_sum.max(dim=1, keepdim=True).values.detach().clamp_min(1e-30)
            # Unlike an earlier attempt, `valid` gates the actual logsumexp
            # sum itself (literal -inf for every non-chunk slot), not just
            # the scale normalizer m -- so a partition's soft-max never sees
            # noise from a chunk slot it doesn't have.
            terms = torch.where(valid, chunk_sum / m / temperature, neg_inf)
            partition_noise = (m.squeeze(1) * temperature) * torch.logsumexp(terms, dim=1)

            # Each partition's TRUE probability, entirely outside any
            # exponential: log-space product of p at cut boundaries and
            # (1-p) at uncut ones, exponentiated once per partition.
            logp = torch.log(p.clamp_min(1e-300))
            log1mp = torch.log((1.0 - p).clamp_min(1e-300))
            bits_f = bits.to(dtype)
            log_partition_prob = bits_f @ logp + (1.0 - bits_f) @ log1mp
            partition_prob = torch.exp(log_partition_prob)

            effective_noise = (partition_prob * partition_noise).sum()

        return linear_noise_to_db(effective_noise).float()
