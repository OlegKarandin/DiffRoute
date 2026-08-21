"""
Differentiable segment combiner for end-to-end GSNR estimation.

Physics: noise accumulates additively along a transparent path. A
regenerator at a boundary splits the path into independent chunks — it
rebuilds the signal, so noise does not carry across it. The end-to-end
noise is the max over these chunks (the whole path must clear its
threshold at its worst chunk, not its sum). Regenerator boundaries are
probabilistic (p ∈ (0,1) per boundary), so the fold returns the exact
expectation of that max over every hard partition of the path into
chunks, weighted by each partition's true realization probability under
independent Bernoulli(p_i) boundary decisions.

That expectation is computed by a polynomial-time dynamic program rather
than by enumerating the 2^(N-1) partitions. Instead of asking "what is the
average worst chunk?", which forces looking at every outcome at once, the
DP asks "what is the probability that every realized chunk stays under a
bar `tau`?" — checkable left to right, because a chunk can never span a
cut, so past a cut the path forgets its history:

    F(tau) = P(every realized chunk has noise sum <= tau)

    U[:, j] = A[j] * prod_{k=j..b-1}(1 - p_k)     (maintained incrementally)
      where A[j] = P(cut immediately before segment j AND every chunk left
                     of it has sum <= tau)

    for b = 0 .. N-2:                             # boundary after segment b
        g_b = p_b * SUM_j [ S(j,b) <= tau ] * U[:, j]
        U   = concat( U * (1 - p_b),  g_b )       # decay existing, append new
    F(tau) = SUM_j [ S(j,N-1) <= tau ] * U[:, j]  # final chunk, no trailing cut

with `S(j,i) = a_j + ... + a_i` the noise of the chunk spanning segments
j..i. The realized max can only ever equal one of the N(N+1)/2 contiguous
chunk sums, so

    E[max] = SUM over sorted tau_r of tau_r * (F(tau_r) - F(tau_{r-1})),
             F(tau_0) = 0

and the N(N+1)/2 thresholds ride a batch dimension, leaving only the
boundary loop in Python.

Two structural properties this buys, which earlier folds only approximated:

* Every probability enters `F` through sums and products alone — `F` is a
  polynomial in the p_k, and no probability ever enters an exponential. An
  earlier attempt at an N-way generalization of the `soft_max` helper below
  put probability *inside* a shared-temperature exponential; chunk-noise
  differences scale as O(1/t) while log-probability differences are O(1), so
  probability got swamped and the fold collapsed toward the no-regen value
  regardless of p. That failure mode is now structurally impossible, not
  merely tested against. See docs/investigations/regen_over_provisioning.md.
* "Regen helps" is provable rather than temperature-dependent: cutting a
  boundary splits one chunk into two no-larger pieces, so E[max] can never
  rise. The shared-temperature fold this replaced overshot a true max by a
  relative t*ln2 and needed annealing to stay sign-correct.

The thresholds are deliberately NOT deduplicated: `torch.sort` is a
permutation and differentiable, while `torch.unique` breaks autograd, and a
duplicated threshold contributes `F(tau_r) - F(tau_{r-1}) = 0` — exactly
zero — so exactly-tied chunks are handled for free with no tolerance.
Likewise the `prod_{k=j..b-1}(1 - p_k)` factors are maintained by the
incremental `U * (1 - p_b)` update and never recovered as a ratio of
cumulative products: production p reaches exactly 1.0 in float32, which
makes such a cumulative product zero and the ratio 0/0. That is the same
hazard the direct-product fix addressed when it replaced a log/`clamp_min`
formulation, whose zero gradient in the clamped region silently flipped the
sign of the gradient on `regen_logits`.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


# Cost of the exact fold is O(N^4) elementwise work over an (R, N) float64
# working set. The forward-pass tensor itself is only ~4*N^3 bytes (~8 MB at
# this cap) — but that figure describes just the final output, not what
# autograd actually retains. Because the boundary loop runs in Python, every
# one of the ~N-1 loop iterations' intermediates (U, g_b, and the concat
# result) stays live in the graph until `.backward()` is called, so the
# retained graph is roughly 2*N^4 bytes per call — about 1.1 GB at the
# 128-segment cap, not 8 MB. `diffopt/train.py` calls `.backward()` once per
# demand, so exactly one such graph is held at a time per demand during
# training (not accumulated across the batch). At real topology path
# lengths — 16-19 segments (ind_132's km-shortest paths — see
# docs/investigations/fold_formula_scalability.md) — 2*N^4 is a few hundred
# KB, so a few hundred demands' worth totals well under 100 MB: not a
# problem in practice. But before raising MAX_EXACT_FOLD_SEGMENTS, redo this
# estimate — the O(N^4) retained-graph cost, not the O(N^3) output size, is
# what will actually bite.
MAX_EXACT_FOLD_SEGMENTS = 128


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

    `SegmentCombiner` no longer calls this: its fold now takes an exact max
    (see the module docstring's dynamic program), which needs no smoothing
    and has no temperature. The helper is kept because it documents, and
    lets tests and `scripts/diagnose_fold_error.py` reconstruct, the
    approximation the exact fold replaced.

    The naive form `t * logsumexp([a/t, b/t])` equals `max(a, b) + t*delta`
    with `0 < delta <= ln2` — an **absolute** overshoot that does not shrink
    as the operands do. That is fatal for this application, because these
    operands are linear noise powers: real segments run ~180-210 km, hence
    ~26 dB, hence noise ~0.0025, while even the sharpest scheduled
    temperature (0.01) overshoots by `0.01 * ln2 = 0.0069`. The error then
    exceeds the signal, `soft_max(a, b)` climbs above `a + b`, and the
    combiner reports that regenerating makes a path *worse* — inverting the
    sign of every gradient reaching `regen_logits`. See
    docs/investigations/regen_placement_not_concentrating.md.

    Normalising by `m = max(a, b)` (detached, so it only rescales and never
    contributes gradient) makes the overshoot `m * t * delta` — proportional
    to the operands instead of absolute. The invariant `soft_max(a,b) < a+b`
    then holds for any `temperature < 1/ln2 ~= 1.44` at ANY noise magnitude,
    so a 0.5 -> 0.01 annealing schedule was sign-correct end to end and no
    longer encoded a hidden assumption about topology-dependent noise scale.
    """
    t = temperature
    m = torch.maximum(a, b).detach().clamp_min(1e-30)
    stacked = torch.stack([a / m / t, b / m / t], dim=0)
    return m * t * torch.logsumexp(stacked, dim=0)


def _expected_max_chunk_noise(n: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Exact E[max over chunks] for independent Bernoulli(p) boundary cuts.

    `n` is the (N,) vector of per-segment linear noises, `p` the (N-1,)
    vector of per-boundary cut probabilities. Implements the module
    docstring's dynamic program; returns a scalar tensor in `n`'s dtype.

    Gradient w.r.t. `n` flows solely through the sorted thresholds `tau`
    (`torch.sort` is a differentiable permutation); the `<=` comparisons are
    plain constants, which is exactly right — `F` is piecewise constant in
    `n` between thresholds, and `sum_r tau_r * (F_r - F_{r-1})` reproduces
    `E[d max / d n]` term by term.
    """
    num_segments = n.shape[0]
    dtype, device = n.dtype, n.device

    # chunk_sums[j, i] = S(j, i) = n_j + ... + n_i, valid for j <= i.
    cumulative = torch.cat(
        [torch.zeros(1, dtype=dtype, device=device), torch.cumsum(n, dim=0)]
    )
    chunk_sums = cumulative[1:].unsqueeze(0) - cumulative[:-1].unsqueeze(1)  # (N, N)

    # Every value the max can take, ascending. Duplicates are kept on
    # purpose (see the module docstring): they contribute exactly zero.
    upper = torch.triu(torch.ones(num_segments, num_segments, dtype=torch.bool, device=device))
    tau, _ = torch.sort(chunk_sums[upper])  # (R,), R = N(N+1)/2

    # U[:, j] = P(a cut sits immediately before segment j, every chunk left
    # of it fits under tau, and no boundary since then has cut) — i.e. the
    # still-open chunk currently starts at segment j.
    u = torch.ones(tau.shape[0], 1, dtype=dtype, device=device)
    for b in range(num_segments - 1):
        # Does the open chunk j..b still fit under each threshold?
        fits = (chunk_sums[: b + 1, b].unsqueeze(0) <= tau.unsqueeze(1)).to(dtype)
        cut = p[b] * (u * fits).sum(dim=1, keepdim=True)
        u = torch.cat([u * (1.0 - p[b]), cut], dim=1)

    # Final chunk j..N-1: no trailing cut to pay for.
    fits = (chunk_sums[:, num_segments - 1].unsqueeze(0) <= tau.unsqueeze(1)).to(dtype)
    cdf = (u * fits).sum(dim=1)  # F(tau_r)

    shifted = torch.cat([torch.zeros(1, dtype=dtype, device=device), cdf[:-1]])
    return (tau * (cdf - shifted)).sum()


# ---------------------------------------------------------------------------
# SegmentCombiner
# ---------------------------------------------------------------------------

class SegmentCombiner(nn.Module):
    """
    Combine per-segment GSNR values into an end-to-end GSNR.

    Stateless and parameter-free: `forward()` takes only the physics
    (per-segment GSNRs and per-boundary regenerator probabilities). There is
    no annealing knob to keep in sync, because the fold is exact.

    Historical note: earlier versions folded chunks through a
    `soft_max_temperature`, first fixed at construction and never annealed,
    then annealed per call by the training loop. Both were approximations of
    a max with an overshoot the caller had to keep small enough for the
    "regenerating helps" physics invariant to survive — at t=0.5 the
    approximation's floor swamped real per-segment noise and made every
    multi-segment path look worse than not regenerating at all. The exact
    fold removes the knob and the invariant with it; see
    docs/investigations/CHANGELOG.md's Phase 1c corrections for the measured
    numbers, and docs/investigations/regen_over_provisioning.md for the
    replacement's derivation.
    """

    def forward(
        self,
        segment_gsnrs_db: List[torch.Tensor],
        regen_probs_at_boundaries: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        segment_gsnrs_db:
            List of N scalar tensors (GSNR in dB for each transparent segment).
        regen_probs_at_boundaries:
            List of N-1 scalar tensors with regenerator probabilities p ∈ (0,1)
            at each inter-segment boundary.

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

        if num_segments == 1:
            return linear_noise_to_db(n[0]).float()

        p = torch.stack([pr.double() for pr in regen_probs_at_boundaries])  # (N-1,)

        is_hard = bool(((p == 0.0) | (p == 1.0)).all().item())

        if is_hard:
            # No real randomness: exactly one partition has nonzero
            # probability. Cut deterministically at p==1 boundaries and take
            # the max over the resulting REAL chunks -- O(N), the only
            # tractable path for very long chains (this file's own
            # float64-precision test uses thousands of segments, all p=0). A
            # genuinely separate code path from the fractional case below,
            # not a special-cased tolerance on the same formula -- see
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
            effective_noise = torch.stack(chunk_sums).max()
        else:
            # Genuinely fractional boundary probabilities: the exact
            # expectation of the max chunk noise, via the polynomial-time
            # dynamic program in the module docstring. Polynomial, but
            # O(N^4) elementwise work -- tractable for any real path and
            # nowhere near tractable at the N=8000 the hard branch above
            # handles, so fail loudly rather than hang if something upstream
            # ever routes a path that long through fractional probabilities.
            if num_segments > MAX_EXACT_FOLD_SEGMENTS:
                raise ValueError(
                    f"SegmentCombiner's exact fractional-probability fold is "
                    f"O(N^4) and is capped at {MAX_EXACT_FOLD_SEGMENTS} "
                    f"segments; this path has {num_segments} (real topology "
                    f"paths measure well under the cap, see "
                    f"docs/investigations/fold_formula_scalability.md). If "
                    f"this is a real path, something upstream changed; if "
                    f"intentional, raise the cap deliberately and check the "
                    f"memory cost first."
                )
            effective_noise = _expected_max_chunk_noise(n, p)

        return linear_noise_to_db(effective_noise).float()
