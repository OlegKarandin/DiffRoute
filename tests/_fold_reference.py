"""
Shared brute-force reference oracle for `SegmentCombiner`'s exact fold.

Enumerates every hard partition of a path into regenerator-bounded chunks
and weights each by its true realization probability under independent
Bernoulli(p_i) boundary decisions -- the exact quantity
`diffopt.qot.segment_combiner._expected_max_chunk_noise`'s polynomial-time
DP computes without enumeration (see that module's docstring). This module
exists so `tests/test_segment_combiner.py` has one shared, structurally
simple (hence trustworthy as ground truth) implementation of that same
quantity, instead of the two near-duplicate inline copies it carried before
the exact DP existed.

Test-only: `exact_expectation_by_enumeration` is O(2^(N-1)) and is guarded
accordingly (see its docstring). Never import this module from production
code.
"""

from __future__ import annotations

from typing import Sequence


def hard_chunk_max(noises: Sequence[float], cuts: Sequence[int]) -> float:
    """Chunk `noises` at the boundaries where `cuts[i] == 1`, sum within
    each chunk, and return the largest chunk sum.

    `noises` are per-segment linear noise values (not dB). `cuts` has one
    entry per inter-segment boundary (`len(noises) - 1` entries):
    `cuts[i] == 1` means a regenerator boundary immediately after segment
    `i` -- the chunk ending at segment `i` completes and a new chunk starts
    at segment `i + 1`. `cuts[i] == 0` means a passthrough boundary --
    segment `i + 1` extends the currently-open chunk.
    """
    if len(cuts) != len(noises) - 1:
        raise ValueError(
            f"expected {len(noises) - 1} cut decisions for {len(noises)} "
            f"segments, got {len(cuts)}"
        )
    chunks = []
    current = noises[0]
    for i, cut in enumerate(cuts):
        n_next = noises[i + 1]
        if cut == 1:
            chunks.append(current)
            current = n_next
        else:
            current += n_next
    chunks.append(current)
    return max(chunks)


def exact_expectation_by_enumeration(
    noises: Sequence[float], probs: Sequence[float]
) -> float:
    """Exact `E[max over chunks]` by brute-force enumeration of all
    `2^(N-1)` regen-decision configurations over the `N - 1` boundaries,
    each weighted by its true realization probability (`prod p_i` over
    regen boundaries, `prod (1 - p_i)` over passthrough boundaries, per
    boundary) under independent Bernoulli(p_i) decisions.

    This is the same quantity `SegmentCombiner`'s DP
    (`_expected_max_chunk_noise`) computes in polynomial time; this
    function computes it the structurally simple, exponential way, as a
    reference oracle for tests.

    Test-only: `2^(N-1)` configurations. Guarded to `len(probs) <= 12`
    (4096 configurations) so a test typo cannot silently enumerate a
    production-scale path (real topology paths run 16-19 segments, see
    docs/investigations/fold_formula_scalability.md) and hang a test run.
    """
    assert len(probs) <= 12, (
        f"exact_expectation_by_enumeration is O(2^(N-1)); refusing to "
        f"enumerate {len(probs)} boundaries ({2 ** len(probs)} "
        f"configurations) -- this oracle is test-only, never call it with "
        f"a production-scale path"
    )
    if len(probs) != len(noises) - 1:
        raise ValueError(
            f"expected {len(noises) - 1} boundary probabilities for "
            f"{len(noises)} segments, got {len(probs)}"
        )

    num_boundaries = len(probs)
    total = 0.0
    for mask in range(2 ** num_boundaries):
        cuts = [(mask >> i) & 1 for i in range(num_boundaries)]
        weight = 1.0
        for cut, p in zip(cuts, probs):
            weight *= p if cut == 1 else (1.0 - p)
        if weight == 0.0:
            continue
        total += weight * hard_chunk_max(noises, cuts)
    return total
