"""
Unit tests for diffopt.qot.segment_combiner.

Test IDs:
  1. Single segment → identity
  2. Two segments, p=1 → worst segment (max noise)
  3. Two segments, p=0 → noise adds
  4. Monotonicity: p ↑ → GSNR ↑ when regen helps
  5. Gradient: ∂gsnr/∂p > 0 when regen helps
  6. Three segments, two boundaries → brute-force check
  7. Numerical stability: extreme GSNR values, no NaN/Inf
  8. Accumulation precision: float64 internal, float32 return
  9. GSNR clamp: inputs outside [-5, 35] dB behave as clamped
  10-13. Multi-segment chunking (docs/investigations/regen_over_provisioning.md):
      max-over-chunks physics, chunk completion before re-accumulation,
      zero marginal value of a redundant regenerator, exact sum at p=0
  14. Fractional p vs a hand-derived expectation (round-2 regression guard,
      docs/investigations/regen_over_provisioning.md)
  15. Gradient regression: wrong-sign gradient when a boundary probability
      saturates to exactly 1.0 alongside a fractional boundary (final-review
      fix wave Fix 1)
  16. Fractional fold with 3+ boundaries (8 partitions) vs brute-force
      enumeration (final-review fix wave Fix 5.1)
  17. The path-length tractability guard actually raises ValueError
      (final-review fix wave Fix 5.2)
"""

import math
import pytest
import torch

from diffopt.qot.segment_combiner import (
    MAX_EXACT_FOLD_SEGMENTS,
    SegmentCombiner,
    _expected_max_chunk_noise,
    _expected_max_chunk_noise_batched,
    db_to_linear_noise,
    linear_noise_to_db,
)
from tests._fold_reference import exact_expectation_by_enumeration, hard_chunk_max


def t(val: float, requires_grad: bool = False) -> torch.Tensor:
    """Convenience: scalar float32 tensor."""
    return torch.tensor(val, dtype=torch.float32, requires_grad=requires_grad)


# ---------------------------------------------------------------------------
# Test 1: single segment → identity
# ---------------------------------------------------------------------------

def test_single_segment_identity():
    combiner = SegmentCombiner()
    gsnr = t(15.0)
    result = combiner([gsnr], [])
    assert torch.isclose(result, gsnr, atol=1e-4), (
        f"Expected {gsnr.item():.4f} dB, got {result.item():.4f} dB"
    )


# ---------------------------------------------------------------------------
# Test 2: two segments, p=1 → output ≈ min(gsnr_1, gsnr_2)
# ---------------------------------------------------------------------------

def test_two_segments_p1_worst_segment():
    """
    With p=1 fully regenerated, accumulated_noise = max(n1, n2).
    Higher noise → lower GSNR → output = min(gsnr_1, gsnr_2).
    """
    combiner = SegmentCombiner()
    gsnr_1 = t(10.0)
    gsnr_2 = t(20.0)
    p = t(1.0)

    result = combiner([gsnr_1, gsnr_2], [p])

    # Analytical: with p=1 the max chunk is the worst segment, min(10, 20) = 10 dB
    expected = min(gsnr_1.item(), gsnr_2.item())
    assert abs(result.item() - expected) < 0.1, (
        f"Expected ~{expected:.2f} dB (worst segment), got {result.item():.4f} dB"
    )


# ---------------------------------------------------------------------------
# Test 3: two segments, p=0 → noise adds
# ---------------------------------------------------------------------------

def test_two_segments_p0_noise_adds():
    """
    With p=0 no regeneration, total noise = n1 + n2.
    GSNR_total: 1/10^(G/10) = 1/10^(G1/10) + 1/10^(G2/10)
    """
    combiner = SegmentCombiner()
    gsnr_1 = t(15.0)
    gsnr_2 = t(18.0)
    p = t(0.0)

    result = combiner([gsnr_1, gsnr_2], [p])

    n1 = 10 ** (-15.0 / 10)
    n2 = 10 ** (-18.0 / 10)
    expected = -10 * math.log10(n1 + n2)

    assert abs(result.item() - expected) < 1e-3, (
        f"Expected {expected:.4f} dB (noise addition), got {result.item():.4f} dB"
    )


# ---------------------------------------------------------------------------
# Test 4: monotonicity — GSNR increases as p increases (when regen helps)
# ---------------------------------------------------------------------------

def test_monotonicity_with_regen():
    """
    When the first segment is much worse (low GSNR), regeneration at the
    boundary resets noise. As p increases from 0→1, the output GSNR should
    be non-decreasing.
    """
    combiner = SegmentCombiner()
    gsnr_1 = t(5.0)   # bad segment
    gsnr_2 = t(25.0)  # good segment

    p_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    results = [combiner([gsnr_1, gsnr_2], [t(p)]).item() for p in p_values]

    for i in range(1, len(results)):
        assert results[i] >= results[i - 1] - 1e-4, (
            f"Non-monotonic at p={p_values[i]}: {results[i]:.4f} < {results[i-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 5: gradient ∂gsnr/∂p > 0 when regen is beneficial
# ---------------------------------------------------------------------------

def test_gradient_wrt_regen_prob():
    """
    When segment 1 has poor GSNR (high noise), regeneration helps.
    ∂gsnr/∂p should be positive: regen gives max(n1,n2) = n1 while no-regen
    gives n1+n2 > n1, so regen strictly reduces noise.
    """
    combiner = SegmentCombiner()
    gsnr_1 = t(5.0)   # bad first segment
    gsnr_2 = t(25.0)
    p = t(0.5, requires_grad=True)

    result = combiner([gsnr_1, gsnr_2], [p])
    result.backward()

    assert p.grad is not None, "Gradient not computed for regen_prob"
    assert p.grad.item() > 0.0, (
        f"Expected positive gradient (regen helps), got {p.grad.item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 5b: same gradient invariant, at the noise scale production actually has
# ---------------------------------------------------------------------------

def test_gradient_wrt_regen_prob_at_production_noise_scale():
    """
    Test 5 uses 5 dB segments (noise 0.316). Real segments are nothing like
    that: `segment_path` cuts at every degree>=3 node, so segments run
    ~180-210 km on both german_17 and ind_132, giving ~26 dB and noise
    ~0.0023-0.0027 — roughly 100x smaller.

    The historical soft-max fold's overshoot was `temperature * ln2`, which
    is ABSOLUTE in linear noise units and did not shrink with the operands.
    At temperature=0.01 that overshoot is 0.00693 — larger than the noise
    itself — so the folded value exceeded `a + b` and the combiner reported
    that regenerating makes the path *worse*. See
    docs/investigations/regen_placement_not_concentrating.md. The fold is an
    exact max now, so no such scale-dependent floor exists; this test keeps
    watching the operating point where the old one broke.

    Two 26 dB segments: no-regen noise is 2 * 0.00251 = 0.00501 (23.0 dB),
    regen noise is 0.00251 (26.0 dB). Regen strictly helps, so
    d(gsnr)/dp must be positive here exactly as it is in test 5.
    """
    combiner = SegmentCombiner()
    gsnr_1 = t(26.0)
    gsnr_2 = t(26.0)
    p = t(0.5, requires_grad=True)

    result = combiner([gsnr_1, gsnr_2], [p])
    result.backward()

    assert p.grad is not None, "Gradient not computed for regen_prob"
    assert p.grad.item() > 0.0, (
        f"Expected positive gradient (regen helps at any noise scale), "
        f"got {p.grad.item():.6f} — the fold has acquired a noise-scale-"
        f"dependent error floor, as the old soft-max one had"
    )


def test_regen_never_increases_noise_across_noise_scales():
    """
    The 'regen helps' invariant must hold at every operating point, not just
    the one the other tests happen to sample. Sweeps segment GSNR across the
    full range the QoT model emits (~6.5-20 dB training range, and the ~26 dB
    short segments the pipeline actually produces) and asserts a fully
    regenerated path is never worse than a transparent one.
    """
    combiner = SegmentCombiner()
    for gsnr_db in [5.0, 10.0, 15.0, 20.0, 23.0, 26.0, 30.0]:
        transparent = combiner([t(gsnr_db), t(gsnr_db)], [t(0.0)])
        regenerated = combiner([t(gsnr_db), t(gsnr_db)], [t(1.0)])
        assert regenerated.item() > transparent.item(), (
            f"At {gsnr_db} dB per segment (noise {10 ** (-gsnr_db / 10):.5f}): "
            f"regenerated path {regenerated.item():.4f} dB is not better than "
            f"transparent {transparent.item():.4f} dB"
        )


# ---------------------------------------------------------------------------
# Test 6: three segments, two boundaries — brute-force check
# ---------------------------------------------------------------------------

def test_three_segments_two_boundaries():
    """
    With 2 boundaries each with probability p, compare against brute-force
    enumeration of 4 binary configs weighted by (p^k * (1-p)^(2-k)), each
    config's noise computed as the TRUE max over the chunks that
    configuration's cuts produce (docs/investigations/
    regen_over_provisioning.md) — i.e. partition at every regen=1 boundary,
    sum noise within each resulting chunk, then take the max chunk. That is
    NOT the same as a single accumulator that takes a max at a cut boundary
    and keeps adding to it afterwards (config (1,0) here — cut at boundary
    0 only — is exactly the case where those two disagree: the correct
    chunks are {0},{1,2}, giving max noise 0.1, not the accumulate-then-max
    recurrence's 0.1316).

    This test previously used that buggy accumulator as its own brute-force
    reference and passed only on a 0.5 dB tolerance (actual deviation
    ~0.37-0.5 dB depending on the fold under test) rather than genuine
    accuracy — the review that caught the exact-partition fold's own
    structural bug (round 2 of this fix, see the module's docstring) also
    caught this latent bug in the test's reference. With the reference
    fixed to true chunk-max physics, and the shipped fold now an exact
    expectation of that same chunk max, the two agree to double precision —
    hence the correspondingly tight tolerance below.
    """
    gsnr_vals = [10.0, 20.0, 15.0]
    p_val = 0.3

    combiner = SegmentCombiner()
    g = [t(v) for v in gsnr_vals]
    p = [t(p_val), t(p_val)]

    result = combiner(g, p).item()

    # Brute force: enumerate regen decisions (0=passthrough, 1=regen) at each
    # boundary via the shared oracle (tests/_fold_reference.py), weighting
    # each config by its true realization probability, using the TRUE max
    # over the chunks its cuts produce.
    noises = [10 ** (-v / 10) for v in gsnr_vals]
    weighted_noise = exact_expectation_by_enumeration(noises, [p_val, p_val])

    expected = -10 * math.log10(weighted_noise)

    assert result == pytest.approx(expected, abs=1e-3), (
        f"3-segment check: expected {expected:.6f} dB, got {result:.6f} dB"
    )


def test_fractional_p_matches_hand_derived_expectation():
    """Round-2 regression guard, docs/investigations/regen_over_provisioning.md:
    a first attempt at generalizing soft_max to N-way chunking put each
    chunk's realization probability INSIDE a shared-temperature exponential.
    Since chunk noise differences scaled as O(1/t) (~50-100 at the
    then-production t=0.01) while log-probability differences are O(1),
    probability got swamped and the fold silently collapsed toward the
    no-regen value regardless of p -- reintroducing a shaped version of the
    original bug.

    Reviewer's exact counterexample: two 26 dB segments (the ~180-210 km
    real-topology segment length), p=0.5. That first attempt gave 23.02 dB;
    the shipped soft-max fold gave 24.229065 dB, still carrying the
    soft-max overshoot. Hand-derived truth -- this is a single boundary, so
    the exact expectation is just `(1-p)*(n1+n2) + p*max(n1,n2)` -- is
    24.239090 dB, which the exact fold now returns.
    """
    combiner = SegmentCombiner()
    gsnr_1 = t(26.0)
    gsnr_2 = t(26.0)
    p = t(0.5)

    result = combiner([gsnr_1, gsnr_2], [p]).item()

    n = 10 ** (-26.0 / 10)
    expected_noise = 0.5 * (2 * n) + 0.5 * max(n, n)
    expected = -10.0 * math.log10(expected_noise)  # 24.239090 dB

    assert result == pytest.approx(expected, abs=1e-3), (
        f"fractional p: expected {expected:.6f} dB, got {result:.6f} dB "
        f"(a ~1 dB-scale gap here is exactly the shape of the round-2 "
        f"regression this test guards against)"
    )


# ---------------------------------------------------------------------------
# Test 7: numerical stability — extreme values, no NaN/Inf
# ---------------------------------------------------------------------------

def test_numerical_stability():
    """
    Extreme GSNR values and near-binary probabilities should not produce NaN/Inf.
    """
    combiner = SegmentCombiner()

    extreme_cases = [
        ([t(0.0), t(30.0)], [t(1e-6)]),
        ([t(0.0), t(30.0)], [t(1.0 - 1e-6)]),
        ([t(30.0), t(0.0)], [t(1e-6)]),
        ([t(30.0), t(0.0)], [t(1.0 - 1e-6)]),
        ([t(0.0), t(0.0)],  [t(0.5)]),
        ([t(30.0), t(30.0)], [t(0.5)]),
    ]

    for gsnrs, probs in extreme_cases:
        result = combiner(gsnrs, probs)
        assert torch.isfinite(result), (
            f"Got non-finite result {result.item()} for "
            f"gsnrs={[g.item() for g in gsnrs]}, p={[p.item() for p in probs]}"
        )


# ---------------------------------------------------------------------------
# Test 8: accumulation precision — docs/architecture/invariants.md:
# "accumulates noise internally in float64 and casts back to float32 on
# return. Do not move this to float32 — overflow on long noisy paths is
# real."
# ---------------------------------------------------------------------------

def test_accumulation_is_float64_not_float32():
    """docs/architecture/invariants.md: 'accumulates noise internally in
    float64 and casts back to float32 on return. Do not move this to
    float32 — overflow on long noisy paths is real.'

    A 40-segment chain (the brief's original guess) does not actually
    discriminate here: per-segment linear noise is bounded to
    [10**-3.5, 10**0.5] by the [-5, 35] dB clamp, so accumulation is at
    most O(N) — nowhere near float32's overflow point — and float32's
    compounding rounding error over only 40 additions is far below any
    reasonable dB tolerance (measured ~1e-7 dB with float64 forced down
    to float32, no distinguishable divergence at N=40). Empirically,
    forcing float32 needs a long chain (thousands of additions, verified
    at N=8000 giving ~5e-4 dB divergence from the float64 answer,
    swamping the ~2e-7 dB the real float64 path measures) before
    compounding rounding error becomes visible in the output dB value.

    One -5 dB (max per-segment noise) segment followed by 8000 35 dB
    (min per-segment noise) segments, no regeneration anywhere (p=0), so
    noise is a plain running sum the whole way — the exact scenario the
    float64-accumulation choice exists to protect.
    """
    combiner = SegmentCombiner()
    n_segments = 8000
    gsnrs = [t(-5.0)] + [t(35.0) for _ in range(n_segments)]
    regen_probs = [t(0.0) for _ in range(n_segments)]

    result = combiner(gsnrs, regen_probs)

    total_noise = 10 ** (5.0 / 10.0) + n_segments * 10 ** (-35.0 / 10.0)
    expected = -10.0 * math.log10(total_noise)
    assert result.item() == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# Test 9: GSNR clamp — docs/architecture/invariants.md: "GSNR inputs are
# clamped to [-5, 35] dB before conversion to linear noise. This is
# intentional."
# ---------------------------------------------------------------------------

def test_gsnr_inputs_are_clamped_to_the_documented_range():
    """Values far outside [-5, 35] dB must behave identically to the
    clamped boundary values — not blow up (e.g. 10**(50/10) in linear
    noise) or otherwise diverge."""
    combiner = SegmentCombiner()

    wild = combiner([t(-50.0), t(80.0)], [t(0.0)])
    clamped = combiner([t(-5.0), t(35.0)], [t(0.0)])
    assert wild.item() == pytest.approx(clamped.item(), abs=1e-4)


# ---------------------------------------------------------------------------
# Tests 10-13: multi-segment chunking — docs/investigations/
# regen_over_provisioning.md. A regenerator rebuilds the signal, so the
# path splits into independent CHUNKS at regenerated boundaries, and
# end-to-end noise is the max over chunks (not a running accumulator that a
# max is occasionally applied to, which is only correct when every
# post-regenerator chunk happens to be a single segment — exactly the blind
# spot the tests below were added to close). All five segments below carry
# identical noise, 0.005 linear (-10*log10(0.005) = 23.0103 dB), so a
# chunk's noise is exactly (segment count in chunk) * 0.005 and the
# analytically-derived expected values below follow directly from that.
# ---------------------------------------------------------------------------

FIVE_SEG_NOISE = 0.005
FIVE_SEG_DB = -10.0 * math.log10(FIVE_SEG_NOISE)  # 23.0103 dB


def five_equal_segments():
    return [t(FIVE_SEG_DB) for _ in range(5)]


def test_multi_segment_chunks_equal_max_over_chunks():
    """p=1 at boundaries 0 and 2 (regen after segment 0 and after segment
    2), p=0 elsewhere -> chunks {0}, {1,2}, {3,4}. The largest chunk has 2
    segments, so effective noise is 2*0.005 and GSNR = -10*log10(0.01) =
    20.000 dB.

    Under the pre-fix single-accumulator fold this measured 18.229 dB (a
    1.771 dB error) because the accumulator kept adding to a stale running
    max instead of restarting a fresh chunk at each regenerated boundary.
    The tolerance used to be loosened to 0.05 dB to absorb the soft-max
    overshoot on this fixture's tied chunks (~0.03 dB, see
    docs/investigations/regen_over_provisioning.md); the fold takes an exact
    max now, ties included, so the answer is exact.

    The fold is exact, so the only residual here is the float32 return cast:
    one ulp at ~20.0 dB is ~1.9e-6, matching the same quantization floor
    `test_chunk_completes_before_next_accumulates` documents. The 1e-5
    tolerance sits just above that floor -- anything larger would be real
    fold error, not rounding. (This still meaningfully checks the fold: the
    historical single-accumulator bug this fixture guards against was off
    by ~1.771 dB, five orders of magnitude above this tolerance.)
    """
    combiner = SegmentCombiner()
    gsnrs = five_equal_segments()
    probs = [t(1.0), t(0.0), t(1.0), t(0.0)]  # cut after seg 0, cut after seg 2

    result = combiner(gsnrs, probs).item()
    expected = -10.0 * math.log10(2 * FIVE_SEG_NOISE)  # 20.000 dB

    assert result == pytest.approx(expected, abs=1e-5), (
        f"chunks {{0}},{{1,2}},{{3,4}}: expected ~{expected:.4f} dB "
        f"(max chunk = 2 segments), got {result:.4f} dB"
    )


def test_chunk_completes_before_next_accumulates():
    """p=1 at boundary 1 only (regen after segment 1), p=0 elsewhere ->
    chunks {0,1}, {2,3,4}. The larger chunk has 3 segments, so effective
    noise is 3*0.005 and GSNR = -10*log10(0.015) = 18.2391 dB.

    Under the pre-fix single-accumulator fold this measured 16.990 dB (a
    1.249 dB error): the accumulator applied a max at the boundary but then
    kept ADDING segments 2-4 onto that max instead of starting a fresh
    chunk, so it never saw that {2,3,4} accumulates to 3 segments' worth of
    noise before being compared to {0,1}.

    The fold is exact, so the only residual here is the float32 return cast:
    one ulp at ~18.24 dB is ~1.9e-6, and the measured deviation is ~6.5e-7.
    The 1e-5 tolerance sits just above that quantization floor -- anything
    larger would be real fold error, not rounding.
    """
    combiner = SegmentCombiner()
    gsnrs = five_equal_segments()
    probs = [t(0.0), t(1.0), t(0.0), t(0.0)]  # cut after seg 1 only

    result = combiner(gsnrs, probs).item()
    expected = -10.0 * math.log10(3 * FIVE_SEG_NOISE)  # 18.2391 dB

    assert result == pytest.approx(expected, abs=1e-5), (
        f"chunks {{0,1}},{{2,3,4}}: expected ~{expected:.4f} dB "
        f"(max chunk = 3 segments), got {result:.4f} dB"
    )


def test_redundant_regenerator_has_zero_marginal_value():
    """This is the property that encodes *why* the pre-fix fold caused
    over-provisioning (docs/investigations/regen_over_provisioning.md):
    under correct chunk-max physics, adding a regenerator on top of one
    that already makes the max-chunk no bigger is worth exactly 0 dB, so
    any positive lambda_regen evicts a redundant node immediately. The
    pre-fix fold reported this redundant regenerator as worth +1.233 dB
    (16.990 -> 18.223), giving it a fake positive marginal value that only
    a large enough price could overcome.

    Boundary 1 alone -> chunks {0,1},{2,3,4}, max chunk = 3 segments.
    Boundaries 0 and 1 -> chunks {0},{1},{2,3,4}, max chunk is STILL 3
    segments (the extra cut at boundary 0 only shrinks the already-smaller
    chunk), so the two configurations must be equal.
    """
    combiner = SegmentCombiner()
    gsnrs = five_equal_segments()

    boundary_1_alone = combiner(gsnrs, [t(0.0), t(1.0), t(0.0), t(0.0)]).item()
    boundary_0_and_1 = combiner(gsnrs, [t(1.0), t(1.0), t(0.0), t(0.0)]).item()

    delta = abs(boundary_0_and_1 - boundary_1_alone)
    assert delta < 1e-9, (
        f"redundant regenerator should add ~0 dB, got {boundary_1_alone:.4f} "
        f"-> {boundary_0_and_1:.4f} dB (delta {delta:.4f} dB)"
    )


def test_zero_regen_probability_is_exact_sum():
    """All boundary probabilities exactly 0: the fold must reduce to a plain
    sum of segment noise, exactly, with no residual contribution from a
    "nothing happened yet" zero sentinel.

    This guards the `any_regen`-style sentinel problem the original
    two-state design in docs/investigations/regen_over_provisioning.md
    called out explicitly: any construction where such a zero is blended
    through a soft_max at loose temperature picks up a ~6% scale-normalised
    excess (soft_max(0, C, t=0.5) = C*1.0635 != C). The exact fold has no
    sentinel and no smoothing -- at p=0 the only partition with nonzero
    probability is the uncut one, whose single chunk is the whole path --
    but the test is kept as a standing regression guard.
    """
    combiner = SegmentCombiner()
    gsnrs = [t(15.0), t(9.0), t(20.0), t(12.5)]
    n_segments = len(gsnrs)
    probs = [t(0.0)] * (n_segments - 1)

    result = combiner(gsnrs, probs).item()

    total_noise = sum(10 ** (-g.item() / 10) for g in gsnrs)
    expected = -10.0 * math.log10(total_noise)

    assert result == pytest.approx(expected, abs=1e-4), (
        f"all p=0: expected exact sum {expected:.4f} dB, got {result:.4f} dB"
    )


# ---------------------------------------------------------------------------
# Test 15: gradient regression — wrong-sign gradient when a boundary
# probability saturates to exactly 1.0 alongside a fractional boundary
# (final-review fix wave Fix 1)
# ---------------------------------------------------------------------------

def test_gradient_correct_sign_when_one_boundary_saturates_to_one():
    """Regression guard for the wrong-sign gradient bug: the fractional
    branch used to compute each partition's probability via
    `logp = log(p.clamp_min(1e-300))`, `log1mp = log((1-p).clamp_min(1e-300))`.
    When a boundary p is exactly 1.0 (reachable: sigmoid(logit/tau)
    saturates to exactly 1.0 in float32 well within the production logit
    range), `(1-p)=0.0` got clamped to 1e-300, and `clamp_min`'s gradient
    is zero in the clamped region -- so `d(log1mp)/dp` came out as 0
    instead of the true (very large) value, flipping the sign of the
    gradient on `regen_logits` at that boundary.

    This exact mixed configuration -- one boundary exactly hard (p=1.0),
    another fractional (p=0.4) -- is what hides the bug: the forward
    computation was already correct at these inputs (a value-only test
    passes both before and after this fix), so only `.backward()` exposes
    it. Pre-fix this measured a gradient of exactly -10/ln(10) =
    -4.342944... (independent of segment values or the other boundary's
    p -- a structural artifact of the clamp, not real physics, since the
    missing term is exactly the gradient contribution from the
    now-clamped-away uncut-at-this-boundary partitions). Post-fix it
    matches finite-difference truth.
    """
    combiner = SegmentCombiner()
    gsnr = [t(26.0), t(26.0), t(26.0)]  # production segment-length noise scale
    p0 = t(1.0, requires_grad=True)  # saturated boundary -- the vertex where the bug fires
    p1 = t(0.4)  # fractional boundary

    result = combiner(gsnr, [p0, p1])
    result.backward()
    grad = p0.grad.item()

    assert grad > 0.0, (
        f"Expected positive gradient (regen should never look harmful), "
        f"got {grad:.6f} -- this is the same class of bug (wrong-sign "
        f"gradient on regen placement) this entire investigation exists "
        f"to eliminate"
    )

    # Finite-difference truth via a 2nd-order one-sided (backward) stencil
    # -- p0 cannot exceed 1.0 physically, so a one-sided estimator is used:
    # f'(x) ~= (3f(x) - 4f(x-h) + f(x-2h)) / (2h)
    eps = 1e-2
    with torch.no_grad():
        r0 = combiner(gsnr, [t(1.0), p1]).item()
        r1 = combiner(gsnr, [t(1.0 - eps), p1]).item()
        r2 = combiner(gsnr, [t(1.0 - 2 * eps), p1]).item()
    fd = (3 * r0 - 4 * r1 + r2) / (2 * eps)

    assert grad == pytest.approx(fd, abs=1e-3), (
        f"autograd gradient {grad:.6f} does not match finite-difference "
        f"truth {fd:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 16: fractional fold with 3+ boundary probabilities (8 partitions),
# vs brute-force enumeration (final-review fix wave Fix 5.1). The existing
# fractional-probability tests only exercise 1-2 boundaries (2 or 4
# partitions); this exercises the DP at a wider fan-out.
# ---------------------------------------------------------------------------

def test_fractional_fold_with_three_boundaries_matches_brute_force():
    """4 segments, 3 fractional boundary probabilities -> 8 hard partitions.
    Compares against brute-force enumeration of all 8 regen-decision
    configs, each weighted by its true realization probability, each
    config's noise computed as the TRUE max over the chunks its cuts
    produce (mirrors test_three_segments_two_boundaries's brute-force
    reference, generalized to 3 boundaries).
    """
    gsnr_vals = [12.0, 22.0, 9.0, 18.0]
    p_vals = [0.2, 0.6, 0.4]

    combiner = SegmentCombiner()
    g = [t(v) for v in gsnr_vals]
    p = [t(v) for v in p_vals]

    result = combiner(g, p).item()

    noises = [10 ** (-v / 10) for v in gsnr_vals]
    weighted_noise = exact_expectation_by_enumeration(noises, p_vals)

    expected = -10 * math.log10(weighted_noise)

    assert result == pytest.approx(expected, abs=1e-3), (
        f"4-segment/3-boundary check: expected {expected:.6f} dB, "
        f"got {result:.6f} dB"
    )


# ---------------------------------------------------------------------------
# Test 17: the path-length tractability guard actually raises ValueError
# (final-review fix wave Fix 5.2) -- the guard exists in the code but was
# previously untested.
# ---------------------------------------------------------------------------

def test_too_many_segments_raises_value_error():
    """A path longer than the fractional fold's segment cap must raise
    ValueError rather than silently running an O(N^4) fold at a length
    nothing upstream can legitimately produce. Deliberately matched on the
    message text and not on the cap's current value, so raising the cap
    later does not break this test."""
    combiner = SegmentCombiner()
    n_segments = MAX_EXACT_FOLD_SEGMENTS + 1
    gsnrs = [t(15.0) for _ in range(n_segments)]
    probs = [t(0.3) for _ in range(n_segments - 1)]

    with pytest.raises(ValueError, match="capped at"):
        combiner(gsnrs, probs)


# ---------------------------------------------------------------------------
# Tests 18-22: DP-vs-oracle validation using the shared brute-force reference
# (tests/_fold_reference.py), gradcheck, the raised segment-count cap, and
# location-aware gradients. Added alongside the oracle extraction above.
# ---------------------------------------------------------------------------

def _dp_expected_noise(gsnr_db_vals, p_vals):
    """Call directly into `_expected_max_chunk_noise` -- the float64 core
    `SegmentCombiner.forward`'s fractional branch calls -- bypassing
    `forward`'s final `.float()` return cast.

    That cast is fine for production (float32 GSNR is plenty for the
    physics), but it caps the public API's relative precision at ~1e-7,
    which is far too loose to check the DP's own arithmetic against the
    brute-force oracle at the ~1e-12 relative floor float64 rounding
    actually leaves. Comparing the pre-cast linear-domain value instead of
    the post-cast dB value is what makes the tight tolerance below
    meaningful rather than trivially satisfied by the cast's rounding.
    """
    n = torch.tensor([10 ** (-v / 10) for v in gsnr_db_vals], dtype=torch.float64)
    p = torch.tensor(p_vals, dtype=torch.float64)
    return _expected_max_chunk_noise(n, p).item()


def _fractional_fold_double(gsnr_db: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Reproduce `SegmentCombiner.forward`'s fractional branch (clamp ->
    dB-to-linear -> `_expected_max_chunk_noise` -> linear-to-dB) entirely in
    float64, without the final `.float()` cast, so `torch.autograd.gradcheck`'s
    numerical Jacobian has the precision it needs. `gsnr_db` and `p` are
    vector tensors (shape (N,) and (N-1,)) rather than `forward`'s lists of
    scalar tensors, since gradcheck needs single tensors to perturb.
    """
    g_clamped = gsnr_db.clamp(-5.0, 35.0)
    n = db_to_linear_noise(g_clamped.double())
    effective_noise = _expected_max_chunk_noise(n, p.double())
    return linear_noise_to_db(effective_noise)


def test_dp_matches_oracle_across_probability_regimes():
    """DP (`_expected_max_chunk_noise`) vs. the brute-force oracle
    (`exact_expectation_by_enumeration`) across four probability regimes,
    each with a handful of cases up to N=10 segments. Compared in the
    linear-domain float64 expectation both sides compute internally (see
    `_dp_expected_noise`'s docstring for why: the public dB API's float32
    cast is too lossy for a 1e-12 relative check), at 1e-12 RELATIVE
    tolerance -- tight because this is a genuine double-precision arithmetic
    check, not a physics-tolerance one; nothing here is soft-maxed or
    annealed anymore (see module docstring of segment_combiner.py).
    """
    regimes = {
        "near-tied": [
            ([10.0, 10.0, 10.05], [0.5, 0.5]),
            ([12.0, 12.0, 12.0, 12.03, 8.0], [0.4, 0.5, 0.5, 0.6]),
            (
                [15.0, 15.01, 14.99, 15.02, 14.98, 15.0, 15.01, 14.99, 15.0, 15.03],
                [0.5] * 9,
            ),
        ],
        "lopsided": [
            ([0.0, 30.0, 30.0, 30.0], [0.5, 0.3, 0.7]),
            ([-5.0, 34.0, 33.0, 32.0, 31.0, 30.0], [0.2, 0.9, 0.1, 0.5, 0.4]),
            ([0.0] + [30.0] * 9, [0.5] * 9),
        ],
        "saturated-p": [
            ([15.0, 20.0, 10.0, 25.0], [1.0, 0.4, 0.0]),
            ([9.0, 18.0, 27.0, 14.0, 22.0], [0.0, 1.0, 0.5, 1.0]),
            ([10.0, 20.0, 15.0, 25.0, 12.0, 18.0], [1.0, 0.0, 0.5, 1.0, 0.3]),
        ],
        "exactly-tied": [
            ([15.0, 15.0, 15.0, 15.0], [0.5, 0.5, 0.5]),
            ([20.0, 20.0, 20.0, 20.0, 20.0], [0.3, 0.3, 0.3, 0.3]),
            ([20.0] * 10, [0.4] * 9),
        ],
    }
    for regime_name, cases in regimes.items():
        for gsnr_vals, p_vals in cases:
            noises = [10 ** (-v / 10) for v in gsnr_vals]
            dp = _dp_expected_noise(gsnr_vals, p_vals)
            oracle = exact_expectation_by_enumeration(noises, p_vals)
            assert dp == pytest.approx(oracle, rel=1e-12), (
                f"[{regime_name}] DP {dp!r} vs oracle {oracle!r} "
                f"(N={len(gsnr_vals)}, gsnr={gsnr_vals}, p={p_vals})"
            )


def test_gradcheck_fractional_branch_wrt_boundary_probabilities():
    """`torch.autograd.gradcheck` on the fractional branch's float64 core
    (`_fractional_fold_double`) w.r.t. the boundary probabilities `p`. Small
    N (5 segments / 4 boundaries) since gradcheck's numerical Jacobian costs
    one forward/backward pass per input element."""
    gsnr_db = torch.tensor([10.0, 22.0, 15.0, 28.0, 9.0], dtype=torch.float64)
    p = torch.tensor([0.3, 0.6, 0.45, 0.8], dtype=torch.float64, requires_grad=True)

    def f(p_):
        return _fractional_fold_double(gsnr_db, p_)

    assert torch.autograd.gradcheck(f, (p,), eps=1e-6)


def test_gradcheck_fractional_branch_wrt_segment_gsnr():
    """Same as `test_gradcheck_fractional_branch_wrt_boundary_probabilities`,
    w.r.t. the segment GSNR values instead of the boundary probabilities."""
    gsnr_db = torch.tensor(
        [10.0, 22.0, 15.0, 28.0, 9.0], dtype=torch.float64, requires_grad=True
    )
    p = torch.tensor([0.3, 0.6, 0.45, 0.8], dtype=torch.float64)

    def f(g_):
        return _fractional_fold_double(g_, p)

    assert torch.autograd.gradcheck(f, (gsnr_db,), eps=1e-6)


def test_long_path_beyond_old_boundary_cap_now_succeeds():
    """Before Task 1's exact DP, the fractional fold was exponential and
    guarded by `num_boundaries > 20` (see 6110f83's diff to this module's
    docstring); a 25-boundary / 26-segment path used to raise ValueError.
    The DP is polynomial-time, and Task 1 replaced that guard with
    `MAX_EXACT_FOLD_SEGMENTS = 128`, so the same path must now succeed --
    and still respect 'regen helps': pushing one boundary's p toward 1.0
    must not make the result worse (module docstring: cutting a boundary
    splits a chunk into two no-larger pieces, so E[max] can never rise as
    any single p increases).
    """
    combiner = SegmentCombiner()
    n_segments = 26
    assert n_segments - 1 > 20, "this case must exceed the OLD 20-boundary cap"
    gsnrs = [t(15.0 + (i % 3)) for i in range(n_segments)]  # mild variation
    probs = [t(0.4) for _ in range(n_segments - 1)]

    result = combiner(gsnrs, probs)
    assert torch.isfinite(result), f"expected a finite result, got {result.item()}"

    probs_boosted = list(probs)
    probs_boosted[10] = t(0.999)  # push one boundary toward full regen
    result_boosted = combiner(gsnrs, probs_boosted)

    assert result_boosted.item() >= result.item() - 1e-6, (
        f"pushing a boundary's p toward regen should never make the path "
        f"worse: {result.item():.4f} -> {result_boosted.item():.4f} dB"
    )


def test_exact_ties_need_no_special_casing_against_oracle():
    """Construct chunk sums that are EXACTLY tied (symmetric noise +
    symmetric p), so `torch.sort`'s duplicate thresholds inside
    `_expected_max_chunk_noise` are actually exercised (module docstring:
    'a duplicated threshold contributes F(tau_r) - F(tau_{r-1}) = 0 --
    exactly zero'). DP-vs-oracle agreement here is checked at the SAME
    tight 1e-12 relative tolerance as the untied regimes above -- no fudge
    factor needed for duplicate thresholds.
    """
    gsnr_vals = [18.0, 18.0, 18.0, 18.0, 18.0]
    p_vals = [0.5, 0.5, 0.5, 0.5]
    noises = [10 ** (-v / 10) for v in gsnr_vals]

    # Confirm the construction actually produces a tie, concretely, via the
    # shared oracle's own chunking primitive: two different cut patterns,
    # both giving a length-2 max chunk over equal per-segment noise, so
    # their hard chunk-max values are identical by construction.
    tie_a = hard_chunk_max(noises, [1, 0, 1, 0])  # chunks {0},{1,2},{3,4}
    tie_b = hard_chunk_max(noises, [0, 1, 0, 1])  # chunks {0,1},{2},{3,4}
    assert tie_a == pytest.approx(tie_b), (
        "construction should produce an exactly tied largest chunk by "
        "symmetry -- otherwise this isn't testing what it claims to"
    )

    dp = _dp_expected_noise(gsnr_vals, p_vals)
    oracle = exact_expectation_by_enumeration(noises, p_vals)

    assert dp == pytest.approx(oracle, rel=1e-12), (
        f"tied-chunk DP {dp!r} vs oracle {oracle!r} (gsnr={gsnr_vals}, p={p_vals})"
    )


def test_gradients_are_location_aware_not_shared_across_boundaries():
    """Two boundaries with the SAME marginal p but structurally different
    surrounding noise must get numerically DISTINCT gradients w.r.t. their
    own p. This is exactly the property round 2's shared-temperature fold
    destroyed (docs/investigations/regen_over_provisioning.md and the module
    docstring: probability entered a shared exponential and collapsed
    location information); the exact DP has no shared knob, so distinct
    boundaries with equal p should not get equal gradients here.

    The GSNR fixture is deliberately asymmetric, not just distinct-valued:
    boundary 0 sits after segment 0, which has no segment to its left at
    all (it's the path start), while boundary 3 sits after segment 3, which
    has a real segment on both sides (25 dB segment 2 to its left, 20 dB
    segment 4 to its right) -- and segment 4's value is deliberately *not*
    equal to segment 1's (25 dB), so boundary 3's right-hand context doesn't
    mirror boundary 0's. A path that were symmetric around its center (e.g.
    [5, 25, 25, 5] with matching boundary p's) would make grad_b0 == grad_b3
    a structural certainty by reflection symmetry, which would pass this
    assertion for the wrong reason -- it would prove the DP respects
    symmetry, not that it is location-aware. This fixture rules that out:
    the two boundaries have no symmetry relating them, so equal gradients
    here could only happen by coincidence, and the DP producing distinct
    ones is real evidence that it tracks each boundary's actual structural
    position rather than collapsing all p=0.4 boundaries onto one shared
    gradient.
    """
    gsnr_db = torch.tensor([5.0, 25.0, 25.0, 5.0, 20.0], dtype=torch.float64)
    p = torch.tensor([0.4, 0.5, 0.5, 0.4], dtype=torch.float64, requires_grad=True)

    result = _fractional_fold_double(gsnr_db, p)
    result.backward()

    grad_b0 = p.grad[0].item()  # boundary right after the bad (5 dB) segment 0
    grad_b3 = p.grad[3].item()  # boundary right after the bad (5 dB) segment 3

    assert grad_b0 != 0.0 and grad_b3 != 0.0, (
        f"expected nonzero gradients at both boundaries, got {p.grad.tolist()}"
    )
    assert grad_b0 != pytest.approx(grad_b3, rel=1e-6), (
        f"boundaries at different structural positions should get distinct "
        f"gradients even though both have p=0.4: got {grad_b0:.8f} vs "
        f"{grad_b3:.8f}"
    )


# ---------------------------------------------------------------------------
# Batched fold
# ---------------------------------------------------------------------------

def _ragged_case():
    """Four demands of 1, 2, 3 and 5 segments -- deliberately ragged, and
    deliberately not all the same GSNR, so padding that leaked between rows
    would change a value rather than cancel."""
    gsnrs = [[21.0], [26.0, 24.0], [18.0, 26.0, 22.5], [26.0, 26.0, 19.0, 23.0, 25.5]]
    probs = [[], [0.4], [0.7, 0.25], [0.5, 0.1, 0.9, 0.35]]
    return gsnrs, probs


def _pack(gsnrs, probs, requires_grad=False):
    """Pack ragged python lists into the (D, J) / (D, J-1) / (D,) triple."""
    d = len(gsnrs)
    j = max(len(g) for g in gsnrs)
    g_mat = torch.zeros(d, j, dtype=torch.float32)
    p_mat = torch.zeros(d, max(j - 1, 0), dtype=torch.float32)
    counts = torch.tensor([len(g) for g in gsnrs], dtype=torch.long)
    for row, (g, p) in enumerate(zip(gsnrs, probs)):
        g_mat[row, : len(g)] = torch.tensor(g)
        if p:
            p_mat[row, : len(p)] = torch.tensor(p)
    g_mat.requires_grad_(requires_grad)
    p_mat.requires_grad_(requires_grad)
    return g_mat, p_mat, counts


def test_batched_fold_matches_the_per_demand_loop_on_ragged_input():
    """The load-bearing equivalence. 1e-6 dB is five orders below the QoT
    model's 0.1909 dB val RMSE and six below the 0.5 dB constraint margin;
    the two paths differ only in float64 reduction order over
    exactly-zero padding."""
    combiner = SegmentCombiner()
    gsnrs, probs = _ragged_case()

    batched = combiner.forward_batched(*_pack(gsnrs, probs))
    looped = torch.stack([
        combiner([t(v) for v in g], [t(v) for v in p])
        for g, p in zip(gsnrs, probs)
    ])

    assert batched.shape == (4,)
    assert torch.allclose(batched, looped, atol=1e-6)


def test_batched_fold_does_not_leak_padding_between_demands():
    """The 1-segment demand batched next to a 5-segment one must give the
    same answer as that demand alone."""
    combiner = SegmentCombiner()
    gsnrs, probs = _ragged_case()

    together = combiner.forward_batched(*_pack(gsnrs, probs))
    alone = combiner.forward_batched(*_pack(gsnrs[:1], probs[:1]))

    assert torch.allclose(together[:1], alone, atol=1e-6)


def test_batched_fold_single_segment_demand_is_its_own_gsnr():
    """A demand padded from 1 real segment up to J must reduce to
    -10*log10(noise) of that one segment -- the same value forward()'s
    num_segments == 1 early return gives."""
    combiner = SegmentCombiner()
    out = combiner.forward_batched(*_pack([[21.0], [26.0, 24.0, 22.0]], [[], [0.5, 0.5]]))
    assert out[0].item() == pytest.approx(21.0, abs=1e-4)


def test_batched_fold_with_no_boundaries_at_all():
    """J == 1: the boundary loop never runs and boundary_probs is (D, 0).
    No NaN, no special case."""
    combiner = SegmentCombiner()
    out = combiner.forward_batched(*_pack([[21.0], [17.5]], [[], []]))
    assert torch.isfinite(out).all()
    assert out[1].item() == pytest.approx(17.5, abs=1e-4)


def test_batched_fold_gradient_matches_the_per_demand_loop():
    """Elementwise on the boundary probabilities, which is what the
    placement head learns through."""
    combiner = SegmentCombiner()
    gsnrs, probs = _ragged_case()

    g_mat, p_mat, counts = _pack(gsnrs, probs, requires_grad=True)
    combiner.forward_batched(g_mat, p_mat, counts).sum().backward()

    for row, (g, p) in enumerate(zip(gsnrs, probs)):
        for col, value in enumerate(p):
            scalar_p = t(value, requires_grad=True)
            boundaries = [t(v) for v in p]
            boundaries[col] = scalar_p
            combiner([t(v) for v in g], boundaries).backward()
            assert p_mat.grad[row, col].item() == pytest.approx(
                scalar_p.grad.item(), abs=1e-6
            )


def test_batched_fold_puts_no_gradient_on_the_padding():
    """Padded entries are masked to exactly 0 before the DP sees them, so
    they must be gradient-dead. A nonzero grad here means the mask was
    applied after the fold instead of before, and the value is wrong too."""
    combiner = SegmentCombiner()
    gsnrs, probs = _ragged_case()
    g_mat, p_mat, counts = _pack(gsnrs, probs, requires_grad=True)

    combiner.forward_batched(g_mat, p_mat, counts).sum().backward()

    for row, count in enumerate(counts.tolist()):
        assert torch.all(g_mat.grad[row, count:] == 0.0)
        assert torch.all(p_mat.grad[row, max(count - 1, 0):] == 0.0)


def test_batched_fold_rejects_a_batch_over_the_segment_cap():
    """The cap now applies to the BATCH maximum, because every demand is
    padded to it. The message must say so -- otherwise someone reads
    'this path has 129 segments' and goes looking for a 129-segment demand
    that does not exist."""
    combiner = SegmentCombiner()
    j = MAX_EXACT_FOLD_SEGMENTS + 1
    with pytest.raises(ValueError, match="BATCH maximum"):
        combiner.forward_batched(
            torch.full((2, j), 20.0),
            torch.full((2, j - 1), 0.5),
            torch.tensor([j, 2], dtype=torch.long),
        )


def test_batched_fold_rejects_zero_num_segments():
    """num_segments[d] == 0 must fail loud, not silently return +inf dB for
    that demand (an all-masked row has zero max-chunk-noise, and
    -10*log10(0) is +inf -- a fail-open failure that clears any margin
    constraint). This is currently unreachable from the production
    pipeline (segment_path() always yields >= 1 segment), but the guard
    should still turn it into a clean ValueError instead of a bare
    IndexError deep in the kernel."""
    combiner = SegmentCombiner()
    with pytest.raises(ValueError, match=r"num_segments must lie in \[1, 3\]"):
        combiner.forward_batched(
            torch.full((2, 3), 20.0),
            torch.full((2, 2), 0.5),
            torch.tensor([3, 0], dtype=torch.long),
        )


def test_scalar_dp_entry_point_is_a_d1_slice_of_the_batched_kernel():
    """There is one implementation of the recurrence. This pins that."""
    n = db_to_linear_noise(torch.tensor([12.0, 22.0, 9.0, 18.0]).double())
    p = torch.tensor([0.2, 0.6, 0.4]).double()

    assert torch.equal(
        _expected_max_chunk_noise(n, p),
        _expected_max_chunk_noise_batched(n.unsqueeze(0), p.unsqueeze(0))[0],
    )
