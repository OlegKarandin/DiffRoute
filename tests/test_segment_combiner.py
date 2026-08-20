"""
Unit tests for diffopt.qot.segment_combiner.

Test IDs:
  1. Single segment → identity
  2. Two segments, p=1 → worst segment (soft_max noise ≈ max noise)
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
"""

import math
import pytest
import torch

from diffopt.qot.segment_combiner import (
    SegmentCombiner,
    db_to_linear_noise,
    linear_noise_to_db,
    soft_max,
)


def t(val: float, requires_grad: bool = False) -> torch.Tensor:
    """Convenience: scalar float32 tensor."""
    return torch.tensor(val, dtype=torch.float32, requires_grad=requires_grad)


# ---------------------------------------------------------------------------
# Test 1: single segment → identity
# ---------------------------------------------------------------------------

def test_single_segment_identity():
    combiner = SegmentCombiner()
    gsnr = t(15.0)
    result = combiner([gsnr], [], temperature=0.5)
    assert torch.isclose(result, gsnr, atol=1e-4), (
        f"Expected {gsnr.item():.4f} dB, got {result.item():.4f} dB"
    )


# ---------------------------------------------------------------------------
# Test 2: two segments, p=1 → output ≈ min(gsnr_1, gsnr_2)
# ---------------------------------------------------------------------------

def test_two_segments_p1_worst_segment():
    """
    With p=1 fully regenerated, accumulated_noise = soft_max(n1, n2) ≈ max(n1, n2).
    Higher noise → lower GSNR → output ≈ min(gsnr_1, gsnr_2).
    """
    combiner = SegmentCombiner()
    gsnr_1 = t(10.0)
    gsnr_2 = t(20.0)
    p = t(1.0)

    result = combiner([gsnr_1, gsnr_2], [p], temperature=0.01)  # very sharp → ≈ hard max

    # Analytical: with p=1 and sharp soft_max, result ≈ min(10, 20) = 10 dB
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

    result = combiner([gsnr_1, gsnr_2], [p], temperature=0.5)

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

    Uses temperature=0.01 so soft_max ≈ true max and the physics is clear.
    """
    combiner = SegmentCombiner()
    gsnr_1 = t(5.0)   # bad segment
    gsnr_2 = t(25.0)  # good segment

    p_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    results = [combiner([gsnr_1, gsnr_2], [t(p)], temperature=0.01).item() for p in p_values]

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
    ∂gsnr/∂p should be positive.

    Uses temperature=0.01 so soft_max ≈ true max: regen gives max(n1,n2) = n1
    while no-regen gives n1+n2 > n1, so regen strictly reduces noise.
    """
    combiner = SegmentCombiner()
    gsnr_1 = t(5.0)   # bad first segment
    gsnr_2 = t(25.0)
    p = t(0.5, requires_grad=True)

    result = combiner([gsnr_1, gsnr_2], [p], temperature=0.01)
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

    soft_max's overshoot is `temperature * ln2`, which is ABSOLUTE in linear
    noise units and does not shrink with the operands. At temperature=0.01
    that overshoot is 0.00693 — larger than the noise itself — so
    `soft_max(a,b)` exceeds `a + b` and the combiner reports that
    regenerating makes the path *worse*. See
    docs/investigations/regen_placement_not_concentrating.md.

    Two 26 dB segments: no-regen noise is 2 * 0.00251 = 0.00501 (23.0 dB),
    regen noise is 0.00251 (26.0 dB). Regen strictly helps, so
    d(gsnr)/dp must be positive here exactly as it is in test 5.
    """
    combiner = SegmentCombiner()
    gsnr_1 = t(26.0)
    gsnr_2 = t(26.0)
    p = t(0.5, requires_grad=True)

    result = combiner([gsnr_1, gsnr_2], [p], temperature=0.01)
    result.backward()

    assert p.grad is not None, "Gradient not computed for regen_prob"
    assert p.grad.item() > 0.0, (
        f"Expected positive gradient (regen helps at any noise scale), "
        f"got {p.grad.item():.6f} — soft_max's absolute error floor "
        f"(temperature*ln2) has swamped the per-segment noise"
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
        transparent = combiner([t(gsnr_db), t(gsnr_db)], [t(0.0)], temperature=0.01)
        regenerated = combiner([t(gsnr_db), t(gsnr_db)], [t(1.0)], temperature=0.01)
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
    enumeration of 4 binary configs weighted by (p^k * (1-p)^(2-k)).
    """
    gsnr_vals = [10.0, 20.0, 15.0]
    p_val = 0.3
    temperature = 0.01  # small so soft_max ≈ hard max; brute-force uses hard max

    combiner = SegmentCombiner()
    g = [t(v) for v in gsnr_vals]
    p = [t(p_val), t(p_val)]

    result = combiner(g, p, temperature=temperature).item()

    # Brute force: enumerate regen decisions (0=passthrough, 1=regen) at each boundary
    # Config (r1, r2): prob = p^(r1+r2) * (1-p)^(2-r1-r2)
    def combine_hard(g_list, regens):
        noises = [10 ** (-v / 10) for v in g_list]
        acc = noises[0]
        for i, r in enumerate(regens):
            n_next = noises[i + 1]
            if r == 0:  # no regen
                acc = acc + n_next
            else:       # regen: worst segment
                acc = max(acc, n_next)
        return -10 * math.log10(acc)

    configs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    weighted_noise = 0.0
    for r1, r2 in configs:
        prob = (p_val ** (r1 + r2)) * ((1 - p_val) ** (2 - r1 - r2))
        gsnr_config = combine_hard(gsnr_vals, [r1, r2])
        noise_config = 10 ** (-gsnr_config / 10)
        weighted_noise += prob * noise_config

    expected = -10 * math.log10(weighted_noise)

    # Soft-max introduces approximation error; allow 0.5 dB tolerance
    assert abs(result - expected) < 0.5, (
        f"3-segment check: expected ~{expected:.3f} dB, got {result:.3f} dB"
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
        result = combiner(gsnrs, probs, temperature=0.5)
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

    result = combiner(gsnrs, regen_probs, temperature=0.01)

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

    wild = combiner(
        [t(-50.0), t(80.0)],
        [t(0.0)],
        temperature=0.01,
    )
    clamped = combiner(
        [t(-5.0), t(35.0)],
        [t(0.0)],
        temperature=0.01,
    )
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
    Tolerance is loosened to 0.05 dB (vs. the 5-segment fixture's own
    verified soft-max chunk-tie overshoot of ~0.03 dB, see
    docs/investigations/regen_over_provisioning.md) rather than the
    single-boundary soft_max's tighter bound, since this fixture ties two
    boundaries at hard p=1 simultaneously.
    """
    combiner = SegmentCombiner()
    gsnrs = five_equal_segments()
    probs = [t(1.0), t(0.0), t(1.0), t(0.0)]  # cut after seg 0, cut after seg 2

    result = combiner(gsnrs, probs, temperature=0.01).item()
    expected = -10.0 * math.log10(2 * FIVE_SEG_NOISE)  # 20.000 dB

    assert result == pytest.approx(expected, abs=0.05), (
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
    """
    combiner = SegmentCombiner()
    gsnrs = five_equal_segments()
    probs = [t(0.0), t(1.0), t(0.0), t(0.0)]  # cut after seg 1 only

    result = combiner(gsnrs, probs, temperature=0.01).item()
    expected = -10.0 * math.log10(3 * FIVE_SEG_NOISE)  # 18.2391 dB

    assert result == pytest.approx(expected, abs=1e-3), (
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

    boundary_1_alone = combiner(
        gsnrs, [t(0.0), t(1.0), t(0.0), t(0.0)], temperature=0.01
    ).item()
    boundary_0_and_1 = combiner(
        gsnrs, [t(1.0), t(1.0), t(0.0), t(0.0)], temperature=0.01
    ).item()

    delta = abs(boundary_0_and_1 - boundary_1_alone)
    assert delta < 0.02, (
        f"redundant regenerator should add ~0 dB, got {boundary_1_alone:.4f} "
        f"-> {boundary_0_and_1:.4f} dB (delta {delta:.4f} dB)"
    )


def test_zero_regen_probability_is_exact_sum_at_loose_temperature():
    """All boundary probabilities exactly 0, at the loose t=0.5 default
    temperature (epochs 1-10, and segment_combiner's own default): the
    fold must reduce to a plain sum of segment noise, exactly, with no
    residual soft-max contribution from the zero sentinel.

    This guards the `any_regen`-style sentinel problem the original
    two-state design in docs/investigations/regen_over_provisioning.md
    called out explicitly: any construction where a "nothing happened yet"
    zero is blended through soft_max at loose temperature picks up a
    ~6% scale-normalised excess (soft_max(0, C, t=0.5) = C*1.0635 != C).
    This N-way weighted log-sum-exp fold has no such sentinel -- at p=0
    every chunk other than the full path has weight exactly 0 (log-weight
    -inf via clamp_min(1e-300)), so the sum degenerates to the single
    active exp() term and cancels exactly against the temperature*log()
    outside -- but the test is kept as a standing regression guard.
    """
    combiner = SegmentCombiner()
    gsnrs = [t(15.0), t(9.0), t(20.0), t(12.5)]
    n_segments = len(gsnrs)
    probs = [t(0.0)] * (n_segments - 1)

    result = combiner(gsnrs, probs, temperature=0.5).item()

    total_noise = sum(10 ** (-g.item() / 10) for g in gsnrs)
    expected = -10.0 * math.log10(total_noise)

    assert result == pytest.approx(expected, abs=1e-4), (
        f"all p=0 at t=0.5: expected exact sum {expected:.4f} dB, "
        f"got {result:.4f} dB"
    )
