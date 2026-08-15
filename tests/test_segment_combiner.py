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
