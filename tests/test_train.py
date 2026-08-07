"""Unit tests for diffopt/train.py's pure annealing helper.

`linear_anneal` (renamed/generalized from `compute_regen_tau`) now drives
both RegenPlacement's `tau` and SegmentCombiner's `soft_max_temperature` —
the fix for the bug where soft_max_temperature was hardcoded at a fixed
0.5 and never annealed (see CLAUDE.md's Phase 1c corrections, and
diffopt/qot/segment_combiner.py's docstring, for the physics diagnosis).
"""
from __future__ import annotations

import pytest

from diffopt.train import linear_anneal


def test_before_anneal_start_returns_start_value():
    assert linear_anneal(epoch=1, start=1.0, end=0.1, anneal_start=5, anneal_end=20) == 1.0
    assert linear_anneal(epoch=5, start=1.0, end=0.1, anneal_start=5, anneal_end=20) == 1.0


def test_after_anneal_end_returns_end_value():
    assert linear_anneal(epoch=20, start=1.0, end=0.1, anneal_start=5, anneal_end=20) == 0.1
    assert linear_anneal(epoch=100, start=1.0, end=0.1, anneal_start=5, anneal_end=20) == 0.1


def test_linear_interpolation_at_midpoint():
    # anneal_start=0, anneal_end=10, epoch=5 -> halfway between start and end
    result = linear_anneal(epoch=5, start=1.0, end=0.0, anneal_start=0, anneal_end=10)
    assert result == pytest.approx(0.5)


def test_monotonic_decrease_across_the_anneal_window():
    values = [
        linear_anneal(epoch=e, start=0.5, end=0.01, anneal_start=5, anneal_end=18)
        for e in range(1, 25)
    ]
    for i in range(1, len(values)):
        assert values[i] <= values[i - 1] + 1e-12, (
            f"Not monotonically non-increasing at index {i}: {values}"
        )
    assert values[0] == 0.5
    assert values[-1] == 0.01


def test_generic_across_different_schedules():
    """The same function must correctly drive both tau (1.0->0.1) and
    soft_max_temperature (0.5->0.01) schedules independently — this is the
    whole point of generalizing it rather than keeping two near-duplicate
    functions."""
    tau = linear_anneal(epoch=10, start=1.0, end=0.1, anneal_start=5, anneal_end=15)
    soft_max_temp = linear_anneal(epoch=10, start=0.5, end=0.01, anneal_start=5, anneal_end=15)

    # Same fractional progress (epoch 10 is 50% through [5,15]) but
    # different start/end -> different absolute values, same fraction.
    tau_frac = (tau - 0.1) / (1.0 - 0.1)
    temp_frac = (soft_max_temp - 0.01) / (0.5 - 0.01)
    assert tau_frac == pytest.approx(temp_frac)
    assert tau_frac == pytest.approx(0.5)
