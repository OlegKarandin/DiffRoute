"""Pure-helper unit tests for the two Task 3 diagnostics: the threshold
sweep and the boundary-headroom histogram. Follows
tests/test_run_placement_ablation.py's style — hand-built inputs against the
small aggregation/formatting functions each script factors out of main(),
never a real DiagContext/pipeline/checkpoint or a real sweep/rollout.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from diagnose_threshold_sweep import (  # noqa: E402
    build_score_delta_grid,
    first_gate1_delta,
    parse_delta_grid,
)
from diagnose_boundary_headroom import (  # noqa: E402
    cut_rate_above_zero,
    demand_oracle_vs_head,
    percentile_summary,
    replay_boundary_feature4,
)


# ---------------------------------------------------------------------------
# diagnose_threshold_sweep.py
# ---------------------------------------------------------------------------

def test_parse_delta_grid_handles_whitespace_and_signs():
    assert parse_delta_grid("1.0, -2.5,0") == [1.0, -2.5, 0.0]


def test_build_score_delta_grid_spans_observed_range_with_margin():
    grid = build_score_delta_grid(-8.7, 2.9, num_points=5, margin_frac=0.1)
    span = 2.9 - (-8.7)
    pad = span * 0.1
    assert len(grid) == 5
    assert grid[0] == pytest.approx(-8.7 - pad)
    assert grid[-1] == pytest.approx(2.9 + pad)
    assert grid == sorted(grid)


def test_build_score_delta_grid_degenerate_span_falls_back():
    """A saturated head (score_min == score_max) must not collapse to a
    single-point or NaN grid — a one-point sweep cannot locate a threshold
    crossing."""
    grid = build_score_delta_grid(5.0, 5.0, num_points=4)
    assert len(grid) == 4
    assert grid[0] == pytest.approx(4.0)
    assert grid[-1] == pytest.approx(6.0)


def test_build_score_delta_grid_nan_input_does_not_raise():
    grid = build_score_delta_grid(math.nan, math.nan, num_points=3)
    assert len(grid) == 3
    assert all(math.isfinite(v) for v in grid)


def test_first_gate1_delta_finds_first_qualifying_row_in_sweep_order():
    # (delta, hard_num_devices, hard_num_violated, oracle_devices)
    rows = [
        (-2.0, 900, 5, 60),   # violated
        (-1.0, 100, 0, 60),   # 100 > 1.25*60=75, fails device cap
        (-0.5, 70, 0, 60),    # 70 <= 75 -> first qualifying row
        (0.0, 65, 0, 60),
    ]
    assert first_gate1_delta(rows) == -0.5


def test_first_gate1_delta_none_when_nothing_qualifies():
    rows = [(-1.0, 900, 3, 60), (0.0, 500, 1, 60)]
    assert first_gate1_delta(rows) is None


def test_first_gate1_delta_boundary_is_inclusive():
    # hard_num_devices == 1.25 * oracle_devices exactly should qualify.
    rows = [(0.0, 75, 0, 60)]
    assert first_gate1_delta(rows) == 0.0


def test_first_gate1_delta_empty_rows():
    assert first_gate1_delta([]) is None


# ---------------------------------------------------------------------------
# diagnose_boundary_headroom.py
# ---------------------------------------------------------------------------

def test_percentile_summary_two_point_list():
    summary = percentile_summary([0.0, 10.0])
    assert summary["count"] == 2
    assert summary["median"] == pytest.approx(5.0)
    assert summary["p10"] == pytest.approx(1.0)
    assert summary["p90"] == pytest.approx(9.0)


def test_percentile_summary_empty_is_nan():
    summary = percentile_summary([])
    assert summary["count"] == 0
    assert math.isnan(summary["median"])
    assert math.isnan(summary["p10"])
    assert math.isnan(summary["p90"])


def test_cut_rate_above_zero_counts_strictly_positive():
    above, total = cut_rate_above_zero([1.0, -0.5, 2.0, 0.0, -3.0])
    assert (above, total) == (2, 5)


def test_cut_rate_above_zero_empty():
    assert cut_rate_above_zero([]) == (0, 0)


def test_demand_oracle_vs_head_buckets_by_oracle_need():
    oracle_counts = [0, 0, 3, 5]
    head_counts = [0, 2, 3, 7]
    summary = demand_oracle_vs_head(oracle_counts, head_counts)
    assert summary == {
        "oracle_zero_demands": 2,
        "oracle_zero_head_buys": 2,
        "oracle_pos_demands": 2,
        "oracle_pos_head_buys": 10,
        "oracle_pos_oracle_wants": 8,
    }


def test_demand_oracle_vs_head_length_mismatch_raises():
    try:
        demand_oracle_vs_head([0, 1], [0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on length mismatch")


# ---------------------------------------------------------------------------
# replay_boundary_feature4 — pure tensor arithmetic, hand-built, no pipeline.
# ---------------------------------------------------------------------------

def test_replay_boundary_feature4_matches_hand_computed_carry():
    """One demand, 3 segments (2 boundaries), noise 0.5/0.3/0.9, bar 0 dB, a
    cut after boundary 0 only. Hand-computed:

      k=0: c = 0.5, f4_0 = -10*log10(0.5 + 0.3) = +0.969 dB, then cut resets c to 0
      k=1: c = 0 + 0.3 = 0.3, f4_1 = -10*log10(0.3 + 0.9) = -0.792 dB
    """
    seg_noise = torch.tensor([[0.5, 0.3, 0.9]])
    num_segments = torch.tensor([3])
    cuts = torch.tensor([[1.0, 0.0]])
    bar_db = torch.tensor([0.0])

    f4, valid = replay_boundary_feature4(seg_noise, num_segments, cuts, bar_db)

    assert valid.tolist() == [[True, True]]
    assert f4[0, 0].item() == pytest.approx(0.9691, abs=1e-3)
    assert f4[0, 1].item() == pytest.approx(-0.7918, abs=1e-3)


def test_replay_boundary_feature4_masks_padded_boundary():
    """num_segments=2 on a J=3 buffer: only boundary 0 is real; boundary 1
    (which would need segment 2, which does not exist) must be marked
    invalid."""
    seg_noise = torch.tensor([[0.5, 0.3, 0.9]])
    num_segments = torch.tensor([2])
    cuts = torch.tensor([[0.0, 0.0]])
    bar_db = torch.tensor([0.0])

    _f4, valid = replay_boundary_feature4(seg_noise, num_segments, cuts, bar_db)

    assert valid.tolist() == [[True, False]]


def test_replay_boundary_feature4_no_cut_accumulates_across_boundary():
    """Without a cut at boundary 0, boundary 1's carry must include BOTH
    prior segments (0.5 + 0.3), not just the most recent one."""
    seg_noise = torch.tensor([[0.5, 0.3, 0.9]])
    num_segments = torch.tensor([3])
    cuts = torch.tensor([[0.0, 0.0]])
    bar_db = torch.tensor([0.0])

    f4, _valid = replay_boundary_feature4(seg_noise, num_segments, cuts, bar_db)

    expected_f4_1 = -10.0 * math.log10(0.5 + 0.3 + 0.9)
    assert f4[0, 1].item() == pytest.approx(expected_f4_1, abs=1e-3)
