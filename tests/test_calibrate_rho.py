"""The rho calibration arithmetic, unit-tested away from a real run.

Mirrors tests/test_calibrate_lambda_dev.py and
tests/test_calibrate_lambda_waste.py, but the anchor is a different KIND of
quantity. Those two set a force RATIO band; this one sets a DISTANCE. The
augmented Lagrangian has a fixed point the hinge does not — at rest a
constrained demand sits exactly on its bar with price
lambda* = lambda_dev / s — and rho converts that price into the width of the
band the penalty stays active over, measured in dB of headroom inside the
feasible region.
"""
import pytest

from calibrate_rho import BAND_TARGET_DB, CEILING_MULTIPLE, implied_rho


def test_rho_makes_the_active_band_exactly_the_target_width():
    """The design claim in one line: band width = lambda*/rho, and rho is
    chosen to set it."""
    lambda_star, rho, _ = implied_rho(dev_push=10.0, unit_feasibility_push=20.0)
    assert lambda_star == pytest.approx(0.5)
    assert rho == pytest.approx(0.5 / 0.35)
    assert lambda_star / rho == pytest.approx(BAND_TARGET_DB)
    assert BAND_TARGET_DB == 0.35, "one value, not a range — spec section 4"


def test_a_narrower_band_target_costs_a_proportionally_larger_rho():
    _, wide, _ = implied_rho(dev_push=10.0, unit_feasibility_push=20.0,
                             band_target_db=0.4)
    _, narrow, _ = implied_rho(dev_push=10.0, unit_feasibility_push=20.0,
                               band_target_db=0.2)
    assert narrow == pytest.approx(2.0 * wide)


def test_ceiling_bounds_one_epochs_dual_step_at_ten_resting_duals():
    """rho is also the dual step, so a 1 dB violation moves a dual by rho.
    The ceiling keeps that under 10x the dual's own resting value."""
    lambda_star, _, ceiling = implied_rho(dev_push=10.0, unit_feasibility_push=20.0)
    assert ceiling == pytest.approx(CEILING_MULTIPLE * lambda_star)
    assert CEILING_MULTIPLE == 10.0


def test_the_default_band_target_sits_under_the_ceiling():
    """At 0.35 dB the band target implies rho = 2.857 * lambda*, comfortably
    inside the 10x cap — so the ceiling is a guard, not the binding
    constraint, and a run that hits it is reporting something unusual."""
    _, rho, ceiling = implied_rho(dev_push=10.0, unit_feasibility_push=20.0)
    assert rho < ceiling


def test_zero_feasibility_push_is_an_error_not_an_infinity():
    """The signed sum has a gradient everywhere the hinge does not, so a zero
    here means the head is disconnected from the graph or fully saturated —
    NOT that the constraint happens to be satisfied."""
    with pytest.raises(ValueError, match="feasibility push is zero"):
        implied_rho(dev_push=10.0, unit_feasibility_push=0.0)


def test_a_non_positive_band_target_is_an_error():
    with pytest.raises(ValueError, match="band_target_db"):
        implied_rho(dev_push=10.0, unit_feasibility_push=20.0, band_target_db=0.0)
