"""The lambda_waste calibration arithmetic, unit-tested away from a real run.

Mirrors tests/test_calibrate_lambda_dev.py. The band here is denominated in
the DEVICE push, not the feasibility push, because the waste term's job is
different: `lambda_dev` prices every device equally, so on its own it sheds
load-bearing and redundant cuts at the same rate. `lambda_waste` exists to
make the shed DISCRIMINATE, and the thing it has to out-pull is the
undiscriminating device push, not the feasibility push.
"""
import pytest

from calibrate_lambda_waste import (
    CEILING_FRACTION,
    TARGET_DISCRIMINATION,
    implied_lambda_waste,
)


def test_band_lands_the_waste_push_at_the_target_multiple_of_the_device_push():
    # device push 10.0, unit-lambda waste push 4.0 -> lambda_waste in
    # [2*10/4, 5*10/4] = [5.0, 12.5]
    lo, hi, _ = implied_lambda_waste(
        dev_push=10.0, waste_push_at_unit_lambda=4.0, feas_l1=1000.0
    )
    assert (lo, hi) == pytest.approx((5.0, 12.5))
    assert TARGET_DISCRIMINATION == (2.0, 5.0)


def test_ceiling_is_where_the_combined_shed_push_reaches_the_feasibility_cap():
    # feasibility 1000 -> cap at 20% = 200. Device push already spends 10,
    # leaving 190 for the waste term; at a unit-lambda push of 4.0 that is
    # lambda_waste = 47.5.
    _, _, ceiling = implied_lambda_waste(
        dev_push=10.0, waste_push_at_unit_lambda=4.0, feas_l1=1000.0
    )
    assert ceiling == pytest.approx(47.5)
    assert CEILING_FRACTION == 0.20


def test_ceiling_is_zero_when_the_device_push_alone_already_exceeds_the_cap():
    """A negative ceiling is not a negative weight — it means there is no room
    left under the cap, and the caller must be told that rather than handed a
    meaningless number."""
    _, _, ceiling = implied_lambda_waste(
        dev_push=500.0, waste_push_at_unit_lambda=4.0, feas_l1=1000.0
    )
    assert ceiling == 0.0


def test_zero_waste_push_is_an_error_not_an_infinity():
    """A zero waste gradient means no allocated cut has positive headroom —
    either the head buys nothing, or every cut it buys is load-bearing.
    Reporting lambda = inf would hide both."""
    with pytest.raises(ValueError, match="waste push is zero"):
        implied_lambda_waste(
            dev_push=10.0, waste_push_at_unit_lambda=0.0, feas_l1=1000.0
        )
