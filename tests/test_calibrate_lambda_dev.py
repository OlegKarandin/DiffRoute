"""The calibration arithmetic, unit-tested away from a real training run."""
import pytest

from calibrate_lambda_dev import implied_lambda_dev, TARGET_BAND


def test_implied_lambda_lands_the_device_push_inside_the_target_band():
    # feasibility push 100.0, unit-lambda device push 4.0 -> lambda in
    # [0.1*100/4, 0.2*100/4] = [2.5, 5.0]
    lo, hi = implied_lambda_dev(feas_l1=100.0, dev_l1_at_unit_lambda=4.0)
    assert (lo, hi) == pytest.approx((2.5, 5.0))
    assert TARGET_BAND == (0.10, 0.20)


def test_zero_device_push_is_an_error_not_an_infinity():
    """A zero device gradient means the head is saturated or disconnected —
    reporting lambda = inf would hide that."""
    with pytest.raises(ValueError, match="device push is zero"):
        implied_lambda_dev(feas_l1=100.0, dev_l1_at_unit_lambda=0.0)
