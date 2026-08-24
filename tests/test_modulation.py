"""Tests for modulation format config and SNR lookup."""
import pytest
from pathlib import Path

from diffopt.modulation import ModulationConfig


@pytest.fixture
def mod_cfg():
    base = Path(__file__).parent.parent
    return ModulationConfig.from_yaml(str(base / "configs/modulation_formats.yaml"))


def test_snr_500_gbps(mod_cfg):
    assert mod_cfg.required_snr_threshold(500) == pytest.approx(9.2, abs=1e-6)


def test_snr_300_gbps(mod_cfg):
    assert mod_cfg.required_snr_threshold(300) == pytest.approx(4.8, abs=1e-6)


def test_snr_800_gbps(mod_cfg):
    assert mod_cfg.required_snr_threshold(800) == pytest.approx(15.1, abs=1e-6)


def test_snr_unknown_raises(mod_cfg):
    with pytest.raises(ValueError):
        mod_cfg.required_snr_threshold(999)


def test_max_feasible_bitrate_14_5_db(mod_cfg):
    # 750 Gbps threshold is 14.1 dB (passes), 800 Gbps threshold is 15.1 dB (fails)
    result = mod_cfg.max_feasible_bitrate(14.5)
    assert result == pytest.approx(750.0, abs=1e-6)


def test_max_feasible_bitrate_below_all(mod_cfg):
    # Below the minimum threshold (4.8 dB)
    result = mod_cfg.max_feasible_bitrate(4.0)
    assert result is None


def test_max_feasible_bitrate_above_all(mod_cfg):
    # Above all thresholds -> highest bitrate
    result = mod_cfg.max_feasible_bitrate(20.0)
    assert result == pytest.approx(800.0, abs=1e-6)


def test_bitrate_options_count(mod_cfg):
    options = mod_cfg.bitrate_options
    assert len(options) == 11
    assert min(options) == 300.0
    assert max(options) == 800.0


def test_bar_db_for_demands_is_threshold_plus_margin_in_demand_order(mod_cfg):
    from diffopt.demands import Demand
    from diffopt.modulation import bar_db_for_demands

    cfg = mod_cfg
    demands = [Demand(0, 1, 2, 400.0), Demand(1, 3, 4, 800.0)]
    bars = bar_db_for_demands(demands, cfg, margin_db=0.5)
    assert bars.shape == (2,)
    for i, d in enumerate(demands):
        assert bars[i].item() == pytest.approx(
            cfg.required_snr_threshold(d.bitrate_gbps) + 0.5
        )
