"""Direct tests for compute_loss — shapes, empty inputs, and metrics keys."""
import torch
import pytest

from diffopt.demands import Demand
from diffopt.loss import compute_loss
from diffopt.modulation import ModulationConfig


def make_mod_config() -> ModulationConfig:
    """Minimal ModulationConfig with a single 400 Gbps format.

    Copied from tests/test_pipeline.py's make_mod_config() helper so this
    file's fixtures match the rest of the suite.
    """
    return ModulationConfig(
        channel_spacing_ghz=100.0,
        symbol_rate_gbaud=64.0,
        num_channels_cband=48,
        cut_channel_index=24,
        formats=[{"bitrate_gbps": 400, "snr_threshold_db": 20.0}],
    )


@pytest.fixture
def simple_loss_inputs():
    """One feasible demand (GSNR above the 20 dB threshold) and one
    infeasible demand (GSNR below it), so num_infeasible == 1."""
    mod_cfg = make_mod_config()
    demands = [
        Demand(id=0, src=0, dst=1, bitrate_gbps=400.0),
        Demand(id=1, src=1, dst=2, bitrate_gbps=400.0),
    ]
    gsnr_preds = {
        0: torch.tensor(25.0),  # feasible: above 20 dB threshold
        1: torch.tensor(10.0),  # infeasible: below 20 dB threshold
    }
    path_noise_costs = {
        0: torch.tensor(1.5),
        1: torch.tensor(2.5),
    }
    regen_probs = torch.tensor([0.1, 0.6, 0.3])
    return dict(
        gsnr_preds=gsnr_preds,
        path_noise_costs=path_noise_costs,
        demands=demands,
        regen_probs=regen_probs,
        modulation_config=mod_cfg,
    )


def test_total_loss_is_a_scalar(simple_loss_inputs):
    total, _ = compute_loss(**simple_loss_inputs)
    assert total.shape == (), f"expected scalar, got shape {tuple(total.shape)}"


def test_empty_demand_list_does_not_crash():
    total, metrics = compute_loss(
        gsnr_preds={},
        path_noise_costs={},
        demands=[],
        regen_probs=torch.zeros(5),
        modulation_config=None,
    )
    assert total.shape == ()
    assert metrics["num_infeasible"] == 0
    assert metrics["path_noise_loss"] == 0.0


def test_metrics_dict_has_the_keys_train_py_logs(simple_loss_inputs):
    """train.py and every milestone claim in CLAUDE.md read these keys."""
    _, metrics = compute_loss(**simple_loss_inputs)
    for key in (
        "feasibility_loss",
        "regen_loss",
        "path_noise_loss",
        "num_regen_soft",
        "num_infeasible",
    ):
        assert key in metrics, f"missing metrics key: {key}"


def test_num_infeasible_counts_only_below_threshold_demands(simple_loss_inputs):
    _, metrics = compute_loss(**simple_loss_inputs)
    assert metrics["num_infeasible"] == 1
