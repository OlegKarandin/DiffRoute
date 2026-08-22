"""Direct tests for compute_loss — shapes, empty inputs, and metrics keys."""
import math

import torch
import pytest

from diffopt.demands import Demand
from diffopt.loss import compute_loss

from tests.test_pipeline import make_mod_config


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
        duals=torch.tensor([10.0, 10.0]),
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
        duals=torch.zeros(0),
    )
    assert total.shape == ()
    assert metrics["num_infeasible"] == 0
    assert metrics["num_violated"] == 0
    assert metrics["path_noise_loss"] == 0.0
    assert math.isnan(metrics["worst_margin_db"])


def test_metrics_dict_has_the_keys_train_py_logs(simple_loss_inputs):
    """train.py and every milestone claim in docs/investigations/CHANGELOG.md read these keys."""
    _, metrics = compute_loss(**simple_loss_inputs)
    for key in (
        "feasibility_loss",
        "regen_loss",
        "path_noise_loss",
        "num_regen_soft",
        "num_infeasible",
        "weighted_feasibility_loss",
        "num_violated",
        "worst_margin_db",
        "shortfalls",
    ):
        assert key in metrics, f"missing metrics key: {key}"


def test_num_infeasible_counts_only_below_threshold_demands(simple_loss_inputs):
    _, metrics = compute_loss(**simple_loss_inputs)
    assert metrics["num_infeasible"] == 1


# ---------------------------------------------------------------------------
# Constrained objective: per-demand duals + margin inside the hinge
# ---------------------------------------------------------------------------

from diffopt.loss import update_duals


def test_margin_keeps_hinge_active_above_threshold():
    """The fix for spec finding #3.

    `relu(thr - gsnr)` is exactly zero the moment a demand clears, so nothing
    pushes for headroom — measured |g_feas|_1 = 0.000 on fully-feasible epochs,
    every logit pushed down and none up. With a margin, a demand sitting at
    thr + delta/2 still contributes loss.
    """
    mod_cfg = make_mod_config()          # single 400 G format, 20 dB threshold
    demands = [Demand(id=0, src=0, dst=1, bitrate_gbps=400.0)]
    margin_db = 0.5
    gsnr = torch.tensor(20.0 + margin_db / 2, requires_grad=True)

    total, metrics = compute_loss(
        gsnr_preds={0: gsnr},
        path_noise_costs={0: torch.tensor(0.0)},
        demands=demands,
        regen_probs=torch.zeros(3),
        modulation_config=mod_cfg,
        duals=torch.tensor([1.0]),
        margin_db=margin_db,
        lambda_regen=0.0,
        lambda_cost=0.0,
    )
    assert total.item() == pytest.approx(margin_db / 2)
    assert metrics["num_violated"] == 1, "violates threshold + margin"
    assert metrics["num_infeasible"] == 0, "but is physically feasible"

    total.backward()
    assert gsnr.grad.item() < 0, (
        "raising GSNR must reduce the loss — without a margin this gradient "
        "is exactly zero above the threshold"
    )


def test_zero_loss_only_once_the_margin_is_cleared():
    mod_cfg = make_mod_config()
    demands = [Demand(id=0, src=0, dst=1, bitrate_gbps=400.0)]
    total, metrics = compute_loss(
        gsnr_preds={0: torch.tensor(20.6)},
        path_noise_costs={0: torch.tensor(0.0)},
        demands=demands,
        regen_probs=torch.zeros(3),
        modulation_config=mod_cfg,
        duals=torch.tensor([1.0]),
        margin_db=0.5,
        lambda_regen=0.0,
        lambda_cost=0.0,
    )
    assert total.item() == pytest.approx(0.0)
    assert metrics["num_violated"] == 0


def test_duals_weight_demands_independently():
    """A per-demand dual, not a global scalar: doubling one demand's dual must
    change the loss by exactly that demand's shortfall."""
    mod_cfg = make_mod_config()
    demands = [
        Demand(id=0, src=0, dst=1, bitrate_gbps=400.0),
        Demand(id=1, src=1, dst=2, bitrate_gbps=400.0),
    ]
    common = dict(
        gsnr_preds={0: torch.tensor(18.0), 1: torch.tensor(19.0)},
        path_noise_costs={0: torch.tensor(0.0), 1: torch.tensor(0.0)},
        demands=demands,
        regen_probs=torch.zeros(3),
        modulation_config=mod_cfg,
        margin_db=0.0,
        lambda_regen=0.0,
        lambda_cost=0.0,
    )
    base, _ = compute_loss(duals=torch.tensor([1.0, 1.0]), **common)
    bumped, _ = compute_loss(duals=torch.tensor([2.0, 1.0]), **common)
    # demand 0 shortfall = 20 - 18 = 2.0
    assert bumped.item() - base.item() == pytest.approx(2.0)


def test_shortfalls_are_returned_indexed_by_demand_id():
    mod_cfg = make_mod_config()
    demands = [
        Demand(id=0, src=0, dst=1, bitrate_gbps=400.0),
        Demand(id=1, src=1, dst=2, bitrate_gbps=400.0),
    ]
    _, metrics = compute_loss(
        gsnr_preds={0: torch.tensor(18.0), 1: torch.tensor(25.0)},
        path_noise_costs={0: torch.tensor(0.0), 1: torch.tensor(0.0)},
        demands=demands,
        regen_probs=torch.zeros(3),
        modulation_config=mod_cfg,
        duals=torch.tensor([1.0, 1.0]),
        margin_db=0.0,
        lambda_regen=0.0,
        lambda_cost=0.0,
    )
    shortfalls = metrics["shortfalls"]
    assert shortfalls.shape == (2,)
    assert not shortfalls.requires_grad, "dual updates must not build a graph"
    assert shortfalls[0].item() == pytest.approx(2.0)
    assert shortfalls[1].item() == pytest.approx(0.0)


def test_worst_margin_db_is_reported_against_the_bare_threshold():
    """worst_margin_db is measured against `thr`, not `thr + delta`, so it
    stays comparable to the pre-change numbers in the spec's evidence."""
    mod_cfg = make_mod_config()
    demands = [
        Demand(id=0, src=0, dst=1, bitrate_gbps=400.0),
        Demand(id=1, src=1, dst=2, bitrate_gbps=400.0),
    ]
    _, metrics = compute_loss(
        gsnr_preds={0: torch.tensor(21.5), 1: torch.tensor(19.4)},
        path_noise_costs={0: torch.tensor(0.0), 1: torch.tensor(0.0)},
        demands=demands,
        regen_probs=torch.zeros(3),
        modulation_config=mod_cfg,
        duals=torch.tensor([1.0, 1.0]),
        margin_db=0.5,
    )
    assert metrics["worst_margin_db"] == pytest.approx(-0.6)


def test_dual_grows_while_violated():
    duals = torch.tensor([10.0, 10.0])
    shortfalls = torch.tensor([1.0, 0.0])
    updated = update_duals(duals, shortfalls, eta=1.0, dual_max=1000.0)
    assert updated[0].item() == pytest.approx(11.0)


def test_dual_stationary_when_satisfied():
    duals = torch.tensor([10.0, 42.0])
    updated = update_duals(duals, torch.zeros(2), eta=1.0, dual_max=1000.0)
    assert updated.tolist() == pytest.approx([10.0, 42.0])


def test_dual_respects_cap():
    duals = torch.tensor([999.0])
    updated = update_duals(duals, torch.tensor([50.0]), eta=1.0, dual_max=1000.0)
    assert updated[0].item() == pytest.approx(1000.0)


def test_dual_never_goes_negative():
    """The hinge makes shortfalls non-negative, so this is belt-and-braces —
    but a negative dual would flip the feasibility term into a reward."""
    updated = update_duals(torch.tensor([1.0]), torch.tensor([-100.0]),
                           eta=1.0, dual_max=1000.0)
    assert updated[0].item() == pytest.approx(0.0)


def test_update_duals_does_not_mutate_its_input():
    duals = torch.tensor([10.0])
    update_duals(duals, torch.tensor([1.0]), eta=1.0, dual_max=1000.0)
    assert duals[0].item() == pytest.approx(10.0)


def test_dual_stationary_when_satisfied_and_decay_zero():
    """decay defaults to 0.0 -- must reproduce the pre-decay ratchet exactly."""
    duals = torch.tensor([10.0, 42.0])
    updated = update_duals(duals, torch.zeros(2), eta=1.0, dual_max=1000.0, decay=0.0)
    assert updated.tolist() == pytest.approx([10.0, 42.0])


def test_dual_decays_only_when_satisfied():
    duals = torch.tensor([10.0, 10.0])
    shortfalls = torch.tensor([1.0, 0.0])
    updated = update_duals(duals, shortfalls, eta=1.0, dual_max=1000.0, decay=0.1)
    # index 0 still violated: ascends undamped, decay does not apply.
    assert updated[0].item() == pytest.approx(11.0)
    # index 1 satisfied: decays by (1 - decay).
    assert updated[1].item() == pytest.approx(9.0)


def test_regen_count_penalty_defaults_to_the_probability_mass(simple_loss_inputs):
    """Regression guard: omitting the argument must reproduce the pre-2026-08-21
    objective exactly, so every previously-measured number stays comparable."""
    explicit = dict(simple_loss_inputs)
    explicit["regen_count_penalty"] = simple_loss_inputs["regen_probs"].sum()

    default_total, default_metrics = compute_loss(**simple_loss_inputs)
    explicit_total, explicit_metrics = compute_loss(**explicit)

    assert torch.allclose(default_total, explicit_total)
    assert default_metrics["regen_loss"] == explicit_metrics["regen_loss"]


def test_regen_count_penalty_overrides_the_mass(simple_loss_inputs):
    """A hard-concrete gate prices the expected COUNT, which is a different
    number from sum(p) — compute_loss must use what it is handed."""
    inputs = dict(simple_loss_inputs)
    inputs["lambda_regen"] = 1.0
    baseline, _ = compute_loss(**inputs)

    inputs["regen_count_penalty"] = simple_loss_inputs["regen_probs"].sum() + 3.0
    bumped, metrics = compute_loss(**inputs)

    assert torch.allclose(bumped - baseline, torch.tensor(3.0))
    assert metrics["regen_loss"] == pytest.approx(
        simple_loss_inputs["regen_probs"].sum().item() + 3.0
    )
