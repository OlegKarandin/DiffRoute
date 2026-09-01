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
    return dict(
        gsnr_preds=gsnr_preds,
        path_noise_costs=path_noise_costs,
        demands=demands,
        device_count=torch.zeros(()),
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
        device_count=torch.zeros(()),
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
        "device_count",
        "path_noise_loss",
        "num_infeasible",
        "weighted_feasibility_loss",
        "num_violated",
        "worst_margin_db",
        "shortfalls",
        "waste_cost",
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
        device_count=torch.zeros(()),
        modulation_config=mod_cfg,
        duals=torch.tensor([1.0]),
        margin_db=margin_db,
        lambda_dev=0.0,
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
        device_count=torch.zeros(()),
        modulation_config=mod_cfg,
        duals=torch.tensor([1.0]),
        margin_db=0.5,
        lambda_dev=0.0,
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
        device_count=torch.zeros(()),
        modulation_config=mod_cfg,
        margin_db=0.0,
        lambda_dev=0.0,
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
        device_count=torch.zeros(()),
        modulation_config=mod_cfg,
        duals=torch.tensor([1.0, 1.0]),
        margin_db=0.0,
        lambda_dev=0.0,
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
        device_count=torch.zeros(()),
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


def test_device_count_is_priced_by_lambda_dev(simple_loss_inputs):
    """The whole point of the stage: the objective's second term is a count
    of DEVICES, sum_{d,k} a[d,k], not a count of sites."""
    inputs = dict(simple_loss_inputs)
    inputs["lambda_dev"] = 2.0
    inputs["device_count"] = torch.tensor(7.0)
    with_devices, m = compute_loss(**inputs)
    inputs["device_count"] = torch.tensor(0.0)
    without, _ = compute_loss(**inputs)
    assert (with_devices - without).item() == pytest.approx(14.0, rel=1e-6)
    assert m["device_count"] == pytest.approx(7.0)


def test_device_term_carries_gradient_to_the_allocation(simple_loss_inputs):
    inputs = dict(simple_loss_inputs)
    inputs["lambda_dev"] = 1.0
    a = torch.tensor([[0.3, 0.7], [0.1, 0.0]], requires_grad=True)
    inputs["device_count"] = a.sum()
    total, _ = compute_loss(**inputs)
    total.backward()
    assert a.grad is not None
    assert torch.allclose(a.grad, torch.ones_like(a))


def test_no_regen_probs_parameter_survives(simple_loss_inputs):
    """Regression guard: a leftover regen_probs kwarg is how site pricing
    creeps back in."""
    import inspect

    params = inspect.signature(compute_loss).parameters
    assert "regen_probs" not in params
    assert "lambda_regen" not in params
    assert "regen_count_penalty" not in params


# ---------------------------------------------------------------------------
# Waste surcharge (arm 4, spec 5.3) — see task-6-brief.md section 9 test 8
# ---------------------------------------------------------------------------

def test_lambda_waste_defaults_to_no_op(simple_loss_inputs):
    """With lambda_waste = 0.0 (the default) the returned total must be
    BIT-IDENTICAL to the pre-change formula, regardless of what waste_cost
    is — 0.0 * anything_finite is exactly +0.0 by IEEE-754, so the no-op case
    must not depend on a branch that happens to skip the term."""
    no_waste_args, _ = compute_loss(**simple_loss_inputs)

    nonzero_waste_zero_weight, _ = compute_loss(
        **simple_loss_inputs,
        waste_cost=torch.tensor(123.456),
        lambda_waste=0.0,
    )

    explicit_defaults, _ = compute_loss(
        **simple_loss_inputs,
        waste_cost=None,
        lambda_waste=0.0,
    )

    assert torch.equal(no_waste_args, nonzero_waste_zero_weight)
    assert torch.equal(no_waste_args, explicit_defaults)


def test_lambda_waste_nonzero_without_waste_cost_raises(simple_loss_inputs):
    """The only silent path is lambda_waste == 0.0 — a non-zero weight with
    no waste_cost supplied must raise, not silently treat it as 0."""
    inputs = dict(simple_loss_inputs)
    inputs["lambda_waste"] = 1.0
    with pytest.raises(ValueError):
        compute_loss(**inputs)


def test_lambda_waste_scales_the_waste_term(simple_loss_inputs):
    inputs = dict(simple_loss_inputs)
    inputs["lambda_waste"] = 2.0
    inputs["waste_cost"] = torch.tensor(5.0)
    with_waste, m = compute_loss(**inputs)
    inputs["waste_cost"] = torch.tensor(0.0)
    without_waste, _ = compute_loss(**inputs)
    assert (with_waste - without_waste).item() == pytest.approx(10.0, rel=1e-6)
    assert m["waste_cost"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Augmented Lagrangian (spec 2026-08-31 sections 2.1 and 2.4)
#
# make_mod_config is a single 400 G format at a 20.0 dB threshold, so with
# margin_db = 0.5 the bar is 20.5 dB and g = 20.5 - gsnr throughout.
# ---------------------------------------------------------------------------

def _one_demand_loss(gsnr_value, *, dual, penalty, rho=None, margin_db=0.5):
    """compute_loss over a single demand, with every term but feasibility
    switched off, returning (total, metrics, the gsnr leaf)."""
    mod_cfg = make_mod_config()
    gsnr = torch.tensor(gsnr_value, requires_grad=True)
    total, metrics = compute_loss(
        gsnr_preds={0: gsnr},
        path_noise_costs={0: torch.tensor(0.0)},
        demands=[Demand(id=0, src=0, dst=1, bitrate_gbps=400.0)],
        device_count=torch.zeros(()),
        modulation_config=mod_cfg,
        duals=torch.tensor([dual]),
        margin_db=margin_db,
        lambda_dev=0.0,
        lambda_cost=0.0,
        penalty=penalty,
        rho=rho,
    )
    return total, metrics, gsnr


def test_augmented_is_active_inside_the_feasible_region():
    """The reason this change exists (spec 1.1).

    A demand 0.2 dB INSIDE its bar contributes exactly zero force under the
    hinge, no matter how large its dual — relu'(g) = 0 for g < 0 — so at an
    optimum where every demand is feasible the only surviving force is
    -lambda_dev, and no non-negative (lambda_dev, lambda_waste) makes that
    optimum a stationary point. Under the augmented penalty the same demand
    at lambda = 10, rho = 20 feels max(0, 10 + 20*(-0.2)) = 6.
    """
    hinge_total, _, hinge_gsnr = _one_demand_loss(20.7, dual=10.0, penalty="hinge")
    hinge_total.backward()
    assert hinge_gsnr.grad.item() == 0.0, "the defect: a satisfied demand pushes nothing"

    aug_total, _, aug_gsnr = _one_demand_loss(20.7, dual=10.0, penalty="augmented", rho=20.0)
    aug_total.backward()
    assert aug_gsnr.grad.item() == pytest.approx(-6.0, abs=1e-4)


def test_augmented_collapses_to_zero_past_the_band():
    """What stops 346 slack demands summing into an over-buy bias, and what
    disqualifies the softplus variant (spec 7): past g < -lambda/rho the
    force is EXACTLY zero, not merely small."""
    # lambda/rho = 10/20 = 0.5 dB, so g = -0.6 is outside the band.
    total, _, gsnr = _one_demand_loss(21.1, dual=10.0, penalty="augmented", rho=20.0)
    total.backward()
    assert gsnr.grad.item() == 0.0


def test_augmented_force_is_max_zero_lambda_plus_rho_g():
    """The derivative in spec 2.1, by direct autograd, on the violated side."""
    total, _, gsnr = _one_demand_loss(20.2, dual=10.0, penalty="augmented", rho=20.0)
    total.backward()
    # g = +0.3 -> lambda + rho*g = 16.0, and raising gsnr lowers the loss.
    assert gsnr.grad.item() == pytest.approx(-16.0)


def test_augmented_matches_the_hinge_force_on_the_bar():
    """Continuity with today at g = 0: the augmented force is exactly lambda,
    the value the hinge takes on its violated side."""
    aug_total, _, aug_gsnr = _one_demand_loss(20.5, dual=10.0, penalty="augmented", rho=20.0)
    aug_total.backward()
    assert aug_gsnr.grad.item() == pytest.approx(-10.0)

    hinge_total, _, hinge_gsnr = _one_demand_loss(20.4, dual=10.0, penalty="hinge")
    hinge_total.backward()
    assert hinge_gsnr.grad.item() == pytest.approx(-10.0)


def test_augmented_term_value_matches_the_closed_form():
    """(relu(lambda + rho*g)^2 - lambda^2) / (2*rho), including the negative
    constant a comfortably-slack demand contributes — that constant is what
    makes the term continuous at the kink."""
    inside, m_inside, _ = _one_demand_loss(20.7, dual=10.0, penalty="augmented", rho=20.0)
    # z = relu(10 + 20*(-0.2)) = 6 -> (36 - 100) / 40 = -1.6
    assert inside.item() == pytest.approx(-1.6, abs=1e-4)
    assert m_inside["weighted_feasibility_loss"] == pytest.approx(-1.6, abs=1e-4)

    outside, _, _ = _one_demand_loss(21.1, dual=10.0, penalty="augmented", rho=20.0)
    # z = relu(10 - 12) = 0 -> (0 - 100) / 40 = -2.5
    assert outside.item() == pytest.approx(-2.5)


def test_feasibility_loss_is_sum_relu_g_in_both_modes():
    """Spec 2.4: the LOGGED diagnostic must stay comparable across arms, so
    only the weighted term changes shape."""
    mod_cfg = make_mod_config()
    demands = [
        Demand(id=0, src=0, dst=1, bitrate_gbps=400.0),
        Demand(id=1, src=1, dst=2, bitrate_gbps=400.0),
    ]
    common = dict(
        gsnr_preds={0: torch.tensor(18.0), 1: torch.tensor(25.0)},
        path_noise_costs={0: torch.tensor(0.0), 1: torch.tensor(0.0)},
        demands=demands,
        device_count=torch.zeros(()),
        modulation_config=mod_cfg,
        duals=torch.tensor([10.0, 10.0]),
        margin_db=0.5,
        lambda_dev=0.0,
        lambda_cost=0.0,
    )
    _, hinge_m = compute_loss(penalty="hinge", **common)
    _, aug_m = compute_loss(penalty="augmented", rho=20.0, **common)
    # bar = 20.5: relu(2.5) + relu(-4.5) = 2.5 in both modes.
    assert hinge_m["feasibility_loss"] == pytest.approx(2.5)
    assert aug_m["feasibility_loss"] == pytest.approx(2.5)


def test_constraint_g_is_signed_and_detached():
    """update_duals' augmented input. A slack demand's entry must be
    NEGATIVE — that is the whole difference from `shortfalls`, and it is what
    lets the dual fall on slack instead of ratcheting."""
    mod_cfg = make_mod_config()
    demands = [
        Demand(id=0, src=0, dst=1, bitrate_gbps=400.0),
        Demand(id=1, src=1, dst=2, bitrate_gbps=400.0),
    ]
    _, metrics = compute_loss(
        gsnr_preds={0: torch.tensor(18.0), 1: torch.tensor(25.0)},
        path_noise_costs={0: torch.tensor(0.0), 1: torch.tensor(0.0)},
        demands=demands,
        device_count=torch.zeros(()),
        modulation_config=mod_cfg,
        duals=torch.tensor([10.0, 10.0]),
        margin_db=0.5,
        lambda_dev=0.0,
        lambda_cost=0.0,
        penalty="augmented",
        rho=20.0,
    )
    g = metrics["constraint_g"]
    assert g.shape == (2,)
    assert not g.requires_grad, "dual updates must not build a graph"
    assert g[0].item() == pytest.approx(2.5)
    assert g[1].item() == pytest.approx(-4.5)
    # `shortfalls` stays one-sided, for hinge mode and the violated count.
    assert metrics["shortfalls"][1].item() == pytest.approx(0.0)


def test_constraint_g_is_returned_in_hinge_mode_too():
    """It costs one subtraction and keeps the metrics dict one shape, so a
    diagnostic reading it does not have to know which mode produced it."""
    _, metrics, _ = _one_demand_loss(20.7, dual=10.0, penalty="hinge")
    assert metrics["constraint_g"][0].item() == pytest.approx(-0.2, abs=1e-4)


def test_augmented_without_rho_raises():
    """Spec 3: never a silent fallback. Mirrors lambda_waste/waste_cost."""
    with pytest.raises(ValueError, match="rho"):
        _one_demand_loss(20.7, dual=10.0, penalty="augmented", rho=None)


def test_non_positive_rho_raises():
    with pytest.raises(ValueError, match="rho"):
        _one_demand_loss(20.7, dual=10.0, penalty="augmented", rho=0.0)


def test_unknown_penalty_raises():
    with pytest.raises(ValueError, match="penalty"):
        _one_demand_loss(20.7, dual=10.0, penalty="softplus")


def test_hinge_mode_is_bit_identical_to_the_default(simple_loss_inputs):
    """Regression guard. `penalty` defaults to hinge, and naming it
    explicitly must change nothing — not the total, not a metric."""
    default_total, default_m = compute_loss(**simple_loss_inputs)
    explicit_total, explicit_m = compute_loss(penalty="hinge", **simple_loss_inputs)
    assert torch.equal(default_total, explicit_total)
    assert default_m["weighted_feasibility_loss"] == explicit_m["weighted_feasibility_loss"]
    assert torch.equal(default_m["shortfalls"], explicit_m["shortfalls"])


def test_augmented_dual_falls_on_slack_under_the_signed_update():
    """Spec 2.2, on the UNCHANGED update_duals: signed g at step rho gives
    lambda <- clamp(lambda + rho*g, 0, dual_max), which decreases on slack by
    construction — the reason dual_decay becomes structurally unnecessary."""
    updated = update_duals(torch.tensor([10.0]), torch.tensor([-0.2]),
                           eta=20.0, dual_max=1000.0)
    assert updated[0].item() == pytest.approx(6.0)


def test_augmented_dual_rises_on_violation_and_respects_the_cap():
    rising = update_duals(torch.tensor([10.0]), torch.tensor([0.5]),
                          eta=20.0, dual_max=1000.0)
    assert rising[0].item() == pytest.approx(20.0)
    capped = update_duals(torch.tensor([990.0]), torch.tensor([5.0]),
                          eta=20.0, dual_max=1000.0)
    assert capped[0].item() == pytest.approx(1000.0)


def test_augmented_dual_floors_at_zero_on_deep_slack():
    """Complementary slackness: a demand that does not need the cut has its
    price decay to 0, its force vanishes, and the cut sheds."""
    updated = update_duals(torch.tensor([1.0]), torch.tensor([-10.0]),
                           eta=20.0, dual_max=1000.0)
    assert updated[0].item() == pytest.approx(0.0)


def test_augmented_composed_seam_dual_dynamics_across_epochs():
    """The seam no per-task review could see: real compute_loss feeding real
    update_duals across epochs, at rho, the way train.py's loop actually
    does it — each function above is only tested in isolation."""
    mod_cfg = make_mod_config()
    demands = [
        Demand(id=0, src=0, dst=1, bitrate_gbps=400.0),  # violated: g = +1.0
        Demand(id=1, src=1, dst=2, bitrate_gbps=400.0),  # deep slack: g = -4.5
    ]
    gsnr_preds = {0: torch.tensor(19.5), 1: torch.tensor(25.0)}
    path_noise_costs = {0: torch.tensor(0.0), 1: torch.tensor(0.0)}
    rho = 1.0
    duals = torch.tensor([0.0, 0.0])
    for epoch in range(3):
        _, metrics = compute_loss(
            gsnr_preds=gsnr_preds, path_noise_costs=path_noise_costs,
            demands=demands, device_count=torch.zeros(()),
            modulation_config=mod_cfg, duals=duals, margin_db=0.5,
            lambda_dev=0.0, lambda_cost=0.0, penalty="augmented", rho=rho,
        )
        g = metrics["constraint_g"]
        # force = max(0, lambda + rho*g); the slack demand's lambda is 0
        # every epoch, so its force is exactly 0, not merely small.
        assert max(0.0, duals[1].item() + rho * g[1].item()) == 0.0
        duals = update_duals(duals, g, eta=rho, dual_max=1000.0)
        assert duals[0].item() == pytest.approx(epoch + 1.0)  # ascends by rho*g=1.0/epoch
        assert duals[1].item() == 0.0  # floored, never ratchets on slack
