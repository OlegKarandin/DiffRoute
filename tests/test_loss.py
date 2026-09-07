"""Direct tests for compute_loss — shapes, empty inputs, and metrics keys."""
import math

import torch
import torch.nn.functional as F
import pytest

from diffopt.demands import Demand
from diffopt.loss import compute_loss, update_duals
from diffopt.modulation import ModulationConfig

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
        rho=20.0,
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
        rho=20.0,
    )
    assert total.shape == ()
    assert metrics["num_infeasible"] == 0
    assert metrics["num_violated"] == 0
    assert metrics["path_noise_loss"] == 0.0
    assert math.isnan(metrics["worst_margin_db"])


def test_metrics_dict_has_the_keys_train_py_logs(simple_loss_inputs):
    """train.py and every published milestone claim read these keys."""
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
    ):
        assert key in metrics, f"missing metrics key: {key}"


def test_num_infeasible_counts_only_below_threshold_demands(simple_loss_inputs):
    _, metrics = compute_loss(**simple_loss_inputs)
    assert metrics["num_infeasible"] == 1


# ---------------------------------------------------------------------------
# Equivalence oracle for the A4 vectorisation (perf task 7): compute_loss used
# to loop `for demand in demands:` in Python, doing ~3 device syncs per
# demand. The loop below is a verbatim copy of that OLD per-demand logic,
# kept here as a manual reference oracle so the vectorized compute_loss can
# be checked against it directly, rather than trusting that a refactor
# preserved behaviour by inspection.
# ---------------------------------------------------------------------------


def test_vectorized_compute_loss_matches_the_old_per_demand_loop():
    """demand-list order is deliberately NOT demand.id order here (ids
    [4, 0, 5, 2, 1, 3] against list positions [0..5]), to exercise the
    index-mismatch trap: `shortfalls`/`constraint_g` are scattered by
    `demand.id`, while `bar_db_for_demands` and the stacked GSNR predictions
    are in demand-list order. Getting the scatter wrong here would pass on
    the existing tests (which all happen to use contiguous, list-order ids)
    but fail here."""
    torch.manual_seed(12345)

    mod_cfg = ModulationConfig(
        channel_spacing_ghz=100.0,
        symbol_rate_gbaud=64.0,
        num_channels_cband=48,
        cut_channel_index=24,
        formats=[
            {"bitrate_gbps": 100, "snr_threshold_db": 10.0},
            {"bitrate_gbps": 200, "snr_threshold_db": 14.0},
            {"bitrate_gbps": 300, "snr_threshold_db": 18.0},
            {"bitrate_gbps": 400, "snr_threshold_db": 22.0},
            {"bitrate_gbps": 500, "snr_threshold_db": 26.0},
        ],
    )
    bitrate_options = mod_cfg.bitrate_options

    num_demands = 6
    id_order = [4, 0, 5, 2, 1, 3]  # list position i -> demand.id id_order[i]
    bitrates = [bitrate_options[i % len(bitrate_options)] for i in range(num_demands)]
    demands = [
        Demand(id=id_order[i], src=i, dst=(i + 1) % num_demands,
               bitrate_gbps=float(bitrates[i]))
        for i in range(num_demands)
    ]

    # GSNR predictions scattered around each demand's own threshold, so the
    # random draw produces a realistic mix of violated/feasible/margin-only
    # demands. Two independent leaf-tensor copies (same values) so the
    # reference loop's backward() and the vectorized call's backward() don't
    # accumulate into the same .grad.
    raw_values = {}
    for d in demands:
        threshold = mod_cfg.required_snr_threshold(d.bitrate_gbps)
        raw_values[d.id] = threshold + torch.randn(()).item() * 3.0
    gsnr_preds_ref = {i: torch.tensor(v, requires_grad=True) for i, v in raw_values.items()}
    gsnr_preds_vec = {i: torch.tensor(v, requires_grad=True) for i, v in raw_values.items()}

    path_noise_costs = {d.id: torch.rand(()) * 2.0 for d in demands}
    duals = torch.rand(num_demands) * 5.0
    device_count = torch.tensor(4.0)
    rho = 3.7
    margin_db = 0.35
    lambda_dev = 0.2
    lambda_cost = 0.05

    # --- Reference oracle: verbatim copy of the OLD per-demand loop ---
    device = duals.device
    weighted_feasibility_ref = torch.zeros((), device=device)
    feasibility_loss_ref = torch.zeros((), device=device)
    shortfalls_ref = torch.zeros(duals.shape[0], device=device)
    constraint_g_ref = torch.zeros(duals.shape[0], device=device)
    num_infeasible_ref = 0
    num_violated_ref = 0
    worst_margin_db_ref = math.inf

    for demand in demands:
        threshold = mod_cfg.required_snr_threshold(demand.bitrate_gbps)
        bar_t = torch.tensor(threshold + margin_db, device=device, dtype=torch.float32)

        g = bar_t - gsnr_preds_ref[demand.id]
        shortfall = F.relu(g)

        z = F.relu(duals[demand.id] + rho * g)
        weighted_feasibility_ref = weighted_feasibility_ref + (
            (z * z - duals[demand.id] ** 2) / (2.0 * rho)
        )

        feasibility_loss_ref = feasibility_loss_ref + shortfall

        shortfall_value = shortfall.item()
        shortfalls_ref[demand.id] = shortfall_value
        constraint_g_ref[demand.id] = g.item()
        if shortfall_value > 0:
            num_violated_ref += 1

        margin = gsnr_preds_ref[demand.id].item() - threshold
        if margin < 0:
            num_infeasible_ref += 1
        worst_margin_db_ref = min(worst_margin_db_ref, margin)

    path_noise_loss_ref = sum(path_noise_costs.values(), torch.zeros((), device=device))
    total_ref = (
        weighted_feasibility_ref
        + lambda_dev * device_count
        + lambda_cost * path_noise_loss_ref
    )
    total_ref.backward()

    # --- Vectorized implementation under test ---
    total, metrics = compute_loss(
        gsnr_preds=gsnr_preds_vec,
        path_noise_costs=path_noise_costs,
        demands=demands,
        device_count=device_count,
        modulation_config=mod_cfg,
        duals=duals,
        rho=rho,
        margin_db=margin_db,
        lambda_dev=lambda_dev,
        lambda_cost=lambda_cost,
    )
    total.backward()

    # Integer counters: exact.
    assert metrics["num_violated"] == num_violated_ref
    assert metrics["num_infeasible"] == num_infeasible_ref

    # Per-demand output vectors, indexed by demand.id: exact.
    assert torch.equal(metrics["shortfalls"], shortfalls_ref)
    assert torch.equal(metrics["constraint_g"], constraint_g_ref)
    assert not metrics["shortfalls"].requires_grad
    assert not metrics["constraint_g"].requires_grad

    # Exact, not approx: both paths compute this bare-threshold margin at
    # float64 precision (the old loop via `.item()` widening, the vectorized
    # version explicitly), and min() is order-independent, so there is no
    # summation-order noise to tolerate here.
    assert metrics["worst_margin_db"] == worst_margin_db_ref

    # Scalar reductions: float32 tolerance (summation-order noise between the
    # accumulating loop and the batched .sum() is expected, not a bug — same
    # rel=1e-6 convention as test_device_count_is_priced_by_lambda_dev above).
    assert metrics["weighted_feasibility_loss"] == pytest.approx(
        weighted_feasibility_ref.item(), rel=1e-6
    )
    assert metrics["feasibility_loss"] == pytest.approx(
        feasibility_loss_ref.item(), rel=1e-6
    )
    assert total.item() == pytest.approx(total_ref.item(), rel=1e-6)

    # Gradient still flows through the batched stack/gather, per demand.
    for d in demands:
        assert gsnr_preds_vec[d.id].grad is not None
        assert gsnr_preds_vec[d.id].grad.item() == pytest.approx(
            gsnr_preds_ref[d.id].grad.item(), rel=1e-6
        )


# ---------------------------------------------------------------------------
# Constrained objective: per-demand duals + margin inside the hinge
# ---------------------------------------------------------------------------



def test_margin_keeps_the_penalty_active_above_threshold():
    """The fix for spec finding #3.

    `relu(thr - gsnr)` is exactly zero the moment a demand clears, so nothing
    pushes for headroom — measured |g_feas|_1 = 0.000 on fully-feasible epochs,
    every logit pushed down and none up. With a margin, a demand sitting at
    thr + delta/2 still contributes force under the augmented penalty too
    (its bar is thr + delta, so g = delta/2 > 0 — still on the violated side).
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
        rho=20.0,
    )
    # z = relu(1.0 + 20.0*0.25) = 6.0 -> (36 - 1) / 40 = 0.875.
    assert total.item() == pytest.approx(0.875)
    assert metrics["num_violated"] == 1, "violates threshold + margin"
    assert metrics["num_infeasible"] == 0, "but is physically feasible"

    total.backward()
    assert gsnr.grad.item() == pytest.approx(-6.0), (
        "raising GSNR must reduce the loss — without a margin this gradient "
        "is exactly zero above the threshold"
    )


def test_num_violated_is_zero_once_the_margin_is_cleared():
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
        rho=20.0,
    )
    # Not exactly 0: the augmented term carries a negative constant on a
    # comfortably-slack demand (test_augmented_term_value_matches_the_closed_form),
    # unlike the removed hinge's exact zero past the bar. What survives is
    # that the demand no longer counts as violated.
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
        rho=20.0,
    )
    base, _ = compute_loss(duals=torch.tensor([1.0, 1.0]), **common)
    bumped, _ = compute_loss(duals=torch.tensor([2.0, 1.0]), **common)
    # Both demands stay violated (g > 0) at both dual values, so the
    # augmented term is LINEAR in dual there — same as the removed hinge —
    # and the difference is exactly demand 0's shortfall, 20 - 18 = 2.0.
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
        rho=20.0,
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
        rho=20.0,
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
# Augmented Lagrangian (spec 2026-08-31 sections 2.1 and 2.4)
#
# make_mod_config is a single 400 G format at a 20.0 dB threshold, so with
# margin_db = 0.5 the bar is 20.5 dB and g = 20.5 - gsnr throughout.
# ---------------------------------------------------------------------------

def _one_demand_loss(gsnr_value, *, dual, rho, margin_db=0.5):
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
        rho=rho,
    )
    return total, metrics, gsnr


def test_augmented_is_active_inside_the_feasible_region():
    """The reason this design exists (spec 1.1): a one-sided hinge term,
    `dual_d * relu(g_d)`, contributes exactly zero force for a demand 0.2 dB
    INSIDE its bar, no matter how large its dual — relu'(g) = 0 for g < 0 —
    so at an optimum where every demand is feasible the only surviving force
    is -lambda_dev, and stationarity would require lambda_dev = 0. The
    augmented penalty does not have this defect: at lambda = 10, rho = 20
    the same demand feels max(0, 10 + 20*(-0.2)) = 6.
    """
    aug_total, _, aug_gsnr = _one_demand_loss(20.7, dual=10.0, rho=20.0)
    aug_total.backward()
    assert aug_gsnr.grad.item() == pytest.approx(-6.0, abs=1e-4)


def test_augmented_collapses_to_zero_past_the_band():
    """What stops 346 slack demands summing into an over-buy bias, and what
    disqualifies the softplus variant (spec 7): past g < -lambda/rho the
    force is EXACTLY zero, not merely small."""
    # lambda/rho = 10/20 = 0.5 dB, so g = -0.6 is outside the band.
    total, _, gsnr = _one_demand_loss(21.1, dual=10.0, rho=20.0)
    total.backward()
    assert gsnr.grad.item() == 0.0


def test_augmented_force_is_max_zero_lambda_plus_rho_g():
    """The derivative in spec 2.1, by direct autograd, on the violated side."""
    total, _, gsnr = _one_demand_loss(20.2, dual=10.0, rho=20.0)
    total.backward()
    # g = +0.3 -> lambda + rho*g = 16.0, and raising gsnr lowers the loss.
    assert gsnr.grad.item() == pytest.approx(-16.0)


def test_augmented_force_at_the_bar_equals_lambda():
    """At g = 0 the force is exactly lambda — the same value a one-sided
    hinge takes on its violated side, so the two agree at the boundary."""
    aug_total, _, aug_gsnr = _one_demand_loss(20.5, dual=10.0, rho=20.0)
    aug_total.backward()
    assert aug_gsnr.grad.item() == pytest.approx(-10.0)


def test_augmented_term_value_matches_the_closed_form():
    """(relu(lambda + rho*g)^2 - lambda^2) / (2*rho), including the negative
    constant a comfortably-slack demand contributes — that constant is what
    makes the term continuous at the kink."""
    inside, m_inside, _ = _one_demand_loss(20.7, dual=10.0, rho=20.0)
    # z = relu(10 + 20*(-0.2)) = 6 -> (36 - 100) / 40 = -1.6
    assert inside.item() == pytest.approx(-1.6, abs=1e-4)
    assert m_inside["weighted_feasibility_loss"] == pytest.approx(-1.6, abs=1e-4)

    outside, _, _ = _one_demand_loss(21.1, dual=10.0, rho=20.0)
    # z = relu(10 - 12) = 0 -> (0 - 100) / 40 = -2.5
    assert outside.item() == pytest.approx(-2.5)


def test_feasibility_loss_is_sum_relu_g():
    """The LOGGED diagnostic — the one-sided sum kept comparable across
    epochs while the duals are non-stationary — stays sum(relu(g))
    regardless of the weighted term's own shape."""
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
        rho=20.0,
    )
    # bar = 20.5: relu(2.5) + relu(-4.5) = 2.5.
    assert metrics["feasibility_loss"] == pytest.approx(2.5)


def test_constraint_g_is_signed_and_detached():
    """update_duals' input. A slack demand's entry must be NEGATIVE — that
    is the whole difference from `shortfalls`, and it is what lets the dual
    fall on slack instead of ratcheting."""
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
        rho=20.0,
    )
    g = metrics["constraint_g"]
    assert g.shape == (2,)
    assert not g.requires_grad, "dual updates must not build a graph"
    assert g[0].item() == pytest.approx(2.5)
    assert g[1].item() == pytest.approx(-4.5)
    # `shortfalls` stays one-sided, for the violated count and the diagnostic.
    assert metrics["shortfalls"][1].item() == pytest.approx(0.0)


def test_compute_loss_requires_rho():
    """Spec 3: never a silent fallback. rho is keyword-only with no
    default, so omitting it is a TypeError at the call boundary, not a
    guessed band width."""
    with pytest.raises(TypeError, match="rho"):
        compute_loss(
            gsnr_preds={0: torch.tensor(20.7)},
            path_noise_costs={0: torch.tensor(0.0)},
            demands=[Demand(id=0, src=0, dst=1, bitrate_gbps=400.0)],
            device_count=torch.zeros(()),
            modulation_config=make_mod_config(),
            duals=torch.tensor([10.0]),
        )


def test_non_positive_rho_raises():
    with pytest.raises(ValueError, match="rho"):
        _one_demand_loss(20.7, dual=10.0, rho=0.0)


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
            lambda_dev=0.0, lambda_cost=0.0, rho=rho,
        )
        g = metrics["constraint_g"]
        # force = max(0, lambda + rho*g); the slack demand's lambda is 0
        # every epoch, so its force is exactly 0, not merely small.
        assert max(0.0, duals[1].item() + rho * g[1].item()) == 0.0
        duals = update_duals(duals, g, eta=rho, dual_max=1000.0)
        assert duals[0].item() == pytest.approx(epoch + 1.0)  # ascends by rho*g=1.0/epoch
        assert duals[1].item() == 0.0  # floored, never ratchets on slack
