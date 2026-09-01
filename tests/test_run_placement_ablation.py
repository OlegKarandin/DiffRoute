"""The arm harness's trajectory scoring, unit-tested away from a real sweep."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_placement_ablation import ARMS, CSV_FIELDNAMES, trajectory_shape_from_counts


def test_a_rising_and_plateauing_run_passes():
    rising = [0, 2, 5, 9, 12, 13, 13, 13, 13]
    assert trajectory_shape_from_counts(rising)["device_plateaued"] is True


def test_a_spike_and_descend_run_is_flagged():
    """The 2.2 saturation failure: 900 devices at epoch 3 against an oracle
    needing ~60, then a long descent. Final value 66 looks fine."""
    spiking = [0, 400, 900, 700, 300, 120, 80, 70, 66]
    shape = trajectory_shape_from_counts(spiking)
    assert shape["device_plateaued"] is False
    assert shape["device_peak"] == 900


def test_empty_trajectory_does_not_raise():
    assert trajectory_shape_from_counts([])["device_plateaued"] is False


def test_no_arm_references_a_deleted_config_key():
    """Regression guard: an arm overriding `placement.gate` would write a
    config key nothing reads, and the arm would silently be a duplicate of
    baseline."""
    dead = {"gate", "gate_dropout_p", "hard_concrete", "lambda_regen", "lr_regen"}
    for arm in ARMS:
        for section in arm["overrides"].values():
            assert not (set(section) & dead), arm["name"]
    assert not (set(CSV_FIELDNAMES) & dead)


def test_the_augmented_arms_are_registered():
    """Spec section 6 needs four runs: al_ste_greedy on three seeds, plus
    baseline-under-AL on one for the no-regression comparison."""
    names = {arm["name"] for arm in ARMS}
    assert {"al_ste_greedy", "al_baseline"} <= names


def test_every_augmented_arm_carries_a_measured_rho_and_a_cold_dual():
    """rho has no default and compute_loss raises without it, so an arm that
    forgot it would fail at epoch 1 rather than silently run the hinge. And
    dual_init must drop to 0: under AL a slack demand's force is
    max(0, lambda + rho*g), so a nonzero starting lambda pushes every demand
    up before any of them has ever been violated."""
    augmented = [
        arm for arm in ARMS
        if arm["overrides"].get("constraint", {}).get("penalty") == "augmented"
    ]
    assert augmented, "no augmented arm registered"
    for arm in augmented:
        constraint = arm["overrides"]["constraint"]
        assert constraint.get("rho", 0.0) > 0.0, arm["name"]
        assert constraint.get("dual_init") == 0.0, arm["name"]


def test_the_augmented_ste_arm_pins_tau_and_turns_on_the_ste_and_residual():
    """The AL defect only EXISTS once the STE removes the phantom violations
    (spec 1.2), so the arm this rho was measured for must carry them."""
    arm = next(a for a in ARMS if a["name"] == "al_ste_greedy")
    assert arm["overrides"]["placement"] == {"alloc_ste": True, "greedy_residual": True}
    assert arm["overrides"]["training"]["alloc_tau_end"] == 1.0


def test_the_augmented_baseline_arm_changes_only_the_penalty():
    """Its whole job is the no-regression comparison against `baseline`, so
    it must differ from it in the constraint block and nowhere else."""
    arm = next(a for a in ARMS if a["name"] == "al_baseline")
    assert set(arm["overrides"]) == {"constraint"}
