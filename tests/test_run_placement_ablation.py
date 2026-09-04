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
    dead = {
        "gate", "gate_dropout_p", "hard_concrete", "lambda_regen", "lr_regen",
        "penalty", "dual_lr", "dual_decay", "alloc_optimizer",
        "alloc_weight_decay", "greedy_residual", "alloc_dropout_p",
        "lambda_waste", "alloc_anti_windup", "alloc_ste_tau10", "alloc_ste_tau30",
    }
    for arm in ARMS:
        for section in arm["overrides"].values():
            assert not (set(section) & dead), arm["name"]
    assert not (set(CSV_FIELDNAMES) & dead)


def test_alloc_ste_arm_pins_tau_and_turns_on_the_ste():
    """The STE defect (spec 1.2) is what the augmented penalty needs fixed
    to have a nonzero band; every arm now runs under the augmented penalty
    via constrained_stress.yaml's own constraint.rho, so this is the arm
    that carries the STE."""
    arm = next(a for a in ARMS if a["name"] == "alloc_ste")
    assert arm["overrides"]["placement"] == {"alloc_ste": True}
    assert arm["overrides"]["training"]["alloc_tau_end"] == 1.0
