"""Unit tests for diffopt/train.py's pure annealing helper.

`linear_anneal` (renamed/generalized from `compute_regen_tau`) now drives
both RegenPlacement's `tau` and SegmentCombiner's `soft_max_temperature` —
the fix for the bug where soft_max_temperature was hardcoded at a fixed
0.5 and never annealed (see CLAUDE.md's Phase 1c corrections, and
diffopt/qot/segment_combiner.py's docstring, for the physics diagnosis).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml

import diffopt.train as train_mod
from diffopt.train import linear_anneal


def test_before_anneal_start_returns_start_value():
    assert linear_anneal(epoch=1, start=1.0, end=0.1, anneal_start=5, anneal_end=20) == 1.0
    assert linear_anneal(epoch=5, start=1.0, end=0.1, anneal_start=5, anneal_end=20) == 1.0


def test_after_anneal_end_returns_end_value():
    assert linear_anneal(epoch=20, start=1.0, end=0.1, anneal_start=5, anneal_end=20) == 0.1
    assert linear_anneal(epoch=100, start=1.0, end=0.1, anneal_start=5, anneal_end=20) == 0.1


def test_linear_interpolation_at_midpoint():
    # anneal_start=0, anneal_end=10, epoch=5 -> halfway between start and end
    result = linear_anneal(epoch=5, start=1.0, end=0.0, anneal_start=0, anneal_end=10)
    assert result == pytest.approx(0.5)


def test_monotonic_decrease_across_the_anneal_window():
    values = [
        linear_anneal(epoch=e, start=0.5, end=0.01, anneal_start=5, anneal_end=18)
        for e in range(1, 25)
    ]
    for i in range(1, len(values)):
        assert values[i] <= values[i - 1] + 1e-12, (
            f"Not monotonically non-increasing at index {i}: {values}"
        )
    assert values[0] == 0.5
    assert values[-1] == 0.01


def test_generic_across_different_schedules():
    """The same function must correctly drive both tau (1.0->0.1) and
    soft_max_temperature (0.5->0.01) schedules independently — this is the
    whole point of generalizing it rather than keeping two near-duplicate
    functions."""
    tau = linear_anneal(epoch=10, start=1.0, end=0.1, anneal_start=5, anneal_end=15)
    soft_max_temp = linear_anneal(epoch=10, start=0.5, end=0.01, anneal_start=5, anneal_end=15)

    # Same fractional progress (epoch 10 is 50% through [5,15]) but
    # different start/end -> different absolute values, same fraction.
    tau_frac = (tau - 0.1) / (1.0 - 0.1)
    temp_frac = (soft_max_temp - 0.01) / (0.5 - 0.01)
    assert tau_frac == pytest.approx(temp_frac)
    assert tau_frac == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Regression: best_loss must not be gated by checkpoint_interval
# ---------------------------------------------------------------------------

_MODULATION_FORMATS_PATH = Path(__file__).parent.parent / "configs/modulation_formats.yaml"


class _DummyTopology:
    """Stand-in for a loaded Topology — only `num_nodes` is read by main()
    before DiffONetPipeline is constructed (which is itself stubbed below)."""
    num_nodes = 5


class _DummyPipeline:
    """Stand-in for DiffONetPipeline. Its physics forward pass is irrelevant
    to the bug under test (a control-flow bug in the checkpoint-save gate),
    so it is replaced with a cheap no-op that satisfies main()'s call shape:
    `pipeline(demands, tau=..., lambda_=..., soft_max_temperature=...)` ->
    (path_noise_costs, gsnr_preds, path_indicators, regen_probs)."""

    def __init__(self, **kwargs) -> None:
        pass

    def to(self, device):
        return self

    def __call__(self, demands, tau=None, lambda_=None, soft_max_temperature=None):
        return {}, {}, {}, torch.zeros(_DummyTopology.num_nodes)


def test_best_loss_tracks_every_epoch_not_only_checkpoint_epochs(tmp_path, monkeypatch):
    """A record-low loss on a non-checkpoint-interval epoch must still be
    recorded and saved as the best checkpoint.

    Regression: train.py's checkpoint block gated the best_loss UPDATE on
    `epoch % checkpoint_interval == 0`, not just the torch.save. A record
    low on a non-interval epoch was therefore discarded *and* never
    recorded, so the saved "best" checkpoint reflected only the best among
    sampled epochs.

    This drives the real diffopt.train.main() epoch loop end to end —
    including the actual checkpoint-gating branch and torch.save call at
    the heart of the bug. Only the numerically expensive / physics-derived
    pieces that are irrelevant to this control-flow bug (topology loading,
    QoT checkpoint loading, demand generation, and the pipeline forward +
    compute_loss) are replaced with cheap deterministic stand-ins, so the
    per-epoch total_loss sequence is exactly controlled: losses =
    [5.0, 1.0, 4.0, 4.0] with checkpoint_interval=3 means the true minimum
    (1.0, epoch 2) lands on a non-interval epoch, and 4.0 (epoch 3) is the
    interval-sampled decoy the pre-fix code would keep instead.
    """
    losses = [5.0, 1.0, 4.0, 4.0]
    call_count = {"n": 0}

    def fake_load_topology(topology_path, modulation_formats_path):
        return _DummyTopology()

    def fake_load_qot_model(checkpoint_path, cfg, device):
        return None

    def fake_generate_demands(topology, num_demands, bitrate_options, seed):
        return []

    def fake_compute_loss(**kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        loss = torch.tensor(losses[idx], requires_grad=True)
        metrics = {
            "feasibility_loss": 0.0,
            "regen_loss": 0.0,
            "path_noise_loss": 0.0,
            "num_regen_soft": 0,
            "num_infeasible": 0,
        }
        return loss, metrics

    monkeypatch.setattr(train_mod, "load_topology", fake_load_topology)
    monkeypatch.setattr(train_mod, "load_qot_model", fake_load_qot_model)
    monkeypatch.setattr(train_mod, "generate_demands", fake_generate_demands)
    monkeypatch.setattr(train_mod, "DiffONetPipeline", _DummyPipeline)
    monkeypatch.setattr(train_mod, "compute_loss", fake_compute_loss)

    config = {
        "topology": "unused",
        "modulation_formats": str(_MODULATION_FORMATS_PATH),
        "qot_checkpoint": "unused",
        "num_demands": 1,
        "bitrate_options": [400],
        "max_spans_per_segment": 60,
        "seed": 42,
        "pipeline": {
            "lambda_regen": 1.0,
            "lambda_infeasible": 10.0,
            "lambda_cost": 0.01,
            "channel_loading_fraction": 0.5,
        },
        "segment_combiner": {
            "soft_max_temperature": 0.5,
            "soft_max_temperature_min": 0.01,
        },
        "training": {
            "lr_edge_net": 1.0e-3,
            "lr_regen": 1.0e-2,
            "epochs_e2e": len(losses),
            "vlastelica_lambda": 10.0,
            "vlastelica_lambda_min": 1.0,
            "vlastelica_lambda_decay": 0.995,
            "regen_tau_start": 1.0,
            "regen_tau_end": 0.1,
            "regen_tau_anneal_start_epoch": 1,
            "regen_tau_anneal_end_epoch": 4,
            "checkpoint_interval": 3,
        },
        "log_dir": str(tmp_path / "logs"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))

    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    train_mod.main()

    ckpt = torch.load(tmp_path / "checkpoints" / "best_e2e.pt", weights_only=False)
    assert ckpt["total_loss"] == pytest.approx(min(losses)), (
        f"saved checkpoint total_loss={ckpt['total_loss']} but true minimum "
        f"over the run was {min(losses)} (epoch {losses.index(min(losses)) + 1}, "
        f"not a multiple of checkpoint_interval=3)"
    )
