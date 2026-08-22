"""Unit tests for diffopt/train.py's pure annealing helper.

`linear_anneal` (renamed/generalized from `compute_regen_tau`) drives
RegenPlacement's `tau`. It stays generic — nothing in it is tau-specific —
because it also used to drive SegmentCombiner's `soft_max_temperature`,
before that fold became exact and lost its temperature entirely (see
docs/investigations/CHANGELOG.md's Phase 1c corrections, and
diffopt/qot/segment_combiner.py's docstring, for that history).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import torch
import yaml

import diffopt.train as train_mod
from diffopt.demands import Demand
from diffopt.placement.regenerator import RegenPlacement
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
    """The same function must drive two different start/end schedules
    independently — this is the whole point of generalizing it rather than
    keeping a near-duplicate function per annealed scalar."""
    tau = linear_anneal(epoch=10, start=1.0, end=0.1, anneal_start=5, anneal_end=15)
    other = linear_anneal(epoch=10, start=0.5, end=0.01, anneal_start=5, anneal_end=15)

    # Same fractional progress (epoch 10 is 50% through [5,15]) but
    # different start/end -> different absolute values, same fraction.
    tau_frac = (tau - 0.1) / (1.0 - 0.1)
    other_frac = (other - 0.01) / (0.5 - 0.01)
    assert tau_frac == pytest.approx(other_frac)
    assert tau_frac == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Checkpoint selection under the constrained objective
# ---------------------------------------------------------------------------

_MODULATION_FORMATS_PATH = Path(__file__).parent.parent / "configs/modulation_formats.yaml"


class _DummyTopology:
    """Stand-in for a loaded Topology. main() reads `num_nodes` (to size
    RegenPlacement) and `regen_candidate_nodes` (to build the
    inert-placement mask) before DiffONetPipeline is constructed, and
    DiffONetPipeline is itself stubbed below.

    Nodes 1 and 3 are regen candidates; 0, 2 and 4 are not — a placement on
    those can never split a path, because pipeline.segment_path only splits
    at candidates.
    """
    num_nodes = 5
    regen_candidate_nodes = [1, 3]


class _DummyPipeline:
    """Stand-in for DiffONetPipeline. Its physics forward pass is irrelevant
    to checkpoint selection and to the inert-placement count, so it is
    replaced with a cheap no-op that satisfies main()'s call shape:
    `pipeline(demands, tau=..., lambda_=...)` ->
    (path_noise_costs, gsnr_preds, path_indicators, regen_probs).

    `regen_probs_value` is a class attribute rather than a constructor
    argument because main() constructs the pipeline itself — the test has no
    handle on the instance.
    """

    regen_probs_value = torch.zeros(_DummyTopology.num_nodes)

    # Per-demand GSNR the stub reports when NO regenerators are placed, and
    # when at least one is. Lets a test script the hard-eval pass
    # independently of the soft one, which is the whole point of the fix.
    # gsnr_unplaced must clear below the 400G demand's real threshold
    # (7.1 dB, from configs/modulation_formats.yaml) plus the 0.5 dB test
    # margin, so the empty placement is genuinely infeasible -- matching
    # what the real bug looks like (an unplaced route failing GSNR).
    gsnr_unplaced = 5.0
    gsnr_placed = 100.0

    def __init__(self, **kwargs) -> None:
        pass

    def to(self, device):
        return self

    def __call__(self, demands, tau=None, lambda_=None, regen_probs_override=None):
        probs = (_DummyPipeline.regen_probs_value
                 if regen_probs_override is None else regen_probs_override)
        if regen_probs_override is None:
            return {}, {}, {}, probs
        value = (_DummyPipeline.gsnr_placed if probs.sum() > 0
                 else _DummyPipeline.gsnr_unplaced)
        gsnr = {d.id: torch.tensor(value) for d in demands}
        return {}, gsnr, {}, probs


def _mod_config():
    from diffopt.modulation import ModulationConfig
    return ModulationConfig.from_yaml(str(_MODULATION_FORMATS_PATH))


def _write_config(tmp_path, *, epochs: int, hard_eval: bool = False) -> Path:
    config = {
        "topology": "unused",
        "modulation_formats": str(_MODULATION_FORMATS_PATH),
        "qot_checkpoint": "unused",
        "num_demands": 1,
        "bitrate_options": [400],
        "max_spans_per_segment": 60,
        "seed": 42,
        "traffic": {
            "scenario": "stress",
            "seed": 0,
            "scale": 1.0e6,
            "holdout_seed": 1,
        },
        "constraint": {
            "margin_db": 0.5,
            "dual_init": 10.0,
            "dual_lr": 1.0,
            "dual_max": 1000.0,
        },
        "pipeline": {
            "lambda_regen": 1.0,
            "lambda_cost": 0.01,
            "channel_loading_fraction": 0.5,
        },
        "training": {
            "lr_edge_net": 1.0e-3,
            "lr_regen": 1.0e-2,
            "epochs_e2e": epochs,
            "vlastelica_lambda": 10.0,
            "vlastelica_lambda_min": 1.0,
            "vlastelica_lambda_decay": 0.995,
            "regen_tau_start": 1.0,
            "regen_tau_end": 0.1,
            "regen_tau_anneal_start_epoch": 1,
            "regen_tau_anneal_end_epoch": epochs,
        },
        "log_dir": str(tmp_path / "logs"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "selection": {"hard_eval": hard_eval},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


def _run_main(tmp_path, monkeypatch, *, scripted, regen_probs=None, hard_eval: bool = False):
    """Drive the real diffopt.train.main() epoch loop with a scripted
    per-epoch (total_loss, num_violated, num_regen_soft) sequence.

    Only the numerically expensive / physics-derived pieces irrelevant to
    checkpoint selection (topology loading, QoT checkpoint loading, traffic
    matrix construction, preflight, and the pipeline forward + compute_loss)
    are replaced with cheap deterministic stand-ins.

    `regen_probs` overrides what the stub pipeline returns, so a test can
    place probability mass on a specific node. Restored afterwards because it
    is class state on _DummyPipeline.
    """
    call_count = {"n": 0}
    if regen_probs is not None:
        monkeypatch.setattr(_DummyPipeline, "regen_probs_value", regen_probs)

    def fake_load_topology(topology_path, modulation_formats_path):
        return _DummyTopology()

    def fake_load_qot_model(checkpoint_path, cfg, device):
        return None

    # A single dummy demand — non-empty, so main()'s "preflight excluded
    # every demand" guard sees a real raw_matrix AND a non-empty kept list
    # (preflight excludes nothing here). Content is irrelevant: compute_loss
    # is stubbed below and never reads it. See
    # test_main_raises_when_preflight_excludes_every_demand for the guard's
    # actual raise path, which this stub deliberately does NOT exercise.
    _dummy_demand = Demand(id=0, src=0, dst=1, bitrate_gbps=400.0)

    def fake_build_traffic_matrix(topology, **kwargs):
        return [_dummy_demand]

    def fake_preflight_filter(topology, demands, **kwargs):
        return list(demands), []

    def fake_compute_loss(**kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        loss_value, num_violated, num_regen = scripted[idx]
        loss = torch.tensor(loss_value, requires_grad=True)
        metrics = {
            "feasibility_loss": 0.0,
            "weighted_feasibility_loss": 0.0,
            "regen_loss": 0.0,
            "path_noise_loss": 0.0,
            "num_regen_soft": num_regen,
            "num_infeasible": num_violated,
            "num_violated": num_violated,
            "worst_margin_db": 0.0,
            "shortfalls": torch.zeros(1),
        }
        return loss, metrics

    monkeypatch.setattr(train_mod, "load_topology", fake_load_topology)
    monkeypatch.setattr(train_mod, "load_qot_model", fake_load_qot_model)
    monkeypatch.setattr(train_mod, "build_traffic_matrix", fake_build_traffic_matrix)
    monkeypatch.setattr(train_mod, "preflight_filter", fake_preflight_filter)
    monkeypatch.setattr(train_mod, "DiffONetPipeline", _DummyPipeline)
    monkeypatch.setattr(train_mod, "compute_loss", fake_compute_loss)

    config_path = _write_config(tmp_path, epochs=len(scripted), hard_eval=hard_eval)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    train_mod.main()
    return torch.load(tmp_path / "checkpoints" / "best_e2e.pt", weights_only=False)


def test_selection_prefers_fewer_violations(tmp_path, monkeypatch):
    """A higher-loss, zero-violation epoch must beat a lower-loss, violated one.

    Selection on total loss alone is already wrong under the OLD objective:
    the shipped checkpoints/e2e_ind132/best_e2e.pt is epoch 40 while its own
    run log records three later improvements ending at epoch 55, because total
    loss is dominated by the tau anneal rather than by solution quality. Under
    a constrained formulation lambda is deliberately non-stationary, so total
    loss becomes still less comparable across epochs.
    """
    #        (total_loss, num_violated, num_regen_soft)
    scripted = [
        (5.0, 3, 4),   # epoch 1 — lowest violations so far
        (1.0, 7, 4),   # epoch 2 — much lower loss, but MORE violated
        (9.0, 0, 6),   # epoch 3 — highest loss, zero violated -> must win
    ]
    ckpt = _run_main(tmp_path, monkeypatch, scripted=scripted)
    assert ckpt["epoch"] == 3
    assert ckpt["num_violated"] == 0
    assert ckpt["total_loss"] == pytest.approx(9.0)


def test_selection_breaks_violation_ties_on_regenerator_count(tmp_path, monkeypatch):
    scripted = [
        (1.0, 0, 9),   # epoch 1 — zero violated, 9 regens
        (8.0, 0, 5),   # epoch 2 — zero violated, 5 regens -> must win
        (0.5, 0, 7),   # epoch 3 — lowest loss, but 7 regens
    ]
    ckpt = _run_main(tmp_path, monkeypatch, scripted=scripted)
    assert ckpt["epoch"] == 2
    assert ckpt["num_regen_soft"] == 5


def test_selection_breaks_full_ties_on_total_loss(tmp_path, monkeypatch):
    scripted = [
        (5.0, 0, 4),
        (2.0, 0, 4),   # identical violations and regens, lower loss -> wins
        (7.0, 0, 4),
    ]
    ckpt = _run_main(tmp_path, monkeypatch, scripted=scripted)
    assert ckpt["epoch"] == 2
    assert ckpt["total_loss"] == pytest.approx(2.0)


def test_duals_are_saved_with_the_checkpoint(tmp_path, monkeypatch):
    """A run can be resumed or audited only if the dual state is persisted."""
    ckpt = _run_main(tmp_path, monkeypatch, scripted=[(1.0, 0, 3)])
    assert "duals" in ckpt
    assert isinstance(ckpt["duals"], torch.Tensor)


def test_selection_key_comes_from_the_hard_placement_not_the_relaxation(
    tmp_path, monkeypatch
):
    """The bug this fixes: at epoch 1 every logit is 0, so the soft pass
    reports (violated=0, regens=0) — an unbeatable key — while the actual
    deployed placement is EMPTY and infeasible. Scripting epoch 1 as the
    soft-optimal epoch and epoch 2 as genuinely better must select epoch 2.

    The dummy pipeline's stubbed loss (see _run_main's fake_compute_loss) is
    a disconnected leaf tensor, so regen_placement.regen_logits never
    receives a real gradient and can't move on its own within this harness.
    Script the DEPLOYED placement directly and independently of the soft
    `scripted` values -- exactly the independence _DummyPipeline's
    docstring calls "the whole point of the fix" -- by driving
    RegenPlacement.hard_placement_mask() off a call counter: it is called
    exactly once per epoch, from hard_placement_metrics's pre-step
    snapshot. Epoch 1 gets the empty placement the real bug ships (matching
    every candidate logit starting at 0); epoch 2 gets one candidate node
    placed.
    """
    masks = [
        torch.zeros(_DummyTopology.num_nodes, dtype=torch.bool),
        torch.tensor([False, True, False, False, False]),
    ]
    call_count = {"n": 0}

    def fake_hard_placement_mask(self):
        idx = min(call_count["n"], len(masks) - 1)
        call_count["n"] += 1
        return masks[idx]

    monkeypatch.setattr(
        RegenPlacement, "hard_placement_mask", fake_hard_placement_mask
    )

    ckpt = _run_main(
        tmp_path,
        monkeypatch,
        # (total_loss, num_violated, num_regen_soft) from the SOFT pass
        scripted=[(1.0, 0, 0), (99.0, 5, 4)],
        hard_eval=True,
    )
    assert ckpt["epoch"] == 2
    assert ckpt["hard_num_violated"] == 0
    assert ckpt["hard_num_placed"] > 0


def test_hard_placement_metrics_counts_against_threshold_plus_margin():
    """num_violated is the margin-inclusive count, matching compute_loss."""
    import diffopt.train as train_mod

    topology = _DummyTopology()
    placement = RegenPlacement(topology.num_nodes)
    with torch.no_grad():
        placement.regen_logits.copy_(torch.tensor([-1.0, 2.0, -1.0, 3.0, -1.0]))
    demands = [Demand(id=0, src=0, dst=1, bitrate_gbps=400.0)]

    metrics = train_mod.hard_placement_metrics(
        _DummyPipeline(), placement, demands, _mod_config(),
        lambda_=10.0, margin_db=0.5,
    )

    assert metrics["hard_num_placed"] == 2
    assert metrics["placement_mask"].tolist() == [False, True, False, True, False]
    # gsnr_placed = 100.0 is far above any threshold + 0.5
    assert metrics["hard_num_violated"] == 0


def test_checkpoint_records_the_placement_mask(tmp_path, monkeypatch):
    """So no consumer has to re-derive the deployed set from raw parameters."""
    ckpt = _run_main(
        tmp_path, monkeypatch, scripted=[(1.0, 0, 0)], hard_eval=True
    )
    assert "placement_mask" in ckpt
    assert ckpt["placement_mask"].dtype == torch.bool
    assert ckpt["placement_mask"].numel() == _DummyTopology.num_nodes


def test_hard_eval_disabled_falls_back_to_the_soft_key(tmp_path, monkeypatch):
    """selection.hard_eval: false must reproduce the old behaviour exactly,
    so the refactor can be proven a no-op before any arm is measured."""
    ckpt = _run_main(
        tmp_path, monkeypatch, scripted=[(1.0, 0, 0), (99.0, 5, 4)],
        hard_eval=False,
    )
    assert ckpt["epoch"] == 1


def _read_log(tmp_path) -> list[dict]:
    with open(tmp_path / "logs" / "e2e_train_log.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_inert_placements_on_non_candidate_nodes_are_counted(tmp_path, monkeypatch):
    """`num_regen_soft` counts probability mass on ALL nodes, but only
    `topology.regen_candidate_nodes` can ever split a path — `segment_path`
    splits nowhere else, so a placement on any other node is physically inert
    while still costing `lambda_regen * p`.

    Measured on the shipped checkpoints/e2e_ind132/best_e2e.pt (epoch 40),
    that contamination is currently ZERO: all 8 placed nodes are candidates,
    and the 84 non-candidates sit pinned in a 0.0069-wide band at ~-0.817,
    because they receive only down-pressure from the regen penalty. (-0.817
    is almost exactly lr_regen * epochs = 0.02 * 40 = 0.80, which is Adam's
    whole reachable excursion for a steady-sign gradient.)

    It is not guaranteed to STAY zero. `regen_probs` also feed EdgeWeightNet's
    edge features at both endpoints of every edge, including non-candidates,
    so feasibility reaches those logits indirectly through routing. That path
    is worth ~0.007 of logit at today's fixed weight of 10 — but the duals now
    scale to `dual_max`, and 100x amplification puts it at ~0.7, comparable to
    the entire +/-1.2 excursion range. Hence the dedicated log column.

    Node 4 is a non-candidate in _DummyTopology, so p=0.9 there must be
    counted by BOTH columns; node 3 is a candidate, so it lands only in
    num_regen_soft.
    """
    probs = torch.tensor([0.1, 0.2, 0.3, 0.8, 0.9])   # nodes 3 and 4 placed
    _run_main(tmp_path, monkeypatch, scripted=[(1.0, 0, 2)], regen_probs=probs)
    row = _read_log(tmp_path)[0]
    assert int(row["num_regen_soft"]) == 2, "nodes 3 and 4 both exceed 0.5"
    assert int(row["num_regen_noncand"]) == 1, "only node 4 is a non-candidate"


def test_no_inert_placements_reports_zero(tmp_path, monkeypatch):
    """The expected steady state — everything placed is a candidate."""
    probs = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3])   # nodes 1 and 3: both candidates
    _run_main(tmp_path, monkeypatch, scripted=[(1.0, 0, 2)], regen_probs=probs)
    row = _read_log(tmp_path)[0]
    assert int(row["num_regen_soft"]) == 2
    assert int(row["num_regen_noncand"]) == 0


def test_main_raises_when_preflight_excludes_every_demand(tmp_path, monkeypatch):
    """main() must fail fast — not silently train on zero demands — when a
    non-empty traffic matrix survives build_traffic_matrix but preflight
    excludes every single one of them (e.g. traffic.scale set so high, or so
    mismatched to the topology, that nothing clears the GSNR bar even under
    the most favourable routing/regen assumptions).

    This is the guard's real purpose, distinct from every other test in this
    module: those use a non-empty raw_matrix that preflight keeps entirely
    (see _run_main's _dummy_demand), so they exercise the guard's happy
    path, not its raise path. Without this test, `if not demands: raise
    ValueError(...)` had zero coverage of the branch it exists for.
    """
    dummy_demand = Demand(id=0, src=0, dst=1, bitrate_gbps=400.0)

    def fake_load_topology(topology_path, modulation_formats_path):
        return _DummyTopology()

    def fake_load_qot_model(checkpoint_path, cfg, device):
        return None

    def fake_build_traffic_matrix(topology, **kwargs):
        return [dummy_demand]

    def fake_preflight_filter(topology, demands, **kwargs):
        # Non-trivial raw matrix, but preflight legitimately excludes
        # everything from it — the exact scenario the guard's message
        # ("Preflight excluded every demand — check traffic.scale...")
        # describes.
        return [], [(dummy_demand, 5.0)]

    monkeypatch.setattr(train_mod, "load_topology", fake_load_topology)
    monkeypatch.setattr(train_mod, "load_qot_model", fake_load_qot_model)
    monkeypatch.setattr(train_mod, "build_traffic_matrix", fake_build_traffic_matrix)
    monkeypatch.setattr(train_mod, "preflight_filter", fake_preflight_filter)
    monkeypatch.setattr(train_mod, "DiffONetPipeline", _DummyPipeline)
    # compute_loss is never reached — main() must raise before the loop.

    config_path = _write_config(tmp_path, epochs=1)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])

    with pytest.raises(ValueError, match="Preflight excluded every demand"):
        train_mod.main()
