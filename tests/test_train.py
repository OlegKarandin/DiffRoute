"""Unit tests for diffopt/train.py.

Two groups:

  `linear_anneal` (renamed/generalized from `compute_regen_tau`) drives the
  AllocationHead's `tau`. It stays generic — nothing in it is tau-specific —
  because it also used to drive SegmentCombiner's `soft_max_temperature`,
  before that fold became exact and lost its temperature entirely (see
  docs/investigations/CHANGELOG.md's Phase 1c corrections, and
  diffopt/qot/segment_combiner.py's docstring, for that history).

  `hard_rollout` and the checkpoint-selection loop it feeds. Selection is
  lexicographic on (hard_num_violated, hard_num_devices,
  -hard_worst_margin_db) — a DEVICE count in the middle slot, never a site
  count: 40 demands regenerating at node 7 need 40 devices, not 1, and a
  site-priced key let an epoch win by touching fewer nodes while buying
  more hardware.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

import diffopt.train as train_mod
from diffopt.demands import Demand
from diffopt.modulation import ModulationConfig
from diffopt.pipeline import DiffONetPipeline
from diffopt.placement.allocation import AllocationHead
from diffopt.qot.model import SpanAttentionQoT
from diffopt.qot.segment_combiner import SegmentCombiner
from diffopt.train import linear_anneal
from tests.test_pipeline import make_demands, make_hub_topology, make_mod_config, make_pipeline


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
# cosine_anneal / step_decay — open_followups.md #7a's lr_alloc schedule
# ---------------------------------------------------------------------------

from diffopt.train import cosine_anneal, step_decay  # noqa: E402


def test_cosine_anneal_before_start_and_after_end_returns_the_endpoints():
    assert cosine_anneal(epoch=1, start=1.0, end=0.1, anneal_start=5, anneal_end=20) == 1.0
    assert cosine_anneal(epoch=5, start=1.0, end=0.1, anneal_start=5, anneal_end=20) == 1.0
    assert cosine_anneal(epoch=20, start=1.0, end=0.1, anneal_start=5, anneal_end=20) == 0.1
    assert cosine_anneal(epoch=100, start=1.0, end=0.1, anneal_start=5, anneal_end=20) == 0.1


def test_cosine_anneal_midpoint_is_the_arithmetic_mean():
    """Cosine's own defining property: at exactly the midpoint of the
    window, cos(pi/2) == 0, so the value is exactly (start+end)/2 — the
    same point linear interpolation would give, even though the path there
    differs."""
    result = cosine_anneal(epoch=5, start=1.0, end=0.0, anneal_start=0, anneal_end=10)
    assert result == pytest.approx(0.5)


def test_cosine_anneal_decays_slower_than_linear_near_the_endpoints():
    """The whole reason to prefer cosine over linear_anneal for lr_alloc:
    it spends more of the budget close to `start` (still exploring) and
    close to `end` (settled), moving fastest through the middle."""
    linear_early = linear_anneal(epoch=2, start=1.0, end=0.0, anneal_start=0, anneal_end=10)
    cosine_early = cosine_anneal(epoch=2, start=1.0, end=0.0, anneal_start=0, anneal_end=10)
    assert cosine_early > linear_early   # cosine has decayed LESS by epoch 2


def test_cosine_anneal_monotonic_decrease_across_the_window():
    values = [
        cosine_anneal(epoch=e, start=0.5, end=0.01, anneal_start=5, anneal_end=18)
        for e in range(1, 25)
    ]
    for i in range(1, len(values)):
        assert values[i] <= values[i - 1] + 1e-12
    assert values[0] == 0.5
    assert values[-1] == 0.01


def test_step_decay_holds_start_within_the_first_step_size_epochs():
    assert step_decay(epoch=1, start=1.0, step_size=10, gamma=0.5) == 1.0
    assert step_decay(epoch=10, start=1.0, step_size=10, gamma=0.5) == 1.0


def test_step_decay_multiplies_by_gamma_each_step():
    assert step_decay(epoch=11, start=1.0, step_size=10, gamma=0.5) == pytest.approx(0.5)
    assert step_decay(epoch=21, start=1.0, step_size=10, gamma=0.5) == pytest.approx(0.25)


def test_step_decay_disabled_below_zero_step_size_returns_start_unchanged():
    assert step_decay(epoch=1000, start=1.0, step_size=0, gamma=0.5) == 1.0


# ---------------------------------------------------------------------------
# hard_rollout, against a REAL pipeline
# ---------------------------------------------------------------------------

def _tiny_e2e_setup():
    """A real (hub topology, pipeline, demands, ModulationConfig).

    Deliberately NOT the stubbed harness below: hard_rollout's whole job is
    to agree with the oracle on how a chunk is scored, and a stub that
    hands both sides the same made-up numbers cannot test that agreement.

    The 400G threshold is 0.0 dB here, NOT tests/test_pipeline.py's 20.0 dB.
    The QoT model is randomly initialised and returns ~1.0-1.4 dB per
    segment, so at a 20.5 dB bar EVERY demand is infeasible under EVERY
    allocation: the oracle's `count` degenerates to a best-effort attempt
    that is not a minimum of anything, and the gap these tests exist to
    measure becomes vacuous. At a 0.5 dB bar the segments clear
    comfortably, the oracle's minimum is a real minimum, and a nonzero gap
    means what it says.
    """
    torch.manual_seed(0)
    topology = make_hub_topology()
    mod_cfg = ModulationConfig(
        channel_spacing_ghz=100.0,
        symbol_rate_gbaud=64.0,
        num_channels_cband=48,
        cut_channel_index=24,
        formats=[{"bitrate_gbps": 400, "snr_threshold_db": 0.0}],
    )
    pipeline = DiffONetPipeline(
        topology=topology,
        qot_model=SpanAttentionQoT(max_spans=60),
        segment_combiner=SegmentCombiner(),
        allocation_head=AllocationHead(),
        modulation_config=mod_cfg,
        margin_db=0.5,
    )
    return topology, pipeline, make_demands(), mod_cfg


def _soft_alloc(pipeline, demands, *, lambda_: float = 10.0, tau: float = 1.0):
    """The soft-pass AllocationOutputs hard_rollout now requires (open_
    followups.md #7b): it reuses this pass's routes/segments/GSNRs instead
    of running its own fresh forward. `tau` doesn't matter to anything
    hard_rollout reads off it — only the routing/segmentation/QoT fields,
    which are tau-independent."""
    with torch.no_grad():
        _, _, _, alloc = pipeline(demands, tau=tau, lambda_=lambda_)
    return alloc


def test_hard_rollout_reports_devices_sites_and_the_oracle_gap():
    """Spec 2.5 + 2.6. hard_num_placed is DELETED; hard_num_sites is
    logged and never priced."""
    topology, pipeline, demands, mod_cfg = _tiny_e2e_setup()
    with torch.no_grad():
        pipeline.allocation_head.net[-1].bias.fill_(5.0)   # cut everywhere
    alloc = _soft_alloc(pipeline, demands)
    hard = train_mod.hard_rollout(
        pipeline, demands, alloc, mod_cfg, margin_db=0.5
    )
    assert "hard_num_placed" not in hard
    assert hard["hard_num_devices"] >= hard["hard_num_sites"]
    assert hard["oracle_gap"] >= 0
    assert hard["oracle_gap"] == hard["hard_num_devices"] - hard["oracle_devices"]


def test_hard_rollout_is_deterministic_and_leaves_no_graph():
    topology, pipeline, demands, mod_cfg = _tiny_e2e_setup()
    alloc = _soft_alloc(pipeline, demands)
    a = train_mod.hard_rollout(pipeline, demands, alloc, mod_cfg, margin_db=0.5)
    b = train_mod.hard_rollout(pipeline, demands, alloc, mod_cfg, margin_db=0.5)
    assert a["hard_num_devices"] == b["hard_num_devices"]
    for p in pipeline.allocation_head.parameters():
        assert p.grad is None


def test_oracle_gap_is_measured_only_on_demands_the_head_made_feasible():
    """`oracle.count[d]` is a lower bound only among allocations that make
    demand d feasible, so two situations produce a meaningless negative
    difference and must be excluded:

      * the head UNDER-buys — epoch 1 has it initialised CLOSED (spec 2.2),
        buying 0 devices while the oracle needs many;
      * nothing works at all — a single segment alone busts the bar
        (`oracle.feasible[d]` False), so `count[d]` is the oracle's
        best-effort attempt, not a minimum.

    Both are real: this setup is the second (a 20.5 dB bar against ~1.4 dB
    segments) on top of the first (a closed head). Left unguarded, the
    negative-gap AssertionError inside `oracle_gap()` fires on epoch 1 of
    every real run. The shortfall is not lost — it IS hard_num_violated.
    """
    torch.manual_seed(0)
    topology = make_hub_topology()
    pipeline = make_pipeline(topology)      # the 20.0 dB threshold config
    demands, mod_cfg = make_demands(), make_mod_config()

    alloc = _soft_alloc(pipeline, demands)
    hard = train_mod.hard_rollout(
        pipeline, demands, alloc, mod_cfg, margin_db=0.5
    )
    assert hard["hard_num_devices"] == 0          # closed head buys nothing
    assert hard["oracle_devices"] > 0             # the floor is not zero
    assert hard["oracle_infeasible"] == len(demands)
    assert hard["hard_num_violated"] == len(demands)
    assert hard["oracle_gap"] == 0                # NOT negative, NOT a crash


def test_oracle_gap_survives_an_under_buying_head_at_epoch_one():
    """The other flavour of the same guard, and the one that would crash a
    REAL run: here the route IS feasible (`oracle_infeasible == 0`, the
    oracle needs 2 devices) but the head is initialised CLOSED and buys 0.
    Unrestricted, `oracle_gap()` would see 0 - 2 and raise on epoch 1 of
    every training run. The shortfall is reported as hard_num_violated.
    """
    topology, pipeline, demands, mod_cfg = _tiny_e2e_setup()
    alloc = _soft_alloc(pipeline, demands)
    hard = train_mod.hard_rollout(
        pipeline, demands, alloc, mod_cfg, margin_db=0.5
    )
    assert hard["oracle_infeasible"] == 0        # the route admits a solution
    assert hard["oracle_devices"] > 0            # and it needs devices
    assert hard["hard_num_devices"] == 0         # the closed head buys none
    assert hard["hard_num_violated"] > 0         # which is where that shows
    assert hard["oracle_gap"] == 0


def test_hard_rollout_counts_violations_against_threshold_plus_margin():
    """num_violated is the margin-inclusive count, matching compute_loss."""
    topology, pipeline, demands, mod_cfg = _tiny_e2e_setup()
    alloc = _soft_alloc(pipeline, demands)
    hard = train_mod.hard_rollout(
        pipeline, demands, alloc, mod_cfg, margin_db=0.5
    )
    threshold = mod_cfg.required_snr_threshold(400.0)
    with torch.no_grad():
        _, gsnr_preds, _, _ = pipeline(demands, lambda_=10.0, hard_alloc=True)
    expected = sum(
        1 for d in demands if gsnr_preds[d.id].item() < threshold + 0.5
    )
    assert hard["hard_num_violated"] == expected


def test_hard_rollout_returns_the_allocation_record_it_evaluated():
    """The viz needs the deployed cuts in PATH ORDER. alloc_by_node (D, N)
    is accumulated and positionless, so the record itself has to come back."""
    from tests.test_pipeline import (
        make_demands, make_hub_topology, make_mod_config, make_pipeline,
    )
    topology = make_hub_topology()
    pipeline = make_pipeline(topology)
    demands = make_demands()
    mod_cfg = make_mod_config()

    with torch.no_grad():
        _, _, _, soft = pipeline(demands, tau=1.0)
    hard = train_mod.hard_rollout(
        pipeline, demands, soft, mod_cfg, margin_db=0.5,
    )

    alloc = hard["alloc"]
    assert alloc.demand_ids == [d.id for d in demands]
    assert alloc.segment_edge_ids  # the route survived the rollout
    # The record is self-consistent with the aggregates beside it.
    assert int(alloc.a.sum().item()) == hard["hard_num_devices"]
    assert torch.equal(alloc.site_view > 0.5, hard["site_mask"])
    # Hard decisions are exactly 0/1, so cut_idx is unambiguous.
    assert torch.all((alloc.a == 0.0) | (alloc.a == 1.0))
    for row in range(len(demands)):
        n_bnd = int(alloc.num_segments[row]) - 1
        cut = alloc.a[row, :n_bnd] > 0.5
        assert torch.all(alloc.boundary_node_ids[row, :n_bnd][cut] >= 0)


# ---------------------------------------------------------------------------
# Stubbed harness for the epoch loop
# ---------------------------------------------------------------------------

_MODULATION_FORMATS_PATH = Path(__file__).parent.parent / "configs/modulation_formats.yaml"


class _DummyEdge:
    """Stand-in for topology.Edge — only the fields frozen_layout() and
    FrameWriter's header need: src/dst/length_km."""

    def __init__(self, src: int, dst: int, length_km: float) -> None:
        self.src = src
        self.dst = dst
        self.length_km = length_km


class _DummyTopology:
    """Stand-in for a loaded Topology. DiffONetPipeline is itself stubbed
    below, so main() only ever passes this object straight through.

    `undirected_edges` is a closed 5-cycle so frozen_layout() (used by
    FrameWriter's header, when viz.dump_frames is on) sees a connected graph
    with finite pairwise distances everywhere."""
    num_nodes = 5
    regen_candidate_nodes = [1, 3]
    undirected_edges = [
        _DummyEdge(0, 1, 100.0),
        _DummyEdge(1, 2, 120.0),
        _DummyEdge(2, 3, 90.0),
        _DummyEdge(3, 4, 110.0),
        _DummyEdge(4, 0, 130.0),
    ]


class _DummyAlloc:
    """Stand-in for AllocationOutputs — one demand, two segments, one
    boundary (node 3).

    The per-segment GSNRs sit comfortably above the 400G bar (7.1 + 0.5 dB)
    and inside the fold's [GSNR_MIN, GSNR_MAX] clamp band, so the REAL
    oracle_allocation run by the REAL hard_rollout needs no cut: the default
    (unscripted) path through this harness has oracle_devices == 0 and
    oracle_gap == 0 rather than tripping oracle_gap's negative-gap
    assertion.
    """

    def __init__(self, *, hard: bool) -> None:
        n = _DummyTopology.num_nodes
        a_value = 0.0 if hard else torch.sigmoid(torch.tensor(-3.0)).item()
        self.a = torch.full((1, 1), a_value)
        self.a_physics = self.a
        # The raw score behind `a_value`, i.e. AllocationHead's own init
        # bias. Carried explicitly for the same reason AllocationOutputs
        # carries it: it is not recoverable from `a` once the head saturates.
        self.score = torch.full((1, 1), -3.0)
        self.alloc_by_node = torch.zeros(1, n)
        self.alloc_by_node[0, 3] = a_value
        self.device_count = self.a.sum()
        self.site_view = self.alloc_by_node.max(dim=0).values
        self.seg_gsnr_db = torch.full((1, 2), 30.0)
        self.seg_noise = torch.full((1, 2), 1.0e-3)
        # FrameWriter's per-segment "seg_km" field reads this directly.
        self.seg_km_matrix = torch.full((1, 2), 50.0)
        self.num_segments = torch.tensor([2])
        self.boundary_node_ids = torch.tensor([[3]])
        self.demand_ids = [0]
        # One demand, two segments, one boundary at node 3 — so two groups.
        self.segment_edge_ids = {0: [[0], [1]]}
        self.ste_clamped_segments = 0
        self.proxy_qot_rank_corr = 1.0


class _DummyPipeline:
    """Stand-in for DiffONetPipeline. Its physics forward pass is irrelevant
    to checkpoint selection, so it is replaced with a cheap deterministic
    no-op satisfying main()'s call shape:
    `pipeline(demands, tau=..., lambda_=...)` ->
    (path_noise_costs, gsnr_preds, path_indicators, AllocationOutputs), plus
    `hard_rollout_from_soft(demands, alloc, ...)` -> (gsnr_preds,
    AllocationOutputs), the reuse path `train.hard_rollout` now calls
    instead of a second `hard_alloc=True` forward (open_followups.md #7b).

    `gsnr_value` is a class attribute rather than a constructor argument
    because main() constructs the pipeline itself — the test has no handle
    on the instance. 100.0 dB clears every threshold in
    configs/modulation_formats.yaml with room to spare.
    """

    gsnr_value = 100.0

    def __init__(self, **kwargs) -> None:
        # main() builds `opt_edge = optim.Adam([pipeline.edge_log_weight], ...)`
        # and snapshots it pre-step every epoch (spec decision 6, Task 7) — an
        # unconnected leaf tensor satisfies both without needing this stub's
        # __call__ to route through it.
        self.edge_log_weight = torch.zeros(5, requires_grad=True)

    def to(self, device):
        return self

    def __call__(self, demands, tau=1.0, lambda_=10.0, *, hard_alloc=False):
        alloc = _DummyAlloc(hard=hard_alloc)
        gsnr = {d.id: torch.tensor(_DummyPipeline.gsnr_value) for d in demands}
        noise = {d.id: torch.zeros(()) for d in demands}
        return noise, gsnr, {}, alloc

    def hard_rollout_from_soft(self, demands, alloc, tau=1.0):
        hard_alloc = _DummyAlloc(hard=True)
        gsnr = {d.id: torch.tensor(_DummyPipeline.gsnr_value) for d in demands}
        return gsnr, hard_alloc


def _write_config(
    tmp_path,
    *,
    epochs: int,
    lookahead: bool = True,
    constraint_overrides: dict | None = None,
    training_overrides: dict | None = None,
    viz_overrides: dict | None = None,
) -> Path:
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
            "dual_init": 0.0,
            "rho": 20.0,
            "dual_max": 1000.0,
        },
        "pipeline": {
            "lambda_dev": 1.0,
            "lambda_cost": 0.01,
            "channel_loading_fraction": 0.5,
        },
        "placement": {
            "lookahead": lookahead,
        },
        "training": {
            "lr_edge_net": 1.0e-3,
            "lr_alloc": 2.0e-2,
            "epochs_e2e": epochs,
            "vlastelica_lambda": 10.0,
            "vlastelica_lambda_min": 1.0,
            "vlastelica_lambda_decay": 0.995,
            "alloc_tau_start": 1.0,
            "alloc_tau_end": 0.1,
            "alloc_tau_anneal_start_epoch": 1,
            "alloc_tau_anneal_end_epoch": epochs,
        },
        "log_dir": str(tmp_path / "logs"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
    }
    # Merged rather than replaced so a test naming only `rho` still gets the
    # margin_db/dual_max the rest of main() reads.
    if constraint_overrides:
        config["constraint"].update(constraint_overrides)
    if training_overrides:
        config["training"].update(training_overrides)
    # No pre-existing `viz` key to merge into, unlike constraint/training
    # above -- a plain assignment is fine here.
    if viz_overrides:
        config["viz"] = viz_overrides
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


def _run_training(tmp_path, *, epochs: int = 1, scripted=None, monkeypatch=None,
                  loss_calls=None, **config_kwargs):
    """Drive the real diffopt.train.main() epoch loop.

    Only the numerically expensive / physics-derived pieces irrelevant to
    checkpoint selection (topology loading, QoT checkpoint loading, traffic
    matrix construction, preflight, and the pipeline forward + compute_loss)
    are replaced with cheap deterministic stand-ins. `hard_rollout` is left
    REAL by default — the stub pipeline feeds it a genuinely feasible
    allocation — so the default path still exercises the oracle. A test that
    needs a specific deployed allocation monkeypatches `train_mod.hard_rollout`
    itself, before calling this.

    `loss_calls`, when given a list, receives the kwargs of every stubbed
    compute_loss call — the only way to assert on what main() PASSED, as
    opposed to what it did with the result.

    `scripted` is a per-epoch (total_loss, num_violated, device_count)
    sequence for the stubbed compute_loss.

    Uses its own MonkeyPatch when the caller does not supply one, so a test
    can patch `hard_rollout` on the pytest fixture and still call this with
    no arguments. The two never collide: this helper patches only the names
    listed below.
    """
    if scripted is None:
        scripted = [(1.0, 0, 0.0)] * epochs
    epochs = len(scripted)
    call_count = {"n": 0}

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
        if loss_calls is not None:
            loss_calls.append(kwargs)
        idx = call_count["n"]
        call_count["n"] += 1
        loss_value, num_violated, device_count = scripted[idx]
        loss = torch.tensor(loss_value, requires_grad=True)
        metrics = {
            "feasibility_loss": 0.0,
            "weighted_feasibility_loss": 0.0,
            "path_noise_loss": 0.0,
            "device_count": float(device_count),
            "num_infeasible": num_violated,
            "num_violated": num_violated,
            "worst_margin_db": 0.0,
            "shortfalls": torch.zeros(1),
            # Sentinel, deliberately distinguishable from `shortfalls`: a
            # test asserting WHICH vector reached update_duals has to be able
            # to tell the two apart, and both are all-zeros in the real
            # feasible case.
            "constraint_g": torch.full((1,), -3.0),
        }
        return loss, metrics

    own = monkeypatch is None
    mp = pytest.MonkeyPatch() if own else monkeypatch
    try:
        mp.setattr(train_mod, "load_topology", fake_load_topology)
        mp.setattr(train_mod, "load_qot_model", fake_load_qot_model)
        mp.setattr(train_mod, "build_traffic_matrix", fake_build_traffic_matrix)
        mp.setattr(train_mod, "preflight_filter", fake_preflight_filter)
        mp.setattr(train_mod, "DiffONetPipeline", _DummyPipeline)
        mp.setattr(train_mod, "compute_loss", fake_compute_loss)

        config_path = _write_config(tmp_path, epochs=epochs, **config_kwargs)
        mp.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
        train_mod.main()
    finally:
        if own:
            mp.undo()

    return torch.load(tmp_path / "checkpoints" / "best_e2e.pt", weights_only=False)


def _run_two_epochs_and_load():
    """Two stubbed epochs in a throwaway directory -> the saved checkpoint."""
    with tempfile.TemporaryDirectory() as tmp:
        return _run_training(Path(tmp), scripted=[(1.0, 0, 0.0), (0.5, 0, 0.0)])


def _read_csv(path) -> list[list[str]]:
    with open(path, newline="") as f:
        return list(csv.reader(f))


def _hard(violated, devices, sites, margin, *, oracle_devices=None, alloc=None,
          gsnr_preds=None):
    """One scripted hard_rollout return value.

    `alloc`/`gsnr_preds` default to a real (if minimal) stand-in
    AllocationOutputs and per-demand GSNR dict, so that when
    viz.dump_frames=True the frame writer's append() -- which reads
    hard["alloc"] and hard["gsnr_preds"] -- has something real to consume.
    Every existing caller is unaffected: both are new, defaulted keys.
    """
    if oracle_devices is None:
        oracle_devices = devices
    if alloc is None:
        alloc = _DummyAlloc(hard=True)
    if gsnr_preds is None:
        gsnr_preds = {did: torch.tensor(30.0) for did in alloc.demand_ids}
    mask = torch.zeros(_DummyTopology.num_nodes, dtype=torch.bool)
    mask[:sites] = True
    return {
        "hard_num_violated": violated,
        "hard_num_devices": devices,
        "hard_num_sites": sites,
        "hard_worst_margin_db": margin,
        "oracle_devices": oracle_devices,
        "oracle_gap": devices - oracle_devices,
        "oracle_infeasible": 0,
        "site_mask": mask,
        "alloc_by_node": torch.zeros(1, _DummyTopology.num_nodes),
        "alloc": alloc,
        "gsnr_preds": gsnr_preds,
    }


def _script_hard_rollout(monkeypatch, scripted):
    calls = {"n": 0}

    def fake_hard_rollout(*a, **kw):
        out = scripted[min(calls["n"], len(scripted) - 1)]
        calls["n"] += 1
        return out

    monkeypatch.setattr(train_mod, "hard_rollout", fake_hard_rollout)


# ---------------------------------------------------------------------------
# Checkpoint selection under the constrained objective
# ---------------------------------------------------------------------------

def test_selection_prefers_fewer_violations(tmp_path, monkeypatch):
    """A device-hungrier, zero-violation epoch must beat a lean, violated one.

    Selection on total loss alone is already wrong under the OLD objective:
    the shipped checkpoints/e2e_ind132/best_e2e.pt is epoch 40 while its own
    run log records three later improvements ending at epoch 55, because total
    loss is dominated by the tau anneal rather than by solution quality. Under
    a constrained formulation lambda is deliberately non-stationary, so total
    loss becomes still less comparable across epochs.
    """
    _script_hard_rollout(monkeypatch, [
        _hard(3, 4, 2, -1.5),    # epoch 1 — fewest devices so far, but violated
        _hard(7, 4, 2, -3.0),    # epoch 2 — MORE violated
        _hard(0, 9, 3, +0.2),    # epoch 3 — most devices, zero violated -> wins
    ])
    ckpt = _run_training(tmp_path, scripted=[(5.0, 3, 4.0), (1.0, 7, 4.0), (9.0, 0, 6.0)])
    assert ckpt["epoch"] == 3
    assert ckpt["hard_num_violated"] == 0
    assert ckpt["hard_num_devices"] == 9


def test_selection_key_is_violated_then_devices_then_margin(tmp_path, monkeypatch):
    """(hard_num_violated, hard_num_devices, -hard_worst_margin_db).

    Epoch 2 improves on epoch 1's violation count and must be checkpointed;
    epoch 3 is feasible too but uses MORE devices, so it must NOT overwrite
    it. This is the whole reason the middle slot changed from a site count:
    under the old key epoch 3 could win by touching fewer NODES while buying
    more hardware.
    """
    scripted = [
        {"hard_num_violated": 1, "hard_num_devices": 0, "hard_num_sites": 0,
         "hard_worst_margin_db": -1.0, "oracle_devices": 0, "oracle_gap": 0,
         "oracle_infeasible": 0, "site_mask": torch.zeros(5, dtype=torch.bool),
         "alloc_by_node": torch.zeros(2, 5)},
        {"hard_num_violated": 0, "hard_num_devices": 2, "hard_num_sites": 2,
         "hard_worst_margin_db": +0.4, "oracle_devices": 2, "oracle_gap": 0,
         "oracle_infeasible": 0,
         "site_mask": torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool),
         "alloc_by_node": torch.zeros(2, 5)},
        {"hard_num_violated": 0, "hard_num_devices": 5, "hard_num_sites": 1,
         "hard_worst_margin_db": +0.9, "oracle_devices": 2, "oracle_gap": 3,
         "oracle_infeasible": 0,
         "site_mask": torch.tensor([1, 0, 0, 0, 0], dtype=torch.bool),
         "alloc_by_node": torch.zeros(2, 5)},
    ]
    _script_hard_rollout(monkeypatch, scripted)
    _run_training(tmp_path, epochs=3)

    ckpt = torch.load(tmp_path / "checkpoints" / "best_e2e.pt", weights_only=False)
    assert ckpt["epoch"] == 2
    assert ckpt["hard_num_devices"] == 2
    # Epoch 3 has MORE devices but FEWER sites. Under the deleted
    # hard_num_placed key it would have won.
    assert ckpt["hard_num_sites"] == 2


def test_selection_breaks_full_ties_on_worst_margin(tmp_path, monkeypatch):
    """Third slot is -hard_worst_margin_db: more headroom wins.

    It used to be loss.item(), which invariants.md itself calls
    non-comparable across epochs.
    """
    _script_hard_rollout(monkeypatch, [
        _hard(0, 4, 2, +0.1),
        _hard(0, 4, 2, +0.9),    # identical violations and devices, most headroom
        _hard(0, 4, 2, +0.5),
    ])
    ckpt = _run_training(tmp_path, epochs=3)
    assert ckpt["epoch"] == 2
    assert ckpt["hard_worst_margin_db"] == pytest.approx(0.9)


def test_selection_key_comes_from_the_hard_rollout_not_the_relaxation(
    tmp_path, monkeypatch
):
    """The bug this fixes: early in training the soft pass is a mean-field
    relaxation whose partition-weighted expectation is systematically more
    optimistic than any single allocation, so it reports an unbeatable
    (violated=0, devices~0) key while the DEPLOYED allocation is empty and
    infeasible. Script epoch 1 as the soft-optimal epoch and epoch 2 as
    genuinely better on the hard rollout; epoch 2 must win.
    """
    _script_hard_rollout(monkeypatch, [
        _hard(1, 0, 0, -2.0),    # deployed: empty and infeasible
        _hard(0, 1, 1, +0.3),    # deployed: one device, feasible
    ])
    ckpt = _run_training(
        tmp_path,
        # (total_loss, num_violated, device_count) from the SOFT pass
        scripted=[(1.0, 0, 0.0), (99.0, 5, 4.0)],
    )
    assert ckpt["epoch"] == 2
    assert ckpt["hard_num_violated"] == 0
    assert ckpt["hard_num_devices"] > 0


def test_duals_are_saved_with_the_checkpoint(tmp_path):
    """A run can be resumed or audited only if the dual state is persisted."""
    ckpt = _run_training(tmp_path)
    assert "duals" in ckpt
    assert isinstance(ckpt["duals"], torch.Tensor)


def test_checkpoint_has_no_gate_keys_and_is_a_hard_break():
    """Spec decision 7. The old keys must be ABSENT, not merely unused —
    a reader that falls back to them would reinterpret a device-priced run
    as a site-priced one."""
    ckpt = _run_two_epochs_and_load()
    assert "alloc_head_state" in ckpt
    assert "gate" not in ckpt
    assert "regen_logits" not in ckpt
    assert "regen_log_alpha" not in ckpt
    assert "hard_num_placed" not in ckpt


def test_checkpoint_records_the_site_mask(tmp_path):
    """So no consumer has to re-derive the deployed set from raw parameters.

    The mask is a DIAGNOSTIC — it is logged and saved, never priced and
    never in the selection key.
    """
    ckpt = _run_training(tmp_path)
    assert "site_mask" in ckpt
    assert ckpt["site_mask"].dtype == torch.bool
    assert ckpt["site_mask"].numel() == _DummyTopology.num_nodes


def test_checkpoint_saves_both_optimizer_states_and_the_head(tmp_path):
    ckpt = _run_training(tmp_path)
    assert "opt_edge_state" in ckpt
    assert "opt_alloc_state" in ckpt
    assert "opt_regen_state" not in ckpt
    # Real AllocationHead parameters, not a stub — main() builds the head
    # itself, only DiffONetPipeline is replaced.
    assert any(k.startswith("net.") for k in ckpt["alloc_head_state"])


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

def test_log_csv_header_is_device_priced(tmp_path):
    _run_training(tmp_path)
    header = _read_csv(tmp_path / "logs" / "e2e_train_log.csv")[0]
    assert header == [
        "epoch", "total_loss", "feasibility_loss", "weighted_feasibility_loss",
        "device_loss", "path_noise_loss", "device_count",
        "num_infeasible", "num_violated",
        "hard_num_violated", "hard_num_devices", "hard_num_sites",
        "hard_worst_margin_db", "oracle_devices", "oracle_gap",
        "oracle_infeasible", "worst_margin_db",
        "lambda_max_observed", "num_at_cap",
        "tau", "vlastelica_lambda", "lr_alloc",
        "alloc_score_mean", "alloc_score_min", "alloc_score_max",
        "lookahead",
        "route_context",
        "ste_clamped_segments", "proxy_qot_rank_corr",
        "alloc_dead_frac", "alloc_grad_norm",
    ]


def test_lr_alloc_schedule_defaults_to_flat(tmp_path):
    """The whole point of defaulting to 'none': every existing config's
    trajectory is unchanged unless it opts into a schedule. lr_alloc must
    read back exactly the configured constant on every epoch."""
    _run_training(tmp_path, epochs=3, training_overrides={"lr_alloc": 0.0005})
    rows = _read_csv(tmp_path / "logs" / "e2e_train_log.csv")
    header, body = rows[0], rows[1:]
    col = header.index("lr_alloc")
    assert len(body) == 3
    for row in body:
        assert float(row[col]) == pytest.approx(0.0005)


def test_lr_alloc_cosine_schedule_decays_monotonically_to_lr_alloc_end(tmp_path):
    _run_training(
        tmp_path, epochs=5,
        training_overrides={
            "lr_alloc": 1.0e-2,
            "lr_alloc_schedule": "cosine",
            "lr_alloc_end": 1.0e-4,
            "lr_alloc_anneal_start_epoch": 1,
            "lr_alloc_anneal_end_epoch": 5,
        },
    )
    rows = _read_csv(tmp_path / "logs" / "e2e_train_log.csv")
    header, body = rows[0], rows[1:]
    col = header.index("lr_alloc")
    values = [float(row[col]) for row in body]
    assert values[0] == pytest.approx(1.0e-2)
    assert values[-1] == pytest.approx(1.0e-4)
    for i in range(1, len(values)):
        assert values[i] <= values[i - 1] + 1e-12


def test_lr_alloc_step_schedule_drops_by_gamma_at_the_step(tmp_path):
    _run_training(
        tmp_path, epochs=4,
        training_overrides={
            "lr_alloc": 1.0e-2,
            "lr_alloc_schedule": "step",
            "lr_alloc_step_size": 2,
            "lr_alloc_step_gamma": 0.1,
        },
    )
    rows = _read_csv(tmp_path / "logs" / "e2e_train_log.csv")
    header, body = rows[0], rows[1:]
    col = header.index("lr_alloc")
    values = [float(row[col]) for row in body]
    assert values[0] == pytest.approx(1.0e-2)
    assert values[1] == pytest.approx(1.0e-2)
    assert values[2] == pytest.approx(1.0e-3)
    assert values[3] == pytest.approx(1.0e-3)


def test_lr_alloc_schedule_rejects_an_unknown_value(tmp_path):
    with pytest.raises(ValueError, match="lr_alloc_schedule"):
        _run_training(
            tmp_path, epochs=1,
            training_overrides={"lr_alloc_schedule": "exponential"},
        )


def test_lr_alloc_actually_drives_the_optimizer_not_only_the_log(tmp_path, monkeypatch):
    """The log column is read off opt_alloc.param_groups[0]['lr'] itself,
    not recomputed separately -- assert on the live optimizer state so a
    version that logs the right number but forgets to set the lr can't
    pass."""
    seen_lrs = []
    real_step = torch.optim.SGD.step

    def spying_step(self, *a, **kw):
        seen_lrs.append(self.param_groups[0]["lr"])
        return real_step(self, *a, **kw)

    monkeypatch.setattr(torch.optim.SGD, "step", spying_step)
    _run_training(
        tmp_path, epochs=3,
        training_overrides={
            "lr_alloc": 1.0,
            "lr_alloc_schedule": "step",
            "lr_alloc_step_size": 1,
            "lr_alloc_step_gamma": 0.1,
        },
    )
    # opt_edge also uses SGD-free Adam, so every spied call here is
    # opt_alloc.step() -- three epochs, three steps, decaying by 0.1 each.
    assert seen_lrs == pytest.approx([1.0, 0.1, 0.01])


def test_log_csv_records_a_non_negative_oracle_gap(tmp_path):
    """The acceptance metric for the whole stage. Non-negative by the
    oracle's exchange argument — a negative value means hard_rollout and
    oracle_allocation are scoring chunks differently."""
    _run_training(tmp_path, scripted=[(1.0, 0, 0.0), (1.0, 0, 0.0)])
    rows = _read_csv(tmp_path / "logs" / "e2e_train_log.csv")
    header, body = rows[0], rows[1:]
    gap = header.index("oracle_gap")
    assert len(body) == 2
    for row in body:
        assert int(row[gap]) >= 0


def test_device_trajectory_csv_records_devices_and_sites(tmp_path):
    _run_training(tmp_path)
    rows = _read_csv(tmp_path / "logs" / "placement_trajectory.csv")
    assert rows[0] == ["epoch", "num_devices", "num_sites", "site_nodes"]


def test_placement_trajectory_has_one_row_per_epoch(tmp_path, monkeypatch):
    """Written every epoch, not only on checkpoint improvements — the whole
    point is to see the oscillation between the epochs that got saved.

    Epoch 2 improves over epoch 1 and gets checkpointed; epoch 3 regresses
    (more devices at the same zero violations) and does NOT — yet must still
    produce a trajectory row.
    """
    _script_hard_rollout(monkeypatch, [
        _hard(1, 0, 0, -1.0),
        _hard(0, 2, 2, +0.4),
        _hard(0, 5, 5, +0.9),
    ])
    _run_training(tmp_path, epochs=3)

    rows = _read_csv(tmp_path / "logs" / "placement_trajectory.csv")
    assert [r[0] for r in rows[1:]] == ["1", "2", "3"]
    assert [r[1] for r in rows[1:]] == ["0", "2", "5"]     # num_devices
    assert [r[2] for r in rows[1:]] == ["0", "2", "5"]     # num_sites
    assert rows[1][3] == ""
    assert rows[3][3] == "0 1 2 3 4"

    ckpt = torch.load(tmp_path / "checkpoints" / "best_e2e.pt", weights_only=False)
    assert ckpt["epoch"] == 2
    assert ckpt["hard_num_devices"] == 2


def test_placement_trajectory_records_an_empty_set_without_crashing(tmp_path):
    """Epoch 1 of a real run allocates nothing — the head starts CLOSED."""
    _run_training(tmp_path)
    rows = _read_csv(tmp_path / "logs" / "placement_trajectory.csv")
    assert rows[1][1] == "0"     # num_devices
    assert rows[1][2] == "0"     # num_sites
    assert rows[1][3] == ""      # site_nodes


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_main_raises_when_preflight_excludes_every_demand(tmp_path, monkeypatch):
    """main() must fail fast — not silently train on zero demands — when a
    non-empty traffic matrix survives build_traffic_matrix but preflight
    excludes every single one of them (e.g. traffic.scale set so high, or so
    mismatched to the topology, that nothing clears the GSNR bar even under
    the most favourable routing/allocation assumptions).

    This is the guard's real purpose, distinct from every other test in this
    module: those use a non-empty raw_matrix that preflight keeps entirely
    (see _run_training's _dummy_demand), so they exercise the guard's happy
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


# ---------------------------------------------------------------------------
# Augmented-Lagrangian wiring (spec 2026-08-31 sections 2.2 and 3; the only
# penalty since open_followups.md item #8 removed the hinge)
# ---------------------------------------------------------------------------


def _spy_on_update_duals(monkeypatch, seen: dict):
    def spy(duals, signal, *, eta, dual_max):
        seen["signal"] = signal.clone()
        seen["eta"] = eta
        return duals
    monkeypatch.setattr(train_mod, "update_duals", spy)


def test_rho_reaches_compute_loss(tmp_path):
    calls = []
    _run_training(tmp_path, loss_calls=calls,
                  constraint_overrides={"rho": 20.0})
    assert calls[0]["rho"] == pytest.approx(20.0)


def test_augmented_dual_step_uses_signed_g_at_step_rho(tmp_path, monkeypatch):
    """Spec 2.2. The augmented dual update is gradient ascent on the dual
    function with step rho, so it takes the SIGNED constraint value and the
    SAME rho the penalty uses — that shared coefficient is what makes the
    step well-scaled against the penalty's curvature."""
    seen = {}
    _spy_on_update_duals(monkeypatch, seen)
    _run_training(tmp_path, monkeypatch=monkeypatch,
                  constraint_overrides={"rho": 20.0})
    assert seen["eta"] == pytest.approx(20.0)
    assert torch.equal(seen["signal"], torch.full((1,), -3.0))


# ---------------------------------------------------------------------------
# alloc_score_stats — the head's raw scores
# ---------------------------------------------------------------------------
#
# The column exists to make saturation visible (see its own docstring and the
# CSV header comment in main()). Recovering the score by inverting the
# sigmoid, `score = tau * logit(a)`, cannot do that under `alloc_ste`, where
# the forward value `a` is EXACTLY 0 or 1 and the logit therefore pins to
# float32's clamps on every boundary at every epoch. Measured on the shipped
# augmented-Lagrangian gate runs (docs/investigations/augmented_lagrangian_gate.md):
# all three `al_ste_greedy` seeds logged alloc_score_min == -87.336545 and
# alloc_score_max == +15.942385 for 60/60 epochs, while the true scores ran to
# -47 with 71% of boundaries below a 1e-6 backward slope. The score has to
# come from the head, not from its own output.


def _pinned_score_pipeline(score: float, *, alloc_ste: bool):
    """A pipeline whose head returns `score` on every boundary.

    Zero final-layer weights plus a constant bias make the score independent
    of the features, so the expected value is exact rather than approximate.
    """
    pipeline = make_pipeline(make_hub_topology(), alloc_ste=alloc_ste)
    with torch.no_grad():
        pipeline.allocation_head.net[-1].bias.fill_(score)
    return pipeline


def test_alloc_score_stats_reports_the_true_score_under_the_ste():
    pipeline = _pinned_score_pipeline(-40.0, alloc_ste=True)
    _, _, _, alloc = pipeline(make_demands(), tau=1.0, lambda_=10.0)

    mean, lo, hi = train_mod.alloc_score_stats(alloc)

    assert mean == pytest.approx(-40.0, abs=1e-3)
    assert lo == pytest.approx(-40.0, abs=1e-3)
    assert hi == pytest.approx(-40.0, abs=1e-3)


# ---------------------------------------------------------------------------
# alloc_dead_fraction — how much of the head can still move
# ---------------------------------------------------------------------------
#
# The score column says WHERE the head sits; this says whether it can still
# get anywhere. Under the STE the backward pass keeps sigmoid'(s/tau)/tau,
# and the head's parameters are SHARED across every boundary — so a boundary
# whose slope has underflowed contributes nothing to the summed parameter
# gradient no matter how badly it needs to move. That is the mechanism spec
# section 8 names for the alloc_ste collapse ("the head opened everything
# while lambda_dev could not reach it"), and no dual-side parameter reaches
# it. Neither device_count nor the score range shows it directly: the gate's
# CLEANEST seed (al_ste_greedy 42, a spotless 10-epoch tail) was also the
# most saturated one measured, at 71% of boundaries below 1e-6.


def test_dead_fraction_is_one_when_every_score_is_saturated():
    pipeline = _pinned_score_pipeline(-40.0, alloc_ste=True)
    _, _, _, alloc = pipeline(make_demands(), tau=1.0, lambda_=10.0)

    # sigmoid'(-40) ~ 4.2e-18, far under the 1e-6 threshold.
    assert train_mod.alloc_dead_fraction(alloc, 1.0) == pytest.approx(1.0)


def test_dead_fraction_is_zero_at_the_decision_boundary():
    pipeline = _pinned_score_pipeline(0.0, alloc_ste=True)
    _, _, _, alloc = pipeline(make_demands(), tau=1.0, lambda_=10.0)

    # sigmoid'(0) = 0.25, the largest slope the surrogate ever delivers.
    assert train_mod.alloc_dead_fraction(alloc, 1.0) == pytest.approx(0.0)


def test_dead_fraction_counts_only_real_boundaries():
    """make_demands() is deliberately ragged — demand 1 (0->3) terminates at
    the sole regen candidate and has NO boundary, so its padded column must
    not be counted alive or dead. Half the (D, J-1) grid here is padding."""
    pipeline = _pinned_score_pipeline(-40.0, alloc_ste=True)
    _, _, _, alloc = pipeline(make_demands(), tau=1.0, lambda_=10.0)

    real = int((alloc.boundary_node_ids >= 0).sum())
    assert real < alloc.score.numel()          # padding really is present
    assert train_mod.alloc_dead_fraction(alloc, 1.0) == pytest.approx(1.0)


def test_dead_fraction_sharpens_as_tau_anneals():
    """tau scales the surrogate's slope, so annealing it kills boundaries the
    head could still move at tau=1. This is the concrete cost behind main()'s
    alloc_ste/alloc_tau_end warning, which today has no measured column."""
    pipeline = _pinned_score_pipeline(-8.0, alloc_ste=True)
    _, _, _, alloc = pipeline(make_demands(), tau=1.0, lambda_=10.0)

    assert train_mod.alloc_dead_fraction(alloc, 1.0) == pytest.approx(0.0)
    assert train_mod.alloc_dead_fraction(alloc, 0.3) == pytest.approx(1.0)



# ---------------------------------------------------------------------------
# Optimizer choice for the allocation head.
#
# Adam's step is a RATIO, m/(sqrt(v)+eps), so a consistently-signed gradient
# gives ~lr HOWEVER SMALL that gradient has become. Two measured consequences
# on the runs in docs/investigations/score_runaway_and_dual_windup.md:
#
#   the objective's own brake is discarded -- lambda_dev * sigmoid'(s/tau)/tau
#   falls 5.0e-3 (s=-3) -> 3.0e-11 (s=-22), and under SGD the step falls with
#   it, but under Adam the drift runs at constant velocity to -131;
#
#   the dual loses authority -- a dual enters the head's loss only as a
#   gradient SCALE, and Adam divides scale back out, so winding 0 -> 21.2
#   moves the head no faster. A rate-limited actuator under an integral
#   controller is the textbook wind-up setup.
#
# SGD is now the only optimizer (open_followups.md item #8 removed Adam,
# AdamW and the alloc_weight_decay knob along with it — Finding 9 found no
# viable weight-decay sizing under SGD anyway).


def _spy_on_optimizers(monkeypatch):
    """Record the (class, param_groups) of every optimizer main() builds."""
    built = []
    for name in ("Adam", "SGD"):
        real = getattr(train_mod.optim, name)

        def make(real=real, name=name):
            def spy(params, **kwargs):
                opt = real(params, **kwargs)
                built.append((name, opt))
                return opt
            return spy

        monkeypatch.setattr(train_mod.optim, name, make())
    return built


def test_head_optimizer_is_always_sgd_with_no_momentum_or_weight_decay(tmp_path, monkeypatch):
    """momentum stays 0 on purpose: Adam's beta1 = 0.9 IS a momentum term,
    and it is what makes the plant second-order. A second-order plant under
    the augmented dual's PI-shaped force is what oscillates."""
    built = _spy_on_optimizers(monkeypatch)
    _run_training(tmp_path, monkeypatch=monkeypatch)

    kinds = [name for name, _ in built]
    assert kinds == ["Adam", "SGD"], f"edge net stays on Adam; head is SGD: {kinds}"
    head = built[1][1]
    assert all(g["momentum"] == 0.0 for g in head.param_groups)
    assert all(g["weight_decay"] == 0.0 for g in head.param_groups)


def _clip_spy(monkeypatch):
    """Record the max_norm every clip_grad_norm_ call receives."""
    calls = []
    real = train_mod.torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, *a, **kw):
        calls.append(max_norm)
        return real(params, max_norm, *a, **kw)

    monkeypatch.setattr(train_mod.torch.nn.utils, "clip_grad_norm_", spy)
    return calls


def test_grad_clip_is_off_by_default_and_the_norm_is_still_logged(tmp_path,
                                                                  monkeypatch):
    """Off means max_norm=inf -- a pure measurement, not a cap. The column
    has to be populated on the default path or it is useless as the
    instrument that sizes an SGD learning rate."""
    calls = _clip_spy(monkeypatch)
    _run_training(tmp_path, monkeypatch=monkeypatch, epochs=2)

    assert calls == [float("inf"), float("inf")]
    rows = _read_csv(tmp_path / "logs" / "e2e_train_log.csv")
    col = rows[0].index("alloc_grad_norm")
    assert [float(r[col]) for r in rows[1:]] == [0.0, 0.0]


def test_alloc_grad_clip_reaches_clip_grad_norm(tmp_path, monkeypatch):
    """Clipping exists so one transient at the epoch 7-9 whipsaw cannot force
    lr down for the remaining 50 epochs -- and, under Adam, cannot be burned
    into exp_avg_sq, which at beta2=0.999 over a 60-STEP run never forgets."""
    calls = _clip_spy(monkeypatch)
    _run_training(tmp_path, monkeypatch=monkeypatch, epochs=2,
                  training_overrides={"alloc_grad_clip": 1.5})
    assert calls == [1.5, 1.5]


def test_negative_grad_clip_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="alloc_grad_clip"):
        _run_training(tmp_path, training_overrides={"alloc_grad_clip": -1.0})


# ---------------------------------------------------------------------------
# viz.dump_frames -- the frame-writer hook
# ---------------------------------------------------------------------------

def test_frame_dump_is_off_by_default(tmp_path, monkeypatch):
    """No viz block, no file. Existing runs must be unaffected."""
    _run_training(tmp_path)
    log_dir = tmp_path / "logs"
    assert not (log_dir / "frames.json").exists()
    assert not (log_dir / "frames.jsonl").exists()


def test_frame_dump_writes_a_readable_document(tmp_path, monkeypatch):
    _run_training(
        tmp_path, epochs=3,
        viz_overrides={"dump_frames": True, "every": 1, "keyframe_every": 2},
    )
    log_dir = tmp_path / "logs"
    doc = json.loads((log_dir / "frames.json").read_text())
    assert [f["epoch"] for f in doc["frames"]] == [1, 2, 3]
    assert len(doc["stats"]["rows"]) == 3
    assert doc["run"]["epochs_e2e"] == 3


def test_selected_epoch_matches_the_epoch_selection_actually_picked(
    tmp_path, monkeypatch,
):
    """Spec section 6.9: the scrubber must mark the selected epoch, or the
    animation ends on a visibly worse state than the result being claimed.
    Drive hard_rollout so epoch 2 wins the lexicographic key outright."""
    _script_hard_rollout(monkeypatch, [
        _hard(1, 5, 3, -0.5),   # epoch 1: violated
        _hard(0, 2, 2, 0.3),    # epoch 2: zero violated, fewest devices -> wins
        _hard(0, 9, 4, 0.1),    # epoch 3: zero violated but more devices
    ])
    ckpt = _run_training(
        tmp_path, epochs=3, monkeypatch=monkeypatch,
        viz_overrides={"dump_frames": True},
    )
    log_dir = tmp_path / "logs"
    doc = json.loads((log_dir / "frames.json").read_text())
    assert doc["run"]["selected_epoch"] == 2 == ckpt["epoch"]


def test_dumping_frames_does_not_change_the_training_trajectory(
    tmp_path, monkeypatch,
):
    """The frame writer consumes no RNG, mutates no tensor and runs outside
    the autograd graph. Byte-compare the training log to prove it."""
    off_dir = tmp_path / "off"
    on_dir = tmp_path / "on"
    off_dir.mkdir()
    on_dir.mkdir()
    _run_training(off_dir, epochs=4)
    _run_training(on_dir, epochs=4, viz_overrides={"dump_frames": True})
    log_off = (off_dir / "logs" / "e2e_train_log.csv").read_bytes()
    log_on = (on_dir / "logs" / "e2e_train_log.csv").read_bytes()
    assert log_off == log_on


def test_a_bad_every_value_fails_loudly(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="viz.every"):
        _run_training(
            tmp_path, epochs=2,
            viz_overrides={"dump_frames": True, "every": 0},
        )
