"""Tests for scripts/_common.py — the shared diagnostic-script setup.

The single most important guarantee this module makes is
`edge_weights_of`: a diagnostic script must route on (and report) the same
edge weights `pipeline.forward` actually routes on. Correction #9's
unit-mean renormalisation was applied in only 2 of 6 places that
recomputed edge_weights outside the real forward() call before this
module existed — see CLAUDE.md's Phase 1c correction #9 and
docs/investigations/edge_weight_scale_collapse.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from diffopt.demands import Demand
from diffopt.train import linear_anneal

from scripts._common import DiagContext, build_context, demands_for, edge_weights_of, schedule_at
from tests.test_pipeline import make_hub_topology, make_mod_config, make_pipeline


_SMALL_TEST_IND132 = Path(__file__).parent.parent / "configs/experiment/small_test_ind132.yaml"


def _context_for(topo, pipeline) -> DiagContext:
    """Build a minimal, fully-populated DiagContext around an existing
    pipeline — the exact objects build_context() would have wired up,
    without loading a real config/checkpoint from disk."""
    return DiagContext(
        cfg={"num_demands": 5, "bitrate_options": [400.0]},
        device=torch.device("cpu"),
        topology=topo,
        mod_cfg=make_mod_config(),
        qot_model=pipeline.qot_model,
        pipeline=pipeline,
        edge_weight_net=pipeline.edge_weight_net,
        regen_placement=pipeline.regen_placement,
        edges=list(topo.undirected_edges),
        regen_candidates=set(topo.regen_candidate_nodes),
        ckpt=None,
    )


# ---------------------------------------------------------------------------
# edge_weights_of — must mirror pipeline.forward exactly
# ---------------------------------------------------------------------------

def test_edge_weights_of_matches_pipeline_forward():
    """The single most important guarantee: a diagnostic must route on the
    same weights the pipeline routes on. Correction #9's unit-mean
    renormalisation was applied in 2 of 6 places before this module
    existed. Captured via a forward hook on a REAL pipeline.forward() call
    (not a reimplementation of its steps), so this fails if edge_weights_of
    ever drifts from what the pipeline actually computed."""
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    ctx = _context_for(topo, pipeline)
    tau = 0.7

    captured = {}

    def hook(_module, _inputs, output):
        captured["raw"] = output.detach().clone()

    handle = pipeline.edge_weight_net.register_forward_hook(hook)
    demands = [Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)]
    pipeline(demands, tau=tau)
    handle.remove()

    expected_raw = captured["raw"].squeeze(-1)
    expected_normalised = expected_raw / expected_raw.mean().clamp_min(1e-12)

    got_normalised = edge_weights_of(ctx, tau, normalised=True)
    got_raw = edge_weights_of(ctx, tau, normalised=False)

    assert torch.allclose(got_raw, expected_raw, atol=1e-6), (
        "edge_weights_of(normalised=False) does not match the raw EdgeWeightNet "
        "output pipeline.forward actually computed"
    )
    assert torch.allclose(got_normalised, expected_normalised, atol=1e-6), (
        "edge_weights_of(normalised=True) does not match pipeline.forward's "
        "unit-mean-renormalised routing weight"
    )
    # And the two must actually differ on a fixture with nonuniform weights —
    # otherwise the normalised/raw distinction this test exists to catch is
    # not being exercised at all.
    assert not torch.allclose(got_raw, got_normalised), (
        "raw and normalised weights are identical on this fixture — the test "
        "would pass even if edge_weights_of silently dropped normalisation"
    )


def test_edge_weights_of_defaults_to_normalised():
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    ctx = _context_for(topo, pipeline)

    default = edge_weights_of(ctx, 0.7)
    explicit = edge_weights_of(ctx, 0.7, normalised=True)
    assert torch.allclose(default, explicit)
    assert abs(default.mean().item() - 1.0) < 1e-5, (
        "normalised edge weights must have unit mean"
    )


# ---------------------------------------------------------------------------
# schedule_at — must replay train.py's per-epoch computation exactly
# ---------------------------------------------------------------------------

def test_schedule_at_matches_train_py_annealing():
    """tau / soft_max_temperature / lambda at a given epoch must equal what
    train.py would compute for that epoch. Independently replays train.py's
    loop (tau/soft_max via linear_anneal, vlastelica_lambda via the
    multiplicative-decay-then-clamp update applied epoch-1 times) rather
    than calling schedule_at for the "expected" side, so a regression in
    schedule_at's own arithmetic is actually caught."""
    cfg = yaml.safe_load(_SMALL_TEST_IND132.read_text())
    t_cfg = cfg["training"]
    sc_cfg = cfg["segment_combiner"]

    for epoch in [1, 5, 10, 25, 40, 60, 61]:
        expected_tau = linear_anneal(
            epoch, t_cfg["regen_tau_start"], t_cfg["regen_tau_end"],
            t_cfg["regen_tau_anneal_start_epoch"], t_cfg["regen_tau_anneal_end_epoch"],
        )
        expected_tsm = linear_anneal(
            epoch, sc_cfg["soft_max_temperature"], sc_cfg["soft_max_temperature_min"],
            t_cfg["regen_tau_anneal_start_epoch"], t_cfg["regen_tau_anneal_end_epoch"],
        )
        # Replicate train.py's per-epoch update loop literally: lambda starts
        # at vlastelica_lambda and is decayed-then-clamped once per epoch
        # *already elapsed* (the epoch-1 update happens after epoch 1 uses
        # the un-decayed starting value).
        expected_lambda = t_cfg["vlastelica_lambda"]
        for _ in range(epoch - 1):
            expected_lambda = max(
                t_cfg["vlastelica_lambda_min"],
                expected_lambda * t_cfg["vlastelica_lambda_decay"],
            )

        tau, tsm, lam = schedule_at(cfg, epoch=epoch)

        assert tau == pytest.approx(expected_tau), f"epoch {epoch}: tau mismatch"
        assert tsm == pytest.approx(expected_tsm), f"epoch {epoch}: soft_max_temperature mismatch"
        assert lam == pytest.approx(expected_lambda), f"epoch {epoch}: vlastelica_lambda mismatch"


def test_schedule_at_none_defaults_to_epoch_one():
    cfg = yaml.safe_load(_SMALL_TEST_IND132.read_text())
    assert schedule_at(cfg, epoch=None) == schedule_at(cfg, epoch=1)


def test_schedule_at_lambda_decays_and_clamps():
    cfg = yaml.safe_load(_SMALL_TEST_IND132.read_text())
    t_cfg = cfg["training"]
    _, _, lam_start = schedule_at(cfg, epoch=1)
    _, _, lam_later = schedule_at(cfg, epoch=30)
    assert lam_start == pytest.approx(t_cfg["vlastelica_lambda"])
    assert lam_later < lam_start
    assert lam_later >= t_cfg["vlastelica_lambda_min"]


# ---------------------------------------------------------------------------
# demands_for
# ---------------------------------------------------------------------------

def test_demands_for_uses_cfg_num_demands_by_default():
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    ctx = _context_for(topo, pipeline)

    demands = demands_for(ctx, seed=1)
    assert len(demands) == ctx.cfg["num_demands"]


def test_demands_for_override_num_demands():
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    ctx = _context_for(topo, pipeline)

    demands = demands_for(ctx, seed=1, num_demands=3)
    assert len(demands) == 3


def test_demands_for_is_seed_deterministic():
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    ctx = _context_for(topo, pipeline)

    d1 = demands_for(ctx, seed=7)
    d2 = demands_for(ctx, seed=7)
    assert d1 == d2


# ---------------------------------------------------------------------------
# build_context — real config, real checkpoint
# ---------------------------------------------------------------------------

def test_build_context_seeds_before_module_construction():
    """Two contexts built from the same seeded config, without loading a
    checkpoint, must produce bit-identical randomly-initialised nets —
    exactly train.py's epoch-0 reproducibility guarantee."""
    cfg = yaml.safe_load(_SMALL_TEST_IND132.read_text())

    ctx_a = build_context(cfg, load_e2e_checkpoint=False)
    ctx_b = build_context(cfg, load_e2e_checkpoint=False)

    for pa, pb in zip(ctx_a.edge_weight_net.parameters(), ctx_b.edge_weight_net.parameters()):
        assert torch.equal(pa, pb)
    assert torch.equal(ctx_a.regen_placement.regen_logits, ctx_b.regen_placement.regen_logits)


def test_build_context_loads_checkpoint_into_pipeline():
    cfg = yaml.safe_load(_SMALL_TEST_IND132.read_text())
    ckpt_path = Path(cfg["checkpoint_dir"]) / "best_e2e.pt"
    if not ckpt_path.exists():
        pytest.skip(f"no checkpoint at {ckpt_path}")

    ctx = build_context(cfg, load_e2e_checkpoint=True)
    assert ctx.ckpt is not None
    assert "edge_weight_net_state" in ctx.ckpt

    # The loaded state must be what pipeline.edge_weight_net actually holds —
    # not merely returned alongside it.
    loaded_first_param = next(iter(ctx.ckpt["edge_weight_net_state"].values()))
    pipeline_first_param = next(iter(ctx.pipeline.edge_weight_net.state_dict().values()))
    assert torch.equal(loaded_first_param, pipeline_first_param)
    assert torch.equal(ctx.regen_placement.regen_logits, ctx.ckpt["regen_logits"])


def test_build_context_max_spans_from_config_not_hardcoded(monkeypatch):
    """pipeline.max_spans must come from cfg, never a bare literal — several
    scripts hardcoded 60 (CLAUDE.md's own max_spans_per_segment default)
    before this module existed, silently drifting from cfg on any config
    that overrides it. Stubs load_qot_model so this doesn't require a real
    checkpoint trained at the nonstandard max_spans=7 used here."""
    import scripts._common as common_mod
    from diffopt.qot.model import SpanAttentionQoT

    def fake_load_qot_model(checkpoint_path, cfg, device):
        return SpanAttentionQoT(max_spans=cfg.get("max_spans_per_segment", 60))

    monkeypatch.setattr(common_mod, "load_qot_model", fake_load_qot_model)

    cfg = yaml.safe_load(_SMALL_TEST_IND132.read_text())
    cfg = dict(cfg)
    cfg["max_spans_per_segment"] = 7
    ctx = build_context(cfg, load_e2e_checkpoint=False)
    assert ctx.pipeline.max_spans == 7
