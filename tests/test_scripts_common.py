"""Tests for scripts/_common.py — the shared diagnostic-script setup.

The single most important guarantee this module makes is
`edge_weights_of`: a diagnostic script must route on (and report) the same
edge weights `pipeline.forward` actually routes on. Correction #9's
unit-mean renormalisation was applied in only 2 of 6 places that
recomputed edge_weights outside the real forward() call before this
module existed — see docs/investigations/CHANGELOG.md#correction-1c-9 and
docs/investigations/edge_weight_scale_collapse.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

import diffopt.pipeline as pipeline_mod
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
        allocation_head=pipeline.allocation_head,
        edges=list(topo.undirected_edges),
        regen_candidates=set(topo.regen_candidate_nodes),
        ckpt=None,
    )


# ---------------------------------------------------------------------------
# edge_weights_of — must mirror pipeline.forward exactly
# ---------------------------------------------------------------------------

def test_edge_weights_of_matches_pipeline_forward(monkeypatch):
    """The single most important guarantee: a diagnostic must route on the
    same weights the pipeline routes on.

    `expected_normalised` comes from monkeypatching
    diffopt.pipeline.surrogate_shortest_path to record its first positional
    argument on its first call — that argument IS the `edge_weights` tensor
    pipeline.forward routes on (diffopt/pipeline.py's
    `surrogate_shortest_path(edge_weights, ...)` call), captured as the
    pipeline actually computed it, not reimplemented from the normalisation
    formula in a second place.

    There is no EdgeWeightNet to hang a register_forward_hook on any more
    (spec decision 6 made routing a free per-edge parameter), so the raw
    half is checked by its DEFINING relation to the captured tensor instead:
    `normalised=False` must return the pre-normalisation quantity, i.e. the
    one that becomes the captured tensor after unit-mean division. That
    still fails if edge_weights_of silently returns the normalised tensor
    for both, and it still does not restate pipeline.py's formula.

    If pipeline.py's normalisation ever changes (e.g. the divisor's clamp
    epsilon, or reintroducing the .detach() correction #9 exists to forbid)
    without a matching update to edge_weights_of, this test must go red —
    see docs/investigations for the regression this guards against.
    """
    topo = make_hub_topology()
    pipeline = make_pipeline(topo)
    ctx = _context_for(topo, pipeline)
    tau = 0.7

    # edge_log_weight is initialised so that softplus(theta) == km/mean(km),
    # whose mean is EXACTLY 1 — at which point raw and normalised coincide
    # and the distinction this test exists to catch is not exercised at all.
    # Displace it the way a training step would, so mean(raw) != 1.
    with torch.no_grad():
        pipeline.edge_log_weight.add_(
            torch.linspace(0.5, 2.0, pipeline.edge_log_weight.numel())
        )

    captured = {}

    original_surrogate = pipeline_mod.surrogate_shortest_path

    def capturing_surrogate(edge_weights, *args, **kwargs):
        if "normalised" not in captured:
            captured["normalised"] = edge_weights.detach().clone()
        return original_surrogate(edge_weights, *args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "surrogate_shortest_path", capturing_surrogate)

    demands = [Demand(id=0, src=0, dst=4, bitrate_gbps=400.0)]
    pipeline(demands, tau=tau)

    expected_normalised = captured["normalised"]

    got_normalised = edge_weights_of(ctx, tau, normalised=True)
    got_raw = edge_weights_of(ctx, tau, normalised=False)

    assert (got_raw > 0).all(), "raw edge weights must be strictly positive (Softplus)"
    assert torch.allclose(
        got_raw / got_raw.mean(), expected_normalised, atol=1e-6
    ), (
        "edge_weights_of(normalised=False) is not the pre-normalisation tensor: "
        "dividing it by its own mean does not reproduce the tensor "
        "pipeline.forward actually routed on"
    )
    assert torch.allclose(got_normalised, expected_normalised, atol=1e-6), (
        "edge_weights_of(normalised=True) does not match the exact tensor "
        "pipeline.forward passed into surrogate_shortest_path — i.e. the "
        "tensor the pipeline actually routed on"
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
    """tau / lambda at a given epoch must equal what train.py would compute
    for that epoch. Independently replays train.py's loop (tau via
    linear_anneal, vlastelica_lambda via the multiplicative-decay-then-clamp
    update applied epoch-1 times) rather than calling schedule_at for the
    "expected" side, so a regression in schedule_at's own arithmetic is
    actually caught."""
    cfg = yaml.safe_load(_SMALL_TEST_IND132.read_text())
    t_cfg = cfg["training"]

    for epoch in [1, 5, 10, 25, 40, 60, 61]:
        expected_tau = linear_anneal(
            epoch, t_cfg["alloc_tau_start"], t_cfg["alloc_tau_end"],
            t_cfg["alloc_tau_anneal_start_epoch"], t_cfg["alloc_tau_anneal_end_epoch"],
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

        tau, lam = schedule_at(cfg, epoch=epoch)

        assert tau == pytest.approx(expected_tau), f"epoch {epoch}: tau mismatch"
        assert lam == pytest.approx(expected_lambda), f"epoch {epoch}: vlastelica_lambda mismatch"


def test_schedule_at_none_defaults_to_epoch_one():
    cfg = yaml.safe_load(_SMALL_TEST_IND132.read_text())
    assert schedule_at(cfg, epoch=None) == schedule_at(cfg, epoch=1)


def test_schedule_at_lambda_decays_and_clamps():
    cfg = yaml.safe_load(_SMALL_TEST_IND132.read_text())
    t_cfg = cfg["training"]
    _, lam_start = schedule_at(cfg, epoch=1)
    _, lam_later = schedule_at(cfg, epoch=30)
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

def test_build_context_seeds_before_module_construction(monkeypatch):
    """Two contexts built from the same seeded config, without loading a
    checkpoint, must produce bit-identical randomly-initialised nets —
    exactly train.py's epoch-0 reproducibility guarantee.

    Stubs load_qot_model the same way test_build_context_max_spans_from_
    config_not_hardcoded does: build_context calls it unconditionally
    (load_e2e_checkpoint only gates the *e2e* checkpoint, not the QoT one),
    and checkpoints/best_qot.pt is gitignored — requiring it here would
    fail this test on a clean clone rather than skip, which is what a
    prior review round caught. This test only needs SOME QoT model to
    exist so build_context can finish constructing the pipeline; it never
    inspects QoT weights.
    """
    import scripts._common as common_mod
    from diffopt.qot.model import SpanAttentionQoT

    def fake_load_qot_model(checkpoint_path, cfg, device):
        return SpanAttentionQoT(max_spans=cfg.get("max_spans_per_segment", 60))

    monkeypatch.setattr(common_mod, "load_qot_model", fake_load_qot_model)

    cfg = yaml.safe_load(_SMALL_TEST_IND132.read_text())

    ctx_a = build_context(cfg, load_e2e_checkpoint=False)
    ctx_b = build_context(cfg, load_e2e_checkpoint=False)

    for pa, pb in zip(ctx_a.allocation_head.parameters(), ctx_b.allocation_head.parameters()):
        assert torch.equal(pa, pb)
    assert torch.equal(ctx_a.pipeline.edge_log_weight, ctx_b.pipeline.edge_log_weight)


def test_build_context_loads_checkpoint_into_pipeline():
    cfg = yaml.safe_load(_SMALL_TEST_IND132.read_text())
    ckpt_path = Path(cfg["checkpoint_dir"]) / "best_e2e.pt"
    if not ckpt_path.exists():
        pytest.skip(f"no checkpoint at {ckpt_path}")

    ctx = build_context(cfg, load_e2e_checkpoint=True)
    assert ctx.ckpt is not None
    assert "alloc_head_state" in ctx.ckpt
    assert "edge_log_weight" in ctx.ckpt

    # The loaded state must be what the pipeline actually holds — not merely
    # returned alongside it.
    loaded_first_param = next(iter(ctx.ckpt["alloc_head_state"].values()))
    head_first_param = next(iter(ctx.pipeline.allocation_head.state_dict().values()))
    assert torch.equal(loaded_first_param, head_first_param)

    # edge_log_weight is a RAW (E,) tensor in the checkpoint, not a
    # state_dict, and its target is a bare nn.Parameter. Loading it with
    # .load_state_dict() (the shape of the call this replaced) would raise;
    # loading it into the wrong object would silently leave the pipeline
    # routing on its length-proportional init.
    assert torch.is_tensor(ctx.ckpt["edge_log_weight"])
    assert torch.equal(ctx.pipeline.edge_log_weight.detach(), ctx.ckpt["edge_log_weight"])


def test_build_context_loads_a_synthetic_checkpoint_into_the_pipeline(monkeypatch, tmp_path):
    """The loading MECHANISM, pinned UNCONDITIONALLY.

    Its sibling above checks the same thing against a real trained
    checkpoint, but `checkpoints/` is gitignored — so on a clean clone, in
    CI, or for anyone who has not just run 60 epochs of training, that test
    SKIPS and the two assertions that matter never execute. That is exactly
    how the bug this test exists to pin survived Task 7: the checkpoint key
    was renamed to `edge_log_weight` while the call stayed
    `edge_weight_net.load_state_dict(...)`, and nothing went red because
    nothing exercised the path without a real checkpoint on disk.

    Two distinct failure modes must both go red here:

      wrong mechanism   `ckpt["edge_log_weight"]` is a RAW (E,) tensor, not
                        a state_dict — `.load_state_dict()` on it raises.
      wrong target      loading it into anything other than
                        `pipeline.edge_log_weight` leaves the pipeline
                        routing on its length-proportional init, silently.
                        The random draw below can never coincide with that
                        init, so the equality assertion catches it.

    Stubs load_qot_model the way every other build_context test in this file
    does, so this needs neither checkpoints/best_qot.pt nor a trained e2e
    checkpoint.
    """
    import scripts._common as common_mod
    from diffopt.placement.allocation import AllocationHead
    from diffopt.qot.model import SpanAttentionQoT
    from diffopt.topology import load_topology

    def fake_load_qot_model(checkpoint_path, cfg, device):
        return SpanAttentionQoT(max_spans=cfg.get("max_spans_per_segment", 60))

    monkeypatch.setattr(common_mod, "load_qot_model", fake_load_qot_model)

    cfg = yaml.safe_load(_SMALL_TEST_IND132.read_text())
    topology = load_topology(cfg["topology"], cfg["modulation_formats"])
    num_edges = len(list(topology.undirected_edges))

    # Randomised, not zeros: the head's output layer initialises to a ZERO
    # weight and edge_log_weight initialises length-proportionally, so a
    # zero/default fixture could pass against a load that never happened.
    torch.manual_seed(1234)
    head_state = {
        k: torch.randn_like(v) for k, v in AllocationHead().state_dict().items()
    }
    assert head_state, "AllocationHead has no state to load — fixture is vacuous"
    edge_log_weight = torch.randn(num_edges)
    ckpt = {
        "epoch": 7,
        "alloc_head_state": head_state,
        "edge_log_weight": edge_log_weight,
        "vlastelica_lambda": 9.5,
    }
    ckpt_path = tmp_path / "best_e2e.pt"
    torch.save(ckpt, ckpt_path)

    ctx = build_context(cfg, load_e2e_checkpoint=True, checkpoint_path=str(ckpt_path))

    assert torch.equal(ctx.pipeline.edge_log_weight.detach(), edge_log_weight)
    # Copied in place, so it is still the trainable Parameter the pipeline
    # routes with — not replaced by a plain tensor no optimizer would see.
    assert isinstance(ctx.pipeline.edge_log_weight, torch.nn.Parameter)
    assert ctx.pipeline.edge_log_weight.requires_grad

    loaded = ctx.pipeline.allocation_head.state_dict()
    for name, value in head_state.items():
        assert torch.equal(loaded[name], value), f"{name} was not loaded"
    # The context's head and the pipeline's head must be ONE object, or a
    # diagnostic would inspect parameters the pipeline never uses.
    assert ctx.allocation_head is ctx.pipeline.allocation_head

    # The raw dict is handed back so callers can read the checkpoint's own
    # saved schedule values rather than recomputing them (see schedule_at).
    assert ctx.ckpt["epoch"] == 7
    assert ctx.ckpt["vlastelica_lambda"] == 9.5


def test_pre_stage_ii_checkpoint_is_rejected_loudly(monkeypatch, tmp_path):
    """A silent fallback is how a site-priced checkpoint gets reported as a
    device-priced result.

    Pre-Stage-II checkpoints were SELECTED under an objective that prices
    sites, so their numbers are not comparable to anything this pipeline
    reports. Any of the three keys that only a RegenPlacement run could have
    written must abort the load with the retrain instruction.

    Stubs load_qot_model the same way its sibling build_context tests do, so
    this doesn't require a real QoT checkpoint on a clean clone — the
    rejection has to happen before anything reads the (absent)
    alloc_head_state, and the stub keeps that the only thing under test.
    """
    import scripts._common as common_mod
    from diffopt.qot.model import SpanAttentionQoT

    def fake_load_qot_model(checkpoint_path, cfg, device):
        return SpanAttentionQoT(max_spans=cfg.get("max_spans_per_segment", 60))

    monkeypatch.setattr(common_mod, "load_qot_model", fake_load_qot_model)

    cfg = yaml.safe_load(_SMALL_TEST_IND132.read_text())

    ckpt = {"edge_weight_net_state": {}, "gate": "sigmoid",
            "regen_logits": torch.zeros(5)}
    torch.save(ckpt, tmp_path / "best_e2e.pt")
    with pytest.raises(ValueError, match="pre-Stage-II checkpoint"):
        build_context(cfg, checkpoint_path=str(tmp_path / "best_e2e.pt"))


@pytest.mark.parametrize("legacy_key", ["regen_logits", "regen_log_alpha", "gate"])
def test_every_pre_stage_ii_marker_is_rejected(monkeypatch, tmp_path, legacy_key):
    """Each of the three markers alone must be enough. A sigmoid run wrote
    `regen_logits`, a hard_concrete run wrote `regen_log_alpha` and no
    `regen_logits`, and both wrote `gate` — so checking only one key would
    let the other flavour through."""
    import scripts._common as common_mod
    from diffopt.qot.model import SpanAttentionQoT

    def fake_load_qot_model(checkpoint_path, cfg, device):
        return SpanAttentionQoT(max_spans=cfg.get("max_spans_per_segment", 60))

    monkeypatch.setattr(common_mod, "load_qot_model", fake_load_qot_model)

    cfg = yaml.safe_load(_SMALL_TEST_IND132.read_text())
    value = "sigmoid" if legacy_key == "gate" else torch.zeros(5)
    torch.save({legacy_key: value}, tmp_path / "best_e2e.pt")
    with pytest.raises(ValueError, match="pre-Stage-II checkpoint"):
        build_context(cfg, checkpoint_path=str(tmp_path / "best_e2e.pt"))


def test_build_context_max_spans_from_config_not_hardcoded(monkeypatch):
    """pipeline.max_spans must come from cfg, never a bare literal — several
    scripts hardcoded 60 (docs/architecture/invariants.md's own max_spans_per_segment default)
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
