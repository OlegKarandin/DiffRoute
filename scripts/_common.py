"""Shared setup for scripts/diagnose_*.py.

Not `diffopt/diagnostics.py`: this is tooling, not shipped surface, and it
reaches into `DiffONetPipeline`'s private members (`_topo_edge_features`,
`_edge_src_ids`, ...) the way the scripts already did before this module
existed.

Before this module existed, the same ~15 lines of topology/model/pipeline
construction were copy-pasted into all 11 diagnose_*.py scripts, and the
copies had quietly drifted apart:

  - seed: `diffopt/train.py` seeds with `cfg.get("seed", 42)` before any
    net is constructed. Several scripts used `torch.manual_seed(0)` (a net
    training never saw) or no seed at all.
  - edge_weights: `pipeline.forward` renormalises EdgeWeightNet's raw
    Softplus output to unit mean before routing on it
    (docs/investigations/CHANGELOG.md#correction-1c-9). That renormalisation
    was reproduced in only 2 of the 6 places scripts recomputed edge_weights
    outside the real forward() call; one script printed the raw output
    labelled `edge_weights`.
  - max_spans: hardcoded `60` in several scripts vs `cfg.get(...)` in
    train.py, for a parameter docs/architecture/invariants.md calls a hard
    architecture parameter.
  - lambda_: hardcoded `10.0` in several scripts instead of
    `cfg["training"]["vlastelica_lambda"]`.

The canonical choices below all match `diffopt/train.py` exactly, so a
diagnostic script describes the deployed system rather than a drifted
approximation of it. Where a script has a documented reason to differ
(e.g. it deliberately reports the raw pre-normalisation weight to
demonstrate the difference from the normalised one), the difference is
kept but made explicit at the call site — see that script's own comments.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import torch

from diffopt.demands import Demand, generate_demands
from diffopt.modulation import ModulationConfig
from diffopt.pipeline import DiffONetPipeline
from diffopt.placement.regenerator import RegenPlacement
from diffopt.qot.model import SpanAttentionQoT
from diffopt.qot.segment_combiner import SegmentCombiner
from diffopt.routing.edge_weight_net import EdgeWeightNet
from diffopt.topology import Edge, Topology, load_topology
from diffopt.traffic import build_traffic_matrix, preflight_filter, scenario_alpha
from diffopt.train import linear_anneal, load_qot_model


@dataclass
class DiagContext:
    """Everything a diagnose_*.py script needs, built the canonical way."""

    cfg: dict
    device: torch.device
    topology: Topology
    mod_cfg: ModulationConfig
    qot_model: SpanAttentionQoT
    pipeline: DiffONetPipeline
    edge_weight_net: EdgeWeightNet
    regen_placement: RegenPlacement
    edges: List[Edge]
    regen_candidates: Set[int]
    ckpt: Optional[dict]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common_args(
    ap: argparse.ArgumentParser,
    *,
    default_config: str = "configs/experiment/small_test_ind132.yaml",
    with_checkpoint: bool = True,
    with_demands: bool = True,
) -> argparse.ArgumentParser:
    """Add the --config/--checkpoint/--num-demands/--seed flags scripts share.

    `default_config` lets a script keep its own default while still gaining
    the flag — three scripts (diagnose_pipeline_walkthrough.py,
    diagnose_segmentation_identity.py, diagnose_topology_segments.py) had no
    --config flag at all before this task, despite CLAUDE.md documenting one.
    """
    ap.add_argument("--config", default=default_config, help="Path to experiment YAML config")
    if with_checkpoint:
        ap.add_argument(
            "--checkpoint", default=None,
            help="E2E checkpoint path. Defaults to <checkpoint_dir>/best_e2e.pt",
        )
    if with_demands:
        ap.add_argument("--num-demands", type=int, default=None,
                        help="Defaults to cfg['num_demands']")
        ap.add_argument("--seed", type=int, default=1, help="Demand-generation seed")
    return ap


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------

def build_context(
    cfg: dict,
    *,
    load_e2e_checkpoint: bool = True,
    checkpoint_path: Optional[str] = None,
    eval_mode: bool = True,
) -> DiagContext:
    """Build topology, models and pipeline exactly the way train.py does.

    Seeds with `cfg.get("seed", 42)` *before* constructing any nn.Module —
    matching train.py's own ordering — so an unloaded EdgeWeightNet/
    RegenPlacement here is bit-identical to training's actual epoch-0 state.
    Reseeding happens on every call, so building two contexts back to back
    (e.g. an "untrained" one and a "trained" one loaded from a checkpoint,
    as diagnose_edge_weight_gradient.py does) reproducibly gives the
    untrained one training's real epoch-0 weights, while the trained one's
    pre-checkpoint random init is irrelevant because load_state_dict
    overwrites it.

    `load_e2e_checkpoint=True` (default) loads `checkpoint_path` (or
    `<checkpoint_dir>/best_e2e.pt` if not given) into `edge_weight_net` and
    `regen_placement` in place — both stay the same objects held by
    `pipeline`, so the loaded weights are what the returned pipeline routes
    with. `ckpt` on the returned context is the raw loaded dict (`None` if
    `load_e2e_checkpoint=False`) so callers can read fields like
    `ckpt["vlastelica_lambda"]` — the checkpoint's own saved value, not a
    recomputation from `schedule_at`, since `train.py` saves on every loss
    improvement rather than only at the final epoch.
    """
    device = torch.device("cpu")

    # Seed before any module construction — see train.py's own comment: an
    # unseeded EdgeWeightNet init changes routing enough to move
    # num_infeasible by an order of magnitude.
    torch.manual_seed(cfg.get("seed", 42))

    topology = load_topology(cfg["topology"], cfg["modulation_formats"])
    mod_cfg = ModulationConfig.from_yaml(cfg["modulation_formats"])
    qot_model = load_qot_model(cfg["qot_checkpoint"], cfg, device)

    edge_weight_net = EdgeWeightNet().to(device)
    pl_cfg = cfg.get("placement", {})
    hc_cfg = pl_cfg.get("hard_concrete", {})
    regen_placement = RegenPlacement(
        topology.num_nodes,
        gate=pl_cfg.get("gate", "sigmoid"),
        beta=hc_cfg.get("beta", 0.5),
        gamma=hc_cfg.get("gamma", -0.1),
        zeta=hc_cfg.get("zeta", 1.1),
    ).to(device)

    ckpt = None
    if load_e2e_checkpoint:
        ckpt_path = Path(checkpoint_path or f"{cfg.get('checkpoint_dir', 'checkpoints')}/best_e2e.pt")
        ckpt = torch.load(ckpt_path, map_location=device)
        edge_weight_net.load_state_dict(ckpt["edge_weight_net_state"])
        # Checkpoints written before 2026-08-21 have no "gate" key and are
        # always sigmoid. Reading the checkpoint's own gate rather than the
        # config's guards against pointing a hard_concrete config at a
        # sigmoid checkpoint and silently reinterpreting its parameters.
        ckpt_gate = ckpt.get("gate", "sigmoid")
        if ckpt_gate != regen_placement.gate:
            raise ValueError(
                f"Checkpoint {ckpt_path} was trained with gate {ckpt_gate!r} but "
                f"the config asks for {regen_placement.gate!r}"
            )
        param_key = "regen_logits" if ckpt_gate == "sigmoid" else "regen_log_alpha"
        with torch.no_grad():
            regen_placement._parameter.copy_(ckpt[param_key].to(device))

    if eval_mode:
        edge_weight_net.eval()
        qot_model.eval()

    pipeline = DiffONetPipeline(
        topology=topology,
        qot_model=qot_model,
        segment_combiner=SegmentCombiner(),
        edge_weight_net=edge_weight_net,
        regen_placement=regen_placement,
        channel_loading_fraction=cfg["pipeline"]["channel_loading_fraction"],
        max_spans=cfg.get("max_spans_per_segment", 60),
    ).to(device)

    return DiagContext(
        cfg=cfg,
        device=device,
        topology=topology,
        mod_cfg=mod_cfg,
        qot_model=qot_model,
        pipeline=pipeline,
        edge_weight_net=edge_weight_net,
        regen_placement=regen_placement,
        edges=list(topology.undirected_edges),
        regen_candidates=set(topology.regen_candidate_nodes),
        ckpt=ckpt,
    )


# ---------------------------------------------------------------------------
# Schedule replay
# ---------------------------------------------------------------------------

def schedule_at(cfg: dict, epoch: Optional[int] = None) -> Tuple[float, float]:
    """(tau, vlastelica_lambda) at a given epoch, replaying
    diffopt/train.py's per-epoch computation exactly.

    `epoch=None` defaults to 1 — the values used for training's first
    optimizer step (regen_tau_start and the raw undecayed
    vlastelica_lambda). This matches what diagnose_surrogate.py reports
    when run without --checkpoint (train.py's epoch-0 state).

    `vlastelica_lambda` decays multiplicatively and is clamped to
    `vlastelica_lambda_min` *after* being used each epoch in train.py's
    loop — it is not a `linear_anneal` schedule like tau — so this
    replays that loop rather than using a closed form.

    A loaded e2e checkpoint's own `vlastelica_lambda` field is the ground
    truth for whatever epoch it was actually saved at (train.py saves on
    every loss improvement, not only at the final epoch) — prefer
    `ctx.ckpt["vlastelica_lambda"]` over recomputing this at a guessed
    epoch when a checkpoint is available.
    """
    t_cfg = cfg["training"]
    e = 1 if epoch is None else epoch

    tau = linear_anneal(
        e,
        t_cfg["regen_tau_start"], t_cfg["regen_tau_end"],
        t_cfg["regen_tau_anneal_start_epoch"], t_cfg["regen_tau_anneal_end_epoch"],
    )

    vlastelica_lambda = t_cfg["vlastelica_lambda"]
    for _ in range(e - 1):
        vlastelica_lambda = max(
            t_cfg["vlastelica_lambda_min"], vlastelica_lambda * t_cfg["vlastelica_lambda_decay"]
        )

    return tau, vlastelica_lambda


# ---------------------------------------------------------------------------
# Demands
# ---------------------------------------------------------------------------

def demands_for(ctx: DiagContext, *, seed: int, num_demands: Optional[int] = None) -> List[Demand]:
    """Generate demands the canonical way: `generate_demands` over `ctx.topology`
    using `ctx.cfg["bitrate_options"]`, defaulting `num_demands` to
    `ctx.cfg["num_demands"]`.

    Predates `train.py`'s fixed-traffic-matrix refactor (commit 849a339) and
    was never updated to match it: this draws an ad hoc random demand set
    unrelated to what any checkpoint trained under `cfg["traffic"]` was
    actually trained or constrained against. Use `fixed_traffic_demands`
    for anything that needs to match a real checkpoint's training set.
    """
    n = ctx.cfg["num_demands"] if num_demands is None else num_demands
    return generate_demands(ctx.topology, n, ctx.cfg["bitrate_options"], seed=seed)


def fixed_traffic_demands(ctx: DiagContext) -> Tuple[List[Demand], List[Tuple[Demand, float]]]:
    """Build the same fixed demand set `train.py` builds from `ctx.cfg["traffic"]`
    and `ctx.cfg["constraint"]`: one `build_traffic_matrix` draw, screened by
    `preflight_filter`. Deterministic (no --seed/--num-demands) because the
    matrix is fully determined by `cfg["traffic"]["seed"]` — this is the
    demand set a checkpoint trained on this config actually saw, unlike
    `demands_for`'s unrelated ad hoc draw."""
    tr_cfg = ctx.cfg["traffic"]
    c_cfg = ctx.cfg["constraint"]
    raw_matrix = build_traffic_matrix(
        ctx.topology,
        seed=tr_cfg["seed"],
        scale=tr_cfg["scale"],
        alpha=scenario_alpha(tr_cfg["scenario"]),
        bitrate_options=ctx.cfg["bitrate_options"],
    )
    return preflight_filter(
        ctx.topology,
        raw_matrix,
        qot_model=ctx.qot_model,
        segment_combiner=SegmentCombiner(),
        modulation_config=ctx.mod_cfg,
        margin_db=c_cfg["margin_db"],
        channel_loading_fraction=ctx.cfg["pipeline"]["channel_loading_fraction"],
        max_spans=ctx.cfg.get("max_spans_per_segment", 60),
    )


# ---------------------------------------------------------------------------
# Edge weights — the single most important guarantee in this module
# ---------------------------------------------------------------------------

def edge_weights_of(ctx: DiagContext, tau: float, *, normalised: bool = True) -> torch.Tensor:
    """The (E,) edge-weight tensor for the current pipeline state at
    regen-decision temperature `tau`, mirroring `pipeline.forward`'s steps
    1-3 exactly:
    regen probabilities -> concatenated edge features -> EdgeWeightNet ->
    unit-mean renormalisation.

    `normalised=True` (default) is what `pipeline.forward` actually routes
    on (correction #9's unit-mean renormalisation, undetached divisor) —
    use this whenever a script reports "edge_weights" as a physical
    quantity. Routing itself (Dijkstra's argmin) is scale-invariant, so
    `normalised=False` never changes which path gets picked — but the two
    tensors differ by orders of magnitude in absolute value, and printing
    the raw one labelled "edge_weights" describes a tensor the pipeline
    never uses (exactly the bug in the old
    diagnose_pipeline_walkthrough.py).

    `normalised=False` returns EdgeWeightNet's raw pre-normalisation
    Softplus output. No migrated script currently passes `normalised=False`
    — diagnose_edge_weight_gradient.py's "D" report compares raw vs
    normalised statistics, but gets its raw values from its own
    `register_forward_hook` (needed anyway for Check B/C/F's live gradient
    decomposition below it), not from this function. This parameter exists
    for the diagnostic that wants a raw, no-grad, forward-value-only
    snapshot without setting up a hook — describe it accurately if a future
    script ends up being that caller.

    This whole function runs under `torch.no_grad()` and returns a
    detached, forward-value-only snapshot either way — it is not a
    substitute for a live `register_forward_hook` on `edge_weight_net`
    during an actual `pipeline(...)` call. Diagnostics that need to
    `torch.autograd.grad` through the raw output (e.g.
    diagnose_edge_weight_gradient.py's Check B/C/F gradient decomposition)
    must keep their own hook — `edge_weights_of` cannot provide a
    graph-connected tensor by construction.
    """
    with torch.no_grad():
        pipe = ctx.pipeline
        regen_probs = pipe.regen_placement.get_regen_probs(tau)
        edge_feats = torch.cat([
            pipe._topo_edge_features,
            regen_probs[pipe._edge_src_ids].unsqueeze(1),
            regen_probs[pipe._edge_dst_ids].unsqueeze(1),
        ], dim=1)
        raw = pipe.edge_weight_net(edge_feats).squeeze(-1)
        if not normalised:
            return raw
        return raw / raw.mean().clamp_min(1e-12)
