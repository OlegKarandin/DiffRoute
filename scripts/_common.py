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
  - edge_weights: `pipeline.forward` renormalises the raw Softplus of
    `edge_log_weight` to unit mean before routing on it
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

Stage II note. The per-node `RegenPlacement` is gone: placement is a
per-(demand, boundary) `AllocationHead`, and routing is a free per-edge
parameter `pipeline.edge_log_weight` rather than an `EdgeWeightNet` over
static features. `DiagContext` therefore carries `allocation_head` and no
`edge_weight_net`, and `build_context` REFUSES to load a pre-Stage-II
checkpoint rather than reinterpreting it — see the ValueError below.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from diffopt.demands import Demand, generate_demands
from diffopt.modulation import ModulationConfig
from diffopt.pipeline import DiffONetPipeline
from diffopt.placement.allocation import AllocationHead
from diffopt.qot.model import SpanAttentionQoT
from diffopt.qot.segment_combiner import SegmentCombiner
from diffopt.topology import Edge, Topology, load_topology
from diffopt.traffic import build_traffic_matrix, preflight_filter, scenario_alpha
from diffopt.train import linear_anneal, load_qot_model


@dataclass
class DiagContext:
    """Everything a diagnose_*.py script needs, built the canonical way.

    No `bar_db` field: the bar is `threshold(bitrate_d) + margin_db` and the
    pipeline computes it internally from the `modulation_config` and
    `margin_db` it was constructed with (`bar_db_for_demands`, the single
    definition). A second copy here would be a second definition.
    """

    cfg: dict
    device: torch.device
    topology: Topology
    mod_cfg: ModulationConfig
    qot_model: SpanAttentionQoT
    pipeline: DiffONetPipeline
    allocation_head: AllocationHead
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
    matching train.py's own ordering — so an unloaded AllocationHead here is
    bit-identical to training's actual epoch-0 state. (`edge_log_weight`'s
    init is deterministic — length-proportional, spec decision 6 — so it
    does not depend on the seed; the head's hidden layers do.) Reseeding
    happens on every call, so building two contexts back to back (e.g. an
    "untrained" one and a "trained" one loaded from a checkpoint, as
    diagnose_edge_weight_gradient.py does) reproducibly gives the untrained
    one training's real epoch-0 weights, while the trained one's
    pre-checkpoint random init is irrelevant because the checkpoint load
    overwrites it.

    `load_e2e_checkpoint=True` (default) loads `checkpoint_path` (or
    `<checkpoint_dir>/best_e2e.pt` if not given) into `allocation_head` and
    `pipeline.edge_log_weight` in place — the head stays the same object
    held by `pipeline`, so the loaded weights are what the returned pipeline
    routes and allocates with. `ckpt` on the returned context is the raw
    loaded dict (`None` if `load_e2e_checkpoint=False`) so callers can read
    fields like `ckpt["vlastelica_lambda"]` — the checkpoint's own saved
    value, not a recomputation from `schedule_at`, since `train.py` saves on
    every loss improvement rather than only at the final epoch.
    """
    device = torch.device("cpu")

    # Seed before any module construction — see train.py's own comment: an
    # unseeded AllocationHead init changes placement enough to move
    # num_violated by an order of magnitude.
    torch.manual_seed(cfg.get("seed", 42))

    topology = load_topology(cfg["topology"], cfg["modulation_formats"])
    mod_cfg = ModulationConfig.from_yaml(cfg["modulation_formats"])
    qot_model = load_qot_model(cfg["qot_checkpoint"], cfg, device)

    pl_cfg = cfg.get("placement", {})
    c_cfg = cfg["constraint"]
    allocation_head = AllocationHead(
        lookahead=pl_cfg.get("lookahead", True),
        route_context=pl_cfg.get("route_context", True),
        # Read from config here, not passed per call, precisely so a
        # diagnostic cannot measure a different relaxation than the run it is
        # describing — the drift class this whole module exists to prevent.
        alloc_ste=pl_cfg.get("alloc_ste", False),
    ).to(device)

    pipeline = DiffONetPipeline(
        topology=topology,
        qot_model=qot_model,
        segment_combiner=SegmentCombiner(),
        allocation_head=allocation_head,
        modulation_config=mod_cfg,
        margin_db=c_cfg["margin_db"],
        channel_loading_fraction=cfg["pipeline"]["channel_loading_fraction"],
        max_spans=cfg.get("max_spans_per_segment", 60),
    ).to(device)

    ckpt = None
    if load_e2e_checkpoint:
        ckpt_path = Path(checkpoint_path or f"{cfg.get('checkpoint_dir', 'checkpoints')}/best_e2e.pt")
        ckpt = torch.load(ckpt_path, map_location=device)

        # No compatibility shim, on purpose. A pre-Stage-II checkpoint was
        # SELECTED under an objective that priced sites, so its numbers are
        # not comparable to anything this pipeline reports; quietly mapping
        # its per-node logits onto anything here would produce a plausible
        # number measured against the wrong metric, which is the single
        # failure mode Stage II exists to remove.
        if "regen_logits" in ckpt or "regen_log_alpha" in ckpt or "gate" in ckpt:
            raise ValueError(
                f"{ckpt_path} is a pre-Stage-II checkpoint. Those were selected "
                f"under an objective that prices SITES, which is the bug Stage II "
                f"removes — loading one would report a plausible number measured "
                f"against the wrong metric. Retrain: "
                f"python -m diffopt.train --config <your config>"
            )

        allocation_head.load_state_dict(ckpt["alloc_head_state"])
        # `ckpt["edge_log_weight"]` is a RAW (E,) tensor, not a state_dict:
        # train.py saves `pipeline.edge_log_weight.detach().clone().cpu()`
        # because the routing parameter is a bare nn.Parameter now, not a
        # submodule. Copy it in place so `pipeline` keeps the same Parameter
        # object (anything already holding a reference to it — an optimizer,
        # a diagnostic's autograd target — stays valid).
        with torch.no_grad():
            pipeline.edge_log_weight.copy_(ckpt["edge_log_weight"].to(device))

    if eval_mode:
        allocation_head.eval()
        qot_model.eval()

    return DiagContext(
        cfg=cfg,
        device=device,
        topology=topology,
        mod_cfg=mod_cfg,
        qot_model=qot_model,
        pipeline=pipeline,
        allocation_head=allocation_head,
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

    `tau` is the ALLOCATION decision temperature — the sharpness of
    `sigmoid(score / tau)` inside `AllocationHead.rollout` — annealed from
    `alloc_tau_start` to `alloc_tau_end`. The old `regen_tau_*` keys named
    the same schedule when it sharpened a per-node sigmoid; they no longer
    exist in any config.

    `epoch=None` defaults to 1 — the values used for training's first
    optimizer step (alloc_tau_start and the raw undecayed
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
        t_cfg["alloc_tau_start"], t_cfg["alloc_tau_end"],
        t_cfg["alloc_tau_anneal_start_epoch"], t_cfg["alloc_tau_anneal_end_epoch"],
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
    """The (E,) edge-weight tensor for the current pipeline state, mirroring
    `pipeline.forward`'s step 3 exactly: `softplus(edge_log_weight)` ->
    unit-mean renormalisation.

    `tau` is accepted but does not affect the result: it is the ALLOCATION
    decision temperature, and routing weights are a function of the free
    per-edge parameter alone (spec decision 6 removed EdgeWeightNet; the
    static edge features it consumed are still built in
    `DiffONetPipeline.__init__` but no longer feed routing). Kept in the
    signature so every existing call site keeps working unchanged.

    `normalised=True` (default) is what `pipeline.forward` actually routes
    on (correction #9's unit-mean renormalisation, undetached divisor) —
    use this whenever a script reports "edge_weights" as a physical
    quantity. Routing itself (Dijkstra's argmin) is scale-invariant, so
    `normalised=False` never changes which path gets picked — but the two
    tensors differ in absolute value, and printing the raw one labelled
    "edge_weights" describes a tensor the pipeline never uses (exactly the
    bug in the old diagnose_pipeline_walkthrough.py).

    `normalised=False` returns the raw pre-normalisation Softplus output.
    No migrated script currently passes `normalised=False` —
    diagnose_edge_weight_gradient.py's "D" report compares raw vs
    normalised statistics, but reconstructs its raw values from
    `pipeline.edge_log_weight` itself (needed anyway for Check B/C/F's live
    gradient decomposition below it), not from this function.

    This whole function runs under `torch.no_grad()` and returns a
    detached, forward-value-only snapshot either way — it is not a
    substitute for differentiating through a live `pipeline(...)` call.
    Diagnostics that need to `torch.autograd.grad` into the routing
    parameter (e.g. diagnose_edge_weight_gradient.py's Check B/C/F gradient
    decomposition) must target `pipeline.edge_log_weight` directly —
    `edge_weights_of` cannot provide a graph-connected tensor by
    construction.
    """
    with torch.no_grad():
        raw = F.softplus(ctx.pipeline.edge_log_weight)
        if not normalised:
            return raw
        return raw / raw.mean().clamp_min(1e-12)
