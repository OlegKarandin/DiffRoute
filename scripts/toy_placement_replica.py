"""Toy replica of the placement problem, with a brute-forceable optimum.

14-node line network, nodes 1..12 are regenerator candidates, 10 demands,
routing FROZEN (there is one path) so this isolates the placement head.
Uses the project's own SegmentCombiner, RegenPlacement and update_duals on
constrained_stress.yaml's exact schedule: 60 epochs, Adam lr=2e-2, tau
1.0 -> 0.1 over epochs 10-40, dual_init=10, dual_lr=1.0, dual_max=1000.

Why this exists: the real network has no ground truth. Greedy leave-one-out
under-reports redundancy -- on this toy it returns 6 where brute force over
all 2^12 subsets finds 3 (at nodes [4, 7, 10]). So this is the only place an
arm can be scored on node IDENTITY rather than count, and identity is where
hard-concrete L0 failed on its first measurement (right count 3, wrong nodes
3/4/5).

Used as a screen: run an arm here in seconds before spending 30 minutes on a
constrained_stress run.

--- Toy construction (build_toy) ---

Nodes 0..13 in a line, 13 links (link i joins node i to node i+1). Node 0
and node 13 are the fixed endpoints -- never regen candidates. Every link
carries the SAME per-link GSNR (10.0 dB); a chunk of k consecutive
un-regenerated links therefore has an exact, closed-form GSNR of
`10 - 10*log10(k)` dB (noise adds linearly in the linear domain SegmentCombiner
works in, so this is exact, not an approximation).

Ten demands, each just a (src, dst, threshold_db) triple -- routing is a
single frozen line-graph path, so there is no EdgeWeightNet/routing head
here at all, only RegenPlacement's boundary probabilities feeding
SegmentCombiner directly.

Three of the ten demands are "forcing" demands: each spans exactly one link
pair straddling a single candidate node (e.g. (3, 5) has ONLY node 4 as an
interior candidate), with a threshold that the raw 2-link chunk fails but
either 1-link half clears. Because each forcing demand's span contains
exactly one interior candidate, ANY feasible placement must include that
node -- there is no other way to satisfy it. Three such demands, at (3,5),
(6,8), (9,11), make {4, 7, 10} a NECESSARY subset of every feasible
placement.

The other five "long" demands (five links or more) fail their raw,
un-regenerated chunk but are satisfied once split at whichever of {4, 7, 10}
falls in their interior -- so {4, 7, 10} is also SUFFICIENT. Necessary and
sufficient at size 3 makes it the unique minimum: every feasible set is a
superset of {4, 7, 10}, so the smallest one both contains it and equals it.

The remaining two demands are short two-link spans that clear their
(lenient) threshold even with zero regenerators anywhere, so they never
constrain placement -- they exist only to keep the "8 of 10 fail with no
regenerators" property (the write-up's stated toy signature) rather than
9 of 10.
"""
from __future__ import annotations

import argparse
import itertools
import math
from collections import namedtuple
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from diffopt.loss import update_duals
from diffopt.placement.regenerator import RegenPlacement
from diffopt.qot.segment_combiner import SegmentCombiner
from diffopt.train import linear_anneal

# ---------------------------------------------------------------------------
# Toy network constants
# ---------------------------------------------------------------------------

NUM_LINE_NODES = 14              # nodes 0..13
CANDIDATE_NODES = list(range(1, 13))   # 12 regen candidates: nodes 1..12
NUM_LINKS = NUM_LINE_NODES - 1    # 13 links; link i joins node i to node i+1

SEGMENT_GSNR_DB = 10.0            # uniform per-link GSNR
MARGIN_DB = 0.5                   # delta added inside the hinge, as in loss.py

# Three distinct demand thresholds -- see the module docstring for why each
# exists. Values are chosen with the exact closed-form chunk GSNR
# `10 - 10*log10(k)` dB for a k-link chunk:
#   k=1 -> 10.00 dB   k=2 -> 6.99 dB   k=3 -> 5.23 dB
#   k=5 -> 3.01 dB    k=8 -> 0.97 dB
T_FORCE = 8.0   # passes only a 1-link chunk (10.00 >= 8.0); a 2-link chunk
                # (6.99 dB) fails -- forces a regen at the single interior
                # candidate of a 2-link demand.
T_LONG = 4.5    # passes any chunk of <=3 links (5.23 dB); fails >=4 links
                # (3.98 dB and below) -- the "long" demands' bar.
T_LOW = 6.0     # passes a bare 2-link chunk (6.99 dB) unconditionally --
                # these two demands are never actually constrained.

ToyDemand = namedtuple("ToyDemand", ["id", "src", "dst", "threshold_db"])

# Training schedule, copied verbatim from configs/experiment/constrained_stress.yaml
# (training.* / constraint.* blocks) -- this toy has no config file of its
# own, per the brief, but must mimic that exact schedule.
EPOCHS = 60
LR_REGEN = 2.0e-2
TAU_START = 1.0
TAU_END = 0.1
TAU_ANNEAL_START_EPOCH = 10
TAU_ANNEAL_END_EPOCH = 40
DUAL_INIT = 10.0
DUAL_LR = 1.0
DUAL_MAX = 1000.0


def build_toy() -> Tuple[torch.Tensor, List[ToyDemand]]:
    """Return (segment_gsnr_db, demands).

    `segment_gsnr_db` is a (NUM_LINKS,) float32 tensor, link i's GSNR in dB
    for the transparent link joining node i to node i+1. `demands` is the
    fixed list of 10 ToyDemand triples described in the module docstring.
    """
    segment_gsnr_db = torch.full((NUM_LINKS,), SEGMENT_GSNR_DB, dtype=torch.float32)

    demands = [
        # -- Forcing demands: each has exactly one interior candidate node,
        # so any feasible placement MUST include it. -----------------------
        ToyDemand(0, 3, 5, T_FORCE),    # forces node 4
        ToyDemand(1, 6, 8, T_FORCE),    # forces node 7
        ToyDemand(2, 9, 11, T_FORCE),   # forces node 10
        # -- Long demands: fail raw, satisfied once split at whichever of
        # {4, 7, 10} lie in their interior. Deliberately WIDE spans (up to
        # the full line) so a single soft-infeasible demand's feasibility
        # gradient reaches nearly every candidate node at once -- this is
        # what makes the fractional relaxation's "smearing" broad rather
        # than local, which is what the baseline arm needs to reproduce
        # Signature A (over-provisioning across nearly all candidates, not
        # just the three that are actually load-bearing). ------------------
        ToyDemand(3, 1, 13, T_LONG),    # full line (12 links); needs 4,7,10
        ToyDemand(4, 2, 13, T_LONG),    # 11 links; needs 4,7,10
        ToyDemand(5, 1, 10, T_LONG),    # 9 links; needs 4,7
        ToyDemand(6, 4, 13, T_LONG),    # 9 links; needs 7,10
        ToyDemand(7, 1, 7, T_LONG),     # 6 links; needs 4 (also touches node 2)
        # -- Always-feasible demands: keep the "8 of 10 fail with zero
        # regenerators" property without adding any new constraint. --------
        ToyDemand(8, 1, 3, T_LOW),
        ToyDemand(9, 11, 13, T_LOW),
    ]
    return segment_gsnr_db, demands


def _boundary_nodes(src: int, dst: int) -> List[int]:
    """Interior candidate nodes strictly between src and dst, in order."""
    return list(range(src + 1, dst))


def _hard_forward(
    segment_gsnr_db: torch.Tensor,
    demands: List[ToyDemand],
    combiner: SegmentCombiner,
    node_set: Set[int],
) -> Dict[int, torch.Tensor]:
    """demand_id -> end-to-end GSNR (scalar tensor), boundary probs in {0,1}."""
    gsnrs = {}
    for d in demands:
        boundary_nodes = _boundary_nodes(d.src, d.dst)
        probs = [
            torch.tensor(1.0 if n in node_set else 0.0) for n in boundary_nodes
        ]
        segs = [segment_gsnr_db[i] for i in range(d.src, d.dst)]
        gsnrs[d.id] = combiner(segs, probs)
    return gsnrs


def evaluate(placement_set) -> int:
    """Number of demands violated (GSNR < threshold + margin) under a HARD
    placement -- boundary probabilities in {0.0, 1.0} only."""
    segment_gsnr_db, demands = build_toy()
    combiner = SegmentCombiner()
    gsnrs = _hard_forward(segment_gsnr_db, demands, combiner, set(placement_set))
    return sum(
        1 for d in demands if gsnrs[d.id].item() < d.threshold_db + MARGIN_DB
    )


def _violated_and_margin(
    segment_gsnr_db: torch.Tensor,
    demands: List[ToyDemand],
    combiner: SegmentCombiner,
    node_set: Set[int],
) -> Tuple[int, float]:
    """(num_violated, worst_margin_db) under a hard placement -- shared by
    evaluate() and run_arm()'s per-epoch selection pass."""
    gsnrs = _hard_forward(segment_gsnr_db, demands, combiner, node_set)
    num_violated = 0
    worst_margin_db = math.inf
    for d in demands:
        gsnr = gsnrs[d.id].item()
        margin = gsnr - d.threshold_db
        if gsnr < d.threshold_db + MARGIN_DB:
            num_violated += 1
        worst_margin_db = min(worst_margin_db, margin)
    return num_violated, worst_margin_db


def brute_force_optimum() -> Tuple[int, List[int]]:
    """Smallest subset of range(1, 13) with evaluate(subset) == 0.

    Genuinely searches all 2**12 = 4096 subsets of CANDIDATE_NODES, in
    ascending order of size, returning the first (smallest) subset that
    clears every demand. Not a shortcut -- every subset of every size below
    the answer's size is checked and confirmed infeasible before a larger
    size is tried.
    """
    for size in range(len(CANDIDATE_NODES) + 1):
        for combo in itertools.combinations(CANDIDATE_NODES, size):
            if evaluate(set(combo)) == 0:
                return size, list(combo)
    raise RuntimeError("no feasible placement found among all 2**12 subsets")


def _max_gap(node_set: Set[int]) -> int:
    """Largest run of consecutive CANDIDATE_NODES with nothing placed."""
    best = 0
    run = 0
    for n in CANDIDATE_NODES:
        if n in node_set:
            run = 0
        else:
            run += 1
            best = max(best, run)
    return best


def make_regen_placement(num_candidates: int, gate: str) -> RegenPlacement:
    """RegenPlacement for `gate`. Only 'sigmoid' works today -- 'hard_concrete'
    is Task 8 in the plan and does not exist in RegenPlacement yet."""
    if gate == "sigmoid":
        return RegenPlacement(num_candidates)
    raise NotImplementedError(
        f"gate={gate!r} needs RegenPlacement's gate support, which lands in "
        "Task 8 of the placement-signal plan and is not in this codebase "
        "yet. Only gate='sigmoid' works today."
    )


def first_clamp_epoch(gate: str, trajectory=None) -> Optional[Dict[int, Optional[int]]]:
    """For the `hard_concrete` gate only: the epoch at which each node's
    deterministic gate first hits exactly 0, or None if it never does.

    STUB. `RegenPlacement`'s hard_concrete gate is Task 8 in the plan and
    does not exist in this codebase yet, so there is no deterministic
    z = clamp(...) gate value to inspect. This always returns a per-node
    dict of `None` (or bare `None` for a non-hard_concrete gate), so
    `run_arm`'s call sites and signature already accommodate the real
    check without a future signature change -- only the body needs
    replacing once Task 8 lands.
    """
    if gate != "hard_concrete":
        return None
    return {n: None for n in CANDIDATE_NODES}


def run_arm(
    gate: str,
    dropout_p: float,
    lambda_regen: float,
    seed: int,
    dual_decay: float = 0.0,
) -> dict:
    """Train one arm on the toy for EPOCHS epochs and report the result.

    Mirrors diffopt/train.py's structure at toy scale: a soft forward pass
    (fractional regen probabilities feeding SegmentCombiner) drives the
    gradient; a pre-step hard-evaluation pass (0/1 override) drives
    checkpoint selection and the placement trajectory, exactly as
    train.py's hard_eval path does. There is no EdgeWeightNet/routing head
    here -- the line graph has exactly one path per demand, so placement is
    isolated from routing by construction.

    `dropout_p` applies gate dropout directly in this loop's physics
    forward (this toy does not go through DiffONetPipeline.forward, so it
    cannot inherit Task 7's pipeline-level dropout -- it reimplements the
    same rule: a per-node Bernoulli(1 - dropout_p) keep-mask, drawn once per
    epoch, zeroes dropped nodes' probabilities in the PHYSICS forward only;
    the returned/count-penalised regen_probs are always the undropped ones,
    matching Task 7's contract).
    """
    torch.manual_seed(seed)

    segment_gsnr_db, demands = build_toy()
    combiner = SegmentCombiner()
    optimum_size, optimum_nodes = brute_force_optimum()

    placement = make_regen_placement(len(CANDIDATE_NODES), gate)
    placement.train()
    optimizer = torch.optim.Adam(placement.parameters(), lr=LR_REGEN)
    duals = torch.full((len(demands),), DUAL_INIT)

    trajectory: List[Set[int]] = []  # one hard placement (node ids) per epoch
    best_key = (math.inf, math.inf, math.inf)
    best_nodes: Set[int] = set()
    best_violated = None

    for epoch in range(1, EPOCHS + 1):
        tau = linear_anneal(
            epoch, TAU_START, TAU_END, TAU_ANNEAL_START_EPOCH, TAU_ANNEAL_END_EPOCH
        )

        optimizer.zero_grad()
        regen_probs = placement.get_regen_probs(tau)  # (12,), live in the graph

        if dropout_p > 0.0:
            keep = (torch.rand_like(regen_probs) >= dropout_p).to(regen_probs.dtype)
            regen_probs_physics = regen_probs * keep
        else:
            regen_probs_physics = regen_probs

        shortfalls = torch.zeros(len(demands))
        weighted_feasibility = torch.zeros(())
        for d in demands:
            boundary_nodes = _boundary_nodes(d.src, d.dst)
            probs = [regen_probs_physics[n - 1] for n in boundary_nodes]
            segs = [segment_gsnr_db[i] for i in range(d.src, d.dst)]
            gsnr = combiner(segs, probs)
            shortfall = F.relu((d.threshold_db + MARGIN_DB) - gsnr)
            weighted_feasibility = weighted_feasibility + duals[d.id] * shortfall
            shortfalls[d.id] = shortfall.item()

        loss = weighted_feasibility + lambda_regen * placement.count_penalty(tau)

        # Pre-step hard eval -- mirrors train.py's ordering exactly: the
        # deployed placement (and the checkpoint/selection key it drives)
        # describes the PRE-step parameters, measured before backward()/
        # step() mutate them.
        hard_mask = placement.hard_placement_mask()
        hard_nodes = {
            CANDIDATE_NODES[i] for i in hard_mask.nonzero(as_tuple=True)[0].tolist()
        }
        hard_violated, hard_worst_margin_db = _violated_and_margin(
            segment_gsnr_db, demands, combiner, hard_nodes
        )

        loss.backward()
        optimizer.step()

        duals = update_duals(
            duals, shortfalls, eta=DUAL_LR, dual_max=DUAL_MAX, decay=dual_decay
        )

        trajectory.append(hard_nodes)

        # Lexicographic selection, identical in spirit to train.py: fewest
        # violated, then fewest regenerators, then most headroom.
        selection_key = (hard_violated, len(hard_nodes), -hard_worst_margin_db)
        if selection_key < best_key:
            best_key = selection_key
            best_nodes = hard_nodes
            best_violated = hard_violated

    final_nodes = trajectory[-1] if trajectory else set()

    last_n = trajectory[-20:] if len(trajectory) >= 20 else trajectory
    churn_pairs = [
        len(last_n[i] ^ last_n[i - 1]) for i in range(1, len(last_n))
    ]
    churn_last20 = float(sum(churn_pairs) / len(churn_pairs)) if churn_pairs else 0.0

    return {
        "placed": len(best_nodes),
        "violated": best_violated,
        "nodes": sorted(best_nodes),
        "optimum_nodes": optimum_nodes,
        "optimum_size": optimum_size,
        "churn_last20": churn_last20,
        "final_equals_selected": final_nodes == best_nodes,
        "overlap_with_optimum": len(best_nodes & set(optimum_nodes)),
        "max_gap": _max_gap(best_nodes),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", default="sigmoid", choices=["sigmoid", "hard_concrete"])
    p.add_argument("--dropout", type=float, default=0.0, dest="dropout_p")
    p.add_argument("--lambda-regen", type=float, default=1.0, dest="lambda_regen")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dual-decay", type=float, default=0.0, dest="dual_decay")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_arm(
        gate=args.gate,
        dropout_p=args.dropout_p,
        lambda_regen=args.lambda_regen,
        seed=args.seed,
        dual_decay=args.dual_decay,
    )
    print(
        f"gate={args.gate} dropout={args.dropout_p} lambda_regen={args.lambda_regen} "
        f"seed={args.seed} | "
        f"optimum_size={result['optimum_size']} optimum_nodes={result['optimum_nodes']} | "
        f"placed={result['placed']} violated={result['violated']} "
        f"nodes={result['nodes']} | "
        f"overlap_with_optimum={result['overlap_with_optimum']}/{result['optimum_size']} "
        f"max_gap={result['max_gap']} churn_last20={result['churn_last20']:.2f} "
        f"final_equals_selected={result['final_equals_selected']}"
    )


if __name__ == "__main__":
    main()
