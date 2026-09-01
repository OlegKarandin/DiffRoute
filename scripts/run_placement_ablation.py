"""Compare placement-signal arms on constrained_stress / ind_132.

Stage II replaced the per-node regenerator gate (`placement.gate`,
`gate_dropout_p`, `hard_concrete`, `pipeline.lambda_regen`) with a
per-demand `AllocationHead` priced by `pipeline.lambda_dev` (spec decision
8). `results/placement_arms.csv` was generated under the old gate-based
arms; every one of them overrode a config key nothing reads anymore, so
the arm set is regenerated here rather than re-run. See the comment above
`ARMS` for what each of the three axes below answers and why.

`score()` trains nothing itself: it loads the checkpoint the arm's
training subprocess (`train()`, below) just wrote, rebuilds a
`DiagContext` from the same config, and scores the DEPLOYED allocation
with `diffopt.train.hard_rollout` -- the identical rollout `train.py`
itself uses for checkpoint selection, so this script reports the same
numbers a training run's own log would, not a second definition of them.
It no longer calls the old `diagnose_regen_ablation.ablate()`, which
inspected per-node gate state that no longer exists (the module itself is
deleted).

`device_peak` / `device_final` / `device_plateaued` come from
`placement_trajectory.csv`'s shape (`trajectory_shape`, below) -- spec
9.2's check that a run's device count RISES and PLATEAUS rather than
spiking and descending. See that function's docstring for why the
endpoint alone can look fine while the process was broken.
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

import yaml

from _common import build_context, fixed_traffic_demands
from diffopt.train import hard_rollout

# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

_BASE_CFG_FOR_CAL = yaml.safe_load(
    Path("configs/experiment/constrained_stress.yaml").read_text()
)
_CAL = _BASE_CFG_FOR_CAL["pipeline"]["lambda_dev"]

# Old arms tested `placement.gate` and `gate_dropout_p`, both deleted. The
# three axes that matter now:
#
#   lambda_dev   the calibration is a measurement with a band, not a point.
#                +/-3x brackets it, so the sweep answers "is the device count
#                a property of need or of price?" — the question no
#                lambda_regen value ever answered under L1-on-sites.
#   alloc_dropout
#                gate dropout existed to manufacture node-discriminating
#                gradient a per-node mask could not otherwise get. A
#                per-demand variable already has a demand-specific signal,
#                so this arm tests whether the mechanism is still EARNING
#                its variance rather than assuming it is.
#   lookahead    feature 4 is what makes the greedy-optimal rule exactly
#                representable. Turning it off should raise oracle_gap; if
#                it does not, the head is not using the representability
#                the architecture was chosen for.
ARMS = [
    {"name": "baseline", "overrides": {}},
    {"name": "lambda_dev_0.3x", "overrides": {"pipeline": {"lambda_dev": _CAL * 0.3}}},
    {"name": "lambda_dev_3x",   "overrides": {"pipeline": {"lambda_dev": _CAL * 3.0}}},
    {"name": "alloc_dropout_0.1", "overrides": {"placement": {"alloc_dropout_p": 0.1}}},
    {"name": "alloc_dropout_0.3", "overrides": {"placement": {"alloc_dropout_p": 0.3}}},
    {"name": "no_lookahead", "overrides": {"placement": {"lookahead": False}}},
    {"name": "no_route_context",      "overrides": {"placement": {"route_context": False}}},
    {"name": "greedy_residual",       "overrides": {"placement": {"greedy_residual": True}}},
    {"name": "greedy_residual_waste", "overrides": {"placement": {"greedy_residual": True},
                                                    "pipeline":  {"lambda_waste": 0.1}}},
    # alloc_ste: the training forward pass takes the DEPLOYED decision, so the
    # relaxation and hard_rollout cannot disagree about who is feasible.
    # Measured at an allocation with oracle_gap == 0, the un-STE'd soft pass
    # called 10 of 346 demands violated -- all 10 feasible when deployed -- and
    # those phantoms carried 100% of the feasibility force, 9.7x the combined
    # lambda_dev + lambda_waste shed. alloc_tau_end is pinned to
    # alloc_tau_start because under the STE tau only scales the backward
    # surrogate; train.py warns if an alloc_ste config leaves it annealing.
    {"name": "alloc_ste", "overrides": {"placement": {"alloc_ste": True},
                                        "training":  {"alloc_tau_end": 1.0}}},
    {"name": "alloc_ste_greedy", "overrides": {"placement": {"alloc_ste": True,
                                                             "greedy_residual": True},
                                               "training":  {"alloc_tau_end": 1.0}}},
    # Plain alloc_ste at tau=1.0 collapsed: TRUE scores ran to +110 (measured by
    # intercepting head.score, NOT by alloc_score_stats -- that diagnostic
    # inverts a = sigmoid(s/tau) and is invalid when a is 0/1), the surrogate
    # slope hit exactly 0 on all 4844 boundaries, and the head sat at 1564
    # devices vs an oracle of 33 with no gradient left to shed.
    #
    # Under the STE tau sets the WIDTH OF THE LIVE GRADIENT WINDOW in score
    # units, nothing else -- the forward decision is a sign test. So it wants
    # to be LARGE enough to span the score range the head traverses, the
    # opposite of the anneal's original job. tau=30 puts s=+110 at s/tau=3.7,
    # where sigmoid' is still ~0.024. tau=10 brackets it from below.
    {"name": "alloc_ste_tau10", "overrides": {
        "placement": {"alloc_ste": True},
        "training":  {"alloc_tau_start": 10.0, "alloc_tau_end": 10.0}}},
    {"name": "alloc_ste_tau30", "overrides": {
        "placement": {"alloc_ste": True},
        "training":  {"alloc_tau_start": 30.0, "alloc_tau_end": 30.0}}},
]

ARMS_BY_NAME: Dict[str, dict] = {arm["name"]: arm for arm in ARMS}

CSV_FIELDNAMES = [
    "arm", "seed", "lambda_dev", "alloc_dropout_p", "lookahead",
    "route_context", "greedy_residual", "lambda_waste", "alloc_ste",
    "hard_num_violated", "hard_num_devices", "hard_num_sites",
    "oracle_devices", "oracle_gap", "oracle_infeasible",
    "hard_worst_margin_db", "device_peak", "device_final", "device_plateaued",
    "selected_epoch", "final_equals_selected", "lambda_max_observed",
]


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge `overrides` into a COPY of `base`.

    Nested dicts are merged key-by-key (a `placement` override does not
    clobber sibling `placement` keys the base config already sets, e.g.
    `lookahead`); any non-dict value (including a dict overriding a
    non-dict, or vice versa) replaces the base value outright.
    """
    merged = dict(base)
    for key, value in overrides.items():
        base_value = merged.get(key)
        if isinstance(value, dict) and isinstance(base_value, dict):
            merged[key] = _deep_merge(base_value, value)
        else:
            merged[key] = value
    return merged


def write_arm_config(base_cfg: dict, arm: dict, seed: int, out_dir: Path) -> Path:
    """Deep-merge `arm["overrides"]` into `base_cfg`, set the model-init
    seed, point `log_dir`/`checkpoint_dir` at `out_dir/<arm>_s<seed>/`, and
    write the result to `<run_dir>/config.yaml`.

    `cfg["seed"]` (model-init seed) is set, NOT `cfg["traffic"]["seed"]` —
    arms/seeds vary AllocationHead/edge_log_weight init, not the traffic
    matrix a checkpoint trains against.
    """
    cfg = _deep_merge(base_cfg, arm["overrides"])
    cfg["seed"] = seed

    run_dir = out_dir / f"{arm['name']}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["log_dir"] = (run_dir / "logs").as_posix()
    cfg["checkpoint_dir"] = (run_dir / "checkpoints").as_posix()

    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.dump(cfg))
    return config_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config_path: Path) -> None:
    """Run `diffopt.train` on `config_path` as a real subprocess."""
    subprocess.run(
        [sys.executable, "-m", "diffopt.train", "--config", str(config_path)],
        check=True,
    )


# ---------------------------------------------------------------------------
# Trajectory analysis
# ---------------------------------------------------------------------------

def trajectory_shape_from_counts(counts: List[int]) -> dict:
    """Spec 9.2's trajectory check, on the per-epoch device counts.

    The count must rise from ~0 and PLATEAU. A run that spikes high and then
    descends is the spec's 2.2 saturation failure in disguise: the head
    opened everything while lambda_dev could not reach it, then spent the
    rest of training clawing back. The ENDPOINT of such a run can look
    perfectly fine while the process was broken, and the same config on a
    different seed will not reproduce it — which is exactly why the shape is
    scored and not just the final value.

    plateaued := the peak occurs in the last third of the run
                 AND final >= 0.9 * peak.

    Boundary is inclusive (`>=`, not `>`): `counts.index(peak)` returns the
    FIRST epoch the peak was reached, and a plateau that reaches its max
    exactly at the last-third boundary and holds it thereafter (e.g. 9
    epochs, peak first hit at epoch 6 == (2*9)//3, sustained through epoch
    9) is the textbook rising-and-plateauing shape this function exists to
    pass, not a spike to flag.
    """
    if not counts:
        return {"device_peak": 0, "device_final": 0, "device_plateaued": False}
    peak = max(counts)
    peak_epoch = counts.index(peak) + 1
    final = counts[-1]
    plateaued = peak_epoch >= (2 * len(counts)) // 3 and final >= 0.9 * peak
    return {
        "device_peak": peak,
        "device_final": final,
        "device_plateaued": bool(plateaued),
    }


def trajectory_shape(trajectory_csv: Path) -> dict:
    """trajectory_shape_from_counts over placement_trajectory.csv's
    num_devices column."""
    with open(trajectory_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    return trajectory_shape_from_counts([int(r["num_devices"]) for r in rows])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _site_set(field: str) -> Set[int]:
    """Parse `placement_trajectory.csv`'s space-separated `site_nodes`
    field into a set of node indices."""
    field = field.strip()
    return {int(x) for x in field.split()} if field else set()


def score(config_path: Path) -> dict:
    """Build a DiagContext from the checkpoint `config_path`'s training run
    just produced, and score it with `train.hard_rollout` -- the same
    deployed-allocation rollout `train.py` itself uses for checkpoint
    selection -- combined with the placement_trajectory.csv shape check.
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    ctx = build_context(cfg, load_e2e_checkpoint=True)
    ckpt = ctx.ckpt

    demands, _excluded = fixed_traffic_demands(ctx)

    # `ckpt["vlastelica_lambda"]` is the checkpoint's own saved value for
    # the epoch it was selected at (train.py saves on every loss
    # improvement, not only the final epoch) -- see _common.build_context's
    # docstring for why this is preferred over recomputing a schedule at a
    # guessed epoch.
    hard = hard_rollout(
        ctx.pipeline, demands, ctx.mod_cfg,
        lambda_=ckpt["vlastelica_lambda"],
        margin_db=cfg["constraint"]["margin_db"],
    )

    trajectory_path = Path(cfg["log_dir"]) / "placement_trajectory.csv"
    shape = trajectory_shape(trajectory_path)

    with open(trajectory_path, newline="") as f:
        traj_rows = list(csv.DictReader(f))
    selected_epoch = ckpt["epoch"]
    selected_row = next(
        (r for r in traj_rows if int(r["epoch"]) == selected_epoch), None
    )
    final_row = traj_rows[-1] if traj_rows else None
    # Did training's LAST epoch end in the same deployed state as the
    # SELECTED (checkpointed, best) epoch? False means training kept going
    # -- and drifting -- past the best point it ever found, the per-demand
    # analogue of the old node-identity churn check.
    final_equals_selected = (
        final_row is not None and selected_row is not None
        and _site_set(final_row["site_nodes"]) == _site_set(selected_row["site_nodes"])
    )

    duals = ckpt.get("duals")
    lambda_max_observed = float(duals.max().item()) if duals is not None else math.nan

    pl_cfg = cfg.get("placement", {})

    return {
        "lambda_dev": cfg["pipeline"]["lambda_dev"],
        "alloc_dropout_p": pl_cfg.get("alloc_dropout_p", 0.0),
        "lookahead": pl_cfg.get("lookahead", True),
        "route_context": pl_cfg.get("route_context", True),
        "greedy_residual": pl_cfg.get("greedy_residual", False),
        "lambda_waste": cfg["pipeline"].get("lambda_waste", 0.0),
        "alloc_ste": pl_cfg.get("alloc_ste", False),
        "hard_num_violated": hard["hard_num_violated"],
        "hard_num_devices": hard["hard_num_devices"],
        "hard_num_sites": hard["hard_num_sites"],
        "oracle_devices": hard["oracle_devices"],
        "oracle_gap": hard["oracle_gap"],
        "oracle_infeasible": hard["oracle_infeasible"],
        "hard_worst_margin_db": hard["hard_worst_margin_db"],
        "device_peak": shape["device_peak"],
        "device_final": shape["device_final"],
        "device_plateaued": shape["device_plateaued"],
        "selected_epoch": selected_epoch,
        "final_equals_selected": final_equals_selected,
        "lambda_max_observed": lambda_max_observed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base-config", default="configs/experiment/constrained_stress.yaml",
        help="Base experiment config each arm's overrides are merged into.",
    )
    ap.add_argument(
        "--arms", default=",".join(ARMS_BY_NAME),
        help="Comma-separated arm names (see ARMS in this module).",
    )
    ap.add_argument(
        "--seeds", default="42",
        help="Comma-separated model-init seeds (cfg['seed']).",
    )
    ap.add_argument("--out", required=True, help="Output CSV path.")
    ap.add_argument(
        "--append", action="store_true",
        help="Append to --out instead of overwriting it (no header rewritten).",
    )
    ap.add_argument(
        "--runs-dir", default="runs",
        help="Root directory each arm/seed's config+logs+checkpoints are written under.",
    )
    args = ap.parse_args()

    requested = args.arms.split(",")
    unknown = [name for name in requested if name not in ARMS_BY_NAME]
    if unknown:
        raise ValueError(
            f"Unknown arm(s) {unknown}; choices: {sorted(ARMS_BY_NAME)}"
        )
    # Iterate in ARMS' own order (not the user's --arms order) so output is
    # stable regardless of --arms order.
    requested_set = set(requested)
    ordered_arms = [arm for arm in ARMS if arm["name"] in requested_set]
    seeds = [int(s) for s in args.seeds.split(",")]

    base_cfg = yaml.safe_load(Path(args.base_config).read_text())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs_dir)

    write_header = not (args.append and out_path.exists())
    mode = "a" if args.append else "w"

    with open(out_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
            f.flush()

        for seed in seeds:
            for arm in ordered_arms:
                print(f"=== arm={arm['name']} seed={seed} ===")
                config_path = write_arm_config(base_cfg, arm, seed, runs_dir)
                train(config_path)
                result = score(config_path)

                row = {"arm": arm["name"], "seed": seed, **result}
                writer.writerow(row)
                f.flush()
                print(
                    f"  hard_num_devices={result['hard_num_devices']} "
                    f"(oracle {result['oracle_devices']}, gap {result['oracle_gap']}) "
                    f"hard_num_violated={result['hard_num_violated']} "
                    f"device_plateaued={result['device_plateaued']} "
                    f"final_equals_selected={result['final_equals_selected']}"
                )

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
