"""Compare placement-signal arms on constrained_stress / ind_132.

Arms (lambda_regen held at 1.0 except where stated, so arms are comparable —
retuning it is an explicit non-goal: the sweep is exhausted, no price yields
the feasible optimum):

  baseline           sigmoid, no dropout      — must reproduce today's numbers
  dropout_0.1        sigmoid, dropout 0.1     — toy: 11 -> 6, feasible, nodes
                                                {4,5,6,7,9,12}
  dropout_0.3        sigmoid, dropout 0.3     — toy: 11 -> 6, feasible, nodes
                                                {3,4,7,9,11,12}
  l0_lambda1         hard_concrete            — toy: 7 placed, 1 VIOLATED
  l0_lambda3         hard_concrete, lr 3.0    — toy: 3 placed, 2 VIOLATED
  l0_dropout_0.3     hard_concrete + dropout  — untested; see rationale below
  dropout_0.3_decay  dropout 0.3 + dual_decay — contingency, see below

The two L0 arms are recorded as "right count, wrong nodes", which undersells
the failure: both were INFEASIBLE (1 and 2 violated), so lambda=3 hit the
right cardinality without solving the problem. Its nodes {3,4,5} are adjacent
on a line network — three regenerations over a short stretch with nodes 6-12
left bare — and lambda=1's {1,3,4,5,6,7,8} is the same low-end region, so the
bias is systematic across lambda rather than a bad seed. Both dropout arms,
by contrast, contain two of the three true nodes (4 and 7) and bracket the
third (9, 11 around 10), spread across the whole line. Score node IDENTITY,
not just count.

`l0_dropout_0.3` is not in the write-up's table and is the arm with the
clearest a-priori case: the two remedies address orthogonal halves. L0
changes HOW MANY the objective wants (expected count, not probability mass —
it can express "exactly three" where no lambda_regen under L1-on-mass ever
could). Dropout changes WHICH ones get gradient. L0 alone cannot fix node
identity by construction, because identity is decided solely by the
feasibility term's gradient, which measures exactly 0.0000 on the plateau.
Note the write-up's tested combination was softplus+dropout, which was worse
than either alone; that is NOT evidence about this pair, since the softplus
failure mode was a permanently-active feasibility term bidding the duals up,
which hard-concrete gates do not do.

The contingency arm exists because update_duals is pure ascent by default and
gate dropout manufactures violations on purpose. A dual rising when a
load-bearing node is dropped IS the mechanism, so it must not be suppressed —
but the ratchet can overrun (that is exactly how the softplus-hinge variant
failed: a permanently-active feasibility term kept bidding the duals up until
they overwhelmed lambda_regen). Decision rule: run this arm only if the
dropout arms show lambda_max_observed materially above baseline.

Note on dual_decay: it was falsified as a remedy for over-provisioning
(there is no loss barrier for a relaxed dual to lower). Under dropout it has
a genuinely different job — bleeding off ratchet from violations that were
manufactured on purpose — so that refutation does not carry over.

--- Implementation notes (not part of the plan's docstring text) ---

`l0_lambda3`'s "lr 3.0" is read literally as `training.lr_regen: 3.0`
(constrained_stress.yaml's base value is 2.0e-2) — the toy replica's own
`--lambda-regen` CLI flag is a different, already-named knob
(`pipeline.lambda_regen`, held fixed across every arm per the paragraph
above), so "lr" here can only mean the regen parameter's optimizer learning
rate. `dropout_0.3_decay` uses `constraint.dual_decay: 0.05`, the same value
`open_followups.md` item #3's `_dual_decay.yaml` hypothesis test used.

`score()`'s greedy-drop `order_key` for the `hard_concrete` gate uses each
node's `P(gate open) = sigmoid(log_alpha - beta*log(-gamma/zeta))` — exactly
the per-node quantity `RegenPlacement.count_penalty()` sums, so "drop the
node the model itself is least committed to opening first" is the same
ordering principle as the sigmoid gate's `sigmoid(logit/tau_end)`, just
through the L0 gate's own probability rather than a temperature-sharpened
one (hard_concrete ignores `tau` entirely — see `RegenPlacement.get_regen_probs`).
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

import torch
import yaml

from _common import build_context, fixed_traffic_demands
from diagnose_regen_ablation import ablate

# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

ARMS: List[dict] = [
    {"name": "baseline", "overrides": {}},
    {
        "name": "dropout_0.1",
        "overrides": {"placement": {"gate_dropout_p": 0.1}},
    },
    {
        "name": "dropout_0.3",
        "overrides": {"placement": {"gate_dropout_p": 0.3}},
    },
    {
        "name": "l0_lambda1",
        "overrides": {"placement": {"gate": "hard_concrete"}},
    },
    {
        "name": "l0_lambda3",
        "overrides": {
            "placement": {"gate": "hard_concrete"},
            "training": {"lr_regen": 3.0},
        },
    },
    {
        "name": "l0_dropout_0.3",
        "overrides": {
            "placement": {"gate": "hard_concrete", "gate_dropout_p": 0.3},
        },
    },
    {
        "name": "dropout_0.3_decay",
        "overrides": {
            "placement": {"gate_dropout_p": 0.3},
            "constraint": {"dual_decay": 0.05},
        },
    },
]

ARMS_BY_NAME: Dict[str, dict] = {arm["name"]: arm for arm in ARMS}

CSV_FIELDNAMES = [
    "arm", "seed", "gate", "gate_dropout_p", "lambda_regen", "dual_decay",
    "placed", "placed_on_candidates", "hard_num_violated",
    "minimal_set_size", "minimal_nodes", "over_provisioning_ratio",
    "churn_last20", "final_equals_selected", "lambda_max_observed",
    "selected_epoch", "placed_nodes", "baseline_minimal_overlap",
]


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge `overrides` into a COPY of `base`.

    Nested dicts are merged key-by-key (a `placement` override does not
    clobber sibling `placement` keys the base config already sets, e.g.
    Task 8/9's `hard_concrete: {beta, gamma, zeta}`); any non-dict value
    (including a dict overriding a non-dict, or vice versa) replaces the
    base value outright.
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
    arms/seeds vary EdgeWeightNet/RegenPlacement init, not the traffic
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

def _read_trajectory(trajectory_csv: Path) -> List[Set[int]]:
    """`placement_trajectory.csv` (Task 4: `epoch,num_placed,placed_nodes`,
    space-separated node indices) -> one `set[int]` per row, epoch order."""
    rows: List[Set[int]] = []
    with open(trajectory_csv, newline="") as f:
        for row in csv.DictReader(f):
            raw = row["placed_nodes"].strip()
            rows.append({int(x) for x in raw.split()} if raw else set())
    return rows


def churn_last20(trajectory_csv: Path) -> float:
    """Mean symmetric-difference size between consecutive epochs' placement
    sets, over the last 20 rows of `trajectory_csv` (or fewer, if the run is
    shorter). This is the limit-cycle measure: baseline should churn; a
    remedy that actually breaks the cycle should settle toward 0.

    Handles short trajectories gracefully: fewer than 2 rows total, or a
    `last 20` window of fewer than 2 rows, both return 0.0 rather than
    raising (empty statistics, not "no churn observed" -- see caller for how
    to distinguish the two cases if that matters).
    """
    rows = _read_trajectory(Path(trajectory_csv))
    window = rows[-20:] if len(rows) >= 20 else rows
    if len(window) < 2:
        return 0.0
    diffs = [len(window[i] ^ window[i - 1]) for i in range(1, len(window))]
    return sum(diffs) / len(diffs)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(config_path: Path) -> dict:
    """Build a DiagContext from the checkpoint `config_path`'s training run
    just produced, run the Task 10 ablation against it, and combine with
    churn/trajectory checks.

    Does not know about other arms/seeds -- `baseline_minimal_overlap` is
    filled in by the caller (`main`), which has the cross-arm state this
    function deliberately does not.
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    ctx = build_context(cfg, load_e2e_checkpoint=True)

    demands, _excluded = fixed_traffic_demands(ctx)
    thresholds = {
        d.id: ctx.mod_cfg.required_snr_threshold(d.bitrate_gbps) for d in demands
    }

    ckpt = ctx.ckpt
    R = set(ckpt["placement_mask"].nonzero(as_tuple=True)[0].tolist())
    candidates = ctx.regen_candidates

    gate = ctx.regen_placement.gate
    learned_param = ctx.regen_placement._parameter.detach().clone()
    if gate == "sigmoid":
        tau_end = cfg["training"]["regen_tau_end"]
        order_probs = torch.sigmoid(learned_param / tau_end)
    else:
        # hard_concrete: no `tau` (`get_regen_probs` ignores it for this
        # gate -- `beta` plays that role). Use each node's own P(gate open),
        # the same quantity RegenPlacement.count_penalty() sums -- see the
        # module docstring's implementation-notes section.
        hc_cfg = cfg.get("placement", {}).get("hard_concrete", {})
        beta = hc_cfg.get("beta", 0.5)
        gamma = hc_cfg.get("gamma", -0.1)
        zeta = hc_cfg.get("zeta", 1.1)
        shift = beta * math.log(-gamma / zeta)
        order_probs = torch.sigmoid(learned_param - shift)

    result = ablate(
        ctx.pipeline, demands, thresholds, R, candidates,
        tau=cfg["training"]["regen_tau_end"],
        lambda_=ckpt["vlastelica_lambda"],
        num_nodes=ctx.topology.num_nodes,
        order_key=lambda n: order_probs[n].item(),
    )
    minimal_set = sorted(result["minimal_set"])

    trajectory_path = Path(cfg["log_dir"]) / "placement_trajectory.csv"
    trajectory_rows = _read_trajectory(trajectory_path)
    final_nodes = trajectory_rows[-1] if trajectory_rows else set()

    duals = ckpt.get("duals")
    lambda_max_observed = float(duals.max().item()) if duals is not None else math.nan

    placed = len(R)
    minimal_set_size = len(minimal_set)
    pl_cfg = cfg.get("placement", {})

    return {
        "gate": gate,
        "gate_dropout_p": pl_cfg.get("gate_dropout_p", 0.0),
        "lambda_regen": cfg["pipeline"]["lambda_regen"],
        "dual_decay": cfg.get("constraint", {}).get("dual_decay", 0.0),
        "placed": placed,
        "placed_on_candidates": len(R & candidates),
        "hard_num_violated": ckpt["hard_num_violated"],
        "minimal_set_size": minimal_set_size,
        "minimal_nodes": " ".join(str(n) for n in minimal_set),
        "over_provisioning_ratio": placed / max(1, minimal_set_size),
        "churn_last20": churn_last20(trajectory_path),
        "final_equals_selected": final_nodes == R,
        "lambda_max_observed": lambda_max_observed,
        "selected_epoch": ckpt["epoch"],
        "placed_nodes": " ".join(str(n) for n in sorted(R)),
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
    # Iterate in ARMS' own order (not the user's --arms order): baseline is
    # first there, and baseline_minimal_overlap below needs baseline scored
    # before any other arm at the same seed.
    requested_set = set(requested)
    ordered_arms = [arm for arm in ARMS if arm["name"] in requested_set]
    seeds = [int(s) for s in args.seeds.split(",")]

    base_cfg = yaml.safe_load(Path(args.base_config).read_text())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs_dir)

    write_header = not (args.append and out_path.exists())
    mode = "a" if args.append else "w"

    baseline_minimal_by_seed: Dict[int, Set[int]] = {}

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

                placed_nodes = (
                    {int(x) for x in result["placed_nodes"].split()}
                    if result["placed_nodes"] else set()
                )
                if arm["name"] == "baseline":
                    minimal_nodes = (
                        {int(x) for x in result["minimal_nodes"].split()}
                        if result["minimal_nodes"] else set()
                    )
                    baseline_minimal_by_seed[seed] = minimal_nodes
                    overlap = len(placed_nodes & minimal_nodes)
                elif seed in baseline_minimal_by_seed:
                    overlap = len(placed_nodes & baseline_minimal_by_seed[seed])
                else:
                    print(
                        f"  WARNING: 'baseline' arm was not run for seed={seed} "
                        "(not in --arms) -- baseline_minimal_overlap left blank"
                    )
                    overlap = ""

                row = {
                    "arm": arm["name"],
                    "seed": seed,
                    **result,
                    "baseline_minimal_overlap": overlap,
                }
                writer.writerow(row)
                f.flush()
                print(
                    f"  placed={result['placed']} minimal={result['minimal_set_size']} "
                    f"hard_num_violated={result['hard_num_violated']} "
                    f"churn_last20={result['churn_last20']:.2f} "
                    f"baseline_minimal_overlap={overlap}"
                )

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
