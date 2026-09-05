"""Post-hoc trajectory dump: one frame from a saved checkpoint.

Not the primary path — `viz.dump_frames` in the training config produces the
full per-epoch evolution, which is the whole point of the demo (spec
decision 1). This exists so an already-finished run, or a --holdout matrix,
can be visualized without a 300-epoch retrain, and it shares `FrameWriter`
so the two cannot drift.

It is also how "fully config-driven" (decision 2) is checked on the small
topologies: neither `build_context` nor `hard_rollout` calls `compute_loss`,
so no `constraint.rho` is needed, and every config except
`constrained_stress.yaml` ships the explicit `rho: 0.0` placeholder.

Usage:
    conda activate diffopt
    python scripts/dump_frames.py --config configs/experiment/constrained_stress.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import yaml

# `scripts/` is not a package (no __init__.py) — every diagnose_*.py uses this
# same bare import, which resolves because Python puts a script's own
# directory on sys.path. Do not "fix" this to `from scripts._common import`.
from _common import add_common_args, build_context, fixed_traffic_demands, schedule_at
from diffopt.train import hard_rollout
from diffopt.viz import FrameWriter


def resolve_out_path(cfg: dict, out: Optional[str]) -> Path:
    """--out if given, else <log_dir>/frames.json — the same convention
    placement_trajectory.csv already follows."""
    if out:
        return Path(out)
    return Path(cfg.get("log_dir", "logs")) / "frames.json"


def selected_epoch_of(ckpt: Optional[dict]) -> Optional[int]:
    """A post-hoc dump has exactly one frame, and it IS the selected
    checkpoint — so name its epoch rather than leaving it null."""
    if not ckpt:
        return None
    epoch = ckpt.get("epoch")
    return None if epoch is None else int(epoch)


def main() -> None:
    ap = add_common_args(
        argparse.ArgumentParser(description=__doc__),
        default_config="configs/experiment/constrained_stress.yaml",
        with_demands=False,
    )
    ap.add_argument("--out", default=None, help="Defaults to <log_dir>/frames.json")
    ap.add_argument("--stats", default=None,
                    help="Training CSV to embed. Defaults to "
                         "<log_dir>/e2e_train_log.csv; absent is fine.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ctx = build_context(cfg, checkpoint_path=args.checkpoint)
    demands, excluded = fixed_traffic_demands(ctx)
    if excluded:
        print(f"{len(excluded)} demand(s) excluded by the preflight screen")
    print(f"{len(demands)} demands on {cfg['topology']}")

    epoch = selected_epoch_of(ctx.ckpt)
    _, vlastelica_lambda = schedule_at(cfg, epoch or cfg["training"]["epochs_e2e"])
    with torch.no_grad():
        _, _, _, soft = ctx.pipeline(demands, lambda_=vlastelica_lambda)
    hard = hard_rollout(
        ctx.pipeline, demands, soft, ctx.mod_cfg,
        margin_db=cfg["constraint"]["margin_db"],
    )

    out = resolve_out_path(cfg, args.out)
    writer = FrameWriter(
        out, topology=ctx.topology, demands=demands,
        cfg={**cfg, "_config_path": args.config},
        modulation_config=ctx.mod_cfg,
    )
    writer.append(epoch or 1, hard)
    stats = Path(args.stats) if args.stats else (
        Path(cfg.get("log_dir", "logs")) / "e2e_train_log.csv"
    )
    writer.close(epoch, stats)

    # Reporting rules (spec section 9): name the epoch, and never quote
    # oracle_gap without its violation count beside it.
    print(
        f"epoch {epoch}: {hard['hard_num_violated']} violated, "
        f"{hard['hard_num_devices']} devices at {hard['hard_num_sites']} sites "
        f"(oracle minimum {hard['oracle_devices']}), "
        f"worst margin {hard['hard_worst_margin_db']:+.2f} dB"
    )
    print(f"Frames: {out}")


if __name__ == "__main__":
    main()
