# DiffONet — CLAUDE.md

## What this project is

Differentiable optical network design: jointly optimize **routing** and **regenerator placement** for WDM networks using surrogate gradients (Vlastelica et al. ICLR 2020). A transformer QoT estimator (`SpanAttentionQoT`) is trained standalone on real-GNPy labels, then frozen and used as the differentiable physics surrogate inside the end-to-end pipeline.

## Environment

```bash
# Activate environment (all commands assume this)
conda activate diffopt   # Python 3.11

# Install / reinstall after pyproject changes
pip install -e ".[dev]"
```

## Key commands

```bash
# Re-generate topology JSONs from .dat files (run from project root; .dat
# sources are not distributed in a clean clone — see configs/topology/README.md)
python diffopt/topology_builder.py

# Generate QoT training data
python data/generate_qot_dataset.py --config configs/experiment/small_test.yaml
python data/generate_qot_dataset.py --config configs/experiment/base.yaml

# Run tests — Phase 1a
pytest tests/test_topology.py tests/test_modulation.py tests/test_qot_model.py -v

# Run tests — Phase 1b
pytest tests/test_segment_combiner.py tests/test_surrogate_grad.py -v

# Run tests — Phase 1c
pytest tests/test_pipeline.py -v

# Run the full test suite (everything under tests/, not just the
# phase-grouped subsets above)
pytest tests/ -v

# Diagnostics (scripts/diagnose_*.py) — all take --config, default small_test_ind132.yaml.
# diagnose_segment_noise_scale.py is the fastest check that the combiner's
# "regen helps" invariant still holds at the real per-segment noise scale.
python scripts/diagnose_segment_noise_scale.py --config configs/experiment/base.yaml

# Train QoT model (Phase 1a)
python -m diffopt.qot.train_qot --config configs/experiment/small_test.yaml
python -m diffopt.qot.train_qot --config configs/experiment/base.yaml

# Train end-to-end pipeline (Phase 1c; requires the config's `qot_checkpoint`
# — run train_qot for that config first)
python -m diffopt.train --config configs/experiment/small_test.yaml
python -m diffopt.train --config configs/experiment/base.yaml

# Train with a per-epoch training-trajectory frame dump (Phase 1d viz;
# requires the config's `viz.dump_frames: true` — already set in
# configs/experiment/constrained_stress.yaml). Streams
# logs/<run>/frames.jsonl during training and assembles logs/<run>/frames.json
# at the end.
python -m diffopt.train --config configs/experiment/constrained_stress.yaml

# Build the self-contained trajectory viewer page from a frame dump
python scripts/build_viewer.py \
  --frames logs/constrained_stress/frames.json \
  --out build/trajectory_viewer.html
```

## Current status

Phases 1a-1c complete. Phase 1d (evaluation, baselines, visualization) is
started: the training-trajectory visualization strand (frame dump behind
`viz.dump_frames`, `scripts/build_viewer.py`, `diffopt/viz/viewer.html`) is
done; evaluation and baselines are not.

## Where to look

Two tiers. `docs/architecture/` ships with the repo; `docs/investigations/`
is a **local-only lab notebook** — gitignored, deliberately absent from the
published history, and present only in a working copy that has it. A fresh
clone has the first tier and not the second, so the local-only rows below
simply won't resolve there.

Published (`docs/architecture/`):

| I need… | Read |
|---|---|
| the rules I must not break | `invariants.md` |
| the full pipeline walkthrough | `pipeline.md` — Part 2 |
| how the QoT surrogate is designed and trained | `pipeline.md` — Part 1 |
| the surrogate gradient explained from scratch | `pipeline.md` — Part 3 |
| every tuneable and what it does | `pipeline.md` — Hyperparameters |
| component contracts and config keys | `interfaces.md` |

Local-only (`docs/investigations/`, not in the repo):

| I need… | Read |
|---|---|
| why a rule exists | `CHANGELOG.md` |
| what is still open | `open_followups.md` |
| what past investigations found | `README.md` |

When a change is driven by something recorded only in the local notebook,
carry the reasoning into the published docs (usually `invariants.md`) rather
than citing a file a reader cannot open.
