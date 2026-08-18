# DiffONet — CLAUDE.md

## What this project is

Differentiable optical network design: jointly optimize **routing** and **regenerator placement** for WDM networks using surrogate gradients (Vlastelica et al. ICLR 2020). Phase 1a builds the standalone QoT estimator that will serve as the differentiable surrogate in later phases.

## Environment

```bash
# Activate environment (all commands assume this)
conda activate diffopt   # Python 3.11

# Install / reinstall after pyproject changes
pip install -e ".[dev]"
```

## Key commands

```bash
# Re-generate topology JSONs from .dat files (run from project root)
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

# Run full test suite (150 tests, 150 pass, 0 fail — includes tests/test_optical_bridge.py,
# tests/test_generate_qot_dataset.py, tests/test_edge_noise.py added during the GNPy migration,
# and tests/test_train.py, tests/test_loss.py, tests/test_shortest_path.py,
# tests/test_span_features.py, tests/test_scripts_common.py, tests/conftest.py added since)
pytest tests/ -v

# Diagnostics (scripts/diagnose_*.py) — all take --config, default small_test_ind132.yaml.
# diagnose_segment_noise_scale.py is the fastest check that the combiner's
# "regen helps" invariant still holds at the real per-segment noise scale.
python scripts/diagnose_segment_noise_scale.py --config configs/experiment/base.yaml

# Train QoT model (Phase 1a)
python -m diffopt.qot.train_qot --config configs/experiment/small_test.yaml
python -m diffopt.qot.train_qot --config configs/experiment/base.yaml

# Train end-to-end pipeline (Phase 1c; requires checkpoints/best_qot.pt)
python -m diffopt.train --config configs/experiment/small_test.yaml
python -m diffopt.train --config configs/experiment/base.yaml
```

## Current status

Phases 1a-1c complete; Phase 1d (evaluation, baselines, visualization) not
started. See `docs/investigations/CHANGELOG.md` for the phase milestone
write-ups (measured RMSE, test-pass counts) and `docs/investigations/open_followups.md`
for what's still open.

## Where to look

| I need… | Read |
|---|---|
| the rules I must not break | `docs/architecture/invariants.md` |
| why a rule exists | `docs/investigations/CHANGELOG.md` |
| component contracts and config keys | `docs/architecture/interfaces.md` |
| what is still open | `docs/investigations/open_followups.md` |
| what past investigations found | `docs/investigations/README.md` |
