# DiffONet — CLAUDE.md

## What this project is

Differentiable optical network design: jointly optimize **routing** and **regenerator placement** for WDM networks using surrogate gradients (Vlastelica et al. ICLR 2020). Phase 1a builds the standalone QoT estimator that will serve as the differentiable surrogate in later phases.

## Environment

```bash
# Activate environment (all commands assume this)
conda activate diffopt   # Python 3.11, at /c/Users/olegk/miniconda3/envs/diffopt/

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

# Run full test suite (42 tests; 2 topology tests require .dat source files)
pytest tests/ -v

# Train QoT model (Phase 1a)
python -m diffopt.qot.train_qot --config configs/experiment/small_test.yaml
python -m diffopt.qot.train_qot --config configs/experiment/base.yaml

# Train end-to-end pipeline (Phase 1c; requires checkpoints/best_qot.pt)
python -m diffopt.train --config configs/experiment/small_test.yaml
python -m diffopt.train --config configs/experiment/base.yaml
```

## Phase 1a milestone

Val RMSE < 0.5 dB on held-out segments → ready for Phase 1b (routing + placement integration).

## Phase 1b milestone

All 7 segment combiner tests and all 5 surrogate gradient tests pass (36/36 total) → ready for Phase 1c (end-to-end pipeline assembly).

## Phase 1c milestone

6/6 `test_pipeline.py` tests pass, 40/42 full suite (2 pre-existing topology failures require `.dat` source files not in the repo). Smoke training run confirms regen count decreasing and loss improving → ready for Phase 1d (evaluation, baselines, visualization).

## Architectural constraints — must not break in future phases

**Topology:**
- Nodes are integer IDs only. No names, no coordinates, no `x`/`y` anywhere.
- Edges are undirected and stored as `src < dst`. The `.dat` files have bidirectional entries; deduplication is by `src < dst` at parse time.
- Span splitting is balanced (no short remainder span). Algorithm: find `n` in `[ceil(L/100), ceil(L/40)]` minimising `|L/n - 80|`. Never produce spans < 20 km.
- All span and amplifier parameters are stored per-span in the topology JSON (not aggregated).

**Modulation / demands:**
- `Demand` has no `modulation` field. Modulation is never assigned at demand-generation time.
- Bitrate → SNR threshold is a **direct lookup** (exact float key match) in `ModulationConfig`. There is no interpolation.
- Valid bitrates: 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800 Gbps (11 values).
- WDM grid: 48 channels at 100 GHz spacing, C-band centered at 193.5 THz.

**QoT model:**
- Input is per-span features for a single **transparent segment** only. No cross-segment state.
- `padding_mask` convention: `True` = real span, `False` = padding. This is the **opposite** of PyTorch `TransformerEncoder`'s `src_key_padding_mask` (the bridge inverts internally).
- `max_spans = 60` is a hard architecture parameter; changing it requires regenerating all datasets and retraining.
- No regenerator-related inputs in the model. The QoT model is deliberately unaware of placement.

**GNPy bridge:**
- `simulate_segment` always silently falls back to the analytical GN model on any GNPy exception. Labels in the dataset may come from either source — this is by design for robustness during development.
- `n_channels` is sampled once per transparent segment (not per link). All spans in a segment share the same channel loading.

**Data pipeline:**
- Parquet schema is fixed: columns `span_features_0` … `span_features_{max_spans*5-1}`, `n_spans`, `gsnr_db`. Changing `max_spans` invalidates existing parquet files.
- Feature ordering within each span is fixed: `[span_length_km, fiber_type_idx, amp_nf_db, channel_loading_fraction, accum_dist_km]`. Index 0–4 within each span block.

**Segment combiner:**
- `SegmentCombiner` accumulates noise internally in **float64** and casts back to float32 on return. Do not move this to float32 — overflow on long noisy paths is real.
- GSNR inputs are clamped to `[-5, 35]` dB before conversion to linear noise. This is intentional.
- The default `soft_max_temperature=0.5` is suitable for training (where annealing brings it down). For unit tests that rely on the approximation being tight (i.e., soft_max ≈ hard max), use `temperature=0.01`.

**Surrogate gradient / routing:**
- `surrogate.py` backward uses **SPFA** (not Dijkstra) for the perturbed solve. Perturbed weights `c + λ·grad` can be negative; Dijkstra hangs on negative weights.
- The Vlastelica perturbation is `c_target = w + λ·grad_output` (positive sign). The formula in the paper uses ŷ = −∂L/∂z; since PyTorch's `grad_output = ∂L/∂z`, the sign flips: `c − λŷ = c + λ·grad_output`. Using the wrong sign (minus) gives zero gradient everywhere.
- Gradient on an edge from `DijkstraSurrogate` is **negative** for active-path edges when the loss penalises their use. Gradient descent (`w -= lr·grad`) then increases those edge costs, routing away from them. This is counterintuitive but correct — do not flip the sign in tests.

## Corrections made during Phase 1a

1. **`pyproject.toml` build backend**: generated as `setuptools.backends.legacy:build` (doesn't exist), corrected to `setuptools.build_meta`.
2. **Channel grid**: the old `IMPLEMENTATION_PLAN_PHASE1A.md` specified 96 channels on a 50 GHz ITU grid. The final plan and the actual modulation format data use **48 channels on a 100 GHz grid**. The code uses 48/100 GHz everywhere.
3. **Topology file names**: old plan used `nobel_germany.json` / `european.json`; actual files are `german_17.json` / `eu_19.json` matching the `.dat` source filenames.

## Architectural constraints — Phase 1c additions

**Pipeline:**
- `tau` and `lambda_` are passed per-call to `pipeline.forward()` — never stored as module attributes (prevents stale annealing state across epochs).
- `path_cost_loss = Σ_demands (path_indicator · edge_weights).sum()` is mechanically required — without it, `∂L/∂path_indicator = 0`, the Vlastelica backward returns the same path, and `EdgeWeightNet` receives zero gradient. It is a gradient enabler, not optional regularization.
- `_reconstruct_path` uses `path_indicator.detach()` for the discrete graph walk, but `path_indicator` stays live in the autograd graph for `path_cost` — both uses of the same tensor are intentional.
- `segment_path` requires `demand_dst` to suppress a spurious empty trailing segment when the path ends exactly on a regen candidate node.
- `accum_dist_km` in span features starts at `0.0` before the first span and increments after each span.
- `amp_nf_db` in span features is `edge.amplifier_nf_db[span_idx]` (per-span), not `edge.mean_amp_nf_db`.
- `SpanAttentionQoT.forward` returns `(batch,)` — use `[0]` to extract a scalar before passing to `SegmentCombiner`.
- QoT model frozen via `requires_grad_(False)` at construction, not via `torch.no_grad()` context. `torch.no_grad()` disables the autograd engine for its entire scope; if the boundary slips to include `regen_probs`, all gradients are silently lost.
- Config files are plain `dict` from `yaml.safe_load` — use `cfg["training"]["lr_edge_net"]`, not `cfg.training.lr_edge_net`.
- `channel_loading_fraction = 0.5` is fixed during inference; this is a known distribution gap from Phase 1a training data where it was sampled uniformly in [1/48, 1.0].

## Corrections made during Phase 1b

1. **Vlastelica perturbation sign**: `DIFFOPT_PROJECT_SPEC.md §4` wrote `c_target = w - λ * (∂L/∂p*)`. This is wrong. The correct formula is `c_target = w + λ * grad_output` because the paper's ŷ is the negative gradient. Using minus gives zero surrogate gradient on every call.
2. **Dijkstra in backward pass**: The spec's `shortest_path.py` description did not mention negative weights. The Vlastelica backward produces perturbed weights that can be negative; Dijkstra hangs on these. SPFA was added to handle this.
3. **`soft_max` scale sensitivity**: The spec showed `temperature=0.1` as a default example. At that temperature, `soft_max` still overestimates `max(a,b)` by up to `0.07` for noise values ~0.1–0.3, which is enough to break the "regen helps" invariant. Tests that verify physical correctness (monotonicity, gradient sign) use `temperature=0.01`.

## Corrections made during Phase 1c

1. **Zero gradient to EdgeWeightNet**: The original spec had `path_indicator` consumed only via `.detach()`. With `∂L/∂path_indicator = 0`, the Vlastelica surrogate returns the same path and produces zero gradient — `EdgeWeightNet` would never update. Fix: `path_cost_loss = Σ (path_indicator · edge_weights).sum()` added to the loss, making `∂L/∂path_indicator = edge_weights > 0`.
2. **QoT output shape**: `SpanAttentionQoT.forward` returns `(batch,)`, not a scalar. For `batch=1`, `[0]` indexing is required before passing to `SegmentCombiner`.
3. **Threshold tensor conversion**: `ModulationConfig.required_snr_threshold` returns a Python `float`. Explicit `torch.tensor(threshold, device=..., dtype=torch.float32)` conversion is needed before the `relu` subtraction.
4. **`FIBER_TYPE_INDEX` not in original spec**: Span feature extraction requires importing `FIBER_TYPE_INDEX` from `diffopt.topology` to encode fiber type as an integer.
5. **Hub topology underspecified**: The original plan said "Node 1 has degree 3" without a full graph. Corrected to a 5-node, 5-edge graph with two paths 0→4 via nodes 1 and 2, merging at node 3 (degree 3).
