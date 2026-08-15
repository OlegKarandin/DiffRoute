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

# Run full test suite (107 tests; 105 pass, 2 topology tests require .dat source files
# not in the repo — includes tests/test_optical_bridge.py, tests/test_generate_qot_dataset.py,
# tests/test_edge_noise.py added during the GNPy migration)
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

## Phase 1a milestone

Re-baselined post-GNPy-migration: the original 0.5 dB target was measured against analytically-generated labels, not real GNPy labels, so it no longer means what it used to.

**Re-measured after switching the production topology from `german_17` to `ind_132`** (132 nodes, 168 edges — see "Upstream dependency"/topology sections; source: https://github.com/OlegKarandin/jocn24-multi-fiber.git, Karandin et al., J. Opt. Commun. Netw. 16, H18-H26, 2024). `german_17`'s tiny segment space (~65,904 possible feature vectors) meant a 50k/10k sample draw duplicated ~77% of rows between train and val, so its RMSE number (0.0428 dB, previously recorded here) was flattering but not a trustworthy generalization estimate. `ind_132` measures out to a ~63x larger pool (~4.1M), and only **7.0%** of val rows have any train duplicate at all (703/10,000; unique val vectors seen in train: 6.7%, 665/9,892) — this number is measured almost entirely against segments the model hasn't seen before.

Re-baselined target is **0.25 dB** (matches `configs/experiment/base.yaml`'s `val_rmse_target`, set with ~30% headroom above the measured value below — single run, not yet replicated). Measured against the production `base.yaml` config (50k train / 10k val, real GNPy labels, 100 epochs, `ind_132` topology): **0.1909 dB** best val RMSE (epoch 99, fully converged — epoch 100 was 0.1909 dB too, `train_mse` tracked `val_rmse` closely the entire run with no overfitting signal). Against the dataset's actual GSNR range (~6.5–20.0 dB), that's ~1.4% relative error. Clears the re-baselined 0.25 dB target → ready for Phase 1b (routing + placement integration).

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
- Owned by `diffopt/qot/optical_bridge.py` (the old `diffopt/qot/gnpy_bridge.py`, which silently fell back to an analytical GN-model approximation on any GNPy exception, is deleted). **No fallback**: every `compute_qot` call either returns a real GNPy-derived GSNR or raises. There is no bare `except Exception` anywhere in this module — the AST test `tests/test_optical_bridge.py` enforces this structurally, not just by convention.
- `direction=Direction.FORWARD` always, and `center_freq_hz=CUT_FREQ_HZ` (193.5 THz, slot 21 on the 191.4 THz / 100 GHz / 48-slot grid) is mandatory on every `compute_qot` call — omitting it silently selects the wrong probe channel (measured ~0.05 dB error in one case; keep this warning).
- `power_dbm` is always literal `None` on every `Channel`: every ROADM re-equalizes to a fixed target output power, so per-channel launch power is physically inert. Do not reintroduce a launch-power knob (`launch_power_dbm` is gone from the configs for this reason).
- `n_channels` is still sampled once per transparent segment (not per link). All spans in a segment share the same channel loading — this behavior is unchanged from the old bridge.
- `build_loading`'s channel-selection policy expands outward from the CUT slot (deterministic, not `FillPolicy.FULL`) — see `diffopt/qot/optical_bridge.py`'s own docstring for the exact policy.
- GSNR is invariant across all 11 modulation formats in `configs/modulation_formats.yaml` (they share 87.5 GBaud / 0.15 roll-off) — this is why the dataset generator and the ground-truth test both pick a single arbitrary `mode_id` rather than tracking bitrate.

**Upstream dependency (`multilayer-optical-mcp`):**
- Pinned via git dependency in `pyproject.toml` at commit `2b64361` (`https://github.com/OlegKarandin/multilayer-optical-mcp.git`) — an exact commit, not a version range. `diffopt.topology.Topology` subclasses its `OpticalNetworkModel` directly, the same extension pattern upstream uses for its own `NetworkModel(OpticalNetworkModel)` IP-layer subclass.
- Pinned to an exact commit (not a version) because this project depends on specific behavior verified against that commit's source: the ground-truth 18.85 dB GSNR test in `tests/test_optical_bridge.py`, and the exact `populate_optical`/`split_link_into_spans` algorithms that `diffopt/topology.py` and `diffopt/topology_builder.py` re-export. Bumping the pin requires re-verifying the ground-truth test and the traps above still hold, not just updating a version number.
- Real GNPy (`gnpy==2.14.0`) is a transitive dependency, not declared directly in `pyproject.toml` — nothing in this repo imports `gnpy` directly anymore.
- This project used to compute GSNR via a broken GNPy wrapper (`gnpy_bridge.py`) that silently fell back to an analytical approximation for its entire history before this dependency was adopted — every label in every dataset generated before the migration came from that fallback, and real GNPy never once executed. See git history and this file's "Corrections made during Phase 1a/1b/1c" sections below for the phase-by-phase record of that era; those sections describe what was actually believed true at the time and are left as-is.

**Data pipeline:**
- Parquet schema is fixed: columns `span_features_0` … `span_features_{max_spans*5-1}`, `n_spans`, `gsnr_db`. Changing `max_spans` invalidates existing parquet files.
- Feature ordering within each span is fixed: `[span_length_km, fiber_type_idx, amp_nf_db, channel_loading_fraction, accum_dist_km]`. Index 0–4 within each span block.

**Segment combiner:**
- `SegmentCombiner` accumulates noise internally in **float64** and casts back to float32 on return. Do not move this to float32 — overflow on long noisy paths is real.
- GSNR inputs are clamped to `[-5, 35]` dB before conversion to linear noise. This is intentional.
- **`soft_max` is scale-normalised** — it divides by `max(a,b).detach()` before the log-sum-exp and multiplies back after. Do not "simplify" this back to the plain `t * logsumexp([a/t, b/t])` form. The plain form's approximation error is `t*ln2`, which is **absolute** in linear-noise units and does not shrink with its operands; real per-segment noise is ~0.0025, so it swamped the signal and inverted the "regen helps" invariant (see Phase 1c correction #8). Normalised, the error is `max(a,b)*t*ln2` — a fixed fraction — and the invariant holds for any `temperature < 1/ln2 ≈ 1.44` at any noise magnitude.
- Because the error is now relative, the temperature no longer encodes a hidden assumption about topology-dependent noise scale. `soft_max_temperature=0.5` annealing down to `0.01` is sign-correct end to end; `temperature=0.01` remains the value to use in unit tests that need the approximation tight (soft_max ≈ hard max).
- Any test of the "regen helps" invariant must include a case at the **production** noise scale (two segments at ~26 dB → noise ~0.0025), not only the 5–15 dB fixtures. `tests/test_segment_combiner.py::test_gradient_wrt_regen_prob_at_production_noise_scale` and `::test_regen_never_increases_noise_across_noise_scales` exist for exactly this and are what caught correction #8.

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
- `tau`, `lambda_`, and `soft_max_temperature` are passed per-call to `pipeline.forward()` — never stored as module attributes (prevents stale annealing state across epochs). `SegmentCombiner` is correspondingly stateless: `temperature` is a required `forward()` argument, not a constructor parameter (see the fixed-bug note in "Corrections made during Phase 1c" for why this matters — an earlier version took it at construction and it was never annealed).
- Pre-STE: `path_cost_loss = Σ_demands (path_indicator · edge_weights).sum()` was mechanically required — without it, `∂L/∂path_indicator = 0`, the Vlastelica backward returns the same path, and `EdgeWeightNet` receives zero gradient. It was a gradient enabler, not optional regularization. As of the STE correction (see correction #6), this is no longer the sole gradient enabler — the ASE-noise proxy independently supplies `∂L/∂path_indicator`, which is why `lambda_cost` was demoted to `0.01`.
- `_reconstruct_path` uses `path_indicator.detach()` for the discrete graph walk, but `path_indicator` stays live in the autograd graph for `path_cost` — both uses of the same tensor are intentional.
- `segment_path` requires `demand_dst` to suppress a spurious empty trailing segment when the path ends exactly on a regen candidate node.
- `accum_dist_km` in span features starts at `0.0` before the first span and increments after each span.
- `amp_nf_db` in span features is `edge.amplifier_nf_db[span_idx]` (per-span), not `edge.mean_amp_nf_db`.
- `SpanAttentionQoT.forward` returns `(batch,)` — use `[0]` to extract a scalar before passing to `SegmentCombiner`.
- QoT model frozen via `requires_grad_(False)` at construction, not via `torch.no_grad()` context. `torch.no_grad()` disables the autograd engine for its entire scope; if the boundary slips to include `regen_probs`, all gradients are silently lost.
- Config files are plain `dict` from `yaml.safe_load` — use `cfg["training"]["lr_edge_net"]`, not `cfg.training.lr_edge_net`.
- `channel_loading_fraction = 0.5` is fixed during inference; this is a known distribution gap from Phase 1a training data where it was sampled uniformly in [1/48, 1.0].
- `edge_ase_noise` is a fixed buffer on `DiffONetPipeline`, precomputed from the analytical GN ASE formula and median-normalized (`diffopt/qot/edge_noise.py::compute_edge_ase_noise`). Per-span loss is read per-edge via `topology.get_fiber_type(edge.fiber_type).loss_coef_db_per_km` — not a hardcoded SSMF constant — so this proxy correctly varies across mixed-fiber-type topologies. NLI is excluded by design. It defaults to being computed automatically from the topology if not passed explicitly to `DiffONetPipeline.__init__` — `train.py` and `scripts/diagnose_surrogate.py` rely on this default and do not pass it explicitly. General lesson: the STE proxy only discriminates between edges to the extent `edge_ase_noise` actually varies across them — on a topology whose edges are physically identical, the STE contributes no within-segment routing signal (this is why `tests/test_pipeline.py::make_hub_topology()` uses asymmetric edge lengths).
- The STE expression `qot_gsnr + (proxy_gsnr - proxy_gsnr.detach())` in `pipeline.py`'s per-segment loop must be written on a single line. Splitting it across statements risks inserting a graph break that defeats the gradient redirection.

## Corrections made during Phase 1b

1. **Vlastelica perturbation sign**: `DIFFOPT_PROJECT_SPEC.md §4` wrote `c_target = w - λ * (∂L/∂p*)`. This is wrong. The correct formula is `c_target = w + λ * grad_output` because the paper's ŷ is the negative gradient. Using minus gives zero surrogate gradient on every call.
2. **Dijkstra in backward pass**: The spec's `shortest_path.py` description did not mention negative weights. The Vlastelica backward produces perturbed weights that can be negative; Dijkstra hangs on these. SPFA was added to handle this.
3. **`soft_max` scale sensitivity**: The spec showed `temperature=0.1` as a default example. At that temperature, `soft_max` still overestimates `max(a,b)` by up to `0.07` for noise values ~0.1–0.3, which is enough to break the "regen helps" invariant. Tests that verify physical correctness (monotonicity, gradient sign) use `temperature=0.01`. **Superseded by Phase 1c correction #8**: the "~0.1–0.3" noise range quoted here is the *unit-test fixture* range (5–10 dB segments), not the production one. Real segments are ~26 dB → noise ~0.0025, ~100x smaller, and `0.01` is inverted there too. The temperature constant was never the right lever; the scale normalisation in correction #8 is.

## Corrections made during Phase 1c

1. **Zero gradient to EdgeWeightNet**: The original spec had `path_indicator` consumed only via `.detach()`. With `∂L/∂path_indicator = 0`, the Vlastelica surrogate returns the same path and produces zero gradient — `EdgeWeightNet` would never update. Fix: `path_cost_loss = Σ (path_indicator · edge_weights).sum()` added to the loss, making `∂L/∂path_indicator = edge_weights > 0`.
2. **QoT output shape**: `SpanAttentionQoT.forward` returns `(batch,)`, not a scalar. For `batch=1`, `[0]` indexing is required before passing to `SegmentCombiner`.
3. **Threshold tensor conversion**: `ModulationConfig.required_snr_threshold` returns a Python `float`. Explicit `torch.tensor(threshold, device=..., dtype=torch.float32)` conversion is needed before the `relu` subtraction.
4. **`FIBER_TYPE_INDEX` not in original spec**: Span feature extraction requires importing `FIBER_TYPE_INDEX` from `diffopt.topology` to encode fiber type as an integer.
5. **Hub topology underspecified**: The original plan said "Node 1 has degree 3" without a full graph. Corrected to a 5-node, 5-edge graph with two paths 0→4 via nodes 1 and 2, merging at node 3 (degree 3).
6. **Zero Vlastelica surrogate gradient (STE correction)**: With only `path_cost_loss` providing `grad_output`, the Vlastelica perturbation `c_target = w·(1+λ·λ_cost)` is a uniform scaling and the surrogate always returns the same path (confirmed via `scripts/diagnose_surrogate.py`: Hamming distance 0, `g/w` ratio std ≈ 0). Fix: a straight-through estimator in `pipeline.py`'s per-segment loop blends the QoT-accurate forward value with an analytical per-edge ASE noise proxy (`diffopt/qot/edge_noise.py`) that supplies the backward gradient direction instead. `EdgeWeightNet` now receives a per-edge, threshold-gated, regen-modulated routing signal independent of its own current output. As part of the same fix, `lambda_cost` was demoted from `0.1` to `0.01` in both config files (and in `compute_loss`'s default) since the ASE-noise STE now supplies the dominant routing signal and `path_cost_loss` is a regularizer only.
7. **`soft_max_temperature=0.5` hardcoded and never annealed (found post-GNPy-migration, during a follow-up e2e smoke test — not caught when Phase 1c originally landed)**: `diffopt/train.py`'s `SegmentCombiner(...)` construction fixed the temperature at 0.5 and nothing in the codebase ever touched it again, despite this file's own documented assumption ("0.5 is suitable for training, where annealing brings it down"). At `t=0.5` the soft-max floor (`t·ln2 ≈ 0.347`) caps any regenerated path's predicted GSNR at ≤4.6 dB, inverting the sign of `dGSNR/dp` — the pipeline was told regenerators make links *worse*, not better. Verified against the real GNPy-trained checkpoint (multi-segment paths capped at ~3.5–7.1 dB, matching the closed-form noise-floor prediction) and against a smoke-test checkpoint that had placed zero regenerators after 20 epochs. **Fixed**: `SegmentCombiner` is now stateless — `temperature` is a required `forward()` argument (see the "Pipeline" architectural constraints above), threaded through `DiffONetPipeline.forward(..., soft_max_temperature=...)` exactly like `tau`/`lambda_`, and annealed in `diffopt/train.py` from `segment_combiner.soft_max_temperature` (0.5) down to `segment_combiner.soft_max_temperature_min` (0.01 — the value this file's Phase 1b correction #3 established as where the "regen helps" invariant reliably holds) over the same epoch window as `regen_tau`. Re-running the `small_test.yaml` e2e smoke test post-fix: `num_infeasible` converges to and stays at exactly `0` from epoch 17 onward (it never got below 1–2 in the pre-fix run) — genuine, stable convergence, not a fluke.
8. **`soft_max`'s error floor is absolute, not relative — the "regen helps" invariant was inverted on every topology since Phase 1b (found while investigating why `num_regen_soft` stayed at 0 on `ind_132`; full write-up in `docs/investigations/regen_placement_not_concentrating.md`)**: `soft_max(a,b,t) = max(a,b) + t·δ` with `0 < δ ≤ ln2`. That overshoot is **absolute in linear-noise units** and does not shrink with the operands. Regeneration only helps if `soft_max(a,b) < a+b`, i.e. (for `a ≈ b = n`) if `t·ln2 < n`. Measured per-segment noise is **0.00266** on `ind_132` and **0.00227** on `german_17` (segments run ~180–210 km on both, because 36–59% of nodes are degree ≥ 3 and `segment_path` cuts at every one of them, so segments are short and sit at ~26 dB). The schedule's sharpest temperature, `0.01`, has a floor of `0.00693` — **2.6x larger than the noise itself**. Consequences, all measured: the feasibility gradient on `regen_logits` was large and **positive** (max +160) for epochs 1–17, i.e. pointing the wrong way, so all 132 logits slid down in lockstep at `lr_regen`/epoch (a pure-drift null model with zero node-specific learning reproduced the whole `regen_loss` curve to **3.49%**), and the forward pass underestimated path GSNR by **~15 dB** at `t=0.5`, manufacturing fake infeasibility (98/100 demands "infeasible" that are actually fine). Note this is **not** an `ind_132` regression — `german_17` was marginally worse; the switch only made it *visible*, because its 427 km paths never needed regenerators while `ind_132`'s do. It also reinterprets correction #7: that fix was real, but its reported `num_infeasible → 0` came from `EdgeWeightNet` routing, not placement — `num_regen_soft` was 0 there too. **Fixed** by normalising `soft_max` by `max(a,b).detach()` (see the "Segment combiner" constraints above), plus two regression tests at the production noise scale, raw `regen_logits` logging in `train.py` (the old `regen_loss` metric hid the problem: 87% of its 66 → 18 fall was `tau` annealing under a near-static logit), and `torch.manual_seed` in `train.py` (e2e runs were unseeded and irreproducible — `num_infeasible` at epoch 1 varied 5/100 vs 98/100 between runs). Verified on `small_test_ind132.yaml`, 60 epochs: `num_infeasible` reaches **0 at epoch 30 and holds**, **18** regenerators placed (was 0), max `regen_prob` **0.9994** (was 0.24), logits spread **[−0.944, +0.743]** across zero instead of packing into a 0.085-wide negative band — and all 18 are regen candidates, with mean `p` 0.3690 on candidates vs **0.0001** on non-candidates.
