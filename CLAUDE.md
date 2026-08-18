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

## Phase 1a milestone

Re-baselined post-GNPy-migration: the original 0.5 dB target was measured against analytically-generated labels, not real GNPy labels, so it no longer means what it used to.

**Re-measured after switching the production topology from `german_17` to `ind_132`** (132 nodes, 168 edges — see "Upstream dependency"/topology sections; source: https://github.com/OlegKarandin/jocn24-multi-fiber.git, Karandin et al., J. Opt. Commun. Netw. 16, H18-H26, 2024). `german_17`'s tiny segment space (~65,904 possible feature vectors) meant a 50k/10k sample draw duplicated ~77% of rows between train and val, so its RMSE number (0.0428 dB, previously recorded here) was flattering but not a trustworthy generalization estimate. `ind_132` measures out to a ~63x larger pool (~4.1M), and only **7.0%** of val rows have any train duplicate at all (703/10,000; unique val vectors seen in train: 6.7%, 665/9,892) — this number is measured almost entirely against segments the model hasn't seen before.

Re-baselined target is **0.25 dB** (matches `configs/experiment/base.yaml`'s `val_rmse_target`, set with ~30% headroom above the measured value below — single run, not yet replicated). Measured against the production `base.yaml` config (50k train / 10k val, real GNPy labels, 100 epochs, `ind_132` topology): **0.1909 dB** best val RMSE (epoch 99, fully converged — epoch 100 was 0.1909 dB too, `train_mse` tracked `val_rmse` closely the entire run with no overfitting signal). Against the dataset's actual GSNR range (~6.5–20.0 dB), that's ~1.4% relative error. Clears the re-baselined 0.25 dB target → ready for Phase 1b (routing + placement integration).

## Phase 1b milestone

At the time of that milestone: all 7 segment combiner tests and all 5 surrogate gradient tests pass (36/36 total) → ready for Phase 1c (end-to-end pipeline assembly).

## Phase 1c milestone

At the time of that milestone: 6/6 `test_pipeline.py` tests pass, 40/42 full suite (2 pre-existing topology failures require `.dat` source files not in the repo). Smoke training run confirms regen count decreasing and loss improving → ready for Phase 1d (evaluation, baselines, visualization).

## Architectural constraints — must not break in future phases

**Topology:**
- Nodes are integer IDs only. No names, no coordinates, no `x`/`y` anywhere.
- Edges are undirected and stored as `src < dst`. The `.dat` files have bidirectional entries; deduplication is by `src < dst` at parse time.
- Span splitting is balanced (no short remainder span). Algorithm: find `n` in `[ceil(L/100), ceil(L/40)]` minimising `|L/n - 80|`, subject to every candidate span being ≥ 20 km. If no `n` in that range keeps every span ≥ 20 km, `split_link_into_spans` falls back to `n=1` and the link stays whole. **A span may be shorter than 20 km only if it is the link's sole span and equals the link length exactly** — splitting itself never leaves a short remainder. This is a real, physical case, not a defect: `ind_132` has one such edge (19.0 km) and `jp_70` has seven (8.0–19.0 km). Enforced for every committed topology by `tests/test_topology.py::test_all_committed_spans_ge_20km`.
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

**Upstream dependency (`multilayer-optical-network`):**
- Pinned via git dependency in `pyproject.toml` at tag `v0.1.1` (`https://github.com/OlegKarandin/multilayer-optical-network.git`). `diffopt.topology.Topology` subclasses its `OpticalNetworkModel` directly, the same extension pattern the MCP server uses for its own `NetworkModel(OpticalNetworkModel)` IP-layer subclass.
- This project depends on specific upstream behavior: the ground-truth 18.85 dB GSNR test in `tests/test_optical_bridge.py`, and the exact `populate_optical`/`split_link_into_spans` algorithms that `diffopt/topology.py` and `diffopt/topology_builder.py` re-export. Bumping the pin requires re-verifying the ground-truth test and the traps above still hold, not just updating a version number.
- **Pin a tag, never a branch commit.** This dependency was previously `multilayer-optical-mcp @ 2b64361`, an exact commit on `master`. Upstream later split the simulator out into this package and rebased `master`, orphaning `2b64361` — `git fetch` of that SHA now returns `upload-pack: not our ref`, so the old pin could not be installed by anyone. Tags survive a rebase; branch commits do not.
- Migrated from `multilayer-optical-mcp` after that split (imports renamed `multilayer_optical_mcp` → `multilayer_optical_network` across 7 files). Verified as a no-op: relative to the orphaned `2b64361`, `optical_topology_import.py` differs only by a path comment and import ordering (algorithms byte-identical), `modes.py` only adds a `default_modes()` helper, and `gnpy_adapter/adapter.py` only drops a dead `DEFAULT_EQPT`/`DEFAULT_TOPO` fallback inside the `topo_path is not None or eqpt_path is not None` branch — which `optical_bridge.py` never enters, since it passes neither and so always takes `build_gnpy_network(model)`. Full suite before and after: 113 passed / 2 failed (the same pre-existing `.dat` failures).
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
- `surrogate.py` backward uses **SPFA** (not Dijkstra) for the perturbed solve. Perturbed weights `c + λ·grad` can be negative; Dijkstra assumes distances only ever decrease and hangs (or returns a wrong answer) on negative weights. SPFA does not "handle" negative weights in general, though — the routing graph is **undirected**, so a single negative edge is already a negative 2-cycle, which has no correct shortest path. `spfa` (`diffopt/routing/shortest_path.py`) detects this via a relaxation-count guard and returns `None` instead of hanging; `diffopt/routing/surrogate.py:75-77` then falls back to `path_star`, making the surrogate gradient **exactly zero** for that demand. SPFA's real benefit is graceful failure instead of a hang, not correct negative-cycle resolution.
- The Vlastelica perturbation is `c_target = w + λ·grad_output` (positive sign). The formula in the paper uses ŷ = −∂L/∂z; since PyTorch's `grad_output = ∂L/∂z`, the sign flips: `c − λŷ = c + λ·grad_output`. Using the wrong sign (minus) gives zero gradient everywhere.
- Gradient on an edge from `DijkstraSurrogate` is **negative** for active-path edges when the loss penalises their use. Gradient descent (`w -= lr·grad`) then increases those edge costs, routing away from them. This is counterintuitive but correct — do not flip the sign in tests.
- **`edge_weights` are renormalised to unit mean** in `pipeline.forward`, and
  the divisor is **not** detached. `w = u/mean(u)` makes the loss homogeneous
  of degree 0 in `EdgeWeightNet`'s raw output, so by Euler's theorem the
  scale-direction gradient `Σ uᵢ ∂L/∂uᵢ` is exactly 0. Detaching the divisor
  restores the collapse degeneracy. This is the OPPOSITE of `soft_max`'s
  deliberate `.detach()` in `segment_combiner.py` — the two lines look alike
  and mean opposite things.
- **The path-cost loss term is denominated in `edge_ase_noise`, not
  `edge_weights`.** Fixed physical coefficients make it degree-0 in the
  weights by construction. Do not "simplify" it back to
  `Σ(path_indicator · edge_weights)`; that is degree-1 and is the collapse
  bug. ASE-only and not ASE+NLI is deliberate: NLI depends on channel
  spectrum position, so it is not determined by the route.
- `EdgeWeightNet`'s static topology features are standardised in
  `DiffONetPipeline.__init__` (not inside the net, which stays
  topology-agnostic). Constant columns map to 0 via a clamped population std
  and are not dropped, so mixed-fiber-type topologies still work.

## Corrections made during Phase 1a

1. **`pyproject.toml` build backend**: generated as `setuptools.backends.legacy:build` (doesn't exist), corrected to `setuptools.build_meta`.
2. **Channel grid**: an early draft plan specified 96 channels on a 50 GHz ITU grid. The final plan and the actual modulation format data use **48 channels on a 100 GHz grid**. The code uses 48/100 GHz everywhere.
3. **Topology file names**: old plan used `nobel_germany.json` / `european.json`; actual files are `german_17.json` / `eu_19.json` matching the `.dat` source filenames.

## Architectural constraints — Phase 1c additions

**Pipeline:**
- `tau`, `lambda_`, and `soft_max_temperature` are passed per-call to `pipeline.forward()` — never stored as module attributes (prevents stale annealing state across epochs). `SegmentCombiner` is correspondingly stateless: `temperature` is a required `forward()` argument, not a constructor parameter (see the fixed-bug note in "Corrections made during Phase 1c" for why this matters — an earlier version took it at construction and it was never annealed).
- Pre-STE: `path_cost_loss = Σ_demands (path_indicator · edge_weights).sum()` was mechanically required — without it, `∂L/∂path_indicator = 0`, the Vlastelica backward returns the same path, and `EdgeWeightNet` receives zero gradient. It was a gradient enabler, not optional regularization. As of the STE correction (see correction #6), this is no longer the sole gradient enabler — the ASE-noise proxy independently supplies `∂L/∂path_indicator`, which is why `lambda_cost` was demoted to `0.01`. **Superseded by correction #9** (see below): this quantity is now `path_noise_cost = Σ(path_indicator · edge_ase_noise)`, not `edge_weights` — using `edge_weights` here is the collapse bug this file's correction #9 fixes.
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

1. **Vlastelica perturbation sign**: the original project spec (§4) wrote `c_target = w - λ * (∂L/∂p*)`. This is wrong. The correct formula is `c_target = w + λ * grad_output` because the paper's ŷ is the negative gradient. Using minus gives zero surrogate gradient on every call.
2. **Dijkstra in backward pass**: The spec's `shortest_path.py` description did not mention negative weights. The Vlastelica backward produces perturbed weights that can be negative; Dijkstra hangs on these. SPFA was added to handle this. (Precisely: the routing graph is undirected, so a single negative edge is already a negative 2-cycle with no correct shortest path — SPFA's actual contribution is detecting that via a relaxation-count guard and returning `None` gracefully, which `surrogate.py` then treats as a zero gradient for that demand, rather than hanging like Dijkstra would. See the "Surrogate gradient / routing" architectural constraints above for the precise mechanism.)
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
9. **`EdgeWeightNet`'s learned edge weights collapsed to the `Softplus` floor — a scale degeneracy in the loss, not the self-reinforcing routing dynamic first suspected (found while investigating follow-up #2's ~2x route-length detour; full write-up in `docs/investigations/edge_weight_scale_collapse.md`)**: at the retrained `ind_132` checkpoint, **86%** of edge weights were below 1e-6 (median 5.7e-11, max 6.1e-3, a 6-order-of-magnitude spread), `corr(weight, length_km) = -0.17`, and routes ran **2.07x** longer than shortest-by-km. Root cause: `path_cost_loss = Σ(path_indicator · edge_weights)` is homogeneous of **degree 1** in `edge_weights`, while `feasibility_loss` and `regen_loss` reach `edge_weights` only through Dijkstra's argmin, which is **scale-invariant** (verified: under `0.5*w`, 100/100 demands route identically). The scale-direction derivative `dL(c·w)/dc` at `c=1` — equal to `Σ_e g_e·w_e`, and reproducing `path_cost_loss` itself to 7 significant figures as the Euler-theorem check for degree-1 homogeneity — was **positive at both epoch 0 (+0.388) and epoch 60 (+5.3e-6)**: no operating point ever opposed the shrink. Three refutations of the plan's original hypothesis ("an edge that gets used gets cheaper, which makes it more likely to be reused, self-reinforcing"): (a) **it is pure scale collapse, not restructuring** — Spearman rank-corr(init weights, trained weights) = **+0.999**, median magnitude ratio trained/init = **1.22e-08**, i.e. training preserved the initialization's cost *ordering* almost exactly and only annihilated the magnitude; (b) **training produced no routing improvement at all** — a bit-identical reconstruction of the untrained epoch-0 network routed at **1.965x** vs shortest-by-km, the 60-epoch trained one at **2.066x** (worse), with **68/100** demands routed identically by both; (c) **`lambda_cost` was never the lever** — at epoch 0, `feasibility_loss`'s Vlastelica-surrogate gradient supplied **81%** of the shrink pressure (+0.316 of the total +0.388), not `path_cost_loss` (19%), because the surrogate's `c_target = w + λ·grad_output` uses a fixed additive `λ` that makes even a truly scale-invariant loss term scale-sensitive through the perturbed solve. **Fixed** by three coordinated changes, all under the new "Surrogate gradient / routing" constraints above: `edge_weights` are renormalised to unit mean in `pipeline.forward` with the divisor **not** detached, making the total loss homogeneous of degree 0 in `EdgeWeightNet`'s raw output (Euler's theorem then forces the scale-direction gradient to exactly 0 — the fix is structural, not a rebalanced hyperparameter); the path-cost term is denominated in `edge_ase_noise` (fixed physical coefficients, degree-0 in the weights by construction) rather than `edge_weights`; and `EdgeWeightNet`'s static topology input features are standardised in `DiffONetPipeline.__init__`. Commits: `77caaf4` (unit-mean renormalisation), `7fb3dec` (ASE-denominated path cost), `9393582` (standardised topology inputs), `03671e6` (end-to-end regression test for the collapse), `2246e9f` (gradient diagnostic updated for post-fix expectations). A fix-round correction found during review, not in the original plan: `scripts/diagnose_edge_weight_gradient.py`'s Check F/G originally measured `EdgeWeightNet`'s **raw** output rather than the unit-mean-normalised weight the pipeline actually routes on; corrected in `b7ecffc`, approved by the human partner as going beyond the brief. Verified by retraining `small_test_ind132.yaml` end-to-end for 60 epochs: the primary gate **passes** — `num_infeasible` first reaches 0 at epoch 16 (`num_regen_soft=7` < 18), and the actual saved checkpoint (epoch 40, best-loss) also has `num_infeasible=0`, `num_regen_soft=8` < 18 (caveat: not monotone — it oscillates 0-6 afterward and reads 1 at the final epoch 60). `path_noise_loss` (the renamed, ASE-denominated successor to `path_cost_loss`) plateaus in the **1408-1911** range across training with no collapse toward zero — contrast the old term's pre-fix collapse **7.198 → 0.00056**. Spearman rank-corr(init, trained) fell to **+0.418** (was +0.999), median magnitude ratio rose to **0.913** (was 1.22e-8), and `corr(weight, length_km)` flipped to **+0.982** (was -0.170) — longer edges are now priced higher, the physically sane direction. The Check C direct-gradient component reads **exactly 0** at both the untrained and trained checkpoints (was 14.46 direct vs 2.2e-6 surrogate pre-fix); Check F's perturbation ratio is **0.89** untrained / **0.80** trained, both O(1) (was 7.9e6 at epoch 60 pre-fix); Check G's scale-direction gradient reads **~1e-7** on every term (was +0.388 total at epoch 0 pre-fix). Route length collapsed from 2807.7 km mean / 2.376x mean-of-ratios / 2.066x ratio-of-means (pre-fix) to **1404.0 km / 1.027x / 1.033x** (post-fix) — routes are now close to shortest-path length. Regenerators placed fell from 18 to 8 (load-bearing 14 → 7), and demand 181 — the specific pre-fix-infeasible case this investigation traced to the routing detour — is feasible post-fix. One new open question surfaced by this retrain, not resolved by this fix: on the broader 400-demand held-out set used by `scripts/diagnose_regen_ablation.py`, infeasible demands rose from 1/400 to **9/400** despite far shorter routes and fewer regenerators, suggesting mild under-provisioning now that routes are much shorter (a possible `lambda_regen`/`lambda_cost` interaction) — tracked as a new item in `docs/investigations/open_followups.md`.
