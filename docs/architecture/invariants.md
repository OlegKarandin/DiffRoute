# Architectural Invariants

Rules that must not break in future phases. Each rule's exact wording is
load-bearing — several were written precisely because a loose paraphrase
caused a bug, so **do not reword them**. For the forensic history behind any
rule (what broke, how it was found, how it was verified fixed), see
`docs/investigations/CHANGELOG.md`. For component-level contracts and config
keys, see `docs/architecture/interfaces.md`.

This file was extracted from `CLAUDE.md`'s "Architectural constraints"
sections; nothing below has been reworded in the move.

## Topology

- Nodes are integer IDs only. No names, no coordinates, no `x`/`y` anywhere.
- Edges are undirected and stored as `src < dst`. The `.dat` files have bidirectional entries; deduplication is by `src < dst` at parse time.
- Span splitting is balanced (no short remainder span). Algorithm: find `n` in `[ceil(L/100), ceil(L/40)]` minimising `|L/n - 80|`, subject to every candidate span being ≥ 20 km. If no `n` in that range keeps every span ≥ 20 km, `split_link_into_spans` falls back to `n=1` and the link stays whole. **A span may be shorter than 20 km only if it is the link's sole span and equals the link length exactly** — splitting itself never leaves a short remainder. This is a real, physical case, not a defect: `ind_132` has one such edge (19.0 km) and `jp_70` has seven (8.0–19.0 km). Enforced for every committed topology by `tests/test_topology.py::test_all_committed_spans_ge_20km`.
- All span and amplifier parameters are stored per-span in the topology JSON (not aggregated).

## Modulation / demands

- `Demand` has no `modulation` field. Modulation is never assigned at demand-generation time.
- Bitrate → SNR threshold is a **direct lookup** (exact float key match) in `ModulationConfig`. There is no interpolation.
- Valid bitrates: 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800 Gbps (11 values).
- WDM grid: 48 channels at 100 GHz spacing, C-band centered at 193.5 THz.

## QoT model

- Input is per-span features for a single **transparent segment** only. No cross-segment state.
- `padding_mask` convention: `True` = real span, `False` = padding. This is the **opposite** of PyTorch `TransformerEncoder`'s `src_key_padding_mask` (the bridge inverts internally).
- `max_spans = 60` is a hard architecture parameter; changing it requires regenerating all datasets and retraining.
- No regenerator-related inputs in the model. The QoT model is deliberately unaware of placement.

## GNPy bridge

- Owned by `diffopt/qot/optical_bridge.py` (the old `diffopt/qot/gnpy_bridge.py`, which silently fell back to an analytical GN-model approximation on any GNPy exception, is deleted). **No fallback**: every `compute_qot` call either returns a real GNPy-derived GSNR or raises. There is no bare `except Exception` anywhere in this module — the AST test `tests/test_optical_bridge.py` enforces this structurally, not just by convention.
- `direction=Direction.FORWARD` always, and `center_freq_hz=CUT_FREQ_HZ` (193.5 THz, slot 21 on the 191.4 THz / 100 GHz / 48-slot grid) is mandatory on every `compute_qot` call — omitting it silently selects the wrong probe channel (measured ~0.05 dB error in one case; keep this warning).
- `power_dbm` is always literal `None` on every `Channel`: every ROADM re-equalizes to a fixed target output power, so per-channel launch power is physically inert. Do not reintroduce a launch-power knob (`launch_power_dbm` is gone from the configs for this reason).
- `n_channels` is still sampled once per transparent segment (not per link). All spans in a segment share the same channel loading — this behavior is unchanged from the old bridge.
- `build_loading`'s channel-selection policy expands outward from the CUT slot (deterministic, not `FillPolicy.FULL`) — see `diffopt/qot/optical_bridge.py`'s own docstring for the exact policy.
- GSNR is invariant across all 11 modulation formats in `configs/modulation_formats.yaml` (they share 87.5 GBaud / 0.15 roll-off) — this is why the dataset generator and the ground-truth test both pick a single arbitrary `mode_id` rather than tracking bitrate.

## Upstream dependency (`multilayer-optical-network`)

- Pinned via git dependency in `pyproject.toml` at tag `v0.1.1` (`https://github.com/OlegKarandin/multilayer-optical-network.git`). `diffopt.topology.Topology` subclasses its `OpticalNetworkModel` directly, the same extension pattern the MCP server uses for its own `NetworkModel(OpticalNetworkModel)` IP-layer subclass.
- This project depends on specific upstream behavior: the ground-truth 18.85 dB GSNR test in `tests/test_optical_bridge.py`, and the exact `populate_optical`/`split_link_into_spans` algorithms that `diffopt/topology.py` and `diffopt/topology_builder.py` re-export. Bumping the pin requires re-verifying the ground-truth test and the traps above still hold, not just updating a version number.
- **Pin a tag, never a branch commit.** This dependency was previously `multilayer-optical-mcp @ 2b64361`, an exact commit on `master`. Upstream later split the simulator out into this package and rebased `master`, orphaning `2b64361` — `git fetch` of that SHA now returns `upload-pack: not our ref`, so the old pin could not be installed by anyone. Tags survive a rebase; branch commits do not.
- Migrated from `multilayer-optical-mcp` after that split (imports renamed `multilayer_optical_mcp` → `multilayer_optical_network` across 7 files). Verified as a no-op: relative to the orphaned `2b64361`, `optical_topology_import.py` differs only by a path comment and import ordering (algorithms byte-identical), `modes.py` only adds a `default_modes()` helper, and `gnpy_adapter/adapter.py` only drops a dead `DEFAULT_EQPT`/`DEFAULT_TOPO` fallback inside the `topo_path is not None or eqpt_path is not None` branch — which `optical_bridge.py` never enters, since it passes neither and so always takes `build_gnpy_network(model)`. Full suite before and after: 113 passed / 2 failed (the same pre-existing `.dat` failures).
- Real GNPy (`gnpy==2.14.0`) is a transitive dependency, not declared directly in `pyproject.toml` — nothing in this repo imports `gnpy` directly anymore.
- This project used to compute GSNR via a broken GNPy wrapper (`gnpy_bridge.py`) that silently fell back to an analytical approximation for its entire history before this dependency was adopted — every label in every dataset generated before the migration came from that fallback, and real GNPy never once executed. See git history and `docs/investigations/CHANGELOG.md`'s phase-by-phase record of that era; those entries describe what was actually believed true at the time and are left as-is.

## Data pipeline

- Parquet schema is fixed: columns `span_features_0` … `span_features_{max_spans*5-1}`, `n_spans`, `gsnr_db`. Changing `max_spans` invalidates existing parquet files.
- Feature ordering within each span is fixed: `[span_length_km, fiber_type_idx, amp_nf_db, channel_loading_fraction, accum_dist_km]`. Index 0–4 within each span block.

## Segment combiner

- `SegmentCombiner` accumulates noise internally in **float64** and casts back to float32 on return. Do not move this to float32 — overflow on long noisy paths is real.
- GSNR inputs are clamped to `[-5, 35]` dB before conversion to linear noise. This is intentional.
- **`soft_max` is scale-normalised** — it divides by `max(a,b).detach()` before the log-sum-exp and multiplies back after. Do not "simplify" this back to the plain `t * logsumexp([a/t, b/t])` form. The plain form's approximation error is `t*ln2`, which is **absolute** in linear-noise units and does not shrink with its operands; real per-segment noise is ~0.0025, so it swamped the signal and inverted the "regen helps" invariant (see [Phase 1c correction #8](../investigations/CHANGELOG.md#correction-1c-8)). Normalised, the error is `max(a,b)*t*ln2` — a fixed fraction — and the invariant holds for any `temperature < 1/ln2 ≈ 1.44` at any noise magnitude.
- Because the error is now relative, the temperature no longer encodes a hidden assumption about topology-dependent noise scale. `soft_max_temperature=0.5` annealing down to `0.01` is sign-correct end to end; `temperature=0.01` remains the value to use in unit tests that need the approximation tight (soft_max ≈ hard max).
- Any test of the "regen helps" invariant must include a case at the **production** noise scale (two segments at ~26 dB → noise ~0.0025), not only the 5–15 dB fixtures. `tests/test_segment_combiner.py::test_gradient_wrt_regen_prob_at_production_noise_scale` and `::test_regen_never_increases_noise_across_noise_scales` exist for exactly this and are what caught [correction #8](../investigations/CHANGELOG.md#correction-1c-8).

## Surrogate gradient / routing

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

## Pipeline

- `tau`, `lambda_`, and `soft_max_temperature` are passed per-call to `pipeline.forward()` — never stored as module attributes (prevents stale annealing state across epochs). `SegmentCombiner` is correspondingly stateless: `temperature` is a required `forward()` argument, not a constructor parameter (see [Phase 1c correction #7](../investigations/CHANGELOG.md#correction-1c-7) for why this matters — an earlier version took it at construction and it was never annealed).
- Pre-STE: `path_cost_loss = Σ_demands (path_indicator · edge_weights).sum()` was mechanically required — without it, `∂L/∂path_indicator = 0`, the Vlastelica backward returns the same path, and `EdgeWeightNet` receives zero gradient. It was a gradient enabler, not optional regularization. As of the STE correction (see [correction #6](../investigations/CHANGELOG.md#correction-1c-6)), this is no longer the sole gradient enabler — the ASE-noise proxy independently supplies `∂L/∂path_indicator`, which is why `lambda_cost` was demoted to `0.01`. **Superseded by [correction #9](../investigations/CHANGELOG.md#correction-1c-9):** this quantity is now `path_noise_cost = Σ(path_indicator · edge_ase_noise)`, not `edge_weights` — using `edge_weights` here is the collapse bug correction #9 fixes.
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
