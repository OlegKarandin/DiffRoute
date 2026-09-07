# Architectural Invariants

Rules that must not break in future phases. Each rule's exact wording is
load-bearing — several were written precisely because a loose paraphrase
caused a bug, so **do not reword them**. Each rule states the measured
evidence behind it inline; the correction numbers it cites (`correction #8`
and so on) are historical labels for those fixes, not links. For
component-level contracts and config keys, see
`docs/architecture/interfaces.md`.

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

## Traffic matrix / constraint

- The traffic matrix is **fixed for a whole run**, built once before the epoch
  loop from `(topology, traffic.seed, traffic.scale, alpha)`. `train.py`
  previously called `generate_demands(..., seed=epoch)` and redrew 100 demands
  every epoch — "all demands feasible" cannot be stated against a set replaced
  each epoch, and a per-demand dual is meaningless without demand identity
  persisting across epochs.
- **`Demand.id` indexes the dual vector.** `build_traffic_matrix` emits
  contiguous ids `0..N-1` and `preflight_filter` **renumbers** what it keeps. A
  gap left by an exclusion would attach every later dual to the wrong demand.
- **No committed matrix file.** The matrix regenerates deterministically and
  `tests/test_traffic.py` pins a checksum of it. The checksum is pinned on the
  *pre-preflight* matrix, because the preflight needs a QoT model and pinning
  its output would pin a checkpoint.
- `alpha` is derived from a named `traffic.scenario` (`stress` → 0.0,
  `realistic` → 1.0), never set directly, so the two settings stay reportable
  named things rather than free-floating numbers.
- The preflight routes **shortest-by-km**, not by `EdgeWeightNet` — the
  learned router changes during training, which would make the matrix depend on
  whichever checkpoint happened to build it.
- The preflight is a **necessary, not sufficient** screen. A surviving demand
  may still be unreachable under learned routing; those surface as duals pinned
  at `dual_max` in the end-of-run report, and that report is the intended
  diagnostic — not a silent oscillation.
- **The feasibility term carries a margin**: the bar is `thr + delta`, not
  `thr`. `delta` has one job now that the augmented penalty is the only mode
  `compute_loss` implements (the one-sided hinge this replaced, and its
  `constraint.penalty`/`constraint.dual_lr` switches, were removed 2026-09):
  absorbing QoT surrogate error (0.5 dB ~ 2.6
  sigma on the model's 0.1909 dB val RMSE), a physical-uncertainty allowance
  no penalty shape replaces. It stacks with the penalty's own active band,
  `lambda_d/rho` dB wide and reaching inside the feasible region, which is
  what pushes for headroom now instead of the margin doing that job alone.
  Do not "simplify" the margin away.
- **The dual update takes the SIGNED `constraint_g` (`bar_d - gsnr_d`), at
  `eta=rho`** — `update_duals(duals, metrics["constraint_g"], eta=rho, ...)`.
  Feeding it the clipped `metrics["shortfalls"]` instead would turn the
  augmented update back into a one-sided ratchet and destroy its fixed point:
  a slack demand's dual has to be able to DECREASE (`g_d < 0`), which the
  clipped vector cannot express.
- **The update stays pure PI: a D term was measured and removed.** The
  penalty supplies P, this ascent supplies I, and a `kd * (g - g_prev)` term
  is the only thing that could respond to `dg/dt`. It was implemented as
  `constraint.dual_kd`, measured, and deleted 2026-09-06; do not re-add it.
  **A POSITIVE `kd` amplifies the very oscillation it targets, provably**:
  at period-2, `g_t = -g_{t-1}`,
  so `(g_t - g_{t-1}) = 2*g_t` and the effective gain at that frequency is
  `rho + 2*kd` — at `kd = rho` that is 3x the integral gain (measured: over-buy
  4 -> 18, tail device sd 4.2 -> 16.5). Derivative action cannot damp an
  oscillation sitting at the Nyquist limit of a once-per-epoch sampler. The
  low-pass correction is `kd = -rho/2` (i.e. `lambda += (rho/2)*(g + g_prev)`,
  a 2-point moving average with a zero at that frequency). Measured on all
  three seeds, it is **outcome-neutral — which is a stronger result than
  inert**. Over-buy 4 -> 4 (seed 42), 5 -> 4 (seed 7), 15 -> 15 (seed 13),
  with `dead_frac` and tail device sd unchanged to a decimal; yet every run is
  a materially DIFFERENT trajectory, first diverging at epoch 10-13 with mean
  |device difference| of 10.5-19.2 over the run. Three divergent paths, three
  pinned outcomes: the selected checkpoint's quality is invariant to
  dual-side perturbation of this size, because it is set by the PRIMAL
  geometry (the score population and its saturation), not by dual dynamics.
  Do not read an endpoint tie as an inactive term — and do not expect a
  dual-side knob to move the product.
  `constraint_g_prev` is REQUIRED when `kd` is nonzero rather than defaulting
  to `constraint_g` — that default would silently zero the term and report
  nothing. Epoch 1 has no previous sample, so `train.py` passes that epoch's
  own `g` and the term is exactly 0, not a spurious `kd*g` kick.
- **`feasibility_loss` (`sum relu(g)`) is a logged diagnostic only, not what
  the augmented penalty acts on.** It can read exactly 0 while the augmented
  force (`weighted_feasibility`) is still nonzero — the penalty's active band
  reaches `lambda_d/rho` dB inside the feasible region, past where the
  one-sided `relu` already reads zero. `num_infeasible`/`num_violated` are
  measured against `feasibility_loss`'s bare/margined thresholds so they stay
  comparable to every pre-augmented-penalty number in the investigation
  record; they are not a measure of what's driving the gradient.
- **Duals are not autograd parameters.** They are Lagrange multipliers updated
  by an explicit clamped ascent rule after the primal step, and they are on no
  optimizer.
- **The deployed allocation is `AllocationHead`'s hard rollout** —
  `pipeline.forward(..., hard_alloc=True)`, i.e. `a_k = 1` iff `score_k > 0`,
  run under `torch.no_grad()` (`diffopt/train.py`'s `hard_rollout()`). This
  replaces the old `RegenPlacement.hard_placement_mask()` design entirely —
  there is no per-node gate any more (`sigmoid`/`hard_concrete`), no
  `placement.gate`/`gate_dropout_p`/`hard_concrete.*` config, and no
  `selection.hard_eval` key. Every cross-epoch or cross-arm comparison reads
  this hard rollout, never the soft training-time relaxation: under hard
  decisions the carry `c` is the EXACT chunk noise, so the rollout is
  self-consistent physics, while the soft pass's carry is a partition-weighted
  expectation and disagrees with it by construction (spec 2.5).
- **Checkpoint selection is lexicographic on the DEPLOYED (hard) rollout**:
  fewest `hard_num_violated`, then fewest `hard_num_devices`, then the
  greatest `hard_worst_margin_db` (`diffopt/train.py`'s `selection_key`).
  Selecting on metrics from the training forward pass is wrong for the same
  reason as under the old gated design: at epoch 0 the head is closed
  (`sigmoid(-3) ~ 0.047` deterministically), so the soft pass's relaxed
  metrics are systematically more optimistic than anything deployable.
  The third slot is `worst_margin_db` rather than total loss because total
  loss is dominated by the `tau` anneal and the duals are deliberately
  non-stationary.
- `num_violated` counts against `thr + delta`; `num_infeasible` counts against
  the bare `thr`. Both are logged: the second is what keeps new runs comparable
  to every pre-change number in the investigation record.

## Regenerator allocation (`AllocationHead`, `diffopt/placement/allocation.py`)

- **Amortized, not per-demand.** The head scores route-local FEATURES, never
  a demand id, so `--holdout` stays meaningful and a later stage can score
  backup routes for demands it never trained on.
- **Autoregressive.** A cut's value depends on noise accumulated since the
  PREVIOUS cut (the carry `c`, reset to 0 on a cut), which is what breaks the
  symmetry between adjacent boundary candidates.
- **Closed at init.** The final MLP layer is zero-weight, bias `-3.0`
  (default `AllocationHead(init_bias=-3.0)`), so `a = sigmoid(-3) ~ 0.047`
  deterministically regardless of features — this is what lets `lambda_dev`
  be live from epoch 0 with no warm-up schedule (a warm-up saturates the head
  at ~20-30x the optimum with no gradient left to escape). Do not restore a
  warm-up.

  **The zero width is also a REGULARIZER, and that is the reason to keep it.**
  At the zero-weight init the score population has exactly zero width, so
  the plant `#{score > 0}` has unbounded slope in the population mean and a
  single optimizer step sweeps every boundary across the STE threshold —
  a period-2 few<->all cycle (0 <-> ~1,470 devices against an oracle of ~22)
  lasting 26-42 epochs on seeds 42/7/13. A spread init (random final-layer
  weights, `init_bias` deepened to keep every score negative) removes that
  cycle and **loses anyway**. Measured 2026-09-06, 7 runs of 300 epochs,
  `constrained_stress`; the `placement.alloc_init_weight_std` knob it was
  measured through was **deleted afterwards** under the item #8 rule, so this
  table is the record:

  | init score range | over-buy at selection | epochs >500 devices | `alloc_dead_frac` @150 |
  |---:|---:|---:|---:|
  | 0.0 (shipped) | **4** | 5 | 0.000 |
  | 1.5 | 12 | 1 | 0.140 |
  | 4.9 | 16 | 3 | 0.828 |

  Both columns are monotone in the width and they point opposite ways. Zero
  width starts every boundary at the sigmoid's point of maximum slope, so the
  whole population stays in the live region; any width parks part of it in the
  tail where `lambda_dev * sigmoid'(s/tau)/tau` underflows, and that
  compounds — wider range means more frozen boundaries, so the live ones
  absorb all the gradient and travel further (final score range 50.4 against
  the shipped 17.7). **The 26-42 epoch transient is the price of keeping
  every boundary alive, and it is worth paying.** Do not re-invent the spread
  init without first re-deriving the convergence measurement that retired
  it. If it is ever re-run, calibrate per seed: the readout draw's scale is
  seed-specific (score range at `std = 1.5` was 4.91 on seeds 42/7 and 15.70
  on seed 13), so a fixed `(std, bias)` pair does not give different seeds the
  same dose. **Nor is the transient reachable by slowing the head's final
  bias** — that was tried too, down to freezing it outright, and the transient
  does not move at all (Finding 12). The score population's translation is
  carried by the WEIGHTS, not by the bias: at the zero-weight init one step
  moves the score by `|dL/ds| * ||h||^2` through `w` against `|dL/ds| * 1`
  through `b`. Nothing in the head's parameterization isolates common mode. (A `greedy_residual` carve-out used to move this closed point to
  the oracle's own greedy cut rule at init; removed 2026-09 — it hard-coded
  the `|S|=1` optimal rule into the score, so a result
  obtained with it on would be "the relaxation recovering the optimum because
  we told it the answer." Recoverable from git history if Stage IV
  restoration needs it back.)
- **The carry enters the head's features as a detached observation, never as
  a differentiable function of the head's own earlier decisions.** Undetached,
  `d(a_{k+1})/d(a_k)` runs through `-10*log10(c)` and is both huge and
  sign-indefinite right after a cut — measured on one real checkpoint,
  `d(device_count)/d(bias)` swung -3317..+1163 across a 0.002-wide window in
  that one parameter (i.e. `lambda_dev` could reward MORE devices), where the
  detached value is a steady +89..+92. Forward values are unaffected;
  `c = c * (1 - a_physics)` stays live physics regardless.
- **`route_context=False`/`lookahead=False` mask the MLP's input, they never
  shrink it.** `ALLOC_FEATURE_DIM` stays 8 either way so arm-sweep checkpoints
  stay structurally comparable across the flag.
- **The waste surcharge (arm 4) detaches only its `relu(feature4)`
  coefficient, never the priced allocation multiplying it.** An undetached
  `relu(feature4)` would carry gradient into `n_next` (and therefore routing)
  even though the carry itself is already detached — rewarding routing onto
  noisier next segments to shrink the surcharge. `a_priced` legitimately
  carries gradient into `n_next`/routing via its own dependence on `score`,
  so only the coefficient is wrapped in `.detach()`.
- **`device_count` (what `lambda_dev` prices) is `total_device_cost`:
  `sum_n sum_d a[d,n]`, a per-DEVICE count.** `site_view` (`max_d a[d,n]`) is
  diagnostic only — never priced, never in the selection key, never in the
  loss. Pricing sites instead of devices was the exact bug this design
  replaces: 40 demands regenerating at one node need 40 devices, not 1.

## QoT model

- Input is per-span features for a single **transparent segment** only. No cross-segment state.
- `padding_mask` convention: `True` = real span, `False` = padding. This is the **opposite** of PyTorch `TransformerEncoder`'s `src_key_padding_mask` (the bridge inverts internally).
- `max_spans = 60` is a hard architecture parameter; changing it requires regenerating all datasets and retraining.
- No regenerator-related inputs in the model. The QoT model is deliberately unaware of placement.
- **One QoT surrogate is reused across topologies.** `checkpoints/best_qot.pt`
  (trained on `ind_132` via `base.yaml`) is also the `qot_checkpoint` for
  `constrained_realistic.yaml`, `small_test.yaml`, `small_test_ind132.yaml`,
  and `small_test_jp70.yaml` — safe because `SpanAttentionQoT` consumes
  per-span physical features (length, fiber type, noise figure, span count,
  loading), not topology identity.
  The limit: cross-topology GSNR accuracy is unmeasured. Of those four, only
  `small_test.yaml` (`topology: configs/topology/german_17.json`) and
  `small_test_jp70.yaml` (`topology: configs/topology/jp_70.json`) are
  actually cross-topology reuse; `constrained_realistic.yaml` and
  `small_test_ind132.yaml` both carry `topology: configs/topology/ind_132.json`
  — the SAME topology the checkpoint was trained on, just a different
  traffic scenario, so those two are not a cross-topology test at all. The
  two that genuinely are cross-topology are pipeline-mechanics smoke tests,
  not accuracy benchmarks.

## GNPy bridge

- Owned by `diffopt/qot/optical_bridge.py` (the old `diffopt/qot/gnpy_bridge.py`, which silently fell back to an analytical GN-model approximation on any GNPy exception, is deleted). **No fallback**: every `compute_qot` call either returns a real GNPy-derived GSNR or raises. There is no bare `except Exception` anywhere in this module — the AST test `tests/test_optical_bridge.py` enforces this structurally, not just by convention.
- `direction=Direction.FORWARD` always, and `center_freq_hz=CUT_FREQ_HZ` (193.5 THz, slot 21 on the 191.4 THz / 100 GHz / 48-slot grid) is mandatory on every `compute_qot` call — omitting it silently selects the wrong probe channel (measured ~0.05 dB error in one case; keep this warning).
- `power_dbm` is always literal `None` on every `Channel`: every ROADM re-equalizes to a fixed target output power, so per-channel launch power is physically inert. Do not reintroduce a launch-power knob (`launch_power_dbm` is gone from the configs for this reason).
- `n_channels` is still sampled once per transparent segment (not per link). All spans in a segment share the same channel loading — this behavior is unchanged from the old bridge.
- `build_loading`'s channel-selection policy expands outward from the CUT slot (deterministic, not `FillPolicy.FULL`) — see `diffopt/qot/optical_bridge.py`'s own docstring for the exact policy.
- GSNR is invariant across all 11 modulation formats in `configs/modulation_formats.yaml` (they share 87.5 GBaud / 0.15 roll-off) — this is why the dataset generator and the ground-truth test both pick a single arbitrary `mode_id` rather than tracking bitrate.

## Upstream dependency (`multilayer-optical-network`)

- Pinned via git dependency in `pyproject.toml` at tag `v0.1.2`
  (`https://github.com/OlegKarandin/multilayer-optical-network.git`). The
  bump from `v0.1.1` was required by `diffopt/traffic.py`: `v0.1.2` added
  `generate_demands`' `aggregate` and `undirected` parameters and made node
  derivation IP-layer-optional (falling back to OMS endpoints), without
  which a bare `OpticalNetworkModel` subclass like `Topology` cannot be used
  as a traffic source. `diffopt.topology.Topology` subclasses its
  `OpticalNetworkModel` directly, the same extension pattern the MCP server
  uses for its own `NetworkModel(OpticalNetworkModel)` IP-layer subclass.
- This project depends on specific upstream behavior: the ground-truth 18.85 dB GSNR test in `tests/test_optical_bridge.py`, and the exact `populate_optical`/`split_link_into_spans` algorithms that `diffopt/topology.py` and `diffopt/topology_builder.py` re-export. Bumping the pin requires re-verifying the ground-truth test and the traps above still hold, not just updating a version number.
- **Pin a tag, never a branch commit.** This dependency was previously `multilayer-optical-mcp @ 2b64361`, an exact commit on `master`. Upstream later split the simulator out into this package and rebased `master`, orphaning `2b64361` — `git fetch` of that SHA now returns `upload-pack: not our ref`, so the old pin could not be installed by anyone. Tags survive a rebase; branch commits do not.
- Migrated from `multilayer-optical-mcp` after that split (imports renamed `multilayer_optical_mcp` → `multilayer_optical_network` across 7 files). Verified as a no-op: relative to the orphaned `2b64361`, `optical_topology_import.py` differs only by a path comment and import ordering (algorithms byte-identical), `modes.py` only adds a `default_modes()` helper, and `gnpy_adapter/adapter.py` only drops a dead `DEFAULT_EQPT`/`DEFAULT_TOPO` fallback inside the `topo_path is not None or eqpt_path is not None` branch — which `optical_bridge.py` never enters, since it passes neither and so always takes `build_gnpy_network(model)`. Full suite before and after: 113 passed / 2 failed (the same pre-existing `.dat` failures).
- Real GNPy (`gnpy==2.14.0`) is a transitive dependency, not declared directly in `pyproject.toml` — nothing in this repo imports `gnpy` directly anymore.
- This project used to compute GSNR via a broken GNPy wrapper (`gnpy_bridge.py`) that silently fell back to an analytical approximation for its entire history before this dependency was adopted — every label in every dataset generated before the migration came from that fallback, and real GNPy never once executed. See git history for that era; those commits describe what was actually believed true at the time and are left as-is.

## Data pipeline

- Parquet schema is fixed: columns `span_features_0` … `span_features_{max_spans*5-1}`, `n_spans`, `gsnr_db`. Changing `max_spans` invalidates existing parquet files.
- Feature ordering within each span is fixed: `[span_length_km, fiber_type_idx, amp_nf_db, channel_loading_fraction, accum_dist_km]`. Index 0–4 within each span block.

## Physics layer

- **The physics layer is entirely static.** Learning touches only *which*
  edges are traversed and *where* cuts happen — never what a given segment
  costs. A transparent segment's QoT input is a pure function of its
  **ordered** edge-id tuple: `span_feature_rows` reads only frozen `Edge`
  fields plus the fixed config constant `channel_loading_fraction`, and
  `qot_model` is frozen at construction. That is what makes
  `DiffONetPipeline`'s segment-GSNR memo exact rather than approximate, and
  it is exactly what a future contributor would break by making
  `channel_loading_fraction` demand-dependent. If that ever becomes
  necessary, the memo key must grow to include it, or the memo must go.
  The key is an ordered tuple and never a frozenset: `accum_dist_km`
  differs between `(a, b)` and `(b, a)`.
- **The QoT surrogate must be deterministic and mode-invariant, not merely
  non-differentiable, for the segment-GSNR memo to be exact.**
  `requires_grad_(False)` only guarantees the model won't be updated by
  gradients — it says nothing about whether two calls with the same input
  return the same value. `dropout=0.0` in `SpanAttentionQoT`'s
  `TransformerEncoderLayer` (and the absence of any other stochastic or
  train/eval-mode-dependent layer, e.g. no BatchNorm) is currently
  load-bearing for `DiffONetPipeline._segment_gsnr_cache`'s correctness. Any
  future change introducing stochastic or train/eval-mode-dependent
  behavior into the QoT model must either extend the memo key to capture
  that state or disable the memo (`cache_segment_gsnr=False`).

## Segment combiner

- `SegmentCombiner` accumulates noise internally in **float64** and casts back to float32 on return. Do not move this to float32 — overflow on long noisy paths is real.
- GSNR inputs are clamped to `[-5, 35]` dB before conversion to linear noise. This is intentional.
- **The fold is an exact, polynomial-time dynamic program, not enumeration and not a soft approximation.** Instead of asking "what is the average worst chunk?" — which forces looking at every one of the `2^(N-1)` hard partitions at once — it asks "what is the probability that every realized chunk stays under a bar `tau`?", which is checkable left to right: a chunk can never span a cut, so past a cut the path forgets its history. Vectorizing that question over all `N(N+1)/2` possible chunk-sum thresholds and reassembling `E[max]` from the resulting step function (`F(tau_r) - F(tau_{r-1})` per threshold) gives the exact expectation in polynomial time. `F(tau)` is built purely from sums and products of the per-boundary probabilities `p_k` — it never puts a probability inside an exponential — so the shared-temperature dominance bug two paragraphs down is **structurally impossible** under this fold, not merely tested against. "Regen helps" is consequently **provable, not temperature-conditional**: cutting a boundary splits one chunk into two no-larger pieces, so `E[max]` cannot rise, for any `p`. See `diffopt/qot/segment_combiner.py`'s module docstring (`_expected_max_chunk_noise`) for the DP's exact recurrence.
- **`SegmentCombiner` is stateless and parameter-free — it takes no annealed knob at all.** The module-level `soft_max` helper still exists (kept for `tests/test_segment_combiner.py` and `scripts/diagnose_fold_error.py`, which still need it to reconstruct/replay the approximation the exact fold replaced), but the production fold does not call it, and there is no `temperature` anywhere in `SegmentCombiner.forward`'s signature. **`soft_max` is still scale-normalised** — it divides by `max(a,b).detach()` before the log-sum-exp and multiplies back after. Do not "simplify" this back to the plain `t * logsumexp([a/t, b/t])` form: the plain form's approximation error is `t*ln2`, **absolute** in linear-noise units and does not shrink with its operands; real per-segment noise is ~0.0025, so it swamped the signal and inverted the "regen helps" invariant (see Phase 1c correction #8). This is a live property of the helper itself, in case it is ever reused elsewhere — not a statement about the current fold, which does not use it.
- Any test of the "regen helps" invariant must include a case at the **production** noise scale (two segments at ~26 dB → noise ~0.0025), not only the 5–15 dB fixtures. `tests/test_segment_combiner.py::test_gradient_wrt_regen_prob_at_production_noise_scale` and `::test_regen_never_increases_noise_across_noise_scales` exist for exactly this and are what caught correction #8.
- **The fold is an exact probability-weighted expectation over every hard partition of the path into chunks, not a single running accumulator.** A regenerator rebuilds the signal, so end-to-end effective noise is `-10*log10(max over chunk noises)` — physically, the path only fails at its worst chunk. **Still banned**: the single-accumulator form (`accumulated = (1-p)*(accumulated+next) + p*soft_max(accumulated, next)`, carried across the whole path). At `p=1` it takes a max but then keeps *adding* the next segment on top of that max, so a downstream chunk's noise piles onto a value that was supposed to have replaced its upstream chunk. It under-reported end-to-end GSNR by up to **3.5 dB** on real multi-segment-chunk paths and gave a redundant regenerator a fake **+1.233 dB** marginal value — the root cause of the regenerator over-provisioning in correction #11. That reasoning doesn't change just because the exact formula shipped in a different shape since.
  **Also still banned, and a real regression risk because it looks superficially similar and a future implementer could plausibly reach for it again**: weighting each *chunk* by its probability but blending that weight *inside a single shared-temperature exponential* as the chunk's noise (a log-sum-exp over chunks, rather than a linear sum over partitions). At production temperature (0.01) chunk-noise differences would scale as `O(1/t)` while log-probability differences stay `O(1)`, so probability gets swamped and the fold silently collapses toward the no-regen value regardless of `p` — this was tried, measured (attenuated the placement gradient ~33x, made it identical for every boundary regardless of actual benefit), and rejected; see correction #11. **The currently shipped code does not use `soft_max` for the fold at all** — the DP above weights each hard partition linearly by its true probability, entirely outside any exponential, which is what makes this failure mode structurally impossible rather than merely avoided.
  Note `soft_max(0, C) != C` at loose temperature (`soft_max` is scale-normalised by `max(a,b)`, and `max(0,C)=C` only makes the *normalizer* right — the log-sum-exp term still adds its own `t*ln2`-scale overshoot on top). This is exactly why an earlier two-state design (`M`=worst completed chunk seeded at a `0` sentinel, `C`=current chunk) needed a third `any_regen` scalar just to blend the sentinel out safely, and why that design was still wrong (non-monotonic, up to 2.94 dB off, at fractional `p`) — it blended a fake "no chunk yet" state linearly with a real chunk value and fed that blend into a nonlinear `soft_max`, which does not correctly propagate uncertainty (`soft_max(E[a],E[b]) != E[soft_max(a,b)]`). The shipped DP never uses a zero sentinel at all — every chunk in every partition is a real, complete sum of real segment noises — so this failure mode cannot recur.
  **Cost:** the genuinely-fractional-probability branch is polynomial, `O(N^4)` elementwise work over an `(R, N)` float64 working set (an `O(N)` hybrid path handles the all-hard-vertex case exactly, e.g. thousands of segments all at `p∈{0,1}`). The cap is now on **segments**, not boundaries: `MAX_EXACT_FOLD_SEGMENTS = 128` (`diffopt/qot/segment_combiner.py`) raises `ValueError` above that, rather than hang — real `ind_132` km-shortest paths reach 16-19 segments, so this leaves ~7x headroom. This replaces the earlier exponential fold's `num_boundaries > 20` cap, which had almost no real margin (measured km-shortest paths on `ind_132` reach 18 boundaries) and is resolved, not open — correction #12.
  Enforcing tests: `tests/test_segment_combiner.py` (25 tests as of this change, including 6 added to validate the DP against a shared brute-force oracle in `tests/_fold_reference.py`), among them `test_fractional_p_matches_hand_derived_expectation`, `test_multi_segment_chunks_equal_max_over_chunks`, `test_chunk_completes_before_next_accumulates`, `test_redundant_regenerator_has_zero_marginal_value`, `test_three_segments_two_boundaries` (hard-vertex/multi-boundary coverage), `test_dp_matches_oracle_across_probability_regimes`, `test_gradcheck_fractional_branch_wrt_boundary_probabilities`/`_wrt_segment_gsnr`, `test_long_path_beyond_old_boundary_cap_now_succeeds` (26 segments — beyond the old 20-boundary cap), and `test_exact_ties_need_no_special_casing_against_oracle`.
- **`forward_batched`'s padding is EXACTLY zero, in both slots.** A demand
  shorter than the batch maximum is padded with segment noise exactly 0 and
  boundary probability exactly 0. `p = 0` makes the appended `cut` column
  exactly zero, so no partition cutting at a pad carries weight; adding 0
  does not change a chunk sum, so every threshold a pad contributes
  duplicates one already present (or is 0), and the undeduplicated-threshold
  argument above makes it contribute `F(tau_r) - F(tau_{r-1}) = 0` exactly.
  Padding with a small epsilon "for safety" breaks both halves at once. The
  `MAX_EXACT_FOLD_SEGMENTS` cap applies to the BATCH maximum under this
  entry point, since every demand is padded up to it.
- **`forward_batched` has no `is_hard` fast path and `forward` keeps
  its.** A batch is generally mixed, and the DP is exact at hard vertices
  anyway. The consequence is that all-hard probabilities get the DP's
  limiting gradient rather than `forward`'s exact zero — inert for the two
  callers that pass hard probabilities, both under `torch.no_grad()`. Do not
  "unify" the two entry points: `forward`'s `O(N)` hard branch is what makes
  the 8001-segment float64-precision test tractable.

## Surrogate gradient / routing

- `surrogate.py` backward uses **SPFA** (not Dijkstra) for the perturbed solve. Perturbed weights `c + λ·grad` can be negative; Dijkstra assumes distances only ever decrease and hangs (or returns a wrong answer) on negative weights. SPFA does not "handle" negative weights in general, though — the routing graph is **undirected**, so a single negative edge is already a negative 2-cycle, which has no correct shortest path. `spfa` (`diffopt/routing/shortest_path.py`) detects this via a relaxation-count guard and returns `None` instead of hanging; `diffopt/routing/surrogate.py:75-77` then falls back to `path_star`, making the surrogate gradient **exactly zero** for that demand. SPFA's real benefit is graceful failure instead of a hang, not correct negative-cycle resolution.
- **`_adjacency`'s neighbour ordering is a tie-break, not a detail.**
  `adj[u]` is in ascending edge-id order, interleaving the `u->v` and `v->u`
  directions exactly as the original per-call `for eid in range(E)` rebuild
  produced it. Relaxation uses strict `<`, so a tie between two equal-cost
  predecessors resolves to whichever edge id was visited first. A
  differently-ordered adjacency returns a different — still shortest — path
  and silently moves every downstream number. Pinned by
  `tests/test_shortest_path.py::test_tied_shortest_paths_resolve_to_the_lowest_edge_id`.
  The structure is cached by `edge_index` content; the weights never are.
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

- `tau` and `lambda_` are passed per-call to `pipeline.forward()` — never stored as module attributes (prevents stale annealing state across epochs). `SegmentCombiner` is correspondingly stateless, and takes this further than `tau`/`lambda_` do: it has no annealed parameter at all (not even a per-call one) — the exact fold needs no temperature to keep in sync (Phase 1c correction #7 is why a per-call, never-stored parameter mattered when `SegmentCombiner` still had one; correction #12 removed it).
- Pre-STE: `path_cost_loss = Σ_demands (path_indicator · edge_weights).sum()` was mechanically required — without it, `∂L/∂path_indicator = 0`, the Vlastelica backward returns the same path, and `EdgeWeightNet` receives zero gradient. It was a gradient enabler, not optional regularization. As of the STE correction (correction #6), this is no longer the sole gradient enabler — the ASE-noise proxy independently supplies `∂L/∂path_indicator`, which is why `lambda_cost` was demoted to `0.01`. **Superseded by correction #9:** this quantity is now `path_noise_cost = Σ(path_indicator · edge_ase_noise)`, not `edge_weights` — using `edge_weights` here is the collapse bug correction #9 fixes.
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
