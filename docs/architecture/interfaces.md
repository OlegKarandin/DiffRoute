# Component Interfaces and Data Contracts

## Topology JSON (`configs/topology/*.json`)

```json
{
  "nodes": [{"id": 0}, {"id": 1}, ...],
  "edges": [
    {
      "src": 0, "dst": 1,
      "length_km": 353.0,
      "num_spans": 4,
      "span_lengths_km": [88.25, 88.25, 88.25, 88.25],
      "fiber_type": "SSMF",
      "amplifier_nf_db": [5.5, 5.5, 5.5, 5.5]
    }
  ]
}
```

- `src < dst` always (undirected; bidirectional .dat entries are deduplicated at parse time)
- `num_spans == len(span_lengths_km) == len(amplifier_nf_db)` — these must stay in sync
- `sum(span_lengths_km)` equals `length_km` within 0.01 km (last span adjusted for floating-point)
- No `name`, `x`, `y` fields anywhere
- `fiber_type` currently only `"SSMF"` in generated configs; `FIBER_TYPE_INDEX` in `topology.py` maps it to integer 0

## `Topology` (`diffopt/topology.py`)

`Topology(OpticalNetworkModel)` — subclasses the upstream
`multilayer_optical_mcp.model.optical_network.OpticalNetworkModel` directly; it is not a
standalone dataclass.

| Property / method | Type | Notes |
|---|---|---|
| `num_nodes` | `int` | derived from ROADM ids; raises `ValueError` if node ids are not contiguous `0..N-1` |
| `undirected_edges` | `List[Edge]` | one `Edge` per undirected pair (`src < dst`), NOT `.edges` |
| `edge_index` | `LongTensor (2, E)` | row 0 = src, row 1 = dst; recomputed each call |
| `get_edge_features()` | `FloatTensor (E, 5)` | see feature order below |
| `regen_candidate_nodes` | `List[int]` | nodes with undirected degree ≥ 3 |
| `from_graph_json(topology_path, modulation_formats_path)` | classmethod | builds a `Topology` from a topology JSON + modulation-formats YAML |

Module-level `load_topology(topology_path, modulation_formats_path)` is a thin alias for
`Topology.from_graph_json` — it now takes **two** required arguments, not one.

**Node-id contiguity contract:** node ids must be contiguous `0..N-1`; accessing `num_nodes` on a
topology with gaps raises `ValueError` rather than silently producing a wrong count.

`get_edge_features()` column order: `[mean_span_length_km, fiber_type_idx, mean_amp_nf_db, num_spans, total_length_km]`

## `Demand` namedtuple (`diffopt/demands.py`)

```python
Demand(id: int, src: int, dst: int, bitrate_gbps: float)
```

No `modulation` field. SNR threshold is derived from `bitrate_gbps` at evaluation time via `ModulationConfig`.

## `ModulationConfig` (`diffopt/modulation.py`)

Loaded from `configs/modulation_formats.yaml`. Key contract:

- `required_snr_threshold(bitrate_gbps)` — **exact float key match** against the 11-entry table. Raises `ValueError` for unknown bitrates. Do not pass fractional bitrates not in the table.
- `max_feasible_bitrate(gsnr_db)` — returns `float` or `None`. Used for evaluation, not training.
- Valid bitrates: 300–800 Gbps in 50 Gbps steps.
- SNR thresholds: 4.8–15.1 dB (300–800 Gbps).

## `segment_gsnr_db` (`diffopt/qot/optical_bridge.py`)

Replaces the old (deleted) `diffopt/qot/gnpy_bridge.py::simulate_segment`. This module wraps
`multilayer_optical_mcp.gnpy_adapter.adapter.compute_qot` and runs real GNPy — there is no
analytical fallback and no bare `except Exception` anywhere in the file (enforced by an AST
test in `tests/test_optical_bridge.py`). Every call either returns a real GNPy-derived GSNR or
raises.

```python
segment_gsnr_db(
    topology,                        # diffopt Topology (subclasses OpticalNetworkModel)
    oms_sequence: Tuple[str, ...],    # e.g. from oms_sequence_for_node_path(topology, node_path)
    mode_id: str,                     # upstream modulation-format id
    n_channels: int,                  # active WDM channels, 1–48
    cache: Optional[QoTCache] = None,
) -> float  # GSNR in dB at CUT (193.5 THz)
```

- `oms_sequence_for_node_path(topology, node_path)` converts a diffopt numeric node path into
  the validated OMS id tuple this function expects; each id is checked via `topology.get_oms`,
  which raises `KeyError` for a nonexistent edge rather than deferring the failure into GNPy.
- Internally calls `compute_qot` with `direction=Direction.FORWARD` and
  `center_freq_hz=CUT_FREQ_HZ` (193.5 THz, slot 21 on the 191.4 THz/100 GHz/48-slot grid)
  always set explicitly — omitting `center_freq_hz` silently selects the wrong probe channel
  (measured ~0.05 dB error in one case).
- `build_loading(n_channels, mode_id)` builds a deterministic, CUT-centered `LoadingState`:
  slots expand outward from the CUT slot (not `FillPolicy.FULL`), `power_dbm` is always literal
  `None` on every `Channel` (every ROADM re-equalizes to a fixed target output power, so
  per-channel launch power is physically inert — there is no `launch_power_dbm` knob anymore).
- Raises `RuntimeError` if GNPy returns a non-finite GSNR (physically impossible on a real span).
- GSNR is invariant across all 11 modulation formats in `configs/modulation_formats.yaml` (they
  share 87.5 GBaud / 0.15 roll-off), which is why callers pick a single arbitrary `mode_id`
  rather than tracking bitrate.

WDM grid: fixed C-band grid shared with upstream — anchor 191.4 THz, 100 GHz spacing, 48 slots
(191.4–196.1 THz); CUT is slot 21 (193.5 THz), diffopt's C-band-center convention.

## Parquet dataset schema (`data/datasets/{train,val}.parquet`)

| Column | Type | Notes |
|---|---|---|
| `span_features_0` … `span_features_{max_spans*5-1}` | float32 | flattened; zero-padded beyond `n_spans` |
| `n_spans` | int32 | number of real spans (1 to max_spans) |
| `gsnr_db` | float32 | GSNR at CUT in dB |

Total columns: `max_spans * 5 + 2`. Default `max_spans=60` → 302 columns.

**Per-span feature layout** (5 values per span, row-major in the flat array):

| Index within span | Feature | Units / encoding |
|---|---|---|
| 0 | `span_length_km` | km, float |
| 1 | `fiber_type_idx` | 0=SSMF, 1=LEAF, 2=TWRS |
| 2 | `amp_nf_db` | dB, float |
| 3 | `channel_loading_fraction` | n_channels / 48, range [1/48, 1.0] |
| 4 | `accum_dist_km` | cumulative km of all prior spans (0.0 before the first span; distance to this span's *start*, not its end) |

Padding spans have all five values set to 0.0.

## `SegmentQoTDataset` (`diffopt/qot/dataset.py`)

```python
dataset[i] -> (span_features, padding_mask, gsnr_db)
```

| Tensor | Shape | Dtype | Values |
|---|---|---|---|
| `span_features` | `(max_spans, 5)` | float32 | padded rows are zeros |
| `padding_mask` | `(max_spans,)` | bool | **True = real span**, False = padding |
| `gsnr_db` | scalar | float32 | |

**Critical convention:** `padding_mask=True` means the span is real and should be included. This is the **opposite** of `TransformerEncoder`'s `src_key_padding_mask`, which expects `True` = ignore. The model inverts the mask internally (`src_key_padding_mask = ~padding_mask`). Pass `padding_mask` as returned by the dataset — do not pre-invert.

## `SpanAttentionQoT.forward` (`diffopt/qot/model.py`)

```python
forward(
    span_features: FloatTensor,   # (B, max_spans, 5)
    padding_mask:  BoolTensor,    # (B, max_spans), True=real
) -> FloatTensor                  # (B,) predicted GSNR in dB
```

- `max_spans` must match the value the model was instantiated with (default 60)
- Positional encoding is added by span index (0 to seq_len-1) regardless of actual span lengths; physical position is captured by `accum_dist_km` in the features
- Output is unbounded float; expected range during training ~5–30 dB

## `SegmentCombiner` (`diffopt/qot/segment_combiner.py`)

```python
SegmentCombiner()  # stateless — takes no constructor arguments

forward(
    segment_gsnrs_db:          List[Tensor],  # N scalar float32 tensors (GSNR in dB)
    regen_probs_at_boundaries: List[Tensor],  # N-1 scalar float32 tensors ∈ (0, 1)
) -> Tensor  # scalar float32, end-to-end GSNR in dB
```

- Takes no annealed parameters at all — not `temperature`, not anything else. The fold is exact, so there is no approximation sharpness to keep in sync across calls. An earlier version took `soft_max_temperature` at construction (never annealed anywhere in the codebase), then as a required per-call `forward()` argument (matching `DiffONetPipeline.forward()`'s `tau`/`lambda_` pattern) — that history caused a run of bugs (Phase 1c corrections #7, #8, #11, #12), ending with the parameter's outright removal.
- `len(regen_probs_at_boundaries) == len(segment_gsnrs_db) - 1` is enforced.
- Inputs are clamped to `[-5, 35]` dB internally; caller does not need to clamp.
- Gradient flows through `regen_probs_at_boundaries`; also through `segment_gsnrs_db` if those tensors require grad.
- **Chunk semantics.** A regenerator boundary splits the path into independent chunks — it rebuilds the signal, so noise does not carry across it. `forward`'s result is the **exact probability-weighted expectation over every possible chunking** (every hard partition of the path at the boundaries): `Σ_π P(π) · (chunk-max noise under partition π)`, `π` ranging over all `2^(N-1)` ways the `N-1` boundaries could independently cut — but it is computed by a polynomial-time dynamic program, not by enumerating `π`. The DP asks "what is the probability every realized chunk stays under a bar `tau`?" (checkable left to right, since a chunk can never span a cut), vectorized over all `N(N+1)/2` possible chunk-sum thresholds, then reconstructs `E[max]` from the resulting step function. `F(tau)` is built purely from sums and products of the per-boundary probabilities, never through an exponential, which is also why "regen helps" is provable directly from the DP's structure (cutting a boundary can only split a chunk into two no-larger pieces) rather than depending on a temperature staying small. Exact at `p ∈ {0,1}` for every boundary simultaneously (a separate `O(N)` hybrid path handles this case directly — no enumeration, so this holds even for thousands of segments), and exact at any individual hard vertex (`p_i ∈ {0,1}` for boundary `i` while others stay fractional) via the general DP's own limit — no separate code path is needed for that case. See `diffopt/qot/segment_combiner.py`'s module docstring for the DP's exact recurrence and `docs/architecture/invariants.md`'s "Segment combiner" section for the invariants it guarantees.
- **Cost.** The genuinely-fractional-probability branch is polynomial: `O(N^4)` elementwise work over an `(R, N)` float64 working set, capped at `MAX_EXACT_FOLD_SEGMENTS = 128` **segments** (not boundaries) — `ValueError` above that rather than hang. Real `ind_132` km-shortest paths reach 16-19 segments, so this leaves ~7x headroom (correction #12). This replaces an earlier exponential-fold implementation whose `num_boundaries > 20` cap had almost no real margin against measured real-topology path lengths.

```python
forward_batched(
    segment_gsnrs_db: Tensor,   # (D, J) float32 dB; entries >= num_segments[d] ignored
    boundary_probs:   Tensor,   # (D, J-1) float32; entries >= num_segments[d]-1 forced to 0
    num_segments:     Tensor,   # (D,) long, each in [1, J]
) -> Tensor                     # (D,) float32, end-to-end GSNR in dB
```

- Elementwise equal to calling `forward()` once per demand on that demand's
  real segments. `DiffONetPipeline.forward` uses this; the list-based
  `forward` remains for single-path callers, tests and diagnostics.
- No `is_hard` fast path — see `docs/architecture/invariants.md`, "Segment
  combiner". The `MAX_EXACT_FOLD_SEGMENTS` cap applies to `J`, the batch
  maximum.

Module-level helpers (also importable):

```python
db_to_linear_noise(gsnr_db: Tensor) -> Tensor   # 10^(-gsnr_db/10), preserves dtype
linear_noise_to_db(noise: Tensor) -> Tensor      # -10*log10(noise), preserves dtype
soft_max(a, b, temperature=0.5) -> Tensor
    # m*t*logsumexp([a/m/t, b/m/t]) where m = max(a,b).detach()
    # Scale-normalised deliberately: the plain form's t*ln2 error is ABSOLUTE
    # in linear-noise units and inverts the "regen helps" invariant at the
    # real per-segment noise scale (~0.0025).
```

## `dijkstra` / `spfa` (`diffopt/routing/shortest_path.py`)

```python
dijkstra(
    edge_weights: np.ndarray,   # (E,) float64, must be non-negative
    edge_index:   np.ndarray,   # (2, E) int; src < dst (undirected)
    src: int, dst: int, num_nodes: int,
) -> Optional[np.ndarray]       # (E,) float32 binary indicator, or None if no path

spfa(
    edge_weights: np.ndarray,   # (E,) float64, may be negative
    edge_index:   np.ndarray,   # (2, E) int
    src: int, dst: int, num_nodes: int,
) -> Optional[np.ndarray]       # (E,) float32 binary indicator, or None if no path / negative cycle
```

Both treat edges as undirected (each edge traversable in both directions). Return value indexes the original undirected edge IDs — the indicator is 1 if the edge was traversed in either direction.

## `surrogate_shortest_path` (`diffopt/routing/surrogate.py`)

```python
surrogate_shortest_path(
    edge_weights: Tensor,    # (E,) float32, requires_grad=True for gradient flow
    edge_index:   Tensor,    # (2, E) LongTensor; src < dst (undirected)
    src: int, dst: int, num_nodes: int,
    lambda_: float = 10.0,  # Vlastelica perturbation strength
) -> Tensor                  # (E,) float32 binary path indicator (differentiable)
```

Implemented as `DijkstraSurrogate.apply(...)`. The returned tensor is not a proper float (it is {0.0, 1.0}), but it carries a surrogate gradient via the custom backward.

**Gradient convention:** `grad_weights[e]` is negative for edges that the loss penalises being on the path. Gradient descent increases those edge costs and routes away from them.

**λ guidance:** λ=10 is a reasonable default. Smaller λ → larger gradient signal but noisier; larger λ → smaller signal. Anneal from 10 → 1 during training (`vlastelica_lambda_decay` in config).

## `AllocationOutputs` (`diffopt/pipeline.py`)

Everything the allocation half of a forward pass produced — returned as `DiffONetPipeline.forward()`'s fourth value and by `hard_rollout_from_soft()`. A record rather than a tuple because the oracle and the deployment repair are pure post-processing on these fields and never re-run the pipeline, so every quantity they need has to come back from one call.

| Field | Type | Notes |
|---|---|---|
| `a` | `(D, J-1)` | priced allocations |
| `a_physics` | `(D, J-1)` | what the fold actually saw |
| `score` | `(D, J-1)` | raw pre-activation scores, detached |
| `alloc_by_node` | `(D, N)` | `a` scattered onto boundary nodes |
| `device_count` | scalar | `sum_n sum_d` |
| `site_view` | `(N,)` | `max_d`, diagnostics only |
| `seg_gsnr_db` | `(D, J)` | STE-blended per-segment GSNR |
| `seg_noise` | `(D, J)` | the same, as linear noise |
| `seg_km_matrix` | `(D, J)` | segment lengths in km, same masking as `seg_noise` |
| `num_segments` | `(D,)` long | |
| `boundary_node_ids` | `(D, J-1)` long | `-1` where padded |
| `demand_ids` | `List[int]` | row index -> `Demand.id` |
| `segment_edge_ids` | `Dict[int, List[List[int]]]` | `demand_id` -> ordered edge ids, grouped by transparent segment; concatenating the groups gives the ordered route, and the grouping aligns `seg_gsnr_db[k]` with a stretch of the route. Consumed by the viz frame writer (`diffopt/viz/frames.py`) to draw per-lightpath ribbons/cuts. |
| `ste_clamped_segments` | `int` | count of segments whose `qot_gsnr` fell outside `SegmentCombiner`'s `[-5, 35]` dB clamp band this forward call |
| `proxy_qot_rank_corr` | `float` | Spearman(proxy, qot) over this call's segments |
| `waste_cost` | scalar | `sum_{d,k} a_priced*relu(f4)`, detached from `n_next` |

`hard_rollout` (`diffopt/train.py`) evaluates the DEPLOYED (hard) allocation and returns a `dict` including `hard_num_violated`, `hard_num_devices`, `hard_num_sites`, `hard_worst_margin_db`, `oracle_devices`, `oracle_gap`, `oracle_infeasible`, `site_mask`, `alloc_by_node`, `gsnr_preds`, and `"alloc"` — the hard `AllocationOutputs` record this rollout evaluated (a reference, not a copy). The viz frame writer needs `alloc`'s per-(demand, boundary) cuts in path order (`alloc_by_node` alone loses position along the path) and `gsnr_preds` for margins, so both are read from `hard_rollout`'s return rather than a second forward pass.

## Traffic matrix (diffopt/traffic.py)

The fixed traffic matrix and its preflight screen — the constraint set the
duals in `diffopt/loss.py` act on. See `docs/architecture/invariants.md`,
"Traffic matrix / constraint" for the invariants; this section is contracts
and signatures only.

```python
SCENARIO_ALPHA: Dict[str, float] = {"realistic": 1.0, "stress": 0.0}

def scenario_alpha(scenario: str) -> float
    # Maps a named traffic scenario to its gravity distance exponent.
    # Raises ValueError for any scenario not in SCENARIO_ALPHA.

def build_traffic_matrix(
    topology: Topology,
    *,
    seed: int,
    scale: float,
    alpha: float,
    bitrate_options: List[float],
) -> List[Demand]
    # Wraps upstream generate_demands with aggregate=True, undirected=True,
    # protected_fraction=0.0. Each pair's offered volume is mapped to the
    # nearest member of bitrate_options; pairs below
    # min(bitrate_options) - 25 are dropped. Returns Demand ids contiguous
    # 0..N-1.

def traffic_matrix_checksum(demands: List[Demand]) -> str
    # Stable 16-hex-char sha256 digest of a matrix's (src, dst, bitrate)
    # content. Deliberately excludes `id` (a positional artifact), so the
    # digest is unchanged by a renumbering that preserves content.

def shortest_path_edges_by_km(
    topology: Topology, src: int, dst: int
) -> Optional[List[int]]
    # Edge ids of the minimum-kilometre path, in traversal order src -> dst.
    # Returns None if dst is unreachable from src, [] when src == dst.
    # Deliberately NOT routed by EdgeWeightNet.

def preflight_filter(
    topology: Topology,
    demands: Sequence[Demand],
    *,
    qot_model: torch.nn.Module,
    segment_combiner: torch.nn.Module,
    modulation_config,
    margin_db: float,
    channel_loading_fraction: float = 0.5,
    max_spans: int = 60,
) -> Tuple[List[Demand], List[Tuple[Demand, float]]]
    # Drops demands that are infeasible under the most favourable conditions
    # (all regen candidates active, routed shortest-by-km). Returns
    # (kept, excluded): kept is renumbered with contiguous ids 0..N-1 in
    # input order; excluded is a list of (demand, shortfall_db), where
    # shortfall_db = threshold + margin - best_case_gsnr, or inf when there
    # is no route at all. Necessary, not sufficient — a demand that survives
    # may still be unreachable under the routing the model actually learns.
```

- `Demand.id` contiguity is load-bearing throughout this module: it indexes
  the per-demand dual vector in `diffopt.loss.compute_loss`, so both
  `build_traffic_matrix` and `preflight_filter` renumber `0..N-1` on their
  own output rather than preserving upstream/input ids.
- `preflight_filter`'s `margin_db` is the same `delta` the constrained loss
  adds inside the hinge — a demand that cannot reach `threshold + margin`
  even under the most favourable route/placement can never satisfy the
  constraint, so excluding it here trades a silently non-converging dual for
  a reported exclusion.

## `DiffONetPipeline` segment-GSNR memo (`diffopt/pipeline.py`)

`DiffONetPipeline(..., cache_segment_gsnr: bool = True)` memoises the frozen
QoT model's per-segment dB output on `(ordered edge-id tuple, batch max
spans)`. Exact — see `docs/architecture/invariants.md`, "Physics layer".
Call `clear_segment_gsnr_cache()` after replacing `qot_model`, or construct
with `cache_segment_gsnr=False`. Bounded by
`diffopt.pipeline._SEGMENT_GSNR_CACHE_MAX` (cleared wholesale on overflow).

## `AllocationHead` / regenerator allocation (`diffopt/placement/allocation.py`)

Replaces the old `RegenPlacement` (a single `(num_nodes,)` regen-probability
logit, gated `sigmoid` or `hard_concrete`, deployed via
`hard_placement_mask()`). That class, its `placement.gate` /
`placement.gate_dropout_p` / `placement.hard_concrete.*` config keys, and
`selection.hard_eval` are gone from the codebase — a per-node vector cannot
express "demand 3 regenerates at node 7, demand 9 does not", and pricing
sites rather than devices was the wrong metric (40 demands regenerating at
node 7 need 40 devices, not 1). `AllocationHead` instead scores every
`(demand, boundary)` pair.

```python
AllocationHead(
    hidden: int = 32,
    *,
    lookahead: bool = True,        # unmask features 3-4 (next-segment lookahead)
    route_context: bool = True,    # unmask features 5-7 (km_since_cut, km_remaining, boundaries_remaining)
    alloc_ste: bool = False,       # straight-through estimator: forward value is exactly 0/1
    init_bias: float = -3.0,       # not config-reachable; see invariants.md "Closed at init"
) -> AllocationHead

AllocationHead.score(feats: (..., 8) Tensor) -> (...) Tensor
    # Positive means "cut here".

AllocationHead.rollout(
    seg_noise: Tensor,      # (D, J) linear noise/segment; padding exactly 0
    seg_km: Tensor,         # (D, J) segment lengths km, same masking
    bar_db: Tensor,         # (D,) threshold(bitrate_d) + margin_db
    num_segments: Tensor,   # (D,) long, each in [1, J]
    *,
    tau: float = 1.0,
    hard: bool = False,      # deterministic a_k = 1 if score_k > 0; self-consistent physics
) -> Tuple[Tensor, Tensor, Tensor]  # (a_priced, a_physics, waste), a_priced/a_physics both (D, J-1)
```

- **Feature layout** (`ALLOC_FEATURE_DIM = 8`, order load-bearing): `0` current-chunk GSNR proxy dB, `1` the bar dB, `2` headroom now, `3` next-segment dB `[lookahead]`, `4` headroom after next `[lookahead]` — the column the greedy-optimal policy keys off (`cut iff feature 4 < 0`; see `tests/test_oracle.py`'s representability test), `5` `km_since_cut/1000` `[route_context]`, `6` `km_remaining/1000` `[route_context]`, `7` `(boundaries_remaining)/10` `[route_context]`.
- **`rollout`'s carry `c` (the current chunk's noise) enters features as a detached observation.** Undetached, `d(a_{k+1})/d(a_k)` runs through `-10*log10(c)` and is both huge and sign-indefinite right after a cut — measured swinging a device-count gradient -3317..+1163 across a 0.002-wide window in one bias parameter, vs. a steady +89..+92 detached. See `docs/architecture/invariants.md`, "Regenerator allocation".
- **`waste` is the arm-4 surcharge**: `sum_{d,k} a_priced[d,k] * relu(feature4[d,k]).detach()`, masked by `cut_valid`. Only the `relu(feature4)` coefficient is detached — `a_priced` is not, and legitimately carries gradient into routing via its own dependence on `score`. Returned as `AllocationOutputs.waste_cost` (`diffopt/pipeline.py`) and logged by `compute_loss` as a diagnostic only — it carries no weight in the loss (the `lambda_waste` weight it used to be priced under was removed 2026-09).
- **`alloc_ste=True`** makes the forward value of `a_k` exactly `(s > 0)` instead of `sigmoid(s / tau)`, while the backward pass keeps `sigmoid'(s/tau)/tau` — so the soft (training) pass and the hard deployment rollout agree on which demands are feasible instead of disagreeing, which otherwise let the duals price violations the deployed network doesn't have. `tau` keeps only its backward role under this flag; the anneal has no forward job left, so `constrained_stress.yaml` pins `alloc_tau_end` to `alloc_tau_start` and `train.py` warns when they're not pinned. `AllocationHead.last_scores` (a buffer, not a parameter) records the raw pre-activation score every rollout — under `alloc_ste` the score can't be recovered from `a` itself (inverting the sigmoid on an exact 0/1 just reports float32's clamps), so this is what a saturation diagnostic reads instead.
- `DiffONetPipeline.forward(..., hard_alloc: bool = False)` threads `hard` into the rollout; `hard_alloc=True` runs the whole rollout under `torch.no_grad()` — see `diffopt/train.py`'s `hard_rollout()`, the deployed-allocation evaluator.
- `AllocationOutputs` (`diffopt/pipeline.py`, forward's 4th return value) carries `a` (priced), `a_physics`, `score` (detached raw pre-activation scores — not recoverable from `a` under `alloc_ste`), `alloc_by_node`, `device_count` (= `total_device_cost(alloc_by_node)`), `site_view` (diagnostic-only max-over-demands), `seg_gsnr_db`/`seg_noise`/`seg_km_matrix`/`num_segments`/`boundary_node_ids` (the last three added so `hard_rollout_from_soft` can re-run the allocation head without re-routing), `demand_ids`, `ste_clamped_segments`/`proxy_qot_rank_corr` (QoT straight-through diagnostics), and `waste_cost`. It is a record, not a tuple, because the oracle and deployment repair are pure post-processing on these fields and never re-run the pipeline.

## Experiment YAML config keys used at runtime

| Key | Used by | Notes |
|---|---|---|
| `topology` | `generate_qot_dataset.py`, `train_qot.py` | path to topology JSON |
| `modulation_formats` | `generate_qot_dataset.py`, `diffopt/train.py` (via `load_topology`), `scripts/diagnose_surrogate.py` | path to modulation YAML |
| `max_spans_per_segment` | both | must match dataset and model |
| `num_channels_cband` | `generate_qot_dataset.py` | denominator for channel_loading_fraction |
| `dataset_dir` | both | directory containing train.parquet / val.parquet |
| `batch_size`, `learning_rate`, `epochs` | `train_qot.py` | |
| `checkpoint_dir`, `log_dir` | `train_qot.py` | created if absent |
| `training.vlastelica_lambda` | `DijkstraSurrogate` | perturbation strength start; default 10.0 |
| `training.vlastelica_lambda_min` | `diffopt/train.py` | decay floor; default 1.0 |
| `training.vlastelica_lambda_decay` | `diffopt/train.py` | per-epoch multiplier; default 0.995 |
| `traffic.scenario` | `diffopt/train.py`, `scripts/evaluate_matrix.py` | `stress` (alpha 0.0) or `realistic` (alpha 1.0); mapped by `diffopt.traffic.scenario_alpha` |
| `traffic.seed` | same | matrix identity; a different value is a different constraint set |
| `traffic.scale` | same | total offered load in Gbps, calibrated per topology to ~500-900 demands |
| `traffic.holdout_seed` | `scripts/evaluate_matrix.py --holdout` | the generalisation-gap matrix |
| `constraint.margin_db` | `diffopt/train.py`, `diffopt/loss.py` | delta added inside the augmented penalty's active band; 0.5 dB ≈ 2.6σ on the QoT model's 0.1909 dB val RMSE |
| `constraint.rho` | `diffopt/train.py`, `diffopt/loss.py` | REQUIRED, no default: the augmented-Lagrangian penalty coefficient AND the dual ascent step (`eta=rho`). Sets the active band width `lambda_d/rho` in dB. Measure with `python -m scripts.calibrate_rho`; `compute_loss` raises if `rho <= 0`. The augmented penalty is the only one `compute_loss` implements — the one-sided hinge it replaced (and its `constraint.penalty`/`constraint.dual_lr` switches) was removed 2026-09 |
| `constraint.dual_init` | `diffopt/train.py` | lambda_0; shipped configs use `0.0` (cold start — every dual starts at 0 under the augmented penalty) |
| `constraint.dual_max` | `diffopt/train.py` | cap; demands pinned here are reported at end of run |
| `pipeline.lambda_dev` | `diffopt/train.py`, `diffopt/loss.py` | weight on device count; fixed, not annealed, live from epoch 0 — calibrate with `scripts/calibrate_lambda_dev.py`, do not hand-tune |
| `pipeline.lambda_cost` | same | weight on the ASE-denominated path-noise regulariser; not the primary routing signal |
| `placement.lookahead` | `diffopt/train.py`, `scripts/_common.py` | `AllocationHead(lookahead=...)`; default `true` |
| `placement.route_context` | same | `AllocationHead(route_context=...)`; default `true` |
| `placement.alloc_ste` | same | `AllocationHead(alloc_ste=...)`; default `false`. Straight-through estimator — forward value is exactly 0/1 instead of `sigmoid(s/tau)`, so the soft training pass and the hard deployment rollout agree on feasibility |
| `training.alloc_grad_clip` | `diffopt/train.py` | max-norm clip on the allocation head's SGD gradient; default `0.0` disables it |
| `viz.dump_frames` | `diffopt/train.py` | `false` (default): off, byte-identical `e2e_train_log.csv` to a run that doesn't opt in. `true` streams a per-epoch training-trajectory frame dump (`diffopt.viz.FrameWriter`) to `<log_dir>/frames.jsonl` (sidecar) and assembles `<log_dir>/frames.json` at the end of training. |
| `viz.every` | `diffopt/train.py` | `1` (default): dump every Nth epoch's frame (subsamples frames, not `e2e_train_log.csv`/stats). |
| `viz.keyframe_every` | `diffopt/train.py` | `50` (default): write a full keyframe (not just a delta) at least this often, so `scripts/build_viewer.py`'s viewer can reconstruct any dumped epoch without replaying from epoch 1. Counts *dumped* frames, not raw epochs — when `viz.every > 1` subsamples, e.g. `every: 2` with `keyframe_every: 50` means a keyframe every 100 epochs, not every 50. |
