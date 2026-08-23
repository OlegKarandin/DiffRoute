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

- Takes no annealed parameters at all — not `temperature`, not anything else. The fold is exact, so there is no approximation sharpness to keep in sync across calls. An earlier version took `soft_max_temperature` at construction (never annealed anywhere in the codebase), then as a required per-call `forward()` argument (matching `DiffONetPipeline.forward()`'s `tau`/`lambda_` pattern) — see `docs/investigations/CHANGELOG.md`'s Phase 1c corrections (#7, #8, #11, #12) for the bugs this history caused and the fixes, ending with the parameter's outright removal.
- `len(regen_probs_at_boundaries) == len(segment_gsnrs_db) - 1` is enforced.
- Inputs are clamped to `[-5, 35]` dB internally; caller does not need to clamp.
- Gradient flows through `regen_probs_at_boundaries`; also through `segment_gsnrs_db` if those tensors require grad.
- **Chunk semantics.** A regenerator boundary splits the path into independent chunks — it rebuilds the signal, so noise does not carry across it. `forward`'s result is the **exact probability-weighted expectation over every possible chunking** (every hard partition of the path at the boundaries): `Σ_π P(π) · (chunk-max noise under partition π)`, `π` ranging over all `2^(N-1)` ways the `N-1` boundaries could independently cut — but it is computed by a polynomial-time dynamic program, not by enumerating `π`. The DP asks "what is the probability every realized chunk stays under a bar `tau`?" (checkable left to right, since a chunk can never span a cut), vectorized over all `N(N+1)/2` possible chunk-sum thresholds, then reconstructs `E[max]` from the resulting step function. `F(tau)` is built purely from sums and products of the per-boundary probabilities, never through an exponential, which is also why "regen helps" is provable directly from the DP's structure (cutting a boundary can only split a chunk into two no-larger pieces) rather than depending on a temperature staying small. Exact at `p ∈ {0,1}` for every boundary simultaneously (a separate `O(N)` hybrid path handles this case directly — no enumeration, so this holds even for thousands of segments), and exact at any individual hard vertex (`p_i ∈ {0,1}` for boundary `i` while others stay fractional) via the general DP's own limit — no separate code path is needed for that case. See `diffopt/qot/segment_combiner.py`'s module docstring for the DP's exact recurrence and `docs/architecture/invariants.md`'s "Segment combiner" section for the invariants it guarantees.
- **Cost.** The genuinely-fractional-probability branch is polynomial: `O(N^4)` elementwise work over an `(R, N)` float64 working set, capped at `MAX_EXACT_FOLD_SEGMENTS = 128` **segments** (not boundaries) — `ValueError` above that rather than hang. Real `ind_132` km-shortest paths reach 16-19 segments, so this leaves ~7x headroom; see `docs/investigations/fold_formula_scalability.md`, resolved by `docs/investigations/CHANGELOG.md`'s correction #12. This replaces an earlier exponential-fold implementation whose `num_boundaries > 20` cap had almost no real margin against measured real-topology path lengths.

Module-level helpers (also importable):

```python
db_to_linear_noise(gsnr_db: Tensor) -> Tensor   # 10^(-gsnr_db/10), preserves dtype
linear_noise_to_db(noise: Tensor) -> Tensor      # -10*log10(noise), preserves dtype
soft_max(a, b, temperature=0.5) -> Tensor
    # m*t*logsumexp([a/m/t, b/m/t]) where m = max(a,b).detach()
    # Scale-normalised deliberately: the plain form's t*ln2 error is ABSOLUTE
    # in linear-noise units and inverts the "regen helps" invariant at the
    # real per-segment noise scale (~0.0025). See
    # docs/investigations/CHANGELOG.md#correction-1c-8.
```

## `dijkstra` / `spfa` / `batched_dijkstra` (`diffopt/routing/shortest_path.py`)

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

```python
batched_dijkstra(
    edge_weights: np.ndarray,          # (E,) float64
    edge_index:   np.ndarray,          # (2, E) int
    demands: List[Tuple[int, int]],    # (src, dst) pairs
    num_nodes: int,
) -> np.ndarray                        # (num_demands, E) float32 binary
```

Raises `ValueError` if any demand has no path.

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
| `constraint.margin_db` | `diffopt/train.py`, `diffopt/loss.py` | delta inside the hinge; 0.5 dB ≈ 2.6σ on the QoT model's 0.1909 dB val RMSE |
| `constraint.dual_init` | `diffopt/train.py` | lambda_0; 10.0 matches the removed `lambda_infeasible` so epoch 1 reproduces prior behaviour |
| `constraint.dual_lr` | `diffopt/train.py` | eta in the dual ascent step |
| `constraint.dual_max` | `diffopt/train.py` | cap; demands pinned here are reported at end of run |
| `placement.gate` | `sigmoid` (default) or `hard_concrete` |
| `placement.gate_dropout_p` | Training-only gate dropout probability; `0.0` = off |
| `placement.hard_concrete.{beta,gamma,zeta}` | Hard-concrete stretch parameters; defaults `0.5 / -0.1 / 1.1` |
| `selection.hard_eval` | `true` (default): selection key from a hard-placement forward pass |
