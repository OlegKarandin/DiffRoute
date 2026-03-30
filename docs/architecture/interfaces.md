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

## `Topology` dataclass (`diffopt/topology.py`)

| Property | Type | Notes |
|---|---|---|
| `num_nodes` | `int` | |
| `num_edges` | `int` | undirected edges only |
| `edge_index` | `LongTensor (2, E)` | row 0 = src, row 1 = dst; recomputed each call |
| `get_edge_features()` | `FloatTensor (E, 5)` | see feature order below |
| `regen_candidate_nodes` | `List[int]` | nodes with undirected degree ≥ 3 |

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

## `simulate_segment` (`diffopt/qot/gnpy_bridge.py`)

```python
simulate_segment(
    span_lengths_km: List[float],   # length of each fiber span
    amplifier_nf_db: List[float],   # NF of post-span EDFA; same length as above
    fiber_type: str = "SSMF",       # only "SSMF" is implemented
    n_channels: int = 24,           # active WDM channels, 1–48
    launch_power_dbm: float = -1.0, # per-channel launch power
    seed: Optional[int] = None,     # for channel selection reproducibility
) -> float  # GSNR in dB at CUT (193.5 THz)
```

- `len(span_lengths_km) == len(amplifier_nf_db)` required
- CUT is always index 24 (193.5 THz) on the 100 GHz grid from 191.1 THz
- `n_channels=1` → only CUT active; upper bound is `NUM_CHANNELS=48`
- Falls back silently to `analytical_gsnr_db` on any exception — callers cannot distinguish GNPy vs. analytical labels

WDM grid constants (not configurable at call time):
- `CHANNEL_SPACING_HZ = 100e9`
- `GRID_START_HZ = 191.1e12` (CUT at index 24 → 193.5 THz)
- SSMF: α=0.2 dB/km, D=17 ps/nm/km, γ=1.3 /W/km

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
| 4 | `accum_dist_km` | cumulative km from segment start to end of this span |

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
SegmentCombiner(soft_max_temperature: float = 0.5)

forward(
    segment_gsnrs_db:          List[Tensor],  # N scalar float32 tensors (GSNR in dB)
    regen_probs_at_boundaries: List[Tensor],  # N-1 scalar float32 tensors ∈ (0, 1)
) -> Tensor  # scalar float32, end-to-end GSNR in dB
```

- `len(regen_probs_at_boundaries) == len(segment_gsnrs_db) - 1` is enforced.
- Inputs are clamped to `[-5, 35]` dB internally; caller does not need to clamp.
- Gradient flows through `regen_probs_at_boundaries`; also through `segment_gsnrs_db` if those tensors require grad.

Module-level helpers (also importable):

```python
db_to_linear_noise(gsnr_db: Tensor) -> Tensor   # 10^(-gsnr_db/10), preserves dtype
linear_noise_to_db(noise: Tensor) -> Tensor      # -10*log10(noise), preserves dtype
soft_max(a, b, temperature=0.1) -> Tensor        # t * logsumexp([a/t, b/t])
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

## Experiment YAML config keys used at runtime

| Key | Used by | Notes |
|---|---|---|
| `topology` | `generate_qot_dataset.py`, `train_qot.py` | path to topology JSON |
| `modulation_formats` | `train_qot.py` (future) | path to modulation YAML |
| `max_spans_per_segment` | both | must match dataset and model |
| `num_channels_cband` | `generate_qot_dataset.py` | denominator for channel_loading_fraction |
| `launch_power_dbm` | `generate_qot_dataset.py` | passed to simulate_segment |
| `dataset_dir` | both | directory containing train.parquet / val.parquet |
| `batch_size`, `learning_rate`, `epochs` | `train_qot.py` | |
| `checkpoint_dir`, `log_dir` | `train_qot.py` | created if absent |
| `segment_combiner.soft_max_temperature` | `SegmentCombiner` | annealing start value; default 0.5 |
| `segment_combiner.soft_max_temperature_min` | training loop (future) | annealing end value; default 0.05 |
| `training.vlastelica_lambda` | `DijkstraSurrogate` | perturbation strength start; default 10.0 |
| `training.vlastelica_lambda_min` | training loop (future) | decay floor; default 1.0 |
| `training.vlastelica_lambda_decay` | training loop (future) | per-epoch multiplier; default 0.995 |
