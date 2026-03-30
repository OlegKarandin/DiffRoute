# DiffONet: Implementation Plan — Phase 1a (QoT Estimator)

**Last updated:** 2026-03-28
**Status:** Planning (nothing implemented yet)

---

## User-Provided Inputs Required Before Implementation

Before starting Step 2, the following external files are needed:

| Input | Format | Used In |
|-------|--------|---------|
| Nobel Germany topology | CSV: `link_id, src_id, dst_id, length_km` (Fibers section only) | Step 2 |
| European topology (e.g., COST239) | CSV: `link_id, src_id, dst_id, length_km` (Fibers section only) | Step 2 |
| Modulation format thresholds | YAML or CSV with modulation name, GSNR threshold (dB), bitrate (Gbps) | Step 4 / new config |

---

## Implementation Order

### Step 1 — Project Scaffold

**Files to create:**
- `pyproject.toml` — dependencies: torch, numpy, networkx, gnpy>=2.7, pyyaml, pandas, pyarrow, matplotlib, tqdm; dev: pytest, ruff
- `diffopt/__init__.py`
- `diffopt/qot/__init__.py`
- `diffopt/routing/__init__.py`
- `diffopt/placement/__init__.py`

---

### Step 2 — Topology Configs

**Input format (user-provided):**
Each topology is given as a CSV/table of links:
```
link_id, src_id, dst_id, length_km
0, 0, 1, 320
1, 1, 2, 170
...
```
Nodes have no names or coordinates — use integer IDs only.

**Span splitting algorithm** (`diffopt/topology_builder.py` or inline in the config generation script):

Spans must satisfy:
- Every span ≥ 20 km (hard constraint — too-short spans are unrealistic and cause GNPy issues)
- Spans should be as equal as possible (no "remainder" short span at the end)

Algorithm `split_link_into_spans(length_km, min_span_km=20, target_span_km=80) -> List[float]`:
1. Compute `n_min = ceil(length_km / 100)` and `n_max = ceil(length_km / 40)` to bound the candidate range.
2. For each candidate `n` in `[n_min, n_max]`:
   a. Compute `span_len = length_km / n` (uniform split).
   b. Skip if `span_len < min_span_km` (would produce too-short spans).
   c. Record `(n, span_len, deviation_from_target = abs(span_len - target_span_km))`.
3. Pick the `n` that minimises `deviation_from_target` (closest to 80 km per span while keeping spans balanced).
4. Return `[length_km / n] * n` (all spans equal, rounded to 2 decimal places — total must sum exactly to `length_km`).

Example:
- 170 km → candidates: n=2 (85 km each), n=3 (56.7 km each), n=4 (42.5 km each). Closest to 80 km is n=2 (85 km). ✓ Avoids 80-80-10 split.
- 320 km → n=4 (80 km each). ✓
- 50 km → n=1 (50 km). ✓ (single span, acceptable for metro)
- 25 km → n=1 (25 km). ✓

**Output topology JSON format:**
```json
{
  "nodes": [
    {"id": 0},
    {"id": 1}
  ],
  "edges": [
    {
      "src": 0, "dst": 1,
      "length_km": 320,
      "num_spans": 4,
      "span_lengths_km": [80.0, 80.0, 80.0, 80.0],
      "fiber_type": "SSMF",
      "amplifier_nf_db": [5.5, 5.5, 5.5, 5.5]
    }
  ]
}
```

No `name`, `x`, or `y` fields anywhere. All topology identity is purely by integer node ID.

**Files to create:**
- `configs/topology/nobel_germany.json` — generated from user-provided link CSV
- `configs/topology/european.json` — generated from user-provided link CSV
- `configs/modulation_formats.yaml` — modulation name → GSNR threshold (dB) + bitrate (Gbps) per channel, generated from user-provided thresholds file
- `configs/experiment/base.yaml` — full hyperparameter config (see Section 9 of spec)
- `configs/experiment/small_test.yaml` — small variant (5 demands, 200 samples, fewer epochs)

**`configs/modulation_formats.yaml` format:**
```yaml
# Each format lists: minimum GSNR threshold (dB) for BER < 1e-4 after FEC
# and net bitrate per channel at 32 Gbaud symbol rate
formats:
  - name: DP-BPSK
    gsnr_threshold_db: 6.0
    bitrate_gbps: 50
  - name: DP-QPSK
    gsnr_threshold_db: 9.0
    bitrate_gbps: 100
  - name: DP-8QAM
    gsnr_threshold_db: 13.0
    bitrate_gbps: 150
  - name: DP-16QAM
    gsnr_threshold_db: 16.0
    bitrate_gbps: 200
  - name: DP-32QAM
    gsnr_threshold_db: 19.5
    bitrate_gbps: 250
  - name: DP-64QAM
    gsnr_threshold_db: 22.5
    bitrate_gbps: 300
```
*(Exact values will be replaced with user-provided thresholds.)*

---

### Step 3 — `diffopt/topology.py`

**Purpose:** Load a topology JSON and expose a PyTorch-friendly graph.

**Key contents:**
- `Topology` dataclass with fields: `nodes`, `edges`, `num_nodes`, `num_edges`, `edge_index` (2×E tensor), `edge_features` tensor
- `load_topology(path: str) -> Topology`
- `Topology.edge_src(edge_id)`, `Topology.edge_dst(edge_id)` helpers
- `Topology.regen_candidate_nodes` property — nodes with degree ≥ 3
- `Topology.get_edge_features()` — tensor of shape (E, F): `[span_length_km, fiber_type_idx, amplifier_nf_db, num_spans, edge_length_km]` per edge

No city names, no coordinates, no `x`/`y` anywhere in this module.

---

### Step 4 — `diffopt/demands.py`

**Purpose:** Generate random traffic demands for a topology. Modulation format is **not** part of a demand — it is determined later from the predicted GSNR and the modulation format config.

**Key contents:**
- `Demand` namedtuple: `(id, src, dst, bitrate_gbps)` — no `modulation` field
- `generate_demands(topology, num_demands, min_bitrate, max_bitrate, seed) -> List[Demand]`
  - Ensures src ≠ dst; samples node pairs uniformly

**Modulation assignment** (separate utility in `diffopt/modulation.py`):
- `load_modulation_formats(path: str) -> List[ModulationFormat]`
- `assign_modulation(gsnr_db: float, bitrate_gbps: float, formats: List[ModulationFormat]) -> Optional[str]`
  - Filters formats where `gsnr_db >= threshold` AND `bitrate_per_channel >= bitrate_gbps`
  - Returns the name of the lowest-threshold feasible format (most robust choice, fewest channels needed)
  - Returns `None` if no format is feasible (demand infeasible even with best modulation)
- `required_gsnr_threshold(bitrate_gbps: float, formats: List[ModulationFormat]) -> float`
  - Returns the minimum GSNR threshold among all formats that can carry the given bitrate
  - Used as the per-demand GSNR threshold in the loss function

The `required_gsnr_threshold` function is key: it turns each demand's bitrate into a concrete GSNR requirement, which then drives the feasibility loss. This means different demands may have different GSNR thresholds depending on their bitrate.

---

### Step 5 — `diffopt/qot/gnpy_bridge.py`

**Purpose:** Wrap GNPy to simulate a transparent segment and return GSNR.

**Channel loading design:**

GNPy simulates WDM transmission where nonlinear interference (NLI) depends on the specific frequencies of active channels — channels close in wavelength to the channel under test (CUT) contribute more NLI than distant channels. A scalar "loading fraction" is therefore physically incorrect as a GNPy input.

Instead:
- Define a fixed C-band channel grid: 96 channels on a 50 GHz ITU-T grid (C-band, ~191.35–196.10 THz), as is standard for modern coherent WDM.
- Always include the CUT (center channel, e.g., channel index 48) as active.
- Randomly select additional active channels from the remaining 95.
- The number of active channels (`n_channels`, integer 1–96) is the randomized parameter during data generation.

**Key contents:**
- `C_BAND_FREQUENCIES: List[float]` — 96 center frequencies (Hz) on 50 GHz grid
- `CUT_INDEX: int = 47` — index of the channel under test (center channel)
- `build_channel_list(n_channels: int, seed: Optional[int] = None) -> List[float]`
  - Always includes `C_BAND_FREQUENCIES[CUT_INDEX]`
  - Randomly selects `n_channels - 1` additional channels from the remaining 95
  - Returns sorted list of active channel frequencies
- `simulate_segment(span_lengths_km, amplifier_nf_db, fiber_type, n_channels, launch_power_dbm=-1.0) -> float`
  - Builds a GNPy network of `Fiber + EDFA` elements (one EDFA after each span)
  - Creates `SpectralInformation` with `n_channels` WDM channels at the selected frequencies
  - Propagates signal through the segment (fresh signal — no accumulated noise from previous segments)
  - Extracts GSNR at the CUT at the segment output
  - Returns GSNR in dB
- Fallback: `analytical_gsnr(span_lengths_km, amplifier_nf_db, fiber_type, n_channels, launch_power_dbm) -> float`
  - GN-model closed-form approximation in case GNPy API issues arise

**Per-span feature for the QoT model:**
Replace the scalar `channel_loading` feature (0–1 fraction) with:
- `channel_loading_fraction = n_channels / 96` (scalar, 0–1) — computed from `n_channels`

This retains a scalar feature compatible with the model architecture while correctly reflecting that the GNPy simulation used actual channel positions. The model learns the relationship between channel fill fraction and NLI noise level (which is a well-defined, monotonic relationship in the GN model).

---

### Step 6 — `data/generate_qot_dataset.py`

**Purpose:** Generate segment-level (span_features, GSNR) training pairs.

**Procedure:**
1. Load topology
2. For each sample:
   a. Pick random src-dst pair
   b. Compute k=5 shortest paths, pick one
   c. Randomly assign regeneration points → split into transparent segments
   d. For each link: randomly sample `n_channels` ∈ [1, 96] (uniform integer)
   e. For each segment: call `gnpy_bridge.simulate_segment()` with actual `n_channels`, record GSNR
3. Per-span features (5 features): `[span_length_km, fiber_type_idx, amp_nf_db, channel_loading_fraction, accum_dist_km]`
   - `channel_loading_fraction = n_channels / 96`
   - `accum_dist_km` = cumulative distance from segment start to end of this span
4. Save to `data/datasets/segments_train.parquet` and `segments_val.parquet`

Target: 50,000 train + 10,000 val samples.

---

### Step 7 — `diffopt/qot/dataset.py`

**Purpose:** PyTorch Dataset for segment-level (features, GSNR) pairs.

**Key contents:**
- `SegmentQoTDataset(parquet_path, max_spans=20)`
- Returns `(span_features, padding_mask, gsnr_db)` tensors
- Pads variable-length segments to `max_spans`; mask = True for real spans, False for padding
- No changes from original plan

---

### Step 8 — `diffopt/qot/model.py`

**Purpose:** Span-attention transformer predicting GSNR for one transparent segment.

**Architecture** (unchanged from original plan):
```
Input: (batch, max_spans, feature_dim=5)
Positional encoding: learned embeddings (span index)
Linear projection: feature_dim → model_dim (64)
Transformer encoder: 2 layers, 4 heads, dim=64, ff_dim=128
Mean pooling over non-padded spans
Output MLP: 64 → 32 → 1 (GSNR in dB)
```

**Key contents:**
- `SpanAttentionQoT(feature_dim, model_dim, num_heads, num_layers, max_spans, ff_dim)`
- `forward(span_features, padding_mask) -> gsnr_db`
- No regenerator-related inputs anywhere

---

### Step 9 — `diffopt/qot/train_qot.py`

**Purpose:** Standalone QoT model training script. No changes from original plan.

**Key contents:**
- Load config from YAML
- `train_epoch(model, dataloader, optimizer) -> mse_loss`
- `eval_epoch(model, dataloader) -> rmse_db`
- Training loop with cosine LR schedule
- Save checkpoint when RMSE < 0.5 dB
- CSV log: epoch, train_loss, val_rmse

---

### Step 10 — Tests

**Files to create:**
- `tests/test_qot_model.py`:
  - Shape test: feed known span sequences, verify output shape `(batch,)`
  - No-regen-input test: assert model has no parameter or input path for regen features
  - Mini-training test: train on 100 synthetic samples, verify loss decreases over 10 epochs
- `tests/test_modulation.py`:
  - Test `assign_modulation` selects correct format for boundary GSNR values
  - Test `required_gsnr_threshold` returns correct minimum threshold per bitrate
  - Test `assign_modulation` returns `None` when GSNR is too low for any format
- `tests/test_topology.py`:
  - Test `split_link_into_spans` produces spans ≥ 20 km for various lengths
  - Test that all splits sum exactly to the link length
  - Test specific cases: 170 km → prefer ~85 km (not 80-80-10), 320 km → 4×80 km

---

## Summary of Changes from Original Plan

| Area | Original | Updated |
|------|----------|---------|
| Topology input | Fixed JSON with names/coordinates | CSV (link_id, src, dst, length_km); nodes are ID-only |
| Span generation | Fixed spans in JSON | Computed by balanced-split algorithm during config generation |
| Demand fields | `(id, src, dst, bitrate_gbps, modulation)` | `(id, src, dst, bitrate_gbps)` — no modulation |
| Modulation assignment | Part of demand generation | Post-QoT assignment via `diffopt/modulation.py` using SNR thresholds config |
| Per-demand GSNR threshold | Single global `gsnr_threshold_db` | Per-demand threshold from `required_gsnr_threshold(bitrate_gbps)` |
| Channel loading | Scalar fraction 0–1 passed to GNPy | Actual integer `n_channels` ∈ [1,96] selects channels on 50 GHz ITU grid; scalar fraction `n_channels/96` used as model feature |
| New files | — | `diffopt/topology_builder.py` (span splitting), `diffopt/modulation.py`, `configs/modulation_formats.yaml` |

---

## Verification (unchanged targets)

1. **Topology loads:** `python -c "from diffopt.topology import load_topology; t = load_topology('configs/topology/nobel_germany.json'); print(t.num_nodes, t.num_edges)"`

2. **Span splits are balanced:** `python -c "from diffopt.topology_builder import split_link_into_spans; print(split_link_into_spans(170))"` → should print two spans near 85 km, not `[80, 80, 10]`

3. **GNPy bridge works:** `python -c "from diffopt.qot.gnpy_bridge import simulate_segment; print(simulate_segment([80,80,80], [5.5,5.5,5.5], 'SSMF', n_channels=64))"` — should return a float in [5, 25] dB range

4. **Modulation assignment works:** `python -c "from diffopt.modulation import load_modulation_formats, assign_modulation; fmts = load_modulation_formats('configs/modulation_formats.yaml'); print(assign_modulation(gsnr_db=14.0, bitrate_gbps=100, formats=fmts))"` — should return `'DP-QPSK'` or equivalent

5. **Dataset generates:** `python data/generate_qot_dataset.py --config configs/experiment/small_test.yaml` — produces parquet files

6. **QoT model runs:** `pytest tests/test_qot_model.py tests/test_modulation.py tests/test_topology.py -v`

7. **QoT model trains:** `python -m diffopt.qot.train_qot --config configs/experiment/small_test.yaml` — loss decreasing, checkpoint saved

**Milestone:** Val RMSE < 0.5 dB on held-out segments → ready for Phase 1b.
