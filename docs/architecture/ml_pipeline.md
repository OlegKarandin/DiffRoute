# ML Pipeline Architecture

## Phase 1a scope

One model: `SpanAttentionQoT`. Takes the physical parameters of a single transparent segment (sequence of fiber spans with amplifiers) and predicts GSNR at a fixed channel under test (CUT). No routing, no placement, no cross-segment state.

## Data flow

```
.dat files
  → topology_builder.py       (one-time; produces JSON configs)
  → generate_qot_dataset.py   (produces parquet; uses GNPy or analytical fallback)
  → SegmentQoTDataset         (loads parquet; pads to max_spans=60)
  → SpanAttentionQoT          (predicts GSNR)
  → train_qot.py              (MSE loss, Adam, cosine LR)
```

## SpanAttentionQoT architecture

```
Input: (B, 60, 5)   — batch of segments, up to 60 spans, 5 features/span

Linear(5 → 64)      — shared projection; no bias initialised to 0
  +
Embedding(60, 64)   — learned positional encoding, one embedding per span index

TransformerEncoder  — 2 layers, 4 heads, d_ff=128, dropout=0, batch_first=True

Masked mean pool    — average over real spans only (padding_mask=True for real)

MLP: Linear(64→64) → ReLU → Linear(64→1)   → scalar GSNR (dB)
```

### Why transformer (not LSTM or simple MLP over aggregate stats)

GSNR at a span chain output depends on the cumulative noise accumulated in order. A span's contribution to NLI depends on its position in the chain (NLI from earlier spans is re-amplified). The transformer lets each span attend to all other spans; the learned positional encoding captures the physical meaning of position (noise accumulates directionally). An MLP over aggregate statistics (total length, mean NF) would discard per-span variance and the ordering structure.

### Why learned positional encoding (not sinusoidal)

Sinusoidal encodings are designed for arbitrary-length sequences. Here max_spans=60 is fixed, and the physical semantics of "span 3 of 5" vs "span 3 of 40" matter. Learned embeddings can fit these patterns directly from data.

### Why masked mean pool (not CLS token)

Variable-length sequences with explicit padding. Mean pooling over real spans directly computes the average encoded representation, which is a natural aggregation for a quantity (GSNR) that depends on all spans roughly equally. A CLS token would require the model to learn to funnel information into one position; mean pool provides that signal for free.

### Why dropout=0

With 50k training samples and a small model (64-dim, 2 layers), regularisation via dropout is likely unnecessary and adds a tuning dimension. If overfitting is observed in Phase 1b+ with larger inputs, add dropout then.

## Training

- Loss: MSE over predicted vs GNPy-simulated GSNR (dB)
- Optimizer: Adam, lr=1e-3
- Schedule: cosine annealing over all epochs (no warmup)
- Checkpoint: saved whenever val RMSE improves; filename `checkpoints/best_qot.pt`
- Log: `logs/train_log.csv` — epoch, train_mse, val_rmse

## Data generation design decisions

**One `n_channels` per segment, not per span.** GNPy simulates a WDM comb propagating through a span chain; all spans in the segment see the same set of active channels. Varying n_channels per span would be physically incoherent.

**Channel selection is random per sample.** At each sample, n_channels is drawn uniformly from [1, 48], and the specific channels are randomly selected (always including the CUT). This randomises both the loading level and the specific NLI pattern seen by the CUT, producing a diverse training distribution.

**`accum_dist_km` as a feature.** This encodes the cumulative distance from the segment start to the end of each span. Combined with positional encoding, it gives the model both the index-based and distance-based position of each span. The two are not redundant: the positional encoding captures discrete order; accum_dist captures physical length accumulation (which drives ASE scaling).

**GNPy fallback is silent.** `simulate_segment` catches all exceptions from the GNPy path and falls back to the analytical GN model without logging. Labels in the dataset may come from either source. This is pragmatic for data generation robustness; the analytical model is a reasonable approximation for training data diversity purposes.

## Phase 1b integration point

`SpanAttentionQoT.forward(span_features, padding_mask)` returns a differentiable scalar per segment. In Phase 1b, this is called per segment of each candidate lightpath during the differentiable routing loop. The model weights are frozen; gradients flow through the predicted GSNR into the routing/placement loss.

---

## Phase 1b scope

Two new components wired together: `SegmentCombiner` and the Vlastelica surrogate routing layer.

## Extended data flow (Phase 1b)

```
edge_weights (tensor, grad-tracked)
  → DijkstraSurrogate             (forward: numpy Dijkstra → binary path indicator)
  → path indicator (tensor)
      → segment_path()            (split path at regen-candidate nodes)
      → SpanAttentionQoT × N      (one call per transparent segment; frozen)
      → [gsnr_seg_0, …, gsnr_seg_N]  (list of scalar tensors)
  → SegmentCombiner               (physics-based noise accumulation)
      ← regen_probs[boundary nodes]
  → path_gsnr_db (scalar tensor)
  → Loss
      → .backward()
          → ∂L/∂regen_probs  (through SegmentCombiner soft interpolation)
          → ∂L/∂edge_weights (through DijkstraSurrogate surrogate backward)
```

## SegmentCombiner

No learnable parameters — purely analytical physics.

```
Inputs:
  segment_gsnrs_db:            list of N scalar tensors (GSNR in dB)
  regen_probs_at_boundaries:   list of N-1 scalar tensors ∈ (0, 1)

Per boundary i:
  noise_no_regen = noise[i-1] + noise[i]          (additive passthrough)
  noise_regen    = soft_max(noise[i-1], noise[i])  (worst segment wins)
  noise[i]       = (1-p) * noise_no_regen + p * noise_regen

Output: linear_noise_to_db(accumulated_noise)   — scalar GSNR in dB
```

Internal precision: float64. Output cast to float32.
GSNR inputs clamped to [-5, 35] dB before conversion (overflow guard).

### Why float64 internally

Noise values for a good 25 dB segment are ~0.003. Over a long path with 10 such segments and no regen, accumulated noise = 0.03, still in a comfortable range. But if a segment has GSNR near 0 dB, its noise is ~1.0, and adding many such segments can push values toward float32 overflow. float64 costs nothing for a scalar loop.

### soft_max temperature

`soft_max(a, b, t) = t * logsumexp([a/t, b/t])` overestimates `max(a, b)` by `t * log(1 + exp(-|a-b|/t))`. For noise values ~0.1 and `t=0.5` this overestimate is ~0.21 — larger than the noise itself, making the regen path seem *worse* than no-regen. Keep `t ≤ 0.05` for the approximation to be physically meaningful. The default `t=0.5` in config is the annealing start value; actual training should decay toward `t_min=0.05`.

## DijkstraSurrogate (Vlastelica layer)

```
Forward:
  edge_weights (tensor) → [detach → numpy] → Dijkstra → path (binary numpy)
  → return as float32 tensor

Backward:
  grad_output = ∂L/∂path   (E,) from downstream
  c_target = edge_weights + λ * grad_output   [NOTE: + not -]
  path_target = SPFA(c_target)   [SPFA because c_target may have negative entries]
  grad_weights = -(1/λ) * (path_star - path_target)
  return grad_weights
```

Two path-finding functions live in `shortest_path.py`:
- `dijkstra` — heap-based, positive weights only; used in forward pass.
- `spfa` — Bellman-Ford with deque, handles negative weights; used in backward pass.

### Why the sign is + not -

Vlastelica Theorem 3.1: `c_target = c - λ·ŷ` where `ŷ` is the *improvement direction* (negative gradient). Since PyTorch's `grad_output = ∂L/∂z` (gradient, not negative gradient), `c_target = c - λ·(-grad_output) = c + λ·grad_output`.

With the wrong sign (`c - λ·grad`), the perturbation makes already-active-path edges even cheaper, so the perturbed solve returns the same path → zero gradient everywhere.

### Gradient sign for active-path edges

If the loss penalises an edge being on the path, `grad_output[e] > 0` for that edge. The surrogate backward returns a *negative* gradient on that edge. Gradient descent (`w -= lr·grad`) then *increases* the edge weight, routing future calls away from it. The gradient sign is counterintuitive but correct — do not flip it.
