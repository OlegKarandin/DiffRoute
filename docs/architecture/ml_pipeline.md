# ML Pipeline Architecture

## Phase 1a scope

One model: `SpanAttentionQoT`. Takes the physical parameters of a single transparent segment (sequence of fiber spans with amplifiers) and predicts GSNR at a fixed channel under test (CUT). No routing, no placement, no cross-segment state.

## Data flow

```
.dat files
  → topology_builder.py       (one-time; produces JSON configs)
  → generate_qot_dataset.py   (produces parquet; real GNPy only, no fallback)
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

**`accum_dist_km` as a feature.** Distance starts at `0.0` before the first span; the value recorded for span *i* is the accumulated length of all *prior* spans (not including span *i* itself), and the running total increments only after each span is recorded. So this is the distance to the *start* of each span, not its end. Combined with positional encoding, it gives the model both the index-based and distance-based position of each span. The two are not redundant: the positional encoding captures discrete order; accum_dist captures physical length accumulation (which drives ASE scaling).

**No GNPy fallback.** `diffopt/qot/optical_bridge.py::segment_gsnr_db` (which replaced the old, deleted `diffopt/qot/gnpy_bridge.py::simulate_segment`) has no `try`/`except` around the GNPy call — an AST test (`tests/test_optical_bridge.py`) enforces that no bare `except Exception` exists anywhere in the module. Every label in the dataset is a real GNPy-derived GSNR; a GNPy failure raises and generation stops rather than silently substituting an analytical approximation. This is a correction, not a design choice: for the project's entire history before this migration, the old bridge silently fell back to the analytical GN model on any GNPy exception, so every prior dataset's labels came from that fallback and real GNPy never once executed (see `CLAUDE.md`'s "Upstream dependency" section for the full story).

## Phase 1b integration point

`SpanAttentionQoT.forward(span_features, padding_mask)` returns a differentiable scalar per segment. In Phase 1b, this is called once per training step in **one batched call over every segment of every demand** (padded to the batch's true max span count, not the architectural `max_spans=60`) — not once per segment — during the differentiable routing loop; results are then scattered back per demand. The model weights are frozen; gradients flow through the predicted GSNR into the routing/placement loss.

---

## Phase 1b scope

Two new components wired together: `SegmentCombiner` and the Vlastelica surrogate routing layer.

## Extended data flow (Phase 1b)

```
edge_weights (tensor, grad-tracked)
  → DijkstraSurrogate             (forward: numpy Dijkstra → binary path indicator)
  → path indicator (tensor)
      → segment_path()            (split path at regen-candidate nodes)
      → SpanAttentionQoT          (one batched call over every segment of every
                                    demand, padded to the batch's true max span
                                    count; frozen; results scattered back per demand)
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

`soft_max` is **scale-normalised**: `soft_max(a, b, t) = m * t * logsumexp([a/m/t, b/m/t])` where `m = max(a, b).detach()`. This divides by `m` before the log-sum-exp and multiplies back after, which makes the overshoot above `max(a, b)` a fixed *fraction* of `m` — `m * t * log(1 + exp(-|a-b|/(m*t)))`, bounded above by `m * t * ln2` — rather than an absolute quantity. Because the error is relative, "regen helps" (`soft_max(a, b) < a + b` for `a ≈ b`) holds for any `temperature < 1/ln2 ≈ 1.44`, at any noise magnitude. Do not simplify this back to the plain `t * logsumexp([a/t, b/t])` form — see the "Superseded" subsection below for why that form is broken.

`temperature` is a required `SegmentCombiner.forward()` argument (no constructor default to accidentally rely on), and `diffopt/train.py` anneals it every epoch from `segment_combiner.soft_max_temperature` (config, default 0.5) down to `segment_combiner.soft_max_temperature_min` (config, default **0.01** — the exact value CLAUDE.md's Phase 1b correction #3 validated via `monotonicity`/`gradient-sign` unit tests), over the same epoch window as `regen_tau`. Because the normalisation makes the temperature schedule sign-correct at any noise scale, this anneal is no longer load-bearing for correctness the way it was pre-fix — it remains useful for keeping the approximation tight (`soft_max ≈ hard max`) late in training.

#### Superseded (pre-correction-#8)

The plain (non-normalised) form `soft_max(a, b, t) = t * logsumexp([a/t, b/t])` overestimates `max(a, b)` by `t * log(1 + exp(-|a-b|/t))`, bounded above by `t * ln2`. For noise values ~0.1 and `t=0.5` this overestimate is ~0.21 — larger than the noise itself, making the regen path seem *worse* than no-regen. Framed this way, temperature looked like the lever that had to be kept small enough for the *fixed* noise scale in the unit-test fixtures (5–15 dB segments, noise ~0.1–0.3).

That framing was wrong: the overshoot is **absolute** in linear-noise units and does not shrink with the operands, so it does not track the noise scale actually seen in training. Real per-segment noise on `ind_132` is ~0.0025 (segments run ~26 dB) — the schedule's sharpest temperature, `0.01`, has an absolute floor of `t*ln2 ≈ 0.00693`, over 2x the noise itself, which inverted the "regen helps" invariant on every topology since Phase 1b (see CLAUDE.md's Phase 1c correction #8 for the full diagnosis). An earlier version of `diffopt/train.py` additionally fixed `t=0.5` for the entire run and never annealed it, compounding the problem (Phase 1c correction #7). The fix was the scale normalisation described above, not a smaller temperature — no fixed temperature makes the plain form's absolute error track an unknown noise scale.

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
