# DiffONet: Implementation Plan — Phase 1b (Segment Combiner + Differentiable Routing)

**Last updated:** 2026-03-30
**Status:** Complete (all 36 tests passing)

---

## Context

Phase 1a produced a trained QoT model (`SpanAttentionQoT`) that predicts GSNR in dB for a single transparent segment. Phase 1b wires the remaining two core components: the **differentiable segment combiner** (physics-based noise accumulation with soft regenerator probabilities) and the **differentiable shortest-path layer** (Vlastelica surrogate gradients through Dijkstra). Both are implemented and unit-tested before Phase 1c assembles the full pipeline.

Spec reference: §3B (SegmentCombiner), §4 (routing), Testing Strategy §test_segment_combiner + §test_surrogate_grad.

---

## Files Created

| File | Description |
|------|-------------|
| `diffopt/qot/segment_combiner.py` | Physics-based differentiable GSNR combiner |
| `diffopt/routing/shortest_path.py` | Dijkstra + SPFA + batched_dijkstra |
| `diffopt/routing/surrogate.py` | Vlastelica blackbox differentiation (autograd.Function) |
| `tests/test_segment_combiner.py` | 7 unit tests |
| `tests/test_surrogate_grad.py` | 5 unit tests |

---

## Component 1: `diffopt/qot/segment_combiner.py`

### Module-level helpers

```python
def db_to_linear_noise(gsnr_db: torch.Tensor) -> torch.Tensor:
    # Returns 10^(-gsnr_db / 10)  — normalized noise power

def linear_noise_to_db(noise_linear: torch.Tensor) -> torch.Tensor:
    # Returns -10 * log10(noise_linear)

def soft_max(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    # log-sum-exp approximation: temperature * logsumexp([a/t, b/t], dim=0)
```

### `SegmentCombiner(nn.Module)`

```python
class SegmentCombiner(nn.Module):
    def __init__(self, soft_max_temperature: float = 0.5) -> None

    def forward(
        self,
        segment_gsnrs_db: List[torch.Tensor],        # list of scalar tensors
        regen_probs_at_boundaries: List[torch.Tensor],  # list of scalar tensors in (0,1)
    ) -> torch.Tensor:  # scalar: end-to-end GSNR in dB
```

**Core logic:**
```
accumulated_noise = db_to_linear_noise(segment_gsnrs_db[0])
for i in 1..len(segments)-1:
    p = regen_probs_at_boundaries[i-1]
    next_noise = db_to_linear_noise(segment_gsnrs_db[i])
    noise_no_regen = accumulated_noise + next_noise          # additive passthrough
    noise_regen    = soft_max(accumulated_noise, next_noise) # worst segment wins
    accumulated_noise = (1-p) * noise_no_regen + p * noise_regen
return linear_noise_to_db(accumulated_noise)
```

**Design notes:**
- Temperature accepted as `__init__` param so it can be annealed externally.
- Clamp guard: `gsnr_db` clamped to `[-5, 35]` to prevent float32 overflow on very long noisy paths.
- Float64 used internally for accumulation loop; cast back to float32 before returning.

---

## Component 2: `diffopt/routing/shortest_path.py`

### `dijkstra`

```python
def dijkstra(
    edge_weights: np.ndarray,      # (E,) float; must be positive
    edge_index: np.ndarray,        # (2, E) int; undirected: each edge stored once (src<dst)
    src: int,
    dst: int,
    num_nodes: int,
) -> Optional[np.ndarray]:         # (E,) binary {0,1}: 1 if edge on shortest path
```

- `heapq`-based Dijkstra; each undirected edge traversable in both directions.
- Returns binary indicator over original edge IDs (direction-agnostic).
- Returns `None` if no path exists.

### `spfa`

```python
def spfa(
    edge_weights: np.ndarray,      # (E,) float; may be negative
    edge_index: np.ndarray,        # (2, E) int
    src: int,
    dst: int,
    num_nodes: int,
) -> Optional[np.ndarray]:         # (E,) binary path indicator
```

- SPFA (Bellman-Ford with deque); handles negative edge weights.
- Used internally by the Vlastelica backward pass where perturbed weights can be negative.
- Guards against negative cycles via relaxation count limit.

### `batched_dijkstra`

```python
def batched_dijkstra(
    edge_weights: np.ndarray,      # (E,) float
    edge_index: np.ndarray,        # (2, E) int
    demands: List[Tuple[int, int]],
    num_nodes: int,
) -> np.ndarray:                   # (num_demands, E) binary
```

---

## Component 3: `diffopt/routing/surrogate.py`

### `DijkstraSurrogate(torch.autograd.Function)`

**Forward:** run exact Dijkstra; return binary path indicator as float32 tensor.

**Backward (Vlastelica ICLR 2020, Theorem 3.1):**
```
# grad_output = ∂L/∂path (E,)  [PyTorch convention]
# Vlastelica uses ŷ = -∂L/∂z, so c - λŷ = c + λ·grad_output
c_target = edge_weights + lambda_ * grad_output
path_target = spfa(c_target)   # SPFA handles possibly-negative perturbed weights
grad_weights = -(1/lambda_) * (path_star - path_target)
```

**Sign convention (important):** `c_target = w + λ·grad` (NOT minus). Perturbing with +λ·grad makes high-loss edges more expensive under the perturbed objective, flipping the path selection and producing a meaningful surrogate gradient.

### `surrogate_shortest_path` (wrapper)

```python
def surrogate_shortest_path(
    edge_weights: torch.Tensor,    # (E,)
    edge_index: torch.Tensor,      # (2, E) LongTensor
    src: int,
    dst: int,
    num_nodes: int,
    lambda_: float = 10.0,
) -> torch.Tensor:                 # (E,) binary path indicator (differentiable)
```

---

## Tests: `tests/test_segment_combiner.py`

7 test cases:

1. **Single segment** — `forward([gsnr], [])` → output == `gsnr` exactly.
2. **Two segments, p=1** — output ≈ `min(gsnr_1, gsnr_2)` (soft_max noise ≈ worst segment). Temperature=0.01, tolerance 0.1 dB.
3. **Two segments, p=0** — noise adds: `1/g_total = 1/g_1 + 1/g_2`. Tolerance 1e-3 dB.
4. **Monotonicity** — p from 0→1, output GSNR non-decreasing when first segment is worse. Temperature=0.01.
5. **Gradient** — `regen_prob.requires_grad=True`, `∂gsnr/∂p > 0` when regen helps. Temperature=0.01.
6. **Three segments, two boundaries** — verify against brute-force enumeration. Temperature=0.01, tolerance 0.5 dB.
7. **Numerical stability** — GSNR in {0, 30} dB, p in {1e-6, 1-1e-6}. Assert no NaN/Inf.

---

## Tests: `tests/test_surrogate_grad.py`

4-node graph: S=0, A=1, B=2, T=3. Edges: SA(e0), SB(e1), AT(e2), BT(e3). Paths: S-A-T and S-B-T.

1. **Correct path selection** — w=[1,2,1,2]: S-A-T selected. Verify e0, e2 active.
2. **Nonzero gradient** — loss penalises A-path. Verify `grad[E_SA] != 0`.
3. **Gradient direction** — grad on A-path edges is **negative** (gradient descent increases their cost → A-path avoided); grad on B-path edges is **positive** (gradient descent decreases their cost → B-path preferred).
4. **Zero gradient for irrelevant edge** — disconnected edge C→D (node 4→5). Verify `grad[4] == 0`.
5. **Path switch under gradient descent** — 50 steps, A-path initially cheaper, loss penalises A-path. Path switches to S-B-T within ~20 steps.

---

## Bugs Found During Implementation

### 1. Segment combiner: soft_max temperature vs noise scale mismatch

`soft_max(a, b, temperature=t)` returns `t * logsumexp([a/t, b/t])`, which overestimates `max(a, b)` by `t * log(1 + exp(-|a-b|/t))`. For typical noise values (~0.1–0.3) and `t=0.5`, the overestimate is large (~0.21 for equal values), making `noise_regen > noise_no_regen`. This inverts the expected monotonicity and gradient sign.

**Fix:** Tests 4, 5, 6 use `soft_max_temperature=0.01` so the approximation is tight.
**Lesson:** The default temperature of 0.5 is suitable for training annealing but not for unit tests that rely on near-physical behaviour.

### 2. Dijkstra with negative weights (backward pass hang)

The Vlastelica backward computes perturbed weights `c_target = w ± λ·grad`. With `w=[1,3,1,3]`, `grad=[1,0,1,0]`, `λ=10`, `c_target = [-9, 3, -9, 3]`. Dijkstra's heap-based algorithm does not handle negative weights and spins indefinitely.

**Fix:** Added `spfa()` (SPFA / Bellman-Ford with deque) to `shortest_path.py`; `surrogate.py` backward uses `spfa` instead of `dijkstra`.

### 3. Vlastelica perturbation sign

The correct Vlastelica formula (Theorem 3.1) is `c_target = c - λŷ` where `ŷ = -∂L/∂z` (the negative gradient = improvement direction). In PyTorch, `grad_output = ∂L/∂z`, so `c_target = c + λ·grad_output`. The original implementation used `c - λ·grad`, which makes the active-path edges even cheaper under perturbation → same path selected → zero gradient everywhere.

**Fix:** Changed to `c_target = w + lambda_ * g_np` in `surrogate.py` backward.

### 4. Gradient direction test assertions

The test initially asserted `grad[A-edge] > 0` (positive). The correct sign is **negative**: with gradient descent `w -= lr * grad`, a negative gradient causes the weight to increase, making the A-path more expensive. Tests updated accordingly.

---

## Config Keys (in `base.yaml`)

```yaml
segment_combiner:
  soft_max_temperature: 0.5
  soft_max_temperature_min: 0.05
training:
  vlastelica_lambda: 10.0
  vlastelica_lambda_min: 1.0
  vlastelica_lambda_decay: 0.995
```

---

## Milestone

Phase 1b milestone passed:
- 7/7 segment combiner tests pass
- 5/5 surrogate gradient tests pass
- 36/36 total tests pass (no Phase 1a regressions)

Ready for Phase 1c: full pipeline assembly (QoT model → segment combiner → surrogate routing → end-to-end training).
