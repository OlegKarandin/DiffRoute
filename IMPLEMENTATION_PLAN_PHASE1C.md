# DiffONet: Implementation Plan — Phase 1c (End-to-End Pipeline)

**Last updated:** 2026-03-31
**Status:** Planned (Phase 1b complete, 36/36 tests passing)

---

## Context

Phase 1b produced the differentiable segment combiner and Vlastelica surrogate routing layer, both unit-tested in isolation. Phase 1c assembles all components into a jointly trainable pipeline: the edge weight network biases routing, the surrogate Dijkstra selects a path, the path is segmented at regenerator candidate nodes, the frozen QoT model predicts GSNR per segment, and the segment combiner merges them using soft regenerator probabilities. Gradient descent then jointly updates routing costs and regenerator placement.

Spec reference: §5 (EdgeWeightNet), §6 (RegenPlacement), §7 (Pipeline), §8 (Loss), §9 (Training loop), Testing Strategy §test_pipeline.

---

## Files to Create / Modify

| Action | File |
|--------|------|
| Create | `diffopt/routing/edge_weight_net.py` |
| Create | `diffopt/placement/regenerator.py` |
| Create | `diffopt/pipeline.py` |
| Create | `diffopt/loss.py` |
| Create | `diffopt/train.py` |
| Create | `tests/test_pipeline.py` |
| Edit   | `configs/experiment/base.yaml` |
| Edit   | `configs/experiment/small_test.yaml` |

---

## Component 1: `diffopt/routing/edge_weight_net.py`

```python
class EdgeWeightNet(nn.Module):
    def __init__(self, input_dim: int = 7, hidden1: int = 64, hidden2: int = 32) -> None

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        # (E, 7) → (E, 1), positive weights via Softplus
```

**Input features (7D per edge):**
- Indices 0–4: topology features from `topology.get_edge_features()` — `[mean_span_length_km, fiber_type_idx, mean_amp_nf_db, num_spans, total_length_km]`
- Index 5: `regen_probs[edge.src]` — probability at source node
- Index 6: `regen_probs[edge.dst]` — probability at destination node

**Architecture:** `Linear(7→64) → ReLU → Linear(64→32) → ReLU → Linear(32→1) → Softplus`

Softplus output guarantees strictly positive weights for Dijkstra without clamping.

---

## Component 2: `diffopt/placement/regenerator.py`

```python
class RegenPlacement(nn.Module):
    def __init__(self, num_nodes: int) -> None:
        # self.regen_logits = nn.Parameter(torch.zeros(num_nodes))
        # sigmoid(0) = 0.5  →  neutral "agnostic" start

    def get_regen_probs(self, tau: float = 1.0) -> torch.Tensor:
        # Returns sigmoid(regen_logits / tau), shape (num_nodes,)
```

**Design notes:**
- No `forward()` method — only `get_regen_probs()` to avoid ambiguous call semantics.
- `tau` is always passed explicitly; never stored as an attribute (prevents stale annealing state).
- Init to `zeros` so the optimizer starts from a neutral prior before the loss guides placement.

---

## Component 3: `diffopt/pipeline.py`

### Helper: `segment_path`

```python
def segment_path(
    ordered_edge_ids: List[int],
    start_node: int,
    regen_candidate_set: set,
    topology: Topology,
    demand_dst: int,
) -> Tuple[List[List[int]], List[int]]:
    # Returns (segments, boundary_node_ids)
```

**Algorithm:**
```
current_node = start_node
for eid in ordered_edge_ids:
    u, v = topology.edge_src(eid), topology.edge_dst(eid)
    exit_node = v if current_node == u else u   # resolve undirected traversal
    append eid to current_segment
    current_node = exit_node
    if exit_node in regen_candidate_set AND exit_node != demand_dst:
        close segment; record boundary_node; start new segment
append final trailing segment
```

The `demand_dst` guard prevents a spurious empty segment when the path ends exactly at a regen candidate.

### Helper: `_reconstruct_path` (method on DiffONetPipeline)

Converts binary `(E,)` path indicator → ordered list of edge IDs from `demand.src` to `demand.dst`:

```
1. active_edge_ids = [e for e if path_indicator.detach() > 0.5]
   (detach before numpy — path_indicator stays in the autograd graph)
2. Build undirected adjacency dict from active edges
3. Walk current_node from src to dst, collecting edge IDs in traversal order
```

### `DiffONetPipeline(nn.Module)`

```python
class DiffONetPipeline(nn.Module):
    def __init__(
        self,
        topology: Topology,
        qot_model: SpanAttentionQoT,
        segment_combiner: SegmentCombiner,
        edge_weight_net: EdgeWeightNet,
        regen_placement: RegenPlacement,
        channel_loading_fraction: float = 0.5,
        max_spans: int = 60,
    ) -> None:
        # Freeze QoT: for p in qot_model.parameters(): p.requires_grad_(False)
        # register_buffer: _topo_edge_features (E,5), _edge_index (2,E),
        #                  _edge_src_ids (E,), _edge_dst_ids (E,)
        # Precompute: _regen_candidate_set = set(topology.regen_candidate_nodes)

    def forward(
        self,
        demands: List[Demand],
        tau: float = 1.0,
        lambda_: float = 10.0,
    ) -> Tuple[Dict[int, Tensor], Dict[int, Tensor], Tensor]:
        # Returns: (paths, gsnr_preds, regen_probs)
```

**Forward pass per demand:**
```
1. regen_probs = regen_placement.get_regen_probs(tau)                      # (num_nodes,)
2. edge_feats = cat([_topo_edge_features,                                   # (E, 7)
                     regen_probs[_edge_src_ids].unsqueeze(1),
                     regen_probs[_edge_dst_ids].unsqueeze(1)], dim=1)
3. edge_weights = edge_weight_net(edge_feats).squeeze(-1)                   # (E,)
4. per demand:
   a. path_indicator = surrogate_shortest_path(edge_weights, ..., lambda_=lambda_)
   b. ordered_edges  = _reconstruct_path(path_indicator, demand.src, demand.dst)
   c. segments, boundary_nodes = segment_path(ordered_edges, demand.src, ..., demand.dst)
   d. for each segment: extract span features (see below), call qot_model → scalar GSNR
   e. boundary_probs = [regen_probs[n] for n in boundary_nodes]
   f. path_gsnr = segment_combiner(segment_gsnrs, boundary_probs)
```

**Span feature extraction for one segment:**
```
accum_dist = 0.0
for each edge in segment:
    for each span in edge:
        row = [span_length_km, fiber_type_idx, amp_nf_db,
               channel_loading_fraction,   # fixed at 0.5 during inference
               accum_dist]                 # cumulative km BEFORE this span
        accum_dist += span_length_km
pad to max_spans=60 with zeros
padding_mask[0, :n_spans] = True          # True = real span
```

Device propagation: derive device from `_topo_edge_features` (registered buffer).

**Gradient flow:**
- `regen_logits` gradient: flows through `segment_combiner`'s soft interpolation weights (`boundary_probs`)
- `edge_weight_net` gradient: flows through the Vlastelica surrogate (path selection change)
- QoT model: `requires_grad_(False)` at construction; its output participates in autograd normally (do NOT use `torch.no_grad()` — that would sever the combiner's gradient path)

---

## Component 4: `diffopt/loss.py`

```python
def compute_loss(
    gsnr_preds: Dict[int, Tensor],
    demands: List[Demand],
    regen_probs: Tensor,                  # (num_nodes,)
    modulation_config: ModulationConfig,
    lambda_regen: float = 1.0,
    lambda_infeasible: float = 10.0,
) -> Tuple[Tensor, dict]:
```

**Logic:**
```python
feasibility_loss = torch.zeros(1, device=regen_probs.device)  # tensor, not float 0
for demand in demands:
    threshold = modulation_config.required_snr_threshold(demand.bitrate_gbps)
    shortfall = relu(threshold_tensor - gsnr_preds[demand.id])
    feasibility_loss += shortfall

regen_loss = regen_probs.sum()
total = lambda_infeasible * feasibility_loss + lambda_regen * regen_loss
```

Starting as `torch.zeros(1)` (not Python `0.0`) ensures a proper scalar tensor even when all demands are feasible.

**Returned metrics dict:** `feasibility_loss`, `regen_loss`, `num_regen_soft`, `num_infeasible`

---

## Component 5: `diffopt/train.py`

Entry point: `python -m diffopt.train --config configs/experiment/base.yaml`

```python
def load_qot_model(checkpoint_path, cfg, device) -> SpanAttentionQoT:
    # Loads checkpoint["model_state"] into SpanAttentionQoT

def compute_regen_tau(epoch, tau_start, tau_end, anneal_start, anneal_end) -> float:
    # Linear interpolation between anneal_start and anneal_end epochs
```

**Training loop:**
```
1. Load config, topology, ModulationConfig
2. Load QoT checkpoint → SpanAttentionQoT; freeze all parameters
3. Create: SegmentCombiner(0.5), EdgeWeightNet(), RegenPlacement(num_nodes)
4. Create: DiffONetPipeline(topology, qot_model, ...)
5. opt_edge  = Adam(edge_weight_net.parameters(), lr=cfg.training.lr_edge_net)
6. opt_regen = Adam([regen_placement.regen_logits], lr=cfg.training.lr_regen)
7. for epoch in 1..epochs_e2e:
   a. tau = compute_regen_tau(epoch, ...)
   b. demands = generate_demands(topology, num_demands, bitrate_options, seed=epoch)
   c. zero_grad both optimizers
   d. paths, gsnr_preds, regen_probs = pipeline(demands, tau=tau, lambda_=vlastelica_lambda)
   e. loss, metrics = compute_loss(gsnr_preds, demands, regen_probs, ...)
   f. loss.backward()
   g. opt_edge.step(); opt_regen.step()
   h. vlastelica_lambda = max(lambda_min, vlastelica_lambda * lambda_decay)
   i. log to CSV; print every 10 epochs
   j. save checkpoint if loss improved
```

`seed=epoch` for demand generation: each epoch sees a different demand set (avoids overfitting) but stays reproducible.

**CSV columns:** `epoch, total_loss, feasibility_loss, regen_loss, num_regen_soft, num_infeasible, tau, vlastelica_lambda`

**Checkpoint format:**
```python
{"epoch", "edge_weight_net_state", "regen_logits",
 "opt_edge_state", "opt_regen_state", "vlastelica_lambda", "total_loss"}
```

---

## Tests: `tests/test_pipeline.py`

In-memory topology construction (no JSON files):

```python
def make_linear_topology() -> Topology:
    # Nodes 0-1-2-3-4, 4 edges in a chain, no regen candidates (all degree ≤ 2)
    # Useful for single-segment identity test

def make_hub_topology() -> Topology:
    # Node 1 has degree 3 → is a regen candidate
    # Enables routing choice and regen gradient tests
```

| Test | Description |
|------|-------------|
| `test_forward_pass_shapes` | 3 demands on linear topology; assert GSNR dict size, shapes |
| `test_gradient_flow_edge_weight_net` | Hub topology, backward; all EdgeWeightNet params have nonzero grad |
| `test_gradient_flow_regen_logits` | Hub topology, backward; `regen_logits.grad.abs().sum() > 0` |
| `test_qot_frozen` | All QoT params have `grad is None` after backward |
| `test_single_segment_identity` | Linear topology (no regen candidates) → one segment; pipeline GSNR == direct QoT call within 1e-4 |
| `test_loss_backward_no_nan` | `torch.isfinite(p.grad).all()` for all trainable params |

---

## Config Additions

Add to both `configs/experiment/base.yaml` and `small_test.yaml` (existing flat keys untouched):

```yaml
# Path to pretrained QoT checkpoint (from Phase 1a training)
qot_checkpoint: checkpoints/best_qot.pt

pipeline:
  lambda_regen: 1.0
  lambda_infeasible: 10.0
  channel_loading_fraction: 0.5    # fixed channel loading during inference
  freeze_qot: true

training:
  lr_edge_net: 1.0e-3
  lr_regen: 1.0e-2
  epochs_e2e: 500                  # small_test: 20
  vlastelica_lambda: 10.0
  vlastelica_lambda_min: 1.0
  vlastelica_lambda_decay: 0.995
  regen_tau_start: 1.0
  regen_tau_end: 0.1
  regen_tau_anneal_start_epoch: 100    # small_test: 5
  regen_tau_anneal_end_epoch: 400      # small_test: 18
  checkpoint_interval: 10              # small_test: 5
```

---

## Architectural Constraints (Phase 1c additions to CLAUDE.md)

- `tau` and `lambda_` are passed per-call to `pipeline.forward()` — never stored as module attributes (prevents stale annealing state)
- `_reconstruct_path` must call `.detach()` on `path_indicator` before converting to numpy (path_indicator stays in the autograd graph)
- `segment_path` requires `demand_dst` to suppress terminal node splits (avoids empty trailing segment)
- `accum_dist_km` in span features starts at `0.0` before the first span; increments after each span
- QoT model frozen via `requires_grad_(False)` in `__init__`, not via `torch.no_grad()` context (which would sever the combiner's gradient path to `regen_logits`)
- `channel_loading_fraction = 0.5` is fixed during inference; this is a known distribution gap from training data where it was sampled uniformly in [1/48, 1.0]

---

## Verification

```bash
# Run new tests
pytest tests/test_pipeline.py -v

# Run full suite (36 existing + 6 new = 42 total)
pytest tests/ -v

# Smoke training run (requires pretrained QoT checkpoint)
python -m diffopt.train --config configs/experiment/small_test.yaml
# Expect: total_loss decreasing in logs/e2e_train_log.csv
```

**Phase 1c milestone check:** In the training CSV, verify `total_loss` is lower at epoch 20 than epoch 1, and `num_infeasible` reaches 0 at some point during the `base.yaml` run.

---

## Milestone

Phase 1c milestone:
- `test_pipeline.py` passes (6 tests)
- 42/42 total tests pass
- Training loss decreases over epochs
- Regenerator count decreases over epochs
- Infeasible demand count reaches 0

Ready for Phase 1d: evaluation, baselines, and visualization.
