# DiffONet: Implementation Plan — Phase 1c (End-to-End Pipeline)

**Last updated:** 2026-05-10
**Status:** Revised pre-implementation (Phase 1b complete, 36/36 tests passing)

---

## Context

Phase 1b produced the differentiable segment combiner and Vlastelica surrogate routing layer, both unit-tested in isolation. Phase 1c assembles all components into a jointly trainable pipeline: the edge weight network biases routing, the surrogate Dijkstra selects a path, the path is segmented at regenerator candidate nodes, the frozen QoT model predicts GSNR per segment, and the segment combiner merges them using soft regenerator probabilities. Gradient descent then jointly updates routing costs and regenerator placement.

Spec reference: §5 (EdgeWeightNet), §6 (RegenPlacement), §7 (Pipeline), §8 (Loss), §9 (Training loop), Testing Strategy §test_pipeline.

**Corrections applied during plan review (2026-05-10):**

1. **Path cost term required for EdgeWeightNet gradient.** `path_indicator` is consumed only via `.detach()` in `_reconstruct_path`, so `∂L/∂path_indicator = 0`. With zero gradient into the Vlastelica backward, the perturbed weights equal the original weights, the perturbed solve returns the same path, and the surrogate gives zero gradient to `EdgeWeightNet` — `test_gradient_flow_edge_weight_net` would silently fail. Fix: add `path_cost_loss = Σ_demands (path_indicator · edge_weights).sum()` to `compute_loss`. This creates a live autograd path `loss → path_indicator → edge_weights → EdgeWeightNet.params`, making `∂L/∂path_indicator = edge_weights` (always positive via Softplus), so the Vlastelica backward finds a genuinely different perturbed path and produces a nonzero surrogate gradient. Pipeline now returns `path_costs: Dict[int, Tensor]` and `compute_loss` gains `path_costs` + `lambda_cost` parameters.

2. **QoT output shape.** `SpanAttentionQoT.forward` returns `(batch,)`. Called with `batch=1`, the output is shape `(1,)`, not a scalar. `SegmentCombiner` expects a list of scalar tensors. Fix: `gsnr_scalar = qot_model(span_feats, mask)[0]`.

3. **Threshold tensor conversion.** `ModulationConfig.required_snr_threshold` returns a Python `float`. The `relu(threshold - gsnr_pred)` expression requires a tensor. Fix: `threshold_t = torch.tensor(threshold, device=regen_probs.device, dtype=torch.float32)`.

4. **`FIBER_TYPE_INDEX` import.** Span feature extraction uses `fiber_type_idx`, which requires `FIBER_TYPE_INDEX` from `diffopt.topology`. Not in the original plan.

5. **Config access style.** `yaml.safe_load` returns a plain `dict`. The training loop must use `cfg["training"]["lr_edge_net"]` not `cfg.training.lr_edge_net`.

6. **Hub topology concrete definition.** Original plan said "Node 1 has degree 3" without specifying the full graph. Concrete definition added below.

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
   (detach before numpy — path_indicator stays live in the autograd graph for path_cost)
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
    ) -> Tuple[Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Tensor]:
        # Returns: (path_costs, gsnr_preds, path_indicators, regen_probs)
        # path_costs:     Dict[demand_id → scalar Tensor] — live in autograd graph
        # gsnr_preds:     Dict[demand_id → scalar Tensor]
        # path_indicators: Dict[demand_id → (E,) Tensor]  — for diagnostics
        # regen_probs:    (num_nodes,) Tensor
```

**Forward pass:**
```
1. regen_probs = regen_placement.get_regen_probs(tau)                      # (num_nodes,)
2. edge_feats = cat([_topo_edge_features,                                   # (E, 7)
                     regen_probs[_edge_src_ids].unsqueeze(1),
                     regen_probs[_edge_dst_ids].unsqueeze(1)], dim=1)
3. edge_weights = edge_weight_net(edge_feats).squeeze(-1)                   # (E,) via Softplus > 0
4. per demand:
   a. path_indicator = surrogate_shortest_path(edge_weights, _edge_index,
                                               demand.src, demand.dst,
                                               num_nodes, lambda_=lambda_)
   b. path_cost = (path_indicator * edge_weights).sum()          # scalar, live in graph
   c. ordered_edges = _reconstruct_path(path_indicator, demand.src, demand.dst)
   d. segments, boundary_nodes = segment_path(ordered_edges, demand.src,
                                              _regen_candidate_set, topology, demand.dst)
   e. for each segment:
        span_feats (1, 60, 5), mask (1, 60) from topology constants (see below)
        gsnr_scalar = qot_model(span_feats, mask)[0]             # (1,) → scalar
      segment_gsnrs = list of gsnr_scalar tensors
   f. boundary_probs = [regen_probs[n] for n in boundary_nodes]
   g. path_gsnr = segment_combiner(segment_gsnrs, boundary_probs)
   h. store path_cost, path_gsnr, path_indicator for this demand
```

**Span feature extraction for one segment:**
```python
# Requires: from diffopt.topology import FIBER_TYPE_INDEX
accum_dist = 0.0
rows = []
for eid in segment_edge_ids:
    edge = topology.edges[eid]
    ftype_idx = float(FIBER_TYPE_INDEX.get(edge.fiber_type, 0))
    for span_idx in range(edge.num_spans):
        rows.append([
            edge.span_lengths_km[span_idx],
            ftype_idx,
            edge.amplifier_nf_db[span_idx],   # per-span NF, not mean
            channel_loading_fraction,
            accum_dist,
        ])
        accum_dist += edge.span_lengths_km[span_idx]
n_spans = len(rows)
span_feats = torch.zeros(1, max_spans, 5, device=device)
span_feats[0, :n_spans] = torch.tensor(rows, dtype=torch.float32, device=device)
padding_mask = torch.zeros(1, max_spans, dtype=torch.bool, device=device)
padding_mask[0, :n_spans] = True          # True = real span (opposite of PyTorch convention)
```

Device: derive from `_topo_edge_features` (registered buffer).

**Gradient flow:**
- `regen_logits` gradient: flows via `regen_probs[boundary_nodes]` → `segment_combiner` → loss
- `edge_weight_net` gradient: flows via `path_cost = (path_indicator * edge_weights).sum()` → `loss` → `∂L/∂path_indicator = edge_weights` (nonzero via Softplus) → Vlastelica backward → `edge_weights` → `edge_weight_net.params`
- QoT model: frozen via `requires_grad_(False)` at construction. Do NOT use `torch.no_grad()` — that disables autograd engine-wide for everything in scope, risking accidental detachment of `regen_probs` if the scope is too wide.

---

## Component 4: `diffopt/loss.py`

```python
def compute_loss(
    gsnr_preds: Dict[int, Tensor],
    path_costs: Dict[int, Tensor],         # demand_id → scalar, live in autograd
    demands: List[Demand],
    regen_probs: Tensor,                   # (num_nodes,)
    modulation_config: ModulationConfig,
    lambda_regen: float = 1.0,
    lambda_infeasible: float = 10.0,
    lambda_cost: float = 0.1,
) -> Tuple[Tensor, dict]:
```

**Logic:**
```python
device = regen_probs.device

feasibility_loss = torch.zeros(1, device=device)   # tensor, not float 0
num_infeasible = 0
for demand in demands:
    threshold = modulation_config.required_snr_threshold(demand.bitrate_gbps)
    threshold_t = torch.tensor(threshold, device=device, dtype=torch.float32)
    shortfall = F.relu(threshold_t - gsnr_preds[demand.id])
    feasibility_loss = feasibility_loss + shortfall
    if shortfall.item() > 0:
        num_infeasible += 1

regen_loss = regen_probs.sum()
path_cost_loss = sum(path_costs.values())          # Σ (path_indicator · edge_weights)
total = (lambda_infeasible * feasibility_loss
         + lambda_regen * regen_loss
         + lambda_cost * path_cost_loss)
```

`torch.zeros(1)` (not Python `0.0`) ensures a proper scalar tensor when all demands are feasible. `path_cost_loss` is what activates the Vlastelica surrogate — without it, `∂L/∂path_indicator = 0` and `EdgeWeightNet` receives zero gradient.

**Returned metrics dict:**
```python
{
    "feasibility_loss": feasibility_loss.item(),
    "regen_loss": regen_loss.item(),
    "path_cost_loss": path_cost_loss.item(),
    "num_regen_soft": (regen_probs > 0.5).sum().item(),
    "num_infeasible": num_infeasible,
}
```

---

## Component 5: `diffopt/train.py`

Entry point: `python -m diffopt.train --config configs/experiment/base.yaml`

```python
def load_qot_model(checkpoint_path: str, cfg: dict, device) -> SpanAttentionQoT:
    # ckpt = torch.load(checkpoint_path, map_location=device)
    # model = SpanAttentionQoT(feature_dim=cfg["feature_dim"], ...)
    # model.load_state_dict(ckpt["model_state"])

def compute_regen_tau(epoch: int, tau_start: float, tau_end: float,
                      anneal_start: int, anneal_end: int) -> float:
    # Linear interpolation; clamp to [tau_end, tau_start]
```

**Training loop:**
```python
cfg = yaml.safe_load(Path(args.config).read_text())
topology = load_topology(cfg["topology"])
mod_cfg  = ModulationConfig.from_yaml(cfg["modulation_formats"])

qot_model = load_qot_model(cfg["qot_checkpoint"], cfg, device)
# freeze already done inside load_qot_model via requires_grad_(False)

segment_combiner = SegmentCombiner(soft_max_temperature=0.5)
edge_weight_net  = EdgeWeightNet()
regen_placement  = RegenPlacement(topology.num_nodes)
pipeline = DiffONetPipeline(topology, qot_model, segment_combiner,
                             edge_weight_net, regen_placement,
                             channel_loading_fraction=cfg["pipeline"]["channel_loading_fraction"])

opt_edge  = Adam(edge_weight_net.parameters(),    lr=cfg["training"]["lr_edge_net"])
opt_regen = Adam([regen_placement.regen_logits],  lr=cfg["training"]["lr_regen"])

vlastelica_lambda = cfg["training"]["vlastelica_lambda"]
lambda_min        = cfg["training"]["vlastelica_lambda_min"]
lambda_decay      = cfg["training"]["vlastelica_lambda_decay"]

for epoch in range(1, cfg["training"]["epochs_e2e"] + 1):
    tau = compute_regen_tau(
        epoch,
        cfg["training"]["regen_tau_start"],
        cfg["training"]["regen_tau_end"],
        cfg["training"]["regen_tau_anneal_start_epoch"],
        cfg["training"]["regen_tau_anneal_end_epoch"],
    )
    demands = generate_demands(
        topology,
        cfg["num_demands"],
        cfg["bitrate_options"],
        seed=epoch,        # different set each epoch; reproducible
    )

    opt_edge.zero_grad()
    opt_regen.zero_grad()

    path_costs, gsnr_preds, _, regen_probs = pipeline(
        demands, tau=tau, lambda_=vlastelica_lambda
    )
    loss, metrics = compute_loss(
        gsnr_preds, path_costs, demands, regen_probs, mod_cfg,
        lambda_regen=cfg["pipeline"]["lambda_regen"],
        lambda_infeasible=cfg["pipeline"]["lambda_infeasible"],
        lambda_cost=cfg["pipeline"]["lambda_cost"],
    )

    loss.backward()
    opt_edge.step()
    opt_regen.step()

    vlastelica_lambda = max(lambda_min, vlastelica_lambda * lambda_decay)
    # log to CSV; print every 10 epochs; save checkpoint if loss improved
```

`seed=epoch` for demand generation: each epoch sees a different demand set (avoids overfitting) but stays reproducible.

**CSV columns:** `epoch, total_loss, feasibility_loss, regen_loss, path_cost_loss, num_regen_soft, num_infeasible, tau, vlastelica_lambda`

**Checkpoint format:**
```python
{
    "epoch": epoch,
    "edge_weight_net_state": edge_weight_net.state_dict(),
    "regen_logits": regen_placement.regen_logits.detach().cpu(),
    "opt_edge_state": opt_edge.state_dict(),
    "opt_regen_state": opt_regen.state_dict(),
    "vlastelica_lambda": vlastelica_lambda,
    "total_loss": loss.item(),
}
```

---

## Tests: `tests/test_pipeline.py`

In-memory topology construction (no JSON files):

```python
def make_linear_topology() -> Topology:
    # Chain: 0—1—2—3—4 (4 edges, all nodes degree ≤ 2 → no regen candidates)
    # Useful for single-segment identity test
    def make_edge(src, dst):
        return Edge(src=src, dst=dst, length_km=80.0, num_spans=1,
                    span_lengths_km=[80.0], fiber_type="SSMF", amplifier_nf_db=[5.0])
    edges = [make_edge(i, i + 1) for i in range(4)]
    return Topology(nodes=[{"id": i} for i in range(5)], edges=edges)


def make_hub_topology() -> Topology:
    # 5 nodes, 5 edges:
    #   0—1 (eid 0), 0—2 (eid 1), 1—3 (eid 2), 2—3 (eid 3), 3—4 (eid 4)
    # Degrees: 0→2, 1→2, 2→2, 3→3, 4→1
    # Node 3 has degree 3 → sole regen candidate
    # Two paths from 0 to 4: 0→1→3→4 and 0→2→3→4
    #   → routing choice activates EdgeWeightNet gradient
    #   → boundary at node 3 activates regen_logits gradient
    def make_edge(src, dst):
        return Edge(src=src, dst=dst, length_km=80.0, num_spans=1,
                    span_lengths_km=[80.0], fiber_type="SSMF", amplifier_nf_db=[5.0])
    edges = [
        make_edge(0, 1),   # eid 0
        make_edge(0, 2),   # eid 1
        make_edge(1, 3),   # eid 2
        make_edge(2, 3),   # eid 3
        make_edge(3, 4),   # eid 4
    ]
    return Topology(nodes=[{"id": i} for i in range(5)], edges=edges)
```

| Test | Description |
|------|-------------|
| `test_forward_pass_shapes` | 3 demands on linear topology; assert `gsnr_preds` dict size, scalar shapes |
| `test_gradient_flow_edge_weight_net` | Hub topology, demand 0→4, backward; all EdgeWeightNet params have nonzero grad (requires `path_cost_loss` in loss) |
| `test_gradient_flow_regen_logits` | Hub topology, demand 0→4, backward; `regen_logits.grad.abs().sum() > 0` (boundary at node 3) |
| `test_qot_frozen` | All QoT params have `grad is None` after backward |
| `test_single_segment_identity` | Linear topology, demand 0→4 → one segment; pipeline GSNR == direct QoT call within 1e-4 |
| `test_loss_backward_no_nan` | `torch.isfinite(p.grad).all()` for all trainable params after backward |

---

## Config Additions

Add to both `configs/experiment/base.yaml` and `small_test.yaml` (existing flat keys untouched):

```yaml
# Path to pretrained QoT checkpoint (from Phase 1a training)
qot_checkpoint: checkpoints/best_qot.pt

pipeline:
  lambda_regen: 1.0
  lambda_infeasible: 10.0
  lambda_cost: 0.1               # weight on path_cost_loss; enables EdgeWeightNet gradient
  channel_loading_fraction: 0.5  # fixed channel loading during inference
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
- `_reconstruct_path` uses `path_indicator.detach()` for path reconstruction, but `path_indicator` must remain live in the autograd graph — it is used in `path_cost = (path_indicator * edge_weights).sum()` which is the sole gradient path into `EdgeWeightNet`
- `path_cost_loss = Σ (path_indicator · edge_weights)` is not optional regularization — it is mechanically required for `∂L/∂path_indicator` to be nonzero, which in turn is required for the Vlastelica surrogate to produce a nonzero gradient to `EdgeWeightNet`
- `segment_path` requires `demand_dst` to suppress terminal node splits (avoids empty trailing segment)
- `accum_dist_km` in span features starts at `0.0` before the first span; increments after each span
- `amp_nf_db` in span features is `edge.amplifier_nf_db[span_idx]` (per-span), not `edge.mean_amp_nf_db`
- QoT model frozen via `requires_grad_(False)` in `__init__`, not via `torch.no_grad()` (which disables autograd engine-wide for its scope, risking accidental detachment of `regen_probs` if the scope boundary slips)
- `SpanAttentionQoT.forward` returns `(batch,)` — index `[0]` to extract scalar before passing to `SegmentCombiner`
- `channel_loading_fraction = 0.5` is fixed during inference; this is a known distribution gap from training data where it was sampled uniformly in [1/48, 1.0]
- Config files are loaded as plain `dict` via `yaml.safe_load`; use `cfg["training"]["lr_edge_net"]` not `cfg.training.lr_edge_net`

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
