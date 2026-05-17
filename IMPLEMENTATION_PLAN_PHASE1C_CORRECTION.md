# DiffONet: Implementation Plan — Phase 1c Correction (B'.2 STE)

**Last updated:** 2026-05-17
**Status:** Pre-implementation. Phase 1c assembled and runs end-to-end (40/42 tests passing, smoke training stable), but the Vlastelica surrogate is producing zero routing gradient. This document corrects that.

---

## Why a correction is needed

The diagnostic `scripts/diagnose_surrogate.py` confirmed that the Vlastelica backward returns the same path as the forward (Hamming distance = 0 across all demands). Root cause:

- `path_indicator` reaches the loss only through `path_cost_loss = λ_cost · (path_indicator · edge_weights).sum()`.
- Therefore `grad_output = ∂L/∂path_indicator = λ_cost · edge_weights`.
- Vlastelica perturbs to `c_target = w + λ · grad_output = w · (1 + λ · λ_cost)` — uniform scaling.
- Uniform scaling preserves shortest-path ordering. SPFA returns the same path. Surrogate gradient is zero.

`EdgeWeightNet` therefore only receives the *direct* `path_cost` gradient (`∂(p·w)/∂w = path_indicator`), which teaches it to lower the cost of currently-selected edges but provides no routing-change signal. The threshold and segment-combiner state of the system are invisible to the routing gradient.

The architectural reason: `gsnr_pred` has no live autograd connection to `path_indicator`. The segment structure is reconstructed from `path_indicator.detach()`, and per-segment QoT calls operate on span features built from fixed topology values — so `∂gsnr_pred/∂path_indicator ≡ 0`.

## What this correction changes

We introduce a **straight-through estimator (STE)** at the per-segment level:

- **Forward value:** unchanged — each segment's GSNR is the QoT-accurate `qot_gsnr_k`.
- **Backward gradient (w.r.t. `path_indicator`):** routed through an analytical per-edge ASE noise proxy.

Per-segment GSNR becomes:

```python
proxy_noise_k = sum over e in seg_k of  path_indicator[e] * edge_ase_noise[e]
proxy_gsnr_k  = -10.0 * log10(proxy_noise_k + eps)
segment_gsnr_k = qot_gsnr_k + (proxy_gsnr_k - proxy_gsnr_k.detach())
```

The forward equals `qot_gsnr_k` (the `proxy - proxy.detach()` pair sums to zero numerically). The backward gradient w.r.t. `path_indicator[e]` equals `∂proxy_gsnr_k/∂path_indicator[e]`, which is non-zero and **scales with the edge's analytical ASE noise**.

The downstream `SegmentCombiner` is unchanged — it still consumes segment-GSNR scalars. The `(1-p)·sum + p·soft_max` recurrence now propagates `path_indicator` gradient through the segment-GSNR inputs as well as `regen_probs` gradient through the boundary terms.

The Vlastelica `grad_output` is now non-uniform per edge:

```
grad_output[e] = λ_cost · w[e]            (path_cost term, as before)
               + (∂L/∂gsnr_pred) · (∂gsnr_pred/∂proxy_noise) · edge_ase_noise[e]
```

The second term:
- Has a different value for each edge (varies with `edge_ase_noise[e]`).
- Is zero when the demand is already feasible (`relu` derivative is zero).
- Is modulated by the soft regen state through `∂gsnr_pred/∂proxy_noise` (via the combiner recurrence).

Three things the routing gradient now sees that it did not see before: **the threshold, the per-edge noise, and the regen placement.**

---

## Files to Create / Modify

| Action | File | Purpose |
|--------|------|---------|
| Create | `diffopt/qot/edge_noise.py` | Per-edge ASE noise buffer (precomputed) |
| Modify | `diffopt/pipeline.py` | STE blend per segment; buffer registration |
| Modify | `diffopt/train.py` | Build and pass the buffer at startup |
| Modify | `tests/test_pipeline.py` | Add four STE-specific tests |
| Modify | `configs/experiment/{small_test,base}.yaml` | Lower `lambda_cost` (path_cost is no longer the sole gradient enabler) |
| Modify | `CLAUDE.md` | Document the correction in "Corrections made during Phase 1c" |
| Re-run | `scripts/diagnose_surrogate.py` | Verify Hamming > 0 |

No new modules need to be removed. `loss.py` is unchanged.

---

## Component 1: `diffopt/qot/edge_noise.py` (new)

A small module that computes a fixed per-edge linear ASE noise vector from the topology.

```python
def compute_edge_ase_noise(topology: Topology) -> torch.Tensor:
    """Per-edge linear ASE noise (analytical, normalized).

    For each span: G_ase_span = NF_lin · (10^(α·L/10) - 1)
    Per-edge:      sum over spans of G_ase_span.
    Returns float32 tensor of shape (E,) on CPU.

    Constants h·ν·Δf are absorbed: only relative magnitudes are needed for the
    Vlastelica perturbation. Values are normalized so the median edge has noise=1.
    """
```

Reuses the per-span ASE formula already present in `diffopt/qot/gnpy_bridge.py::analytical_gsnr_db` (lines 188–192). No new physics. NLI is intentionally **excluded** from the proxy — it depends on path length and channel count, which we do not yet model per-edge. ASE alone gives the right sign and approximate magnitudes for gradient direction.

The output is registered as a buffer on `DiffONetPipeline`; it is fixed for the lifetime of the model.

---

## Component 2: `diffopt/pipeline.py` (modify)

### Constructor

Add `edge_ase_noise: torch.Tensor` argument. Register as buffer:

```python
self.register_buffer("_edge_ase_noise", edge_ase_noise)
```

Add a small epsilon constant for the log:
```python
self._proxy_eps = 1e-12
```

### `forward` — per-segment STE assembly

After the existing QoT call inside the per-segment loop (currently lines 273–278), wrap the result in the STE blend:

```python
for seg_edge_ids in segments:
    span_feats, padding_mask = self._extract_span_features(seg_edge_ids, device)
    qot_gsnr = self.qot_model(span_feats, padding_mask)[0]   # forward value

    # Analytical proxy — live in path_indicator
    seg_noise_idx = torch.tensor(seg_edge_ids, dtype=torch.long, device=device)
    proxy_noise = (path_indicator[seg_noise_idx]
                   * self._edge_ase_noise[seg_noise_idx]).sum()
    proxy_gsnr = -10.0 * torch.log10(proxy_noise + self._proxy_eps)

    # STE blend: forward value = qot_gsnr, gradient direction from proxy
    segment_gsnr = qot_gsnr + (proxy_gsnr - proxy_gsnr.detach())
    segment_gsnrs.append(segment_gsnr)
```

No changes downstream — `SegmentCombiner` already accepts a list of scalar GSNR tensors. The segment_combiner backward will now correctly propagate gradient through `segment_gsnr → proxy_gsnr → path_indicator`.

### Failure modes to guard

- **Empty segment.** `segment_path` should not produce an empty segment given the `demand_dst` guard added in Phase 1c. If `seg_edge_ids` were ever empty, `proxy_noise = 0` and `log10(eps) = -12` → `proxy_gsnr` very large. Add an assertion `assert seg_edge_ids` at the top of the loop.
- **NaN/Inf from log.** Bounded by `+ eps`; not a real risk with positive `_edge_ase_noise`.

---

## Component 3: `diffopt/train.py` (modify)

Two-line change after the topology is loaded:

```python
from diffopt.qot.edge_noise import compute_edge_ase_noise

edge_ase_noise = compute_edge_ase_noise(topology).to(device)

pipeline = DiffONetPipeline(
    topology=topology,
    qot_model=qot_model,
    segment_combiner=segment_combiner,
    edge_weight_net=edge_weight_net,
    regen_placement=regen_placement,
    edge_ase_noise=edge_ase_noise,                      # NEW
    channel_loading_fraction=cfg["pipeline"]["channel_loading_fraction"],
    max_spans=cfg.get("max_spans_per_segment", 60),
).to(device)
```

`tests/test_pipeline.py::make_pipeline` is updated similarly.

---

## Component 4: `tests/test_pipeline.py` (extend)

Four new tests, none replacing existing ones. All six current tests must continue to pass.

### Test 7: STE preserves forward value

```python
def test_ste_forward_value_matches_qot():
    """gsnr_pred from STE pipeline equals gsnr_pred from QoT-only pipeline."""
```
Construct a hub topology, run the pipeline twice with identical state — once with the STE blend, once with the proxy term zeroed (or use a small helper that switches the blend off). The forward `gsnr_pred` values must match to within float32 precision (`< 1e-4`).

### Test 8: path_indicator gradient is non-zero and non-uniform

```python
def test_path_indicator_gradient_nonzero():
    """∂feasibility_loss/∂path_indicator is non-zero AND varies across edges."""
```
On the hub topology with a high-bitrate demand (forced infeasible), register a hook on `path_indicator`, run `loss.backward()`. Assert:
- `path_indicator.grad` exists and has non-zero norm.
- `path_indicator.grad.std() > 0` (gradient is non-uniform — captures the per-edge noise variation).

### Test 9: Vlastelica produces non-zero EdgeWeightNet gradient via per-edge perturbation

```python
def test_edge_weight_net_grad_via_per_edge_perturbation():
    """In a topology with two paths of differing noise, the surrogate produces a non-uniform grad on EdgeWeightNet."""
```
Construct a topology with two parallel paths of equal hop count but unequal total length (one "noisy" path, one "clean"). Run the pipeline with one infeasible demand. Verify that:
- `EdgeWeightNet.parameters().grad` is non-zero (existing Test 2 already covers this loosely).
- The grad differs from what you would get with a zeroed proxy (sanity that the surrogate term contributes, not just the direct `path_cost` term). This can be checked by snapshotting two grads and asserting they differ.

### Test 10: feasible demand contributes zero feasibility gradient

```python
def test_feasible_demand_no_routing_signal():
    """For a demand whose path is already above threshold, feasibility_loss provides zero gradient to EdgeWeightNet."""
```
Use a linear topology and a low-bitrate demand for which any path is feasible. After `backward()`, the gradient norm on EdgeWeightNet parameters coming from `feasibility_loss` alone (zero out `path_cost_loss` and `regen_loss` for this test) must be zero. Validates that the threshold relu correctly gates the new gradient path.

---

## Configuration changes

`configs/experiment/small_test.yaml` and `configs/experiment/base.yaml`:

```yaml
pipeline:
  lambda_cost: 0.01        # was 0.1; demoted to a regularizer
```

Rationale: `path_cost_loss` is no longer the sole gradient enabler for `EdgeWeightNet`. Keeping `lambda_cost = 0.1` would let the path-cost term dominate the surrogate `grad_output` and re-create a near-uniform perturbation (the same failure mode in milder form). Reducing it by an order of magnitude lets the per-edge ASE noise term dominate the routing signal while still serving as a length regularizer when the system is feasible.

No new hyperparameters are introduced by the STE itself. The proxy is parameter-free.

---

## Verification milestones

In order:

1. **Unit tests pass.** `pytest tests/test_pipeline.py -v` → 10/10 (existing 6 + new 4).
2. **Diagnostic confirms surrogate fires.** Re-run `python scripts/diagnose_surrogate.py --config configs/experiment/small_test.yaml`. Expect:
   - At least one demand with Hamming > 0 (typically all infeasible demands).
   - `g/w ratio std` > 0 (gradient is no longer proportional to edge weights).
   - Reported `EdgeWeightNet` total grad norm noticeably larger than the previous run.
3. **Smoke training run improves feasibility.** `python -m diffopt.train --config configs/experiment/small_test.yaml` for 20 epochs. Expect:
   - `feasibility_loss` decreasing monotonically (modulo demand re-sampling noise).
   - `num_infeasible` trending down.
   - `regen_loss` and `num_regen_soft` still decreasing (regen pathway not broken by the change).

If any of these fail, the failure mode points at a specific component (test → unit-level; diagnostic → autograd graph; smoke run → loss-balance / hyperparameter).

---

## Out of scope for this correction

- **B'.1 (replace QoT entirely with the analytical proxy).** Loses QoT modeling fidelity in the forward pass. Not justified given STE is cheap.
- **B'.3 (soft-membership span features into QoT).** Would require retraining QoT on fractional-membership inputs — high risk, no clear gain.
- **Option B (per-edge QoT evaluation).** O(E) extra QoT calls per demand. Too expensive.
- **Per-edge NLI in the proxy.** Edge-local NLI requires per-edge channel-loading context. ASE-only is a reasonable first cut; NLI can be added later if smoke runs reveal routing biases in NLI-dominated regimes.
- **Removing `path_cost_loss`.** Keep it as a small-weight regularizer; it provides routing gradient when all demands are feasible (where the STE term zeros out via the relu).

---

## Risks and notes

- **STE bias.** The backward gradient direction comes from the analytical proxy, not the true `∂qot_gsnr/∂path_indicator`. The two agree in sign for ASE-dominated regimes but can disagree in magnitude. The forward GSNR is QoT-accurate, so the *decision* of whether to optimize is correct; only the *direction* of optimization is approximate.
- **Normalization.** Normalize the proxy noise buffer so the median edge has `edge_ase_noise = 1`. Without normalization, raw ASE values can vary across many orders of magnitude (long edges with high NF dominate), which would couple the routing-gradient scale to the topology and force `λ` re-tuning per dataset. With median-normalization, `lambda_cost` and the implicit `lambda_diff` (which here equals 1) stay topology-agnostic.
- **Detach correctness.** The `proxy_gsnr - proxy_gsnr.detach()` pattern is sensitive to ordering: both terms must share the same forward value within one forward pass. Writing it as a single expression on a single line is safer than splitting across statements.
- **Eps choice.** `1e-12` is far below any physically plausible per-segment noise. It exists only to prevent `log10(0)` in degenerate test cases.

---

## Documentation updates after implementation

Append to `CLAUDE.md` under **Corrections made during Phase 1c**:

> 6. **Zero Vlastelica surrogate gradient (B'.2 STE)**: With only `path_cost_loss` providing `grad_output`, the Vlastelica perturbation `c_target = w·(1+λ·λ_cost)` is a uniform scaling and the surrogate returns zero gradient. Fix: introduce a straight-through estimator at the per-segment GSNR level, blending the QoT-accurate forward value with an analytical per-edge ASE noise proxy that supplies the backward gradient direction. EdgeWeightNet now receives a per-edge, threshold-gated, regen-modulated routing signal.

Append to **Architectural constraints — Phase 1c additions**:

> - `edge_ase_noise` is a fixed buffer on `DiffONetPipeline`, precomputed from the analytical GN formula and median-normalized. NLI is excluded by design. The buffer is referenced by the STE blend in `forward` and must not be moved off the model's device.
> - The STE expression `qot_gsnr + (proxy_gsnr - proxy_gsnr.detach())` must be written on a single line. Splitting it across statements risks inserting a graph break that defeats the gradient redirection.
