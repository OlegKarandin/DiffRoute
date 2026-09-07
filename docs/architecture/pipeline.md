# Pipeline architecture

This document covers the whole system in the order it was built: first the
standalone QoT surrogate that is trained on its own and then frozen, then the
end-to-end pipeline that consumes it, then a worked-by-hand walkthrough of the
one piece of that pipeline that is not ordinary autograd (the surrogate
gradient through Dijkstra), and finally the hyperparameter reference.

## Part 1 — the QoT surrogate

One model, `SpanAttentionQoT` (`diffopt/qot/model.py`), trained by itself
before any end-to-end training happens. It takes the physical parameters of a
single transparent segment — a sequence of fiber spans with amplifiers — and
predicts GSNR at a fixed channel under test (CUT). No routing, no placement,
no cross-segment state; those all live in Part 2, which calls this model
frozen.

### Data flow

```
.dat files
  → topology_builder.py       (one-time; produces the committed topology JSONs)
  → generate_qot_dataset.py   (produces parquet; real GNPy only, no fallback)
  → SegmentQoTDataset         (loads parquet; reshapes to (max_spans=60, 5))
  → SpanAttentionQoT          (predicts GSNR)
  → train_qot.py              (MSE loss, Adam, cosine LR)
```

### Architecture

```
Input: (B, 60, 5)   — batch of segments, up to 60 spans, 5 features/span

Linear(5 → 64)      — shared projection; Xavier-uniform weight, zero bias
  +
Embedding(60, 64)   — learned positional encoding, one embedding per span
                      index; normal(std=0.02) init

TransformerEncoder  — 2 layers, 4 heads, d_ff=128, dropout=0, batch_first=True

Masked mean pool    — average over real spans only (padding_mask=True for real)

MLP: Linear(64 → ff_dim//2=64) → ReLU → Linear(64 → 1)   → scalar GSNR (dB)
```

The five per-span features are fixed in one place,
`diffopt/qot/span_features.py`: `[span_length_km, fiber_type_idx, amp_nf_db,
channel_loading_fraction, accum_dist_km]`. That ordering is an architectural
invariant — the on-disk parquet schema (`span_features_0` ..
`span_features_{max_spans*5-1}`) is laid out against it, so the generator, the
dataset loader and the live pipeline must all agree.

**Why a transformer (not an LSTM or an MLP over aggregate stats).** GSNR at a
span chain's output depends on noise accumulated *in order*. A span's
contribution to NLI depends on its position in the chain, since NLI from
earlier spans is re-amplified downstream. The transformer lets each span
attend to every other span, and the learned positional encoding captures the
physical meaning of position (noise accumulates directionally). An MLP over
aggregate statistics — total length, mean NF — would discard both per-span
variance and the ordering structure.

**Why a learned positional encoding (not sinusoidal).** Sinusoidal encodings
are designed for arbitrary-length sequences. Here `max_spans=60` is fixed, and
the physical semantics of "span 3 of 5" versus "span 3 of 40" genuinely
differ. Learned embeddings fit those patterns directly from data.

**Why masked mean pool (not a CLS token).** Sequences are variable-length with
explicit padding. Mean pooling over real spans directly computes the average
encoded representation, a natural aggregation for a quantity that depends on
all spans roughly equally. A CLS token would require the model to *learn* to
funnel information into one position; mean pool supplies that for free.

**Why `dropout=0`.** With 50k training samples and a small model (64-dim, 2
layers), dropout regularisation is likely unnecessary and adds a tuning
dimension. It is also load-bearing downstream: the pipeline memoizes this
model's outputs per segment (Part 2, step 4), which is only sound because the
model is deterministic — `SpanAttentionQoT` hardcodes `dropout=0.0` and has no
other train/eval-dependent layer, so `requires_grad_(False)` plus that
hardcoding is what makes the memo safe.

### Training

- Loss: MSE over predicted vs GNPy-simulated GSNR (dB)
- Optimizer: Adam, `lr=1e-3`
- Schedule: cosine annealing over all epochs (no warmup)
- Checkpoint: saved whenever val RMSE improves, to `checkpoints/best_qot.pt`
- Log: `logs/train_log.csv` — epoch, train_mse, val_rmse

Measured on `base.yaml` (`ind_132`, 50k train / 10k val, 100 epochs): best val
RMSE **0.1909 dB**, against a dataset GSNR range of ~6.5-20.0 dB — about 1.4%
relative error. Train MSE tracked val closely for the whole run, with no
overfitting. That number is a fairly honest generalisation estimate: only ~7%
of val rows have any exact-feature duplicate in train on `ind_132` (it was
~77% on the smaller `german_17`, where the same metric was largely a
lookup-table artifact — which is why the default topology changed).

### Dataset-generation decisions

**One `n_channels` per segment, not per span.** GNPy simulates a WDM comb
propagating through a span chain; every span in the segment sees the same set
of active channels. Varying `n_channels` per span would be physically
incoherent. `n_channels` is drawn uniformly from `[1, num_channels_cband]`
(48) once per segment, and reaches the model only as `channel_loading_fraction
= n_channels / 48`, one of the five per-span features.

**Which channels are lit is deterministic, not random.**
`optical_bridge.build_loading` grows the comb outward from the CUT (`CUT`,
`CUT+1`, `CUT-1`, `CUT+2`, …, skipping off-grid candidates), so a given
`n_channels` always produces the same slot set. NLI from a neighbouring
channel falls off with spectral distance, so a CUT-centered comb is the
physically meaningful way to grow load; an arbitrary block at one edge of the
grid would leave the CUT's immediate neighbourhood empty at low loads and
understate cross-phase effects. A consequence worth naming: for `n_channels ≥
2` the CUT is not first in tuple order, so `compute_qot` must be told which
carrier to probe via `center_freq_hz`.

**One `mode_id`, fixed for the whole run.** GSNR is mode-invariant here — all
eleven modulation formats share 87.5 GBaud and 0.15 roll-off, and
`tests/test_optical_bridge.py` pins that GSNR is identical across all eleven.
The generator therefore fixes `mode_id` once rather than resampling per
segment, which would add variance without adding information.

**Segment lengths target a measured regenerator reach.**
`split_path_into_segments` walks a k-shortest path accumulating real per-edge
distance and splits at the first regen candidate reached past a target reach
sampled fresh per segment from `[min_regen_reach_km, max_regen_reach_km]` =
`[250, 3700]` km. That range is measured, not a rule of thumb: real coherent
reach against this project's own `modulation_formats.yaml`, via real GNPy on a
synthetic 80 km-span chain, runs from ~320 km (800 Gbps, 15.1 dB threshold) to
beyond 3600 km (300 Gbps, 4.8 dB) at full loading. The earlier policy — split
at 0-2 randomly chosen regen candidates regardless of resulting distance —
produced segments far longer than realistic regenerator spacing allows on
`ind_132`'s diameter.

**`accum_dist_km` is the distance to the *start* of each span.** It begins at
`0.0` before the first span; the value recorded for span *i* is the
accumulated length of all *prior* spans, and the running total increments only
after the row is written. Combined with the positional encoding, this gives
the model both index-based and distance-based position. The two are not
redundant: the positional encoding captures discrete order, `accum_dist_km`
captures physical length accumulation, which is what drives ASE scaling.
Accumulated distance resets at every segment boundary in the live pipeline.

**No GNPy fallback.** `diffopt/qot/optical_bridge.py::segment_gsnr_db` has no
`try`/`except` around the GNPy call, and an AST test
(`tests/test_optical_bridge.py::test_bridge_source_contains_no_try_except`)
enforces that no `try` or `except` exists anywhere in the module. Every label
in the dataset is a real GNPy-derived GSNR; a GNPy failure raises and
generation stops rather than silently substituting an analytical
approximation. This is a correction, not a design choice: for the project's
entire history before this migration, the old bridge silently fell back to the
analytical GN model on any GNPy exception, so every prior dataset's labels
came from that fallback and real GNPy never once executed. See
`docs/architecture/invariants.md`'s "Upstream dependency" section for the full
story.

## Part 2 — the end-to-end pipeline

The forward pass (`diffopt/pipeline.py`, `DiffONetPipeline.forward`) chains
these differentiable/quasi-differentiable stages:

```
free per-edge weight → surrogate Dijkstra → segment at regen candidates
  → frozen QoT surrogate per segment (batched) → AllocationHead rollout
  → SegmentCombiner (exact fold) → constrained loss
```

**1. Every edge gets a free, directly-learned weight.** `self.edge_log_weight`
(`diffopt/pipeline.py`), an `nn.Parameter` of shape `(E,)`, one raw number per
edge, passed through `Softplus` to stay strictly positive: `u =
softplus(edge_log_weight)`. Initialized at length-proportional weights (`km /
mean(km)`, i.e. shortest-by-km routing — the documented baseline
`preflight_filter` screens against), not at zero: a uniform init would tie
every edge and hand routing entirely to Dijkstra's tie-break order.

An earlier design (`EdgeWeightNet`) computed `u` from a small MLP over 5
static topology features plus the regenerator probability at each edge's two
endpoints, the only channel through which it saw `RegenPlacement`'s (the
placement design described below has since replaced) per-node state.
Per-(demand, boundary) allocation has no single per-node probability left to
feed it, and with the other features held constant, `EdgeWeightNet(constant)`
was just a fixed function of its own ~5000 parameters — a costlier
reparameterization of `E` numbers, not a source of extra signal.
`edge_log_weight` replaces it directly; the class has since been removed from
the codebase.

The pipeline then renormalizes edge costs to unit mean *without detaching
the divisor* (`w = u / mean(u)`, `diffopt/pipeline.py`). Concretely: if `u =
[2, 4, 6]`, then `mean(u) = 4` and `w = [0.5, 1.0, 1.5]`. Scale every raw
cost by *any* constant `k > 0` — say `k = 1000`, giving `u' = [2000, 4000,
6000]` — and `w' = u'/mean(u') = [0.5, 1.0, 1.5]`, identical to `w`. So
Dijkstra's chosen path, and every downstream loss term computed from `w`,
are completely unaffected by uniformly scaling the parameter's raw output up
or down.

This matters because an earlier version of the loss did *not* have that
property: a term (`path_cost_loss`, since replaced — see step 7) summed
`path_indicator · edge_weights` directly, without normalizing first, so it
scaled linearly with `u` — doubling every `u` doubled that term. Nothing
opposed shrinking `u` toward zero: the routing decision (Dijkstra's argmin)
doesn't care about absolute scale, so shrinking cost nothing there, but it
directly reduced this one loss term, and gradient descent found and exploited
that free direction. On a real training run (`ind_132`, 60 epochs) 86% of
edge weights fell below `1e-6` (median `5.7e-11`), while the *ranking* of
edges by weight barely moved (Spearman rank-correlation with the random
initialization stayed at `+0.999`) — confirming this was pure scale collapse,
not the network learning anything about routing; routes ended up 2.07x longer
than shortest-by-km, no better than before training.

`w = u / mean(u)` fixes this structurally: it makes the whole loss
homogeneous of degree 0 in `u` (scaling `u` by any `k` leaves `w`, and
therefore the loss, byte-for-byte unchanged), so by Euler's homogeneous-
function theorem `Σᵢ uᵢ · ∂L/∂uᵢ ≡ 0` — the "shrink everything" gradient
direction is annihilated as an algebraic identity, not merely discouraged by
a competing term.

**2. A surrogate Dijkstra routes each demand.** `diffopt/routing/surrogate.py`
wraps Dijkstra in a `torch.autograd.Function` called `DijkstraSurrogate` —
named for its forward pass, which runs real, exact Dijkstra
(`diffopt.routing.shortest_path.dijkstra`) and returns a binary `(E,)`
path-indicator vector `z`. Dijkstra itself has no gradient (it's a discrete
argmin over paths), so the backward pass instead implements the Vlastelica
construction: it builds an imagined, slightly different cost vector
`c_target = w + λ·grad_output` (`surrogate.py:71`; the `+` sign follows from
the paper's `ŷ = -∂L/∂z` and PyTorch's `grad_output = ∂L/∂z` convention) and
re-solves shortest path *on that*. Concretely, "perturbing the edge costs"
means: for every edge whose use the loss wants discouraged (using it more
would make the loss worse, so `grad_output[e] > 0`), its cost in `c_target`
goes *up* — less attractive to the re-solve; for every edge the loss wants
used more (`grad_output[e] < 0`), its cost goes *down* — more attractive.
Comparing the original path `z` to the path chosen under this perturbed cost
— which edges appeared, which disappeared — *is* the surrogate gradient:
`-(1/λ)·(path* - path_target)`.

Perturbed costs can go negative, and because the graph is undirected, a
negative edge is already a negative 2-cycle (crossable back and forth for
free) — so this re-solve uses SPFA (Bellman-Ford-style), not Dijkstra, purely
to fail gracefully: SPFA's relaxation-count guard detects the cycle and
returns no path instead of hanging, and the surrogate gradient for that
demand then falls back to exactly zero rather than crashing.
`DijkstraSurrogate` keeps "Dijkstra" in its name because that's the operation
it's *differentiating through* — the thing it actually computes and returns
on the forward pass. SPFA is only an internal detail of how `backward()`
answers its own auxiliary question ("what would the solver pick under these
nudged costs?").

**3. The path is cut into transparent segments.** `segment_path` (in
`diffopt/pipeline.py`) walks the chosen route and splits it at every *regen
candidate* node — any node of undirected degree ≥ 3
(`Topology.regen_candidate_nodes`) — the only places a regenerator may
physically be placed. Each resulting segment is a run of fiber spans crossed
purely optically, with no electrical regeneration; noise accumulates along
it.

**4. A frozen QoT surrogate predicts each segment's GSNR, batched across every
demand in one call.** `SpanAttentionQoT` (`diffopt/qot/model.py`) is
pretrained separately (Part 1 above; the root `README.md`'s Key commands has
the invocation) on real-GNPy-labeled per-segment data and then frozen with
`requires_grad_(False)` inside
the pipeline — it is never updated by end-to-end gradients. All demands are
routed and segmented first (step 2-3), and every resulting segment across the
whole batch is sent through the model in one padded call rather than one call
per segment, memoized on `(ordered edge-id tuple, batch max spans)` since a
segment's QoT input is a pure function of its edges. Because segment
membership is a discrete function of the (detached) reconstructed path, and
its span features come from static topology data, the frozen model's output
`qot_gsnr` has zero live gradient with respect to `path_indicator` — on its
own it would not expose a usable gradient back to which edges were chosen.
The pipeline works around this with a straight-through estimator
(`diffopt/pipeline.py`):

```python
proxy_noise = (path_indicator[seg_idx] * self._edge_ase_noise[seg_idx]).sum()
proxy_gsnr = -10.0 * torch.log10(proxy_noise + eps)
segment_gsnr = qot_gsnr + (proxy_gsnr - proxy_gsnr.detach())
```

`self._edge_ase_noise` is a fixed per-edge analytical ASE-noise coefficient
(`diffopt/qot/edge_noise.py`): for each span, `noise_figure_linear ·
(span_loss_linear - 1)`, summed over the segment's spans and normalized by
the topology's median edge — the standard closed-form ASE term from the
Gaussian-Noise model, with the physical constants that only set an overall
scale dropped. `proxy_gsnr` is a smooth function of `path_indicator` because
`self._edge_ase_noise` is a fixed, topology-derived vector multiplying it. In
the *forward* pass, `(proxy_gsnr - proxy_gsnr.detach())` is numerically zero,
so `segment_gsnr` equals `qot_gsnr` exactly — the physics-model's accurate
prediction. In the *backward* pass, `.detach()` blocks gradient through the
second `proxy_gsnr`, so the only gradient that survives is
`∂proxy_gsnr/∂path_indicator` — the analytical proxy's, not the frozen
model's. This is what actually supplies `edge_log_weight` with a per-edge
routing signal.

**5. `AllocationHead` rolls autoregressively along each demand's boundaries,
deciding where to regenerate.** (`diffopt/placement/allocation.py`.) Replaces
`RegenPlacement`, a single learnable logit per node: a per-node probability
cannot express "demand 3 regenerates at node 7, demand 9 does not", and
pricing a per-node *site* rather than a per-demand *device* was the wrong
metric (40 demands regenerating at one node need 40 devices, not 1).
`AllocationHead` instead scores every `(demand, boundary)` pair from an
8-feature vector — current-chunk GSNR proxy, the bitrate's dB bar, headroom
now, next-segment lookahead, headroom-after-next lookahead, and three
route-context features (distance since the last cut, distance remaining,
boundaries remaining) — walking each demand's boundaries in order so a cut's
value depends on noise accumulated since the *previous* cut (the carry `c`,
reset on a cut, entering the features as a detached observation so the
gradient cannot exploit its own earlier decisions). The head is **closed at
init**: its final layer starts at zero weight and a fixed negative bias
(`sigmoid(-3) ≈ 0.047`), so every boundary starts at a small constant
regeneration probability regardless of features — this is what lets the
device-count penalty (step 7) be live from epoch 0 with no warm-up schedule.
An earlier `greedy_residual` carve-out moved this closed-init point to
exactly reproduce the oracle's own greedy cut rule (`cut iff` headroom-after-
next `< 0`); it has been removed, since it hard-codes the optimal rule into
the score rather than letting the relaxation find it.

A **hard rollout** (`a_k = 1` iff `score_k > 0`, run under `torch.no_grad()`)
is what every checkpoint-selection and cross-epoch comparison reads; the
**soft** pass used during training (`sigmoid(score / tau)`) is a mean-field
relaxation that is allowed to disagree with it. By default (`placement.
alloc_ste`, `true` in every shipped config) the soft pass is instead a
**straight-through estimator**: its forward value is exactly the hard 0/1
decision (so the deployed rollout and the training-time relaxation always
agree on which demands are feasible), while `sigmoid'(score/tau)/tau`
survives on the backward pass so the head still learns. Without it, measured
on `constrained_stress` at an allocation with zero oracle gap and zero hard
violations, the soft pass still read 10 of 346 demands as violated — phantom
violations the deployed network does not have — and those 10 demands carried
essentially all of the feasibility force. Since the forward value under
`alloc_ste` is exactly 0/1, `tau` keeps only a backward role; training warns
if `alloc_tau_end` differs from `alloc_tau_start` under this arm, and every
shipped config pins them equal. `AllocationOutputs.waste_cost` — `sum_{d,k}
a_priced · relu(headroom-after-next).detach()`, a diagnostic for "a device
bought with headroom to spare" — is still computed and logged every epoch,
but carries no weight in the loss.

**6. `SegmentCombiner` folds every demand's boundaries into one end-to-end
GSNR.** (`diffopt/qot/segment_combiner.py`, `forward_batched`.) Per-segment
GSNRs are converted to linear noise and accumulated according to the
physics: if there's no regenerator at a boundary, noise from the two
neighboring segments simply adds; if there is one, only the worse
(higher-noise) segment matters, because regeneration resets accumulated
noise. Since the allocation is a soft probability during training,
`SegmentCombiner` returns the **exact** probability-weighted expectation of
the worst chunk's noise over every hard partition of the path at the
boundaries — not an approximation, and not a running accumulator. It
computes this exactly via a polynomial-time dynamic program rather than by
enumerating all `2^(N-1)` partitions: instead of asking "what is the average
worst chunk?" it asks "what is the probability every chunk stays under a bar
`tau`?", checkable left to right since a chunk can never span a cut,
vectorized over all `N(N+1)/2` possible chunk-sum thresholds, and folds every
demand in the batch in one call. `SegmentCombiner` is stateless and takes no
annealed parameter at all — there is no temperature to keep sharp or in
sync, because the fold has no approximation error to control. Accumulation
runs in float64 internally and the result is cast back to float32: a good
25 dB segment carries noise ~0.003, but a segment near 0 dB carries ~1.0, and
summing many of those approaches float32's range. Segment GSNRs are clamped to
`[-5, 35]` dB before the dB→linear conversion as an overflow guard — the
allocation head and the oracle clamp identically, since they must agree with
the fold on what a chunk's noise is or `oracle_gap` stops measuring the head
and starts measuring a units mismatch.

**7. The loss enforces feasibility as a constraint, via per-demand duals and
an augmented-Lagrangian penalty, rather than pricing it as a fixed-weight
penalty.** `diffopt/loss.py`'s `compute_loss` looks up the required SNR for
each demand's requested bitrate via `ModulationConfig` (a direct dictionary
lookup — 11 fixed bitrates, no interpolation) against a **fixed traffic
matrix** (`diffopt/traffic.py`, built once per run from `traffic.seed`/
`traffic.scale`/`traffic.scenario`, not redrawn every epoch) and combines
three terms as a weighted sum:

$$
L = \sum_{d \in D} \frac{\mathrm{ReLU}(\lambda_d + \rho\, g_d)^2 - \lambda_d^2}{2\rho}
  \;+\; \lambda_{\text{dev}} \sum_{n} \sum_{d} a_{d,n}
  \;+\; \lambda_{\text{cost}} \sum_{d \in D} \sum_{e} z_{d,e}\,\nu_e
$$

For each demand `d` requesting bitrate `b_d`: `g_d = bar_d - ĜSNR_d` is the
**signed** shortfall against `bar_d = τ(b_d) + δ` (`τ(b_d)` the SNR
threshold, `δ` = `constraint.margin_db` a fixed margin so the term stays
active with a gradient even after a demand clears the bare threshold, and
`ĜSNR_d` the pipeline's predicted end-to-end GSNR), negative once a demand
has headroom. This is the augmented Lagrangian (method of multipliers —
Hestenes / Powell / Rockafellar): the force `ReLU(λ_d + ρ·g_d)` is nonzero
for a band of width `λ_d/ρ` dB *inside* the feasible region (wider for
demands whose duals have grown) and exactly zero past it, unlike a one-sided
hinge `λ_d·ReLU(g_d)` whose derivative is exactly zero for any satisfied
demand no matter how large its dual — at an optimum where every demand is
satisfied and a marginal cut is load-bearing, a hinge leaves no force but
`-λ_dev` on that cut's allocation variable, so stationarity would require
`λ_dev = 0`. The augmented penalty's nonzero band inside the feasible region
is what lets a satisfied demand defend the cut that satisfies it. `λ_d` is
demand `d`'s **dual variable** — not a fixed hyperparameter, but a per-demand
Lagrange multiplier persisted across epochs and updated by an explicit
clamped ascent step *after* the optimizer step (`diffopt/loss.py`'s
`update_duals`, not on any optimizer): `λ_d ← clamp(λ_d + ρ·g_d, 0,
λ_max)` (`constraint.rho`, `constraint.dual_max`) — `ρ`, the augmented
penalty's own coefficient, doubles as the dual step size, which is gradient
ascent on the dual function with a step the penalty's own curvature makes
well-scaled; there is no separate dual learning rate. A demand that keeps
falling short gets a rising, individually-targeted penalty; one with slack
(`g_d < 0`) sees its own dual *decrease* by construction — no separate decay
term is needed, unlike a one-sided hinge dual that only ever ascends.
`a_{d,n}` is `AllocationHead`'s priced per-(demand, node) allocation (step
5); `Σ_n Σ_d a_{d,n}` (`device_count`, from `total_device_cost`) is a
per-*device* count, at a fixed weight `λ_dev` — feasibility no longer
competes with it on a tuned exchange rate, since the duals buy feasibility
directly. `z_{d,e}` is demand `d`'s (surrogate-differentiable) binary path
indicator on edge `e`, and `ν_e` is the fixed per-edge ASE-noise coefficient
from step 4 — this route-noise term is a regularizer, not the primary
routing signal (that's the straight-through estimator in step 4); it exists
so that once every demand is feasible (the augmented term's force drops to
zero past its band) some signal still reaches `edge_log_weight` favoring
lower-noise routes. Default weights (`configs/experiment/base.yaml`): `δ =
0.5` dB, `λ_0 = 0.0` (cold start — every dual starts at zero under the
augmented penalty, unlike a hinge dual which needs a nonzero starting force),
`ρ` measured per topology by `scripts/calibrate_rho.py` (`0.258220` on
`base`, `0.310338` on `constrained_stress` — required, no default, since a
guessed `ρ` silently sets the penalty's band width), `λ_dev = 0.11`
(measured by `scripts/calibrate_lambda_dev.py`), `λ_cost = 0.01`.

**8. Training can dump one JSON frame per epoch for the trajectory
viewer.** (`diffopt/viz/frames.py`, `diffopt/train.py`.) When `viz.
dump_frames: true` (set in `configs/experiment/constrained_stress.yaml`),
each epoch's hard-rollout allocation and per-lightpath GSNR are streamed as
one line to a `frames.jsonl` sidecar (`viz.every` controls the stride, `viz.
keyframe_every` how often a full — rather than delta-encoded — frame is
written), so a crashed run keeps every frame written so far. At the end of
training, `close()` assembles the sidecar plus `e2e_train_log.csv` into a
single `frames.json`. `scripts/build_viewer.py --frames <frames.json> --out
<out.html>` then inlines that JSON directly into `diffopt/viz/viewer.html`'s
template, producing one self-contained HTML file with no server and no
external data file — the artifact served from `docs/demo/`.

```
FORWARD
───────
edge_log_weight (E,) free parameter
                                  │
                            Softplus
                                  ▼
                              u (E,) > 0
                                  │
                        w = u / mean(u)                                     [unit-mean renorm, step 1]
                                  ▼
                  DijkstraSurrogate.forward = exact Dijkstra(w)             [step 2]
                                  ▼
                        z (E,) binary path indicator
                                  │
                  segment_path(z) at regen-candidate nodes                  [step 3]
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                        ▼
   frozen SpanAttentionQoT(spans, batched)   proxy_gsnr = -10·log10(Σ z·ν_e)  [step 4]
        → qot_gsnr (forward value)              (gradient source only)
              └───────────────────┬───────────────────┘
                                  ▼
      segment_gsnr = qot_gsnr + (proxy_gsnr - proxy_gsnr.detach())          [straight-through, step 4]
                                  ▼
      AllocationHead.rollout: autoregressive per-(demand,boundary)          [step 5]
          score → a ∈ {0,1} (STE forward) / (0,1) (soft, backward only)
                                  ▼
      SegmentCombiner.forward_batched: exact DP fold, E[max chunk noise]    [step 6]
                                  ▼
                          path GSNR (dB), per demand
                                  ▼
      loss = Σ_d (relu(λ_d+ρ·g_d)²-λ_d²)/(2ρ) + λ_dev·Σa + λ_cost·Σ z·ν_e   [step 7]
                                  ▼
                              scalar loss L

BACKWARD  (gradient flows bottom-to-top, mirroring the arrows above)
────────
∂L/∂g_d = relu(λ_d+ρ·g_d)/1   (nonzero for a λ_d/ρ-wide band inside the feasible region, zero past it)
      │
      ▼
∂L/∂segment_gsnr ──through the DP fold's exact partition weighting──   also: ∂L/∂a ──through AllocationHead.score──► its MLP directly
      │                                                                    (the carry feeding score is detached, so this
      ▼                                                                     gradient never reaches an earlier boundary's a)
∂L/∂qot_gsnr → discarded (frozen model, nothing upstream to update)
∂L/∂z ← ∂proxy_gsnr/∂z            [straight-through swap — the ONLY path back into z]
      │
      ▼
DijkstraSurrogate.backward:
   c_target = w + λ·(∂L/∂z)       (cost↑ for edges the loss wants dropped, cost↓ for edges it wants kept)
   path* = SPFA(c_target)          (not Dijkstra — perturbed costs can go negative)
   grad_w = -(1/λ)·(z - path*)
      │
      ▼
∂L/∂w ──through w = u/mean(u)──  (non-detached mean ⇒ Euler's theorem kills the pure-scale component)
      │
      ▼
∂L/∂u  ──Softplus'(·)──►  ∂L/∂edge_log_weight
```

Two parameter groups receive gradient from this single scalar loss:
`AllocationHead`'s MLP sits directly in the autograd graph via the sigmoid
probabilities used throughout steps 5-6, while `edge_log_weight` is reached
only indirectly, through `DijkstraSurrogate`'s backward pass converting
whatever gradient lands on the path indicator (from the straight-through
estimator in step 4, and from the route-noise regularizer in step 7) into a
gradient on edge costs.

## Part 3 — the surrogate gradient, worked through by hand

Step 2 above states what `DijkstraSurrogate.backward` does. This section is
the same mechanism at a slower pace, on a four-edge toy graph with concrete
numbers — read it first if step 2 went by too fast, and skip it otherwise. It
assumes you know what a gradient is and what gradient descent does, but skips
the formal theorem.

### The problem: Dijkstra is a wall

Training a network means differentiating a loss with respect to parameters and
nudging them downhill. That works as long as every operation in the chain is
differentiable. Dijkstra's algorithm is not. It takes edge weights, runs a
discrete search, and returns a binary vector: 1 for edges on the shortest
path, 0 otherwise. There is no slope. Raise edge weight S-A from 2.0 to 2.001
and the output path either stays exactly the same (gradient 0) or jumps to a
completely different path (gradient undefined). There is nothing in between.

This is the general problem of differentiating through a combinatorial solver
— the same wall you hit with argmax, sorting, or an LP.

### What we want the gradient to say

Suppose the network selected path S-A-T, and the downstream loss says "this
path is bad, route via S-B-T instead." We want a signal that tells the edge
weights:

- increase the cost of the S-A-T edges, so S-A-T stops being chosen;
- decrease the cost of the S-B-T edges, so S-B-T starts being chosen.

The exact numerical value matters less than the direction. Get the sign right
and keep the magnitude reasonable, and gradient descent does the rest.

### The trick: ask a perturbed question

Rather than differentiate the solver, **run it twice** and use the difference
between the two outputs as the gradient. During the backward pass:

**1.** We already have the forward path, `path_star`. Here that is S-A-T,
encoded over the four edges as `[1, 0, 1, 0]`.

**2.** Autograd hands us `grad_output = ∂L/∂path`. If the loss penalises the
S-A-T edges, it looks like `[1, 0, 1, 0]` — positive on the edges of the
current (bad) path, zero elsewhere.

**3.** Build perturbed costs, `c_target = edge_weights + λ·grad_output`. This
makes the edges the loss is complaining about *more expensive*: with `λ = 10`
and the S-A edge costing 1.0, its perturbed cost is 11.0.

**4.** Re-solve on `c_target`. Because the penalised edges are now expensive,
the solver picks something else — say S-B-T, `[0, 1, 0, 1]`. Call it
`path_target`. (The re-solve uses SPFA rather than Dijkstra, because
`c_target` can go negative; see step 2 for why that is about failing
gracefully rather than about negative-cycle correctness.)

**5.** The surrogate gradient is the scaled difference:

```
grad_weights = -(1/λ) * (path_star - path_target)
             = -(1/10) * ([1,0,1,0] - [0,1,0,1])
             = [-0.1, 0.1, -0.1, 0.1]
```

Gradient descent then applies `edge_weights -= lr * grad_weights`: the S-A-T
edges (gradient `-0.1`) get *more* expensive and are chosen less; the S-B-T
edges (gradient `+0.1`) get cheaper and are chosen more. Over many steps the
solver comes to prefer S-B-T and the loss falls.

### Why the gradient sign looks backwards

Seeing `grad_weights[SA] = -0.1` and thinking "negative gradient, so descent
decreases the S-A weight, so S-A gets cheaper" is the common confusion — and
it is backwards. Descent is `w_new = w - lr·grad`:

```
w_SA_new = 1.0 - 0.5 * (-0.1) = 1.05
```

The weight *increased*, because of the minus sign in the update rule. So: a
negative surrogate gradient on an edge makes that edge more expensive and less
likely to be selected. Counterintuitive, correct, and not to be flipped.

### Why the perturbation sign is `+` and not `−`

Vlastelica's Theorem 3.1 writes the perturbation as `c − λŷ`, where `ŷ` is the
*improvement* direction — the negative gradient, the way you want to move.
PyTorch's `grad_output` is `∂L/∂path`, the direction in which loss *increases*.
The two minus signs cancel: `c − λ·(−grad_output) = c + λ·grad_output`.

Get this wrong and the failure is silent. With `c − λ·grad`, the perturbation
makes the already-active path's edges *cheaper*, the perturbed solve returns
the same path, `path_star − path_target` is zero, every gradient is zero, and
nothing learns. No error, no NaN — just a model that never improves.

### Where `∂L/∂path` comes from

`path` is the `(E,)` binary path indicator, carrying a surrogate gradient via
the custom autograd function. Downstream it is used to pick out which spans
are on the route (feeding the QoT surrogate and the combiner, step 4) and to
price route properties in the loss (step 7). When `loss.backward()` runs,
PyTorch accumulates `∂L/∂path` by the time it reaches `DijkstraSurrogate` —
that is `grad_output`. A positive `grad_output[e]` means "if edge `e` were on
the path more, the loss would be higher": having it on the path is bad, and we
should discourage it by raising its weight.

### Recap

| Step | What happens |
|------|-------------|
| Forward | Dijkstra selects the shortest path; output is a binary {0,1} edge indicator |
| Downstream loss | Computes a scalar from the path (via QoT, allocation, combiner) |
| Backward arrives at surrogate | PyTorch hands over `grad_output = ∂L/∂path` |
| Perturb | `c_target = w + λ·grad_output` — penalised edges become more expensive |
| Re-solve | SPFA finds the shortest path under perturbed costs → `path_target` |
| Surrogate gradient | `-(1/λ)·(path_star − path_target)` — negative on active penalised edges |
| Gradient descent | Raises those edge costs → the next forward pass may route differently |

## Hyperparameters

All defaults below are from `configs/experiment/base.yaml`;
`small_test.yaml` uses the same keys at a smaller scale.

| Section | Key | Default | Controls |
|---|---|---|---|
| `constraint` | `margin_db` | 0.5 | delta added inside the feasibility force (step 7) |
| `constraint` | `dual_init` | 0.0 | lambda_0 — cold start; every dual starts at zero under the augmented penalty |
| `constraint` | `rho` | 0.258220 | augmented-penalty coefficient AND the dual ascent step size (`eta = rho`); required, no default — measured per topology by `scripts/calibrate_rho.py` |
| `constraint` | `dual_max` | 1000.0 | lambda_max — diagnostic cap; demands pinned here are reported at end of run |
| `traffic` | `scenario` | `stress` | `stress` (alpha 0.0) or `realistic` (alpha 1.0); mapped by `diffopt.traffic.scenario_alpha` |
| `traffic` | `seed` | 0 | matrix identity — a different value is a different constraint set |
| `traffic` | `scale` | 1300000.0 | total offered load in Gbps |
| `traffic` | `holdout_seed` | 1 | a different matrix, for the generalisation gap (`scripts/evaluate_matrix.py --holdout`) |
| `pipeline` | `lambda_dev` | 0.11 | Weight on the device-count penalty (step 7); measured, not hand-tuned — see `scripts/calibrate_lambda_dev.py` |
| `pipeline` | `lambda_cost` | 0.01 | Weight on the ASE-noise route regularizer (step 7) |
| `pipeline` | `channel_loading_fraction` | 0.5 | Channel loading assumed for span-feature extraction |
| `placement` | `lookahead` | true | `AllocationHead`'s next-segment lookahead features (step 5) |
| `placement` | `route_context` | true | `AllocationHead`'s km-since-cut/km-remaining/boundaries-remaining features (step 5) |
| `placement` | `alloc_ste` | false (code default); `true` in every shipped config | straight-through estimator — forward value is exactly the hard 0/1 decision (step 5); load-bearing for the headline result |
| `training` | `lr_edge_net` | 1e-3 | Adam LR for `edge_log_weight` |
| `training` | `lr_alloc` | 5.0e-4 | **SGD** LR for `AllocationHead`'s parameters (`opt_alloc = optim.SGD`) |
| `training` | `alloc_grad_clip` | 100.0 | Gradient-norm clip on `AllocationHead`'s parameters; `0.0` disables it |
| `training` | `lr_alloc_schedule` → `_end`/`_anneal_start_epoch`/`_anneal_end_epoch`/`_step_size`/`_step_gamma` | `"none"` | Optional cosine/step schedule on `lr_alloc`; `"none"` reproduces a flat LR exactly |
| `training` | `epochs_e2e` | 500 (300 in `constrained_stress.yaml`) | Number of end-to-end training epochs |
| `training` | `vlastelica_lambda` → `vlastelica_lambda_min` | 10.0 → 1.0 (×0.995/epoch) | `DijkstraSurrogate`'s perturbation strength λ (step 2), decaying every epoch |
| `training` | `alloc_tau_start` → `alloc_tau_end` | 1.0 → 1.0 (pinned, not annealed) | `AllocationHead`'s sigmoid temperature (step 5); kept equal under `alloc_ste`, since tau only affects the backward pass there |

How they interact:

- **`alloc_tau` no longer anneals in any shipped config.** An earlier
  version scheduled `alloc_tau_start → alloc_tau_end` down from 1.0 to 0.3
  over `alloc_tau_anneal_start_epoch`–`_end_epoch` (100–400), in step with a
  matching `soft_max_temperature` anneal in `SegmentCombiner`'s fold. Both
  are gone: `SegmentCombiner` is now an exact dynamic program with no
  approximation sharpness to anneal, and `alloc_ste` makes the forward pass
  exactly 0/1 regardless of `tau`, so every shipped config pins
  `alloc_tau_end` to `alloc_tau_start` and `train.py` warns if they differ.
  A floor of 0.3 (never 0.1) is what a still-annealing config should use if
  it ever revives the schedule: float32's sigmoid underflows once a score
  drifts past `tau`'s usable band (`|s| < 0.5` at `tau=0.1`), and once every
  boundary saturates there is no gradient left to escape.
- **`alloc_ste` trades a small amount of soft/hard disagreement for
  removing phantom violations.** Without it, the soft relaxation and the
  hard rollout can disagree about which demands are feasible — measured on
  `constrained_stress` at an allocation with zero oracle gap and zero hard
  violations, 10 of 346 demands still read as violated in the soft pass,
  and those 10 carried essentially all of the feasibility force the duals
  saw. Annealing `tau` made this worse, not better, because a barely-needed
  cut sits at score ≈ 0, where `sigmoid(0/tau) == 0.5` at every temperature.
- **`vlastelica_lambda` trades perturbation size against gradient
  magnitude.** A larger λ perturbs costs further, making it more likely some
  edge actually flips in the re-solve (real signal instead of a zero
  gradient) — but the returned gradient is also divided by λ, so a larger λ
  that does flip an edge still shrinks the resulting gradient. It starts
  loose (10.0) and decays toward a floor (1.0) over training.
- **`lr_alloc` is SGD, not Adam, and two orders of magnitude smaller than
  the old Adam LR it replaced.** Adam normalizes every parameter's step by
  its own gradient magnitude, so a single LR works across parameters of very
  different scale; plain SGD does not, so the same nominal LR that was safe
  under Adam is far too large under SGD. Measured on `constrained_stress`:
  SGD with `lr_alloc=5e-4` reaches a 300-epoch regenerator over-buy of 4
  devices above oracle-optimal, against 24 for the old Adam + hinge
  combination on the same seed — but the advantage only appears past ~60-150
  epochs of budget; below that the old combination still wins, since Adam's
  normalization buys speed at the cost of the runaway this change fixes.

See `docs/architecture/invariants.md` for the full history behind these
values.
