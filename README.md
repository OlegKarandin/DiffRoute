# DiffONet

DiffONet (package: `diffopt`) jointly optimizes **routing** and **regenerator
placement** for WDM (wavelength-division-multiplexed) optical networks by
backpropagating through a discrete shortest-path solver. Both decisions are
learned end to end, driven by a physically grounded, per-path GSNR
(generalized signal-to-noise ratio) estimate, using the blackbox-solver
surrogate gradient of Vlastelica et al., *Differentiation of Blackbox
Combinatorial Solvers* (ICLR 2020).

## How it works

The forward pass (`diffopt/pipeline.py`, `DiffONetPipeline.forward`) chains
five differentiable/quasi-differentiable stages:

```
EdgeWeightNet → surrogate Dijkstra → segment at regen candidates
  → frozen QoT surrogate per segment → SegmentCombiner → loss
```

**1. `EdgeWeightNet` prices every edge.** A small MLP
(`diffopt/routing/edge_weight_net.py`) maps a 7-dimensional per-edge feature
vector to a strictly positive routing cost via `Softplus`. Five of the seven
features are static topology properties, read directly off each edge
(`Topology.get_edge_features`, `diffopt/topology.py`): mean span length (km),
fiber type (categorical index), mean amplifier noise figure (dB), span count,
and total edge length (km) — each z-scored across the topology before use.
The other two are the current regenerator probability at the edge's source
and destination node (two separate scalars, not one value duplicated) — the
only channel through which `EdgeWeightNet` sees `RegenPlacement`'s state, so
it can price an edge differently depending on whether a regenerator is likely
to sit at either end.

The pipeline then renormalizes all edge costs to unit mean *without
detaching the divisor* (`w = u / mean(u)`, `diffopt/pipeline.py:311`).
Concretely: if `EdgeWeightNet` outputs raw costs `u = [2, 4, 6]`, then
`mean(u) = 4` and `w = [0.5, 1.0, 1.5]`. Scale every raw cost by *any*
constant `k > 0` — say `k = 1000`, giving `u' = [2000, 4000, 6000]` — and
`w' = u'/mean(u') = [0.5, 1.0, 1.5]`, identical to `w`. So Dijkstra's chosen
path, and every downstream loss term computed from `w`, are completely
unaffected by uniformly scaling the network's raw output up or down.

This matters because an earlier version of the loss did *not* have that
property: a term (`path_cost_loss`, since replaced — see step 6) summed
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
than shortest-by-km, no better than before training. See
`docs/investigations/edge_weight_scale_collapse.md`.

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

**4. A frozen QoT surrogate predicts each segment's GSNR.** `SpanAttentionQoT`
(`diffopt/qot/model.py`) is pretrained separately (see Key commands below) on
real-GNPy-labeled per-segment data and then frozen with
`requires_grad_(False)` inside the pipeline — it is never updated by
end-to-end gradients. Because segment membership is a discrete function of
the (detached) reconstructed path, and its span features come from static
topology data, the frozen model's output `qot_gsnr` has zero live gradient
with respect to `path_indicator` — on its own it would not expose a usable
gradient back to which edges were chosen. The pipeline works around this with
a straight-through estimator (`diffopt/pipeline.py:429-440`):

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
model's. This is what actually supplies `EdgeWeightNet` with a per-edge,
regen-modulated routing signal.

**5. `SegmentCombiner` folds segments into one end-to-end GSNR.**
(`diffopt/qot/segment_combiner.py`.) Per-segment GSNRs are converted to
linear noise and accumulated according to the physics: if there's no
regenerator at a boundary, noise from the two neighboring segments simply
adds; if there is one, only the worse (higher-noise) segment matters, because
regeneration resets accumulated noise. Since regenerator placement is a soft
probability during training, `SegmentCombiner` returns the **exact**
probability-weighted expectation of the worst chunk's noise over every hard
partition of the path at the boundaries — not an approximation, and not a
running accumulator. It computes this exactly via a polynomial-time dynamic
program rather than by enumerating all `2^(N-1)` partitions: instead of
asking "what is the average worst chunk?" it asks "what is the probability
every chunk stays under a bar `tau`?", checkable left to right since a chunk
can never span a cut, vectorized over all `N(N+1)/2` possible chunk-sum
thresholds. `SegmentCombiner` is stateless and takes no annealed
parameter at all — there is no temperature to keep sharp or in sync,
because the fold has no approximation error to control.

**6. The loss enforces feasibility as a constraint, via per-demand duals,
rather than pricing it as a fixed-weight penalty.** `diffopt/loss.py`'s
`compute_loss` looks up the required SNR for each demand's requested bitrate
via `ModulationConfig` (a direct dictionary lookup — 11 fixed bitrates, no
interpolation) against a **fixed traffic matrix** (`diffopt/traffic.py`,
built once per run from `traffic.seed`/`traffic.scale`/`traffic.scenario`,
not redrawn every epoch) and combines three terms as a weighted sum:

$$
L = \sum_{d \in D} \lambda_d \, \mathrm{ReLU}\big(\tau(b_d) + \delta - \widehat{GSNR}_d\big)
  \;+\; \lambda_{\text{regen}} \sum_{n} p_n
  \;+\; \lambda_{\text{cost}} \sum_{d \in D} \sum_{e} z_{d,e}\,\nu_e
$$

For each demand `d` requesting bitrate `b_d`: `τ(b_d)` is its SNR threshold,
`δ` (`constraint.margin_db`) is a fixed margin added inside the hinge so the
term stays active with a gradient even after a demand clears the bare
threshold, and `ĜSNR_d` is the pipeline's predicted end-to-end GSNR; a `ReLU`
penalizes falling short of `τ(b_d) + δ` and costs nothing once it's met.
`λ_d` is demand `d`'s **dual variable** — not a fixed hyperparameter, but a
per-demand Lagrange multiplier persisted across epochs and updated by an
explicit clamped ascent step *after* the optimizer step (`diffopt/loss.py`'s
`update_duals`, not on any optimizer): `λ_d ← clamp(λ_d + η·shortfall_d, 0,
λ_max)` (`constraint.dual_lr`, `constraint.dual_max`). A demand that keeps
falling short gets a rising, individually-targeted penalty; one that is
feasible sees its dual relax. This replaces a single fixed `λ_infeasible`
that priced every demand's shortfall the same regardless of how persistently
it failed. `p_n = sigmoid(logit_n / τ)` is `RegenPlacement`'s soft
regenerator-presence probability at node `n` — summing it is a soft count,
penalizing more regenerators, at a fixed weight (`λ_regen` stays fixed;
feasibility no longer competes with it on a tuned exchange rate — the duals
buy feasibility directly). `z_{d,e}` is demand `d`'s (surrogate-
differentiable) binary path indicator on edge `e`, and `ν_e` is the fixed
per-edge ASE-noise coefficient from step 4 — this route-noise term is a
regularizer, not the primary routing signal (that's the straight-through
estimator in step 4); it exists so that once every demand is feasible (the
ReLU term's gradient is zero everywhere) some signal still reaches
`EdgeWeightNet` favoring lower-noise routes. Default weights
(`configs/experiment/base.yaml`): `δ = 0.5` dB, `λ_0 = 10.0` (matching the
old fixed `λ_infeasible`, so epoch 1 reproduces prior behaviour), `λ_regen =
1.0`, `λ_cost = 0.01`.

```
FORWARD
───────
regen_logits ──sigmoid(·/τ)──► p ∈ (0,1)ⁿ                                    [RegenPlacement]
                                  │
      5 static feats (z-scored) ─┤
                                  ▼
                    edge_feats (E,7) = [static | p[src] | p[dst]]
                                  │
                    Linear→ReLU→Linear→ReLU→Linear→Softplus                 [EdgeWeightNet, step 1]
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
   frozen SpanAttentionQoT(spans)          proxy_gsnr = -10·log10(Σ z·ν_e)   [step 4]
        → qot_gsnr (forward value)              (gradient source only)
              └───────────────────┬───────────────────┘
                                  ▼
      segment_gsnr = qot_gsnr + (proxy_gsnr - proxy_gsnr.detach())          [straight-through, step 4]
                                  ▼
      SegmentCombiner: exact DP fold, E[max chunk noise] over partitions    [step 5]
                                  ▼
                          path GSNR (dB), per demand
                                  ▼
      loss = Σ_d λ_d·ReLU(τ(b_d)+δ - GSNR_d) + λ_regen·Σp + λ_cost·Σ z·ν_e   [step 6]
                                  ▼
                              scalar loss L

BACKWARD  (gradient flows bottom-to-top, mirroring the arrows above)
────────
∂L/∂GSNR = -λ_d if shortfall > 0, else 0    (ReLU'(τ+δ-GSNR) gates the -1 from d(τ+δ-GSNR)/dGSNR; λ_d is demand d's dual)
      │
      ▼
∂L/∂segment_gsnr ──through the DP fold's exact partition weighting──   also: ∂L/∂p ──sigmoid'(·)──► regen_logits directly
      │
      ▼
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
∂L/∂u   (gradient w.r.t. EdgeWeightNet's pre-Softplus output)
      │
      ▼
backprop through EdgeWeightNet's own layers, last to first —
Softplus'(·) at the output layer, then Linear, then ReLU'(·) at each
hidden layer, then Linear — accumulating a gradient on every Linear
layer's weight matrix along the way
```

Both regenerator placement (`RegenPlacement`, a learnable logit per node in
`diffopt/placement/regenerator.py`) and `EdgeWeightNet` receive gradient from
this single scalar loss: `RegenPlacement`'s logits sit directly in the
autograd graph via the sigmoid probabilities used throughout steps 3 and 5,
while `EdgeWeightNet` is reached only indirectly, through `DijkstraSurrogate`'s
backward pass converting whatever gradient lands on the path indicator (from
the straight-through estimator in step 4, and from a small route-noise
regularizer) into a gradient on edge costs.

## Hyperparameters

All defaults below are from `configs/experiment/base.yaml`;
`small_test.yaml` uses the same keys at a smaller scale.

| Section | Key | Default | Controls |
|---|---|---|---|
| `constraint` | `margin_db` | 0.5 | delta added inside the feasibility hinge (step 6) |
| `constraint` | `dual_init` | 10.0 | lambda_0 — matches the old fixed `lambda_infeasible`, so epoch 1 reproduces prior behaviour |
| `constraint` | `dual_lr` | 1.0 | eta — dual ascent step size (a 1 dB shortfall moves a dual 10% of lambda_0) |
| `constraint` | `dual_max` | 1000.0 | lambda_max — cap; demands pinned here are reported at end of run |
| `traffic` | `scenario` | `stress` | `stress` (alpha 0.0) or `realistic` (alpha 1.0); mapped by `diffopt.traffic.scenario_alpha` |
| `traffic` | `seed` | 0 | matrix identity — a different value is a different constraint set |
| `traffic` | `scale` | 1300000.0 | total offered load in Gbps |
| `traffic` | `holdout_seed` | 1 | a different matrix, for the generalisation gap (`scripts/evaluate_matrix.py --holdout`) |
| `pipeline` | `lambda_regen` | 1.0 | Weight on the regenerator-count penalty (step 6) |
| `pipeline` | `lambda_cost` | 0.01 | Weight on the ASE-noise route regularizer (step 6) |
| `pipeline` | `channel_loading_fraction` | 0.5 | Channel loading assumed for span-feature extraction |
| `training` | `lr_edge_net` | 1e-3 | Adam LR for `EdgeWeightNet` |
| `training` | `lr_regen` | 1e-2 | Adam LR for `RegenPlacement`'s logits |
| `training` | `epochs_e2e` | 500 | Number of end-to-end training epochs |
| `training` | `vlastelica_lambda` → `vlastelica_lambda_min` | 10.0 → 1.0 (×0.995/epoch) | `DijkstraSurrogate`'s perturbation strength λ (step 2), decaying every epoch |
| `training` | `regen_tau_start` → `regen_tau_end` | 1.0 → 0.1 | `RegenPlacement`'s sigmoid temperature, annealed over `regen_tau_anneal_start_epoch`–`_end_epoch` (100–400) |

How they interact:

- **`regen_tau` anneals alone now; `SegmentCombiner` has no matching knob to
  keep in step with it.** An earlier version of the fold annealed a
  `soft_max_temperature` over the same epoch window (100-400) as
  `regen_tau`, so the physics evaluation (step 5) sharpened at the same rate
  as the placement decisions (step 3) it was evaluating. That approximation
  is gone: `SegmentCombiner`'s fold is now an exact dynamic program with no
  approximation sharpness to anneal, so `regen_tau` is the only schedule left
  in this window. The approximation this replaced had a real failure mode
  worth remembering — its error was an *absolute* offset in linear-noise
  units (`temperature · ln2`), and at this topology's real per-segment noise
  scale (~0.0025, i.e. ~26 dB segments) even the tightest scheduled
  temperature (0.01) overshot by 2.6x the real signal and *inverted the
  sign* of every gradient into regenerator placement — see
  `docs/investigations/CHANGELOG.md`'s corrections #8 and #12.
- **`vlastelica_lambda` trades perturbation size against gradient
  magnitude.** A larger λ perturbs costs further, making it more likely some
  edge actually flips in the re-solve (real signal instead of a zero
  gradient) — but the returned gradient is also divided by λ, so a larger λ
  that does flip an edge still shrinks the resulting gradient. It starts
  loose (10.0) and decays toward a floor (1.0) over training.
- **`lr_regen`'s effective range depends on `epochs_e2e`, not just its own
  value.** Adam's step size is ~lr whenever a logit's gradient sign is
  steady, so a regenerator logit can move at most `lr_regen · epochs_e2e`
  total over a run — `1e-2 · 500 = 5.0` at the defaults, enough to cross the
  sigmoid's zero-logit decision boundary from most initializations. A short
  run (say, 20 epochs) gives only `0.2` total excursion, which isn't enough —
  regenerators can fail to place at all purely from the schedule being too
  short, independent of whether the gradient signal is otherwise correct.
- **`lambda_regen` vs. `lambda_cost` is a documented open question, not a
  resolved balance.** On a broader held-out demand set, infeasible demands
  rose after the scale-collapse fix (routes got much shorter, leaving less
  noise margin) — noted in `docs/investigations/open_followups.md` as a
  possible `lambda_regen`/`lambda_cost` interaction that hasn't been tuned or
  confirmed.

See `docs/architecture/invariants.md` and `docs/investigations/CHANGELOG.md`
for the full history behind these values.

## Install

```bash
conda create -n diffopt python=3.11 && conda activate diffopt
pip install -e ".[dev]"
```

## Key commands

```bash
# Run the test suite
pytest tests/

# Generate a QoT training set — runs real GNPy per segment, so this can take
# a while depending on sample count. Topology JSONs are already committed
# under configs/topology/, so no extra setup is required.
python data/generate_qot_dataset.py --config configs/experiment/small_test.yaml
python data/generate_qot_dataset.py --config configs/experiment/base.yaml

# Train the QoT surrogate on the dataset produced above (reads
# <dataset_dir>/train.parquet and val.parquet from the same config)
python -m diffopt.qot.train_qot --config configs/experiment/small_test.yaml
python -m diffopt.qot.train_qot --config configs/experiment/base.yaml

# Train the end-to-end routing + placement pipeline. Requires a QoT
# checkpoint at the config's `qot_checkpoint` path (produced by the command
# above) to already exist.
python -m diffopt.train --config configs/experiment/small_test.yaml
python -m diffopt.train --config configs/experiment/base.yaml
```

## Repo map

```
diffopt/
  pipeline.py         DiffONetPipeline — wires the stages above together
  loss.py              compute_loss — feasibility / regen-count / route-noise terms
  topology.py           Topology (extends OpticalNetworkModel)
  demands.py             Demand generation
  modulation.py           Bitrate -> SNR threshold lookup
  routing/               EdgeWeightNet, the Vlastelica surrogate, shortest-path solvers
  qot/                    QoT model, GNPy bridge, segment combiner, dataset generation/training
  placement/              RegenPlacement (soft regenerator logits)
configs/
  experiment/             Experiment YAML configs (small_test*.yaml, base.yaml)
  topology/                Committed topology JSONs + provenance (README.md)
  modulation_formats.yaml   WDM channel grid + bitrate/SNR table
data/
  generate_qot_dataset.py  QoT dataset generator (calls diffopt.qot.optical_bridge)
scripts/
  diagnose_*.py            Standalone diagnostics for individual pipeline stages
tests/                     pytest suite
docs/architecture/         Architecture notes (interfaces, ML pipeline)
```

## Citations

- Vlastelica, M., Paulus, A., Musil, V., Martius, G., Rolínek, M.
  *Differentiation of Blackbox Combinatorial Solvers.* ICLR 2020.
- Karandin, O. et al. *J. Opt. Commun. Netw.* **16**, H18-H26 (2024).
  https://github.com/OlegKarandin/jocn24-multi-fiber — source of the
  `ind_132` and `jp_70` topologies under `configs/topology/`.
- `multilayer-optical-network` — upstream `OpticalNetworkModel` /
  GNPy-integration dependency this project extends and pins via git
  dependency in `pyproject.toml`.
  https://github.com/OlegKarandin/multilayer-optical-network

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
