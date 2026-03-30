# How the Surrogate Gradient Works

This document explains the Vlastelica surrogate gradient approach used in DiffONet. It assumes you know what a gradient is and what gradient descent does, but skips the formal theorem proofs.

---

## The problem: Dijkstra is a wall

Training a neural network means taking the derivative of a loss with respect to parameters, then nudging those parameters downhill. That works as long as every operation in the chain is differentiable.

Dijkstra's algorithm is not. It takes a list of edge weights, runs a discrete search, and returns a binary vector: 1 for edges on the shortest path, 0 for everything else. There is no slope to follow. If you increase edge weight SA from 2.0 to 2.001, the output path either stays exactly the same (gradient = 0) or jumps to a completely different path (gradient = undefined). There is no in-between.

This is the same problem that comes up any time you have a discrete decision inside a neural network — argmax, sorting, linear programming. The field calls it "differentiating through a combinatorial solver."

---

## What we actually want the gradient to say

Imagine the network has selected path S-A-T through the graph. The downstream loss says "this path is bad — you should route via S-B-T instead." We want a gradient signal that tells the edge weight parameters:

- "Increase the cost of S-A-T edges" (so S-A-T stops being chosen)
- "Decrease the cost of S-B-T edges" (so S-B-T starts being chosen)

The exact numerical value of that signal matters less than its direction. If we can get the sign right and keep the magnitude reasonable, gradient descent will do its job.

---

## The Vlastelica trick: ask a perturbed question

The key idea (from Vlastelica et al., ICLR 2020) is: instead of trying to differentiate through the solver directly, **run the solver twice** and use the difference in outputs as a gradient approximation.

Here is the sequence during a backward pass:

**Step 1.** We already have the path from the forward pass: `path_star`. In our example, that's S-A-T, encoded as `[1, 0, 1, 0]` (1 = on path, 0 = not).

**Step 2.** We know the gradient of the loss with respect to the path indicator: `∂L/∂path`. Call this `grad_output`. If the loss penalises the S-A edges, `grad_output` is something like `[1, 0, 1, 0]` — high for the edges that are on the current (bad) path, zero for the rest.

**Step 3.** Build a set of *perturbed* edge weights:

```
c_target = edge_weights + λ * grad_output
```

This makes the edges the loss is complaining about *more expensive*. With λ=10 and the S-A edge initially costing 1.0, the perturbed cost becomes 11.0.

**Step 4.** Run the solver again on `c_target`. Call the result `path_target`. Because the edges that were penalised are now more expensive, the solver picks a different path — in our example, S-B-T: `[0, 1, 0, 1]`.

**Step 5.** The surrogate gradient is:

```
grad_weights = -(1/λ) * (path_star - path_target)
             = -(1/10) * ([1,0,1,0] - [0,1,0,1])
             = [-0.1, 0.1, -0.1, 0.1]
```

Gradient descent then does `edge_weights -= lr * grad_weights`:
- S-A-T edges (`grad = -0.1`): weight *increases* (becomes more expensive → chosen less)
- S-B-T edges (`grad = +0.1`): weight *decreases* (becomes cheaper → chosen more)

Over many steps, the solver starts preferring S-B-T. The loss goes down.

---

## Why the gradient sign looks backwards

If you see `grad_weights[SA] = -0.1` and think "negative gradient, gradient descent decreases SA weight, SA becomes cheaper" — that's the opposite of what we want. The confusion is common.

Gradient descent is `w_new = w - lr * grad`. Substituting:
```
w_SA_new = 1.0 - 0.5 * (-0.1) = 1.0 + 0.05 = 1.05
```

The weight *increased*. The negative gradient causes an increase because of the minus sign in the update rule. So: negative surrogate gradient on an edge → edge becomes more expensive → less likely to be selected. This is correct.

---

## The sign in the perturbation formula

The formula `c_target = w + λ * grad_output` has a positive sign. This is easy to get wrong. The original Vlastelica paper writes it as `c - λŷ` where `ŷ` is defined as the *negative* gradient (the direction you want to move, not the direction loss is increasing). Since PyTorch's `grad_output` is `∂L/∂path` (the direction loss *is* increasing), the signs cancel and you get a plus.

With the wrong sign (`c - λ * grad`), the perturbation *reduces* the cost of penalised edges, making the perturbed solver return the same path as before. The difference `path_star - path_target` is zero. Every gradient is zero. Nothing learns. This is a silent failure — no error, no NaN, just a model that doesn't improve.

---

## Why SPFA instead of Dijkstra in the backward pass

After adding `λ * grad_output` to edge weights, some perturbed weights can go negative. (Example: edge cost 1.0, `grad_output` entry 2.0, λ=10 → but wait, the perturbation adds, so 1 + 10*2 = 21. Actually for negative, consider: a different setup where we subtract — but even with addition, other components of the system might have negative weights for other reasons.)

Actually the issue arises because once SPFA is needed for robustness: some graph configurations with the perturbation applied can produce edge weights less than zero. Dijkstra's heap-based approach makes an implicit assumption that distances only ever decrease, and will loop or produce wrong results on negative weights. SPFA (a BFS-style Bellman-Ford) relaxes edges repeatedly until no further improvements are found — it handles negative weights correctly as long as there are no negative-weight cycles.

---

## What ∂L/∂path means and where it comes from

`path` is a `(E,)` tensor of 0s and 1s — the path indicator. It has a surrogate gradient attached via the custom autograd function.

Downstream of `surrogate_shortest_path`, the path indicator is used to:
- Extract which segments are on the selected route (for the QoT model and combiner)
- Penalise certain path properties in the loss (e.g., "penalise paths through nodes with no regenerator")

When `loss.backward()` is called, PyTorch walks backward through the computation graph. By the time it reaches `DijkstraSurrogate`, it has accumulated `∂L/∂path` — the gradient of the total loss with respect to each element of the path indicator. This is `grad_output`.

A positive `grad_output[e]` means "if edge e were on the path more (path[e] were higher), the loss would be higher." Equivalently: having edge e on the path is bad, and we should discourage it by increasing its weight.

---

## How this connects to Phase 1c

In Phase 1c, the full pipeline looks like this:

```
regen_logits  (trainable, one per node)
     ↓ sigmoid
regen_probs   (soft placement decision per node)
     ↓
edge_weight_net (trainable MLP)
     ↓
edge_weights
     ↓
surrogate_shortest_path      ← Vlastelica layer here
     ↓
path_indicator
     ↓
[split path into segments at regen-candidate nodes]
     ↓
SpanAttentionQoT × N segments  ← frozen, called once per segment
     ↓
[gsnr_seg_0, ..., gsnr_seg_N]
     ↓
SegmentCombiner ← regen_probs at boundary nodes flow in here
     ↓
path_gsnr_db
     ↓
Loss (feasibility penalty + regen count penalty)
```

Two gradient pathways run in parallel:

**Pathway 1 — Placement:** `∂L/∂regen_logits` flows through `SegmentCombiner`. The combiner mixes `noise_no_regen` and `noise_regen` with weight `p` at each boundary. If placing a regenerator at a node reduces the loss (by enabling a long infeasible path), `∂L/∂p > 0` at that node, and since `p = sigmoid(logit)`, the logit gets pushed up. The L1 penalty on `sum(regen_probs)` pushes logits down everywhere. The optimizer finds the subset of nodes where placement actually helps.

**Pathway 2 — Routing:** `∂L/∂edge_weights` flows through the Vlastelica surrogate. If the current path leads to poor QoT, the combiner and feasibility loss produce a high loss, and `grad_output` for the path indicator is nonzero on the active-path edges. The surrogate backward computes the gradient on edge weights, which backpropagates into `edge_weight_net`. The network learns to assign high costs to edges that lead to bad outcomes (long spans, no regenerators nearby) and low costs to edges that lead to feasible paths.

These two pathways are coupled: routing improves once regenerator placement is sensible, and regenerator placement becomes more targeted once routing has found good candidate paths. In practice, they are jointly optimized from the start, and convergence emerges from their interaction.

---

## The λ hyperparameter

λ controls how large a perturbation the backward pass applies. It shows up in two places:

- **Perturbation size:** larger λ → `c_target` deviates more from `c` → more likely the perturbed path differs from the original → nonzero gradient. If λ is too small, small perturbations don't flip any paths, and the gradient stays near zero.
- **Gradient magnitude:** the output gradient is divided by λ. Larger λ → smaller gradient magnitude. This creates a tension: you want λ large enough to flip the path, but not so large that the gradient is tiny.

Typical practice is to start with λ=10 (strong signal, somewhat noisy) and decay to λ=1 as training matures (finer updates). The config has `vlastelica_lambda_decay=0.995` per epoch for this.

---

## Summary

| Step | What happens |
|------|-------------|
| Forward | Dijkstra selects the shortest path; output is binary {0,1} edge indicator |
| Downstream loss | Computes a scalar from the path (via QoT, combiner, etc.) |
| Backward arrives at surrogate | PyTorch hands over `grad_output = ∂L/∂path` |
| Perturb | `c_target = w + λ * grad_output` — make penalised edges more expensive |
| Re-solve | SPFA finds shortest path under perturbed costs → `path_target` |
| Surrogate gradient | `-(1/λ) * (path_star - path_target)` — negative on active penalised edges |
| Gradient descent | Increases those edge costs → next forward pass may route differently |
