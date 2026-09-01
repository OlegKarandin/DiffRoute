"""Loss functions for the end-to-end DiffONet pipeline."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from diffopt.demands import Demand
from diffopt.modulation import ModulationConfig


def compute_loss(
    gsnr_preds: Dict[int, torch.Tensor],
    path_noise_costs: Dict[int, torch.Tensor],
    demands: List[Demand],
    device_count: torch.Tensor,
    modulation_config: ModulationConfig,
    duals: torch.Tensor,
    margin_db: float = 0.5,
    lambda_dev: float = 1.0,
    lambda_cost: float = 0.01,
    waste_cost: Optional[torch.Tensor] = None,
    lambda_waste: float = 0.0,
    penalty: str = "hinge",
    rho: Optional[float] = None,
) -> Tuple[torch.Tensor, dict]:
    """Compute the constrained training loss.

        L = sum_d lambda_d * relu(thr_d + delta - gsnr_d)
          + lambda_dev   * sum_n sum_d a[d,n]
          + lambda_waste * waste_cost
          + lambda_cost  * path_noise

    This replaces a weighted sum of three soft penalties in which feasibility
    competed with regenerator count on a fixed exchange rate. `lambda_dev`
    stays fixed; the duals rise until feasibility is bought, so regenerator
    count becomes an outcome rather than a tuned trade-off.

    Args:
        gsnr_preds:   demand_id -> scalar GSNR tensor (dB).
        path_noise_costs: demand_id -> scalar (path_indicator · edge_ase_noise).sum().
                      Accumulated ASE noise along the chosen route, in fixed
                      physical units. Denominating this in learned
                      edge_weights instead made the loss degree-1 in those
                      weights and collapsed them to the Softplus floor — see
                      docs/investigations/edge_weight_scale_collapse.md.
        demands:      List of Demand namedtuples, from the FIXED traffic
                      matrix (diffopt/traffic.py). Their `id`s must be
                      contiguous 0..N-1 — they index `duals`.
        device_count: The scalar `lambda_dev` multiplies — how many
                      REGENERATOR DEVICES the allocation buys,
                      `sum_n sum_d a[d,n]`, from
                      `diffopt.placement.allocation.total_device_cost`.
                      This replaces a per-SITE count, which was the wrong
                      metric: 40 demands regenerating at node 7 need 40
                      devices, not 1, and pricing sites is what made the
                      placement count track lambda's magnitude rather than
                      need. Live in the autograd graph — the gradient
                      through it is the only downward pressure on the
                      allocation head.
        modulation_config: Bitrate -> SNR threshold lookup.
        duals:        (num_demands,) tensor of per-demand multipliers, indexed
                      by `Demand.id` and persisted across epochs by the caller.
                      Per-demand rather than one adaptive scalar because Adam
                      bounds its step at ~lr and saturates past a ~20:1 force
                      ratio, so a larger weight does not buy a proportionally
                      larger move — what it buys is RECRUITMENT: epochs where a
                      node's feasibility signal was too small to flip the sign
                      of its total gradient get pushed across that threshold.
                      A per-demand dual does that selectively for the demands
                      that keep failing; a global scalar does it for every node
                      at once and over-places.
        margin_db:    delta — added inside the hinge so the term stays active
                      above the threshold. Without it `relu(thr - gsnr)` is
                      exactly zero the moment a demand clears, nothing pushes
                      for headroom, and the system sits on the boundary by
                      construction (measured |g_feas|_1 = 0.000 on
                      fully-feasible epochs). Also absorbs QoT surrogate error:
                      0.5 dB is ~2.6 sigma on the QoT model's 0.1909 dB val
                      RMSE plus headroom over soft_max's ~0.03 dB relative
                      error at t=0.01.
        lambda_dev:   Weight on the device count. Fixed, not annealed, and
                      LIVE FROM EPOCH 0 — see spec 2.2 for why a warm-up
                      saturates the head at ~20-30x the optimum with no
                      gradient left to escape. Calibrate with
                      scripts/calibrate_lambda_dev.py; do not hand-tune.
        lambda_cost:  Weight on the ASE-denominated path-noise regulariser. Not the
                      primary routing signal -- the STE in pipeline.forward supplies that.
        waste_cost:   Scalar `AllocationOutputs.waste_cost` —
                      sum_{d,k} a_priced[d,k] * relu(feature4[d,k]), live in
                      the autograd graph but gradient-detached from
                      n_next/routing per `AllocationHead.rollout`'s own
                      docstring (only the priced allocation's own dependence
                      on the score carries gradient). None (the default) is
                      the "arm doesn't use this term" case; required
                      whenever `lambda_waste != 0.0`.
        lambda_waste: Weight on `waste_cost`. Default 0.0 is a true no-op —
                      see `test_lambda_waste_defaults_to_no_op`.
        penalty:      "hinge" (the default and the shipped behaviour) or
                      "augmented". The hinge prices feasibility with
                      `dual_d * relu(g_d)`, whose derivative is
                      `dual_d * 1{g_d > 0}` — exactly zero for a satisfied
                      demand, no matter how large its dual. At an optimum
                      every demand is satisfied AND the marginal cut is
                      load-bearing, so the only surviving force on an
                      allocation variable is `-lambda_dev`, and stationarity
                      would require `lambda_dev = 0`. No non-negative
                      (lambda_dev, lambda_waste) makes that optimum a
                      stationary point; `lambda_waste` cancels out of the
                      condition entirely. "augmented" is the canonical
                      method of multipliers (Hestenes / Powell /
                      Rockafellar): the term becomes
                      `(relu(lambda_d + rho*g_d)^2 - lambda_d^2) / (2*rho)`
                      on the SIGNED `g_d = bar_d - gsnr_d`, so the force is
                      `max(0, lambda_d + rho*g_d)` — nonzero for a band of
                      width `lambda_d/rho` dB INSIDE the feasible region,
                      wider for demands whose duals grew, and EXACTLY zero
                      past it. That last exactness is the point: a merely
                      small force on 346 slack demands sums into systematic
                      over-buy, which is why a softplus hinge was rejected.
        rho:          Augmented-Lagrangian penalty coefficient, and ALSO the
                      dual step: `update_duals` is called with `eta=rho` in
                      that mode, which is gradient ascent on the dual
                      function with a step the penalty's own curvature makes
                      well-scaled. Required whenever `penalty="augmented"`
                      and ignored otherwise; there is no default, because a
                      guessed rho sets the band width. MEASURE it with
                      `python -m scripts.calibrate_rho`; do not hand-tune.

    Returns:
        (total_loss, metrics_dict). `metrics["shortfalls"]` is a detached
        (num_demands,) tensor indexed by `Demand.id`, to be fed straight into
        `update_duals` after the optimizer step in HINGE mode.
        `metrics["constraint_g"]` is the same tensor unclipped — the SIGNED
        `bar_d - gsnr_d`, negative when the demand has headroom — and is what
        `update_duals` consumes in AUGMENTED mode. Both are returned in both
        modes, so the dict has one shape regardless of penalty.
    """
    device = duals.device

    # Named errors, never a silent fallback — the same rule the
    # lambda_waste/waste_cost pair below follows. A mistyped penalty that
    # quietly ran the hinge would produce a plausible number measured
    # against the wrong objective, and an unset rho would silently pick a
    # band width nobody measured.
    if penalty not in ("hinge", "augmented"):
        raise ValueError(
            f"unknown penalty {penalty!r}; expected 'hinge' or 'augmented'"
        )
    if penalty == "augmented":
        if rho is None:
            raise ValueError(
                "penalty='augmented' requires rho — the penalty coefficient, "
                "which is also the dual step. There is no default: measure it "
                "with `python -m scripts.calibrate_rho`."
            )
        if rho <= 0.0:
            raise ValueError(
                f"rho must be > 0 under penalty='augmented' (it divides the "
                f"penalty term and scales the dual step), got {rho}"
            )

    # Start as zero tensors (not float 0) so the graph is valid even when
    # all demands are feasible and no shortfall terms are added.
    weighted_feasibility = torch.zeros((), device=device)
    feasibility_loss = torch.zeros((), device=device)
    # Detached record for the dual update — deliberately not part of the graph.
    shortfalls = torch.zeros(duals.shape[0], device=device)
    # The same quantity UNCLIPPED. `update_duals` needs the one-sided version
    # under the hinge (a slack demand's violation is 0, not -3) and the signed
    # one under the augmented penalty (where falling on slack is the point).
    constraint_g = torch.zeros(duals.shape[0], device=device)

    num_infeasible = 0
    num_violated = 0
    worst_margin_db = math.inf

    for demand in demands:
        threshold = modulation_config.required_snr_threshold(demand.bitrate_gbps)
        # The bar the constraint enforces is threshold + margin.
        bar_t = torch.tensor(threshold + margin_db, device=device, dtype=torch.float32)

        # SIGNED and live in the graph. The relu is taken separately below
        # for the two consumers that genuinely want a one-sided quantity —
        # the logged `feasibility_loss` and the hinge-mode dual record —
        # because the augmented term needs `g` itself: carrying force while
        # g < 0 is its entire purpose.
        g = bar_t - gsnr_preds[demand.id]
        shortfall = F.relu(g)

        if penalty == "augmented":
            z = F.relu(duals[demand.id] + rho * g)
            weighted_feasibility = weighted_feasibility + (
                (z * z - duals[demand.id] ** 2) / (2.0 * rho)
            )
        else:
            weighted_feasibility = weighted_feasibility + duals[demand.id] * shortfall

        # Unweighted sum kept so the logged feasibility_loss column stays
        # comparable across epochs while the duals are deliberately
        # non-stationary.
        feasibility_loss = feasibility_loss + shortfall

        shortfall_value = shortfall.item()
        shortfalls[demand.id] = shortfall_value
        constraint_g[demand.id] = g.item()
        if shortfall_value > 0:
            num_violated += 1

        # num_infeasible is measured against the BARE threshold, so it stays
        # directly comparable to every pre-change number in the investigation
        # record. num_violated is the stricter, margin-inclusive count the
        # duals actually act on.
        margin = gsnr_preds[demand.id].item() - threshold
        if margin < 0:
            num_infeasible += 1
        worst_margin_db = min(worst_margin_db, margin)

    # sum() over dict values — each is a scalar tensor live in the autograd graph.
    # Seed with a zero tensor so an empty demand list still yields a tensor
    # (bare sum() returns int 0, and .item() below would then raise).
    path_noise_loss = sum(path_noise_costs.values(), torch.zeros((), device=device))

    if lambda_waste != 0.0 and waste_cost is None:
        raise ValueError(
            "lambda_waste is non-zero but waste_cost was not provided"
        )
    waste_term = waste_cost if waste_cost is not None else torch.zeros((), device=device)

    total = (
        weighted_feasibility
        + lambda_dev * device_count
        + lambda_waste * waste_term
        + lambda_cost * path_noise_loss
    )

    metrics = {
        "feasibility_loss": feasibility_loss.item(),
        "weighted_feasibility_loss": weighted_feasibility.item(),
        "path_noise_loss": path_noise_loss.item(),
        "waste_cost": float(waste_term.item()),
        # The soft (mean-field) device count. NOT tau-invariant the way the
        # old num_regen_soft was — sigmoid(score/tau) moves with tau even on
        # frozen scores — so this is a within-epoch diagnostic only. The
        # cross-epoch comparison belongs to train.py's hard_num_devices,
        # which is a count of actual decisions and has no tau in it.
        "device_count": float(device_count.item()),
        "num_infeasible": num_infeasible,
        "num_violated": num_violated,
        "worst_margin_db": worst_margin_db if demands else math.nan,
        "shortfalls": shortfalls,
        "constraint_g": constraint_g,
    }
    return total, metrics


def update_duals(
    duals: torch.Tensor,
    shortfalls: torch.Tensor,
    *,
    eta: float,
    dual_max: float,
    decay: float = 0.0,
) -> torch.Tensor:
    """Dual ascent step: `lambda_d <- clamp(lambda_d + eta * shortfall_d, 0, lambda_max)`.

    Returns a NEW tensor; `duals` is not mutated, so a caller can keep a
    previous epoch's vector for diagnostics.

    The cap prevents runaway on a demand that is unsatisfiable for reasons the
    traffic-matrix preflight could not see — e.g. satisfiable only under a
    placement the router never reaches. Demands pinned at the cap are reported
    at the end of the run by `diffopt/train.py`, so a non-converging constraint
    surfaces as a named list rather than as silent oscillation.

    `decay` (default 0.0, i.e. no decay — pure ratchet, the original
    behaviour) multiplicatively relaxes a dual by `(1 - decay)` on any epoch
    where its shortfall is exactly 0. Without this, a dual that was bid up to
    buy feasibility during an early crisis never comes back down even once
    its demand is comfortably feasible, which pins `lambda_regen`'s (fixed,
    weak) downward pressure out of contention indefinitely — see
    open_followups.md item #3's over-provisioning investigation. Only
    satisfied demands decay; a demand still in shortfall keeps ascending,
    undamped, same as before.

    No autograd here by design: duals are Lagrange multipliers updated by an
    explicit ascent rule, not parameters optimised by Adam.
    """
    ascended = duals + eta * shortfalls
    relaxed = torch.where(shortfalls > 0, ascended, ascended * (1.0 - decay))
    return torch.clamp(relaxed, min=0.0, max=dual_max)
