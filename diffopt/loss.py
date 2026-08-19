"""Loss functions for the end-to-end DiffONet pipeline."""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from diffopt.demands import Demand
from diffopt.modulation import ModulationConfig


def compute_loss(
    gsnr_preds: Dict[int, torch.Tensor],
    path_noise_costs: Dict[int, torch.Tensor],
    demands: List[Demand],
    regen_probs: torch.Tensor,
    modulation_config: ModulationConfig,
    duals: torch.Tensor,
    margin_db: float = 0.5,
    lambda_regen: float = 1.0,
    lambda_cost: float = 0.01,
) -> Tuple[torch.Tensor, dict]:
    """Compute the constrained training loss.

        L = sum_d lambda_d * relu(thr_d + delta - gsnr_d)
          + lambda_regen * sum_n p_n
          + lambda_cost  * path_noise

    This replaces a weighted sum of three soft penalties in which feasibility
    competed with regenerator count on a fixed exchange rate. `lambda_regen`
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
        regen_probs:  (num_nodes,) tensor from RegenPlacement.get_regen_probs().
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
        lambda_regen: Weight on regenerator count penalty. Fixed, not tuned.
        lambda_cost:  Weight on the ASE-denominated path-noise regulariser. Not the
                      primary routing signal -- the STE in pipeline.forward supplies that.

    Returns:
        (total_loss, metrics_dict). `metrics["shortfalls"]` is a detached
        (num_demands,) tensor indexed by `Demand.id`, to be fed straight into
        `update_duals` after the optimizer step.
    """
    device = regen_probs.device

    # Start as zero tensors (not float 0) so the graph is valid even when
    # all demands are feasible and no shortfall terms are added.
    weighted_feasibility = torch.zeros((), device=device)
    feasibility_loss = torch.zeros((), device=device)
    # Detached record for the dual update — deliberately not part of the graph.
    shortfalls = torch.zeros(duals.shape[0], device=device)

    num_infeasible = 0
    num_violated = 0
    worst_margin_db = math.inf

    for demand in demands:
        threshold = modulation_config.required_snr_threshold(demand.bitrate_gbps)
        # The bar the constraint enforces is threshold + margin.
        bar_t = torch.tensor(threshold + margin_db, device=device, dtype=torch.float32)
        shortfall = F.relu(bar_t - gsnr_preds[demand.id])

        weighted_feasibility = weighted_feasibility + duals[demand.id] * shortfall
        # Unweighted sum kept so the logged feasibility_loss column stays
        # comparable across epochs while the duals are deliberately
        # non-stationary.
        feasibility_loss = feasibility_loss + shortfall

        shortfall_value = shortfall.item()
        shortfalls[demand.id] = shortfall_value
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

    regen_loss = regen_probs.sum()

    # sum() over dict values — each is a scalar tensor live in the autograd graph.
    # Seed with a zero tensor so an empty demand list still yields a tensor
    # (bare sum() returns int 0, and .item() below would then raise).
    path_noise_loss = sum(path_noise_costs.values(), torch.zeros((), device=device))

    total = (
        weighted_feasibility
        + lambda_regen * regen_loss
        + lambda_cost * path_noise_loss
    )

    metrics = {
        "feasibility_loss": feasibility_loss.item(),
        "weighted_feasibility_loss": weighted_feasibility.item(),
        "regen_loss": regen_loss.item(),
        "path_noise_loss": path_noise_loss.item(),
        # (regen_probs > 0.5) is exactly (regen_logits > 0) for any tau > 0,
        # since sigmoid is monotone and sigmoid(0) = 0.5 — so this count is
        # tau-invariant and safe to compare across an annealing run, unlike
        # regen_loss (87% of whose observed 66 -> 18 fall came from tau).
        "num_regen_soft": int((regen_probs > 0.5).sum().item()),
        "num_infeasible": num_infeasible,
        "num_violated": num_violated,
        "worst_margin_db": worst_margin_db if demands else math.nan,
        "shortfalls": shortfalls,
    }
    return total, metrics


def update_duals(
    duals: torch.Tensor,
    shortfalls: torch.Tensor,
    *,
    eta: float,
    dual_max: float,
) -> torch.Tensor:
    """Dual ascent step: `lambda_d <- clamp(lambda_d + eta * shortfall_d, 0, lambda_max)`.

    Returns a NEW tensor; `duals` is not mutated, so a caller can keep a
    previous epoch's vector for diagnostics.

    The cap prevents runaway on a demand that is unsatisfiable for reasons the
    traffic-matrix preflight could not see — e.g. satisfiable only under a
    placement the router never reaches. Demands pinned at the cap are reported
    at the end of the run by `diffopt/train.py`, so a non-converging constraint
    surfaces as a named list rather than as silent oscillation.

    No autograd here by design: duals are Lagrange multipliers updated by an
    explicit ascent rule, not parameters optimised by Adam.
    """
    return torch.clamp(duals + eta * shortfalls, min=0.0, max=dual_max)
