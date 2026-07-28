"""Loss functions for the end-to-end DiffONet pipeline."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from diffopt.demands import Demand
from diffopt.modulation import ModulationConfig


def compute_loss(
    gsnr_preds: Dict[int, torch.Tensor],
    path_costs: Dict[int, torch.Tensor],
    demands: List[Demand],
    regen_probs: torch.Tensor,
    modulation_config: ModulationConfig,
    lambda_regen: float = 1.0,
    lambda_infeasible: float = 10.0,
    lambda_cost: float = 0.01,
) -> Tuple[torch.Tensor, dict]:
    """Compute combined training loss.

    Args:
        gsnr_preds:   demand_id → scalar GSNR tensor (dB).
        path_costs:   demand_id → scalar (path_indicator · edge_weights).sum().
                      Must be live in the autograd graph — this is the sole
                      gradient path into EdgeWeightNet via the Vlastelica surrogate.
        demands:      List of Demand namedtuples.
        regen_probs:  (num_nodes,) tensor from RegenPlacement.get_regen_probs().
        modulation_config: Bitrate → SNR threshold lookup.
        lambda_regen:     Weight on regenerator count penalty.
        lambda_infeasible: Weight on GSNR feasibility shortfall.
        lambda_cost:      Weight on path cost; enables EdgeWeightNet gradient.

    Returns:
        (total_loss, metrics_dict)
    """
    device = regen_probs.device

    # Start as a zero tensor (not float 0) so the graph is valid even when
    # all demands are feasible and no shortfall terms are added.
    feasibility_loss = torch.zeros(1, device=device)
    num_infeasible = 0

    for demand in demands:
        threshold = modulation_config.required_snr_threshold(demand.bitrate_gbps)
        threshold_t = torch.tensor(threshold, device=device, dtype=torch.float32)
        shortfall = F.relu(threshold_t - gsnr_preds[demand.id])
        feasibility_loss = feasibility_loss + shortfall
        if shortfall.item() > 0:
            num_infeasible += 1

    regen_loss = regen_probs.sum()

    # sum() over dict values — each is a scalar tensor live in the autograd graph
    path_cost_loss = sum(path_costs.values())

    total = (
        lambda_infeasible * feasibility_loss
        + lambda_regen * regen_loss
        + lambda_cost * path_cost_loss
    )

    metrics = {
        "feasibility_loss": feasibility_loss.item(),
        "regen_loss": regen_loss.item(),
        "path_cost_loss": path_cost_loss.item(),
        "num_regen_soft": int((regen_probs > 0.5).sum().item()),
        "num_infeasible": num_infeasible,
    }
    return total, metrics
