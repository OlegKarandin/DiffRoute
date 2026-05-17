"""RegenPlacement: learnable soft regenerator placement via sigmoid-gated logits."""
from __future__ import annotations

import torch
import torch.nn as nn


class RegenPlacement(nn.Module):
    """Soft regenerator placement model.

    Maintains one logit per node. Probabilities are obtained via sigmoid with
    an optional temperature parameter for annealing.

    No forward() method — call get_regen_probs() explicitly to avoid
    ambiguous semantics when this module is used as a sub-component.
    """

    def __init__(self, num_nodes: int) -> None:
        super().__init__()
        # zeros → sigmoid(0) = 0.5, neutral agnostic prior
        self.regen_logits = nn.Parameter(torch.zeros(num_nodes))

    def get_regen_probs(self, tau: float = 1.0) -> torch.Tensor:
        """Return per-node regenerator probabilities.

        Args:
            tau: Temperature for sigmoid. Lower values push probabilities toward
                 0 or 1 (sharper placement decisions). Never stored as an
                 attribute — always passed explicitly to prevent stale state.
        Returns:
            (num_nodes,) tensor of probabilities in (0, 1).
        """
        return torch.sigmoid(self.regen_logits / tau)
