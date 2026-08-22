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

    def hard_placement_mask(self) -> torch.Tensor:
        """The DEPLOYED placement: a deterministic (num_nodes,) bool mask.

        This is the single definition of "which regenerators are placed",
        used by checkpoint selection, the placement trajectory log, and
        every diagnostic. `sigmoid(logit / tau) > 0.5` is exactly
        `logit > 0` for any tau > 0, so thresholding the logit directly is
        tau-invariant by construction — no argument required, and no `tau`
        to pass in and get wrong.

        Detached: callers forward on this mask to evaluate a placement, and
        that evaluation must never build a graph back into the placement
        head.
        """
        return (self.regen_logits > 0.0).detach()

    def count_penalty(self, tau: float = 1.0) -> torch.Tensor:
        """The scalar `lambda_regen` multiplies — "how many regenerators".

        For this gate that is the probability MASS, `sum(p)`, which is what
        `compute_loss` has always summed. It is factored out here because
        the quantity is gate-specific: a hard-concrete L0 gate prices the
        expected COUNT instead, which is a different function of the
        parameters. Keeping the name gate-agnostic means `compute_loss` and
        `train.py` never branch on the gate.
        """
        return self.get_regen_probs(tau).sum()
