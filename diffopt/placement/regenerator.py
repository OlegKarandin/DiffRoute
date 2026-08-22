"""RegenPlacement: learnable soft regenerator placement, sigmoid or L0 gated."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

SIGMOID = "sigmoid"
HARD_CONCRETE = "hard_concrete"


class RegenPlacement(nn.Module):
    """Soft regenerator placement model.

    Two gate parameterizations behind one contract:

    `sigmoid` (default, unchanged): one logit per node, probabilities via
    `sigmoid(logit / tau)` with `tau` annealed by the caller. `lambda_regen`
    prices the probability MASS, `sum(p)`.

    `hard_concrete` (Louizos et al., ICLR 2018): stretch-then-clamp gates
    that can be exactly 0 or exactly 1, and a penalty on the expected COUNT
    rather than the mass. This matters because L1-on-mass is what made the
    placement count track `lambda_regen`'s magnitude rather than need — no
    price in a three-order-of-magnitude sweep produced the true optimum. It
    retires `tau`: `beta` plays that role, and sparsity comes from the clamp
    rather than from sharpening a sigmoid.

    Known limitation, measured: on the toy replica this gate found the right
    COUNT (3) and the wrong three NODES (adjacent 3/4/5 instead of 4/7/10).
    It repairs the objective's semantics; it does not create the missing
    node-discriminating signal, which is gate dropout's job. Signal first,
    semantics second.

    No forward() method — call get_regen_probs() explicitly to avoid
    ambiguous semantics when this module is used as a sub-component.
    """

    def __init__(
        self,
        num_nodes: int,
        *,
        gate: str = SIGMOID,
        beta: float = 0.5,
        gamma: float = -0.1,
        zeta: float = 1.1,
    ) -> None:
        super().__init__()
        if gate not in (SIGMOID, HARD_CONCRETE):
            raise ValueError(
                f"Unknown gate {gate!r}; expected {SIGMOID!r} or {HARD_CONCRETE!r}"
            )
        self.gate = gate
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta

        if gate == SIGMOID:
            # zeros → sigmoid(0) = 0.5, neutral agnostic prior
            self.regen_logits = nn.Parameter(torch.zeros(num_nodes))
        else:
            # zeros → deterministic z = 0.5, the same neutral prior. Note
            # this leaves every gate OPEN under hard_placement_mask() at
            # epoch 0 (the open/closed boundary is at z > 0, not z > 0.5),
            # which is the L0 convention: gates start open and the count
            # penalty closes them. Checkpoint selection reports that epoch
            # honestly as num_placed = num_nodes and never selects it.
            self.log_alpha = nn.Parameter(torch.zeros(num_nodes))

    @property
    def _parameter(self) -> nn.Parameter:
        """The learned parameter, whichever gate is active. Lets train.py
        build its optimizer without branching on the gate."""
        return self.regen_logits if self.gate == SIGMOID else self.log_alpha

    def _open_threshold(self) -> float:
        """The value `sigmoid(log_alpha / beta)` must exceed for z > 0.

        z = clamp(s*(zeta-gamma) + gamma, 0, 1) is > 0 exactly when
        s > -gamma / (zeta - gamma). With the defaults that is 0.1/1.2.
        """
        return -self.gamma / (self.zeta - self.gamma)

    def get_regen_probs(self, tau: float = 1.0) -> torch.Tensor:
        """Return per-node regenerator probabilities.

        Args:
            tau: Temperature for the sigmoid gate. Lower values push
                 probabilities toward 0 or 1. Never stored as an attribute —
                 always passed explicitly to prevent stale state. IGNORED by
                 the hard_concrete gate, where `beta` plays this role.
        Returns:
            (num_nodes,) tensor. In (0, 1) for the sigmoid gate; in [0, 1]
            inclusive for hard_concrete, which can produce exact 0 and 1.
        """
        if self.gate == SIGMOID:
            return torch.sigmoid(self.regen_logits / tau)

        if self.training:
            # Binary concrete with a stretched interval, then clamped.
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1.0 - 1e-6)
            s = torch.sigmoid(
                (torch.log(u) - torch.log1p(-u) + self.log_alpha) / self.beta
            )
        else:
            s = torch.sigmoid(self.log_alpha / self.beta)
        return (s * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0)

    def hard_placement_mask(self) -> torch.Tensor:
        """The DEPLOYED placement: a deterministic (num_nodes,) bool mask.

        Gate-agnostic by design — this is the single definition of "which
        regenerators are placed", used by checkpoint selection, the placement
        trajectory log, and every diagnostic. Do not reintroduce
        `sigmoid(logit / tau) > 0.5` at a call site: it is correct only
        for the sigmoid gate, where it equals `logit > 0`.

        Detached: callers forward on this mask to evaluate a placement, and
        that evaluation must never build a graph back into the placement
        head.
        """
        if self.gate == SIGMOID:
            return (self.regen_logits > 0.0).detach()
        s = torch.sigmoid(self.log_alpha / self.beta)
        return (s > self._open_threshold()).detach()

    def count_penalty(self, tau: float = 1.0) -> torch.Tensor:
        """The scalar `lambda_regen` multiplies — "how many regenerators".

        Gate-specific by necessity. For `sigmoid` it is the probability MASS,
        `sum(p)`, which is what compute_loss has always summed. For
        `hard_concrete` it is the expected COUNT, `sum P(gate open)` — a
        different function of the parameters, and the reason that gate can
        express "three regenerators" in a way an L1-on-mass penalty cannot.

        `tau` is used only by the sigmoid gate.
        """
        if self.gate == SIGMOID:
            return self.get_regen_probs(tau).sum()
        shift = self.beta * math.log(-self.gamma / self.zeta)
        return torch.sigmoid(self.log_alpha - shift).sum()
