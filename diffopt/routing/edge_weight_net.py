"""EdgeWeightNet: maps per-edge features to positive routing costs."""
from __future__ import annotations

import torch
import torch.nn as nn


class EdgeWeightNet(nn.Module):
    """Predict positive edge weights from topology + regenerator probability features.

    Input: (E, 7) — 5 topology features + regen_prob at src + regen_prob at dst.
    Output: (E, 1) — strictly positive weights via Softplus.
    """

    def __init__(self, input_dim: int = 7, hidden1: int = 64, hidden2: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
            nn.Softplus(),
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_features: (E, 7)
        Returns:
            (E, 1) positive weights
        """
        return self.net(edge_features)
