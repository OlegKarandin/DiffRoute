"""
Vlastelica et al. (ICLR 2020) blackbox differentiation through Dijkstra.

Reference: "Differentiation of Blackbox Combinatorial Solvers"
           Vlastelica et al., ICLR 2020
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from diffopt.routing.shortest_path import dijkstra, spfa


class DijkstraSurrogate(torch.autograd.Function):
    """
    Autograd Function wrapping Dijkstra with Vlastelica surrogate gradients.

    Forward: run exact Dijkstra, return binary path indicator.
    Backward: perturb edge weights by +lambda * grad_output, re-run Dijkstra,
              compute finite-difference surrogate gradient.
    """

    @staticmethod
    def forward(
        ctx,
        edge_weights: torch.Tensor,
        edge_index: torch.Tensor,
        src: int,
        dst: int,
        num_nodes: int,
        lambda_: float,
    ) -> torch.Tensor:
        # Run Dijkstra on detached numpy arrays
        w_np = edge_weights.detach().cpu().numpy().astype(np.float64)
        ei_np = edge_index.detach().cpu().numpy()

        path_np = dijkstra(w_np, ei_np, src, dst, num_nodes)
        if path_np is None:
            raise ValueError(f"No path found from {src} to {dst}")

        path = torch.tensor(path_np, dtype=torch.float32, device=edge_weights.device)

        # Save tensors and scalars for backward
        ctx.save_for_backward(edge_weights, edge_index, path)
        ctx._src = src
        ctx._dst = dst
        ctx._num_nodes = num_nodes
        ctx._lambda = lambda_
        # Backward needs the same numpy edge_index. Converting it there
        # repeats a .detach().cpu().numpy() per demand per step for a
        # tensor that is a registered buffer and never changes.
        ctx._ei_np = ei_np

        return path

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        edge_weights, edge_index, path = ctx.saved_tensors
        src = ctx._src
        dst = ctx._dst
        num_nodes = ctx._num_nodes
        lambda_ = ctx._lambda

        # Perturbed weights: c_target = w + lambda * grad_output
        w_np = edge_weights.detach().cpu().numpy().astype(np.float64)
        g_np = grad_output.detach().cpu().numpy().astype(np.float64)

        # Vlastelica (ICLR 2020, Theorem 3.1): perturb by +λ*ŷ where ŷ = -∂L/∂z.
        # Since grad_output = ∂L/∂z (PyTorch convention), ŷ = -grad_output, so
        # c - λ*ŷ = c - λ*(-grad) = c + λ*grad.
        c_target = w_np + lambda_ * g_np

        # Use SPFA in backward: perturbed weights can be negative
        path_target_np = spfa(c_target, ctx._ei_np, src, dst, num_nodes)
        if path_target_np is None:
            # If perturbed graph has no path, use zero gradient
            path_target_np = path.detach().cpu().numpy().astype(np.float64)

        path_star = path.detach().cpu().numpy().astype(np.float64)

        # Surrogate gradient: -(1/lambda) * (path_star - path_target)
        grad_weights_np = -(1.0 / lambda_) * (path_star - path_target_np.astype(np.float64))
        grad_weights = torch.tensor(
            grad_weights_np, dtype=edge_weights.dtype, device=edge_weights.device
        )

        # Return gradients: (edge_weights, edge_index, src, dst, num_nodes, lambda_)
        # Only edge_weights is a tensor; rest get None
        return grad_weights, None, None, None, None, None


def surrogate_shortest_path(
    edge_weights: torch.Tensor,
    edge_index: torch.Tensor,
    src: int,
    dst: int,
    num_nodes: int,
    lambda_: float = 10.0,
) -> torch.Tensor:
    """
    Convenience wrapper for DijkstraSurrogate.apply().

    Parameters
    ----------
    edge_weights:
        (E,) tensor with requires_grad=True for gradient flow.
    edge_index:
        (2, E) LongTensor; edges stored as src < dst.
    src, dst:
        Source and destination node IDs.
    num_nodes:
        Total number of nodes.
    lambda_:
        Vlastelica perturbation strength. Higher → larger gradient signal,
        less accurate surrogate. Typical range: 1–100.

    Returns
    -------
    (E,) float tensor: binary path indicator (differentiable via surrogate).
    """
    return DijkstraSurrogate.apply(edge_weights, edge_index, src, dst, num_nodes, lambda_)
