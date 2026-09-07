"""
Unit tests for the Vlastelica surrogate gradient through Dijkstra.

4-node graph topology:
  Nodes: S=0, A=1, B=2, T=3
  Edges (undirected, stored as src < dst):
    e0: S-A (0-1)
    e1: S-B (0-2)
    e2: A-T (1-3)
    e3: B-T (2-3)

  Two paths:
    S-A-T: edges e0, e2
    S-B-T: edges e1, e3
"""

import torch

from diffopt.routing.surrogate import surrogate_shortest_path


# Graph constants
NUM_NODES = 4
S, A, B, T = 0, 1, 2, 3
# edge_index: (2, 4) — edges stored as src < dst
EDGE_INDEX = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long)
# Edge names for readability
E_SA, E_SB, E_AT, E_BT = 0, 1, 2, 3


def make_weights(*vals) -> torch.Tensor:
    """Create edge weight tensor (E=4) from positional args."""
    return torch.tensor(list(vals), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Test 1: Correct path selection
# ---------------------------------------------------------------------------

def test_correct_path_selection():
    """S-A-T cheaper than S-B-T: path indicator should select e0, e2."""
    weights = make_weights(1.0, 2.0, 1.0, 2.0)  # S-A-T cost=2, S-B-T cost=4
    path, ordered_edges = surrogate_shortest_path(weights, EDGE_INDEX, S, T, NUM_NODES)

    assert path[E_SA].item() == 1.0, "Expected S-A edge on path"
    assert path[E_AT].item() == 1.0, "Expected A-T edge on path"
    assert path[E_SB].item() == 0.0, "Expected S-B edge NOT on path"
    assert path[E_BT].item() == 0.0, "Expected B-T edge NOT on path"


def test_ordered_edges_is_src_to_dst_traversal_order():
    """`ordered_edges` must be e0, e2 in that order —
    the S->A->T walk, not just the same two edges in any order."""
    weights = make_weights(1.0, 2.0, 1.0, 2.0)
    _, ordered_edges = surrogate_shortest_path(weights, EDGE_INDEX, S, T, NUM_NODES)
    assert ordered_edges == [E_SA, E_AT]


# ---------------------------------------------------------------------------
# Test 2: Non-zero gradient for path edges
# ---------------------------------------------------------------------------

def test_nonzero_gradient():
    """
    Loss penalises A-path edges (wants B-path). Gradient on A-path
    edges should be nonzero.
    """
    weights = make_weights(1.0, 3.0, 1.0, 3.0)
    weights.requires_grad_(True)

    path, _ = surrogate_shortest_path(weights, EDGE_INDEX, S, T, NUM_NODES, lambda_=10.0)

    # Loss: penalise use of A-path edges (push optimizer to prefer B-path)
    loss = path[E_SA] + path[E_AT]
    loss.backward()

    assert weights.grad is not None, "No gradient computed"
    # A-path edges should have nonzero gradient
    assert weights.grad[E_SA].item() != 0.0, "Gradient on S-A edge should be nonzero"
    assert weights.grad[E_AT].item() != 0.0, "Gradient on A-T edge should be nonzero"


# ---------------------------------------------------------------------------
# Test 3: Gradient direction — A-path grad positive, B-path grad negative
# ---------------------------------------------------------------------------

def test_gradient_direction():
    """
    Loss = path[SA] + path[AT] (penalise A-path).
    Vlastelica backward gives ∂L/∂w where w are edge costs.
    Gradient descent (w -= lr * grad) should increase A-edge costs and
    decrease B-edge costs → so grad must be NEGATIVE on A-edges and POSITIVE
    on B-edges (then w -= lr*neg → w increases on A-edges).
    """
    weights = make_weights(1.0, 3.0, 1.0, 3.0)
    weights.requires_grad_(True)

    path, _ = surrogate_shortest_path(weights, EDGE_INDEX, S, T, NUM_NODES, lambda_=10.0)
    loss = path[E_SA] + path[E_AT]
    loss.backward()

    grad = weights.grad
    assert grad[E_SA].item() < 0.0, (
        f"Expected negative gradient on S-A (grad descent → cost increases), got {grad[E_SA].item()}"
    )
    assert grad[E_AT].item() < 0.0, (
        f"Expected negative gradient on A-T (grad descent → cost increases), got {grad[E_AT].item()}"
    )
    assert grad[E_SB].item() > 0.0, (
        f"Expected positive gradient on S-B (grad descent → cost decreases), got {grad[E_SB].item()}"
    )
    assert grad[E_BT].item() > 0.0, (
        f"Expected positive gradient on B-T (grad descent → cost decreases), got {grad[E_BT].item()}"
    )


# ---------------------------------------------------------------------------
# Test 4: Zero gradient for disconnected / irrelevant edge
# ---------------------------------------------------------------------------

def test_zero_gradient_for_irrelevant_edge():
    """
    Add a disconnected edge C→D (node 4→5). Its gradient should be exactly 0
    since it is never on any path.
    """
    # Extended graph with 6 nodes; extra edge e4: C(4)-D(5)
    num_nodes_ext = 6
    edge_index_ext = torch.tensor(
        [[0, 0, 1, 2, 4],
         [1, 2, 3, 3, 5]],
        dtype=torch.long
    )
    weights_ext = make_weights(1.0, 3.0, 1.0, 3.0, 1.0)  # 5 edges
    weights_ext.requires_grad_(True)

    path, _ = surrogate_shortest_path(
        weights_ext, edge_index_ext, S, T, num_nodes_ext, lambda_=10.0
    )
    # Penalise A-path edges: this creates non-zero gradients on SA/AT/SB/BT
    # but the disconnected edge C-D (index 4) must still have zero gradient
    loss = path[E_SA] + path[E_AT]
    loss.backward()

    # Edge e4 (C-D) is disconnected from S-T; gradient must be zero
    assert weights_ext.grad[4].item() == 0.0, (
        f"Disconnected edge should have zero gradient, got {weights_ext.grad[4].item()}"
    )


# ---------------------------------------------------------------------------
# Test 5: Path switch under gradient descent
# ---------------------------------------------------------------------------

def test_path_switch_under_gradient_descent():
    """
    Start with S-A-T as cheapest path. Apply gradient descent steps with a
    loss that penalises A-path edges. After enough steps, weights should
    favour the B-path.
    """
    # Initial: S-A-T cheaper
    w = torch.tensor([1.0, 5.0, 1.0, 5.0], dtype=torch.float32)

    for step in range(50):
        w = w.detach().clone().requires_grad_(True)
        path, _ = surrogate_shortest_path(w, EDGE_INDEX, S, T, NUM_NODES, lambda_=5.0)
        # Loss: strongly penalise A-path edges
        loss = 10.0 * (path[E_SA] + path[E_AT])
        loss.backward()

        with torch.no_grad():
            w = w - 0.5 * w.grad

        # Ensure weights stay positive (required for Dijkstra)
        w = torch.clamp(w, min=0.01)

        # Check if B-path is now selected
        with torch.no_grad():
            test_path, _ = surrogate_shortest_path(
                w.requires_grad_(False), EDGE_INDEX, S, T, NUM_NODES
            )
            if test_path[E_SB].item() == 1.0 and test_path[E_BT].item() == 1.0:
                return  # path switched — test passes

    # If we didn't switch, check final state
    final_path, _ = surrogate_shortest_path(
        w.requires_grad_(False), EDGE_INDEX, S, T, NUM_NODES
    )
    assert final_path[E_SB].item() == 1.0 and final_path[E_BT].item() == 1.0, (
        f"Path did not switch to B-path after gradient descent. "
        f"Final weights: {w.tolist()}, path: {final_path.tolist()}"
    )
