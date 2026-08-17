"""
Direct tests for spfa — the reason Dijkstra was replaced in the Vlastelica
backward pass (CLAUDE.md, Phase 1b correction #2): perturbed weights
c + lambda*grad can be negative, and Dijkstra hangs or is wrong on those.

Real signature (read from diffopt/routing/shortest_path.py; the brief's
`spfa(num_nodes=..., edges=[(u, v, w), ...], src=..., dst=...) -> (dist, path)`
was a guess and does not match):

    spfa(edge_weights: np.ndarray, edge_index: np.ndarray, src: int,
         dst: int, num_nodes: int) -> Optional[np.ndarray]

- `edge_weights` is a flat (E,) float array; `edge_index` is (2, E) int,
  one row per undirected edge stored once as (src < dst) — same shape/
  convention as `dijkstra`'s.
- The return value is a single (E,) binary float32 **path indicator**
  (1.0 for edges on the shortest path), not a (distance, node-path) tuple.
- "No path" and "negative cycle detected" are NOT distinguished in the
  return value: both produce `None` (see shortest_path.py:63-68 — the
  relaxation-count guard at lines 63-65 returns `None` directly on a
  suspected negative cycle; falling through with `dist[dst] == inf`
  after the queue drains returns `None` too, at line 68).

Undirected-graph subtlety that shapes every negative-weight test below:
because every edge is relaxed in both directions from a single row of
`edge_index`, a single negative-weight edge is *always* a negative cycle
here (walk it forward, then back, for an unbounded decrease). So a
negative edge can never legitimately shorten a path in this codebase —
the only correct behaviour when one appears is to detect the cycle and
refuse to answer, which is what these tests check `spfa` actually does
(and `dijkstra` actually does not).
"""
import numpy as np
import pytest

from diffopt.routing.shortest_path import dijkstra, spfa


def _edges(pairs):
    """Build the (2, E) edge_index array `spfa`/`dijkstra` expect from a
    list of (src, dst) pairs with src < dst."""
    return np.array([[p[0] for p in pairs], [p[1] for p in pairs]])


# ---------------------------------------------------------------------------
# Baseline: all-positive weights, spfa must agree with dijkstra exactly.
# ---------------------------------------------------------------------------

def test_spfa_matches_dijkstra_on_positive_weights():
    # 0-1-2 costs 1+1=2, cheaper than the direct 0-2 edge at 5.
    edge_index = _edges([(0, 1), (1, 2), (0, 2)])
    edge_weights = np.array([1.0, 1.0, 5.0])

    spfa_path = spfa(edge_weights, edge_index, src=0, dst=2, num_nodes=3)
    dijkstra_path = dijkstra(edge_weights, edge_index, src=0, dst=2, num_nodes=3)

    assert spfa_path is not None
    np.testing.assert_array_equal(
        spfa_path, np.array([1.0, 1.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(spfa_path, dijkstra_path)


# ---------------------------------------------------------------------------
# No path: dst genuinely unreachable, no negative weights involved at all.
# ---------------------------------------------------------------------------

def test_spfa_returns_none_when_dst_is_unreachable():
    # Node 2 is disconnected from {0, 1}.
    edge_index = _edges([(0, 1)])
    edge_weights = np.array([1.0])

    assert spfa(edge_weights, edge_index, src=0, dst=2, num_nodes=3) is None
    assert dijkstra(edge_weights, edge_index, src=0, dst=2, num_nodes=3) is None


# ---------------------------------------------------------------------------
# The case the brief asked for: a negative edge that genuinely misleads
# Dijkstra, not just a graph that happens to work.
# ---------------------------------------------------------------------------

def test_spfa_detects_the_negative_cycle_dijkstra_is_fooled_by():
    """
    Edges: 0-1 (5.0), 1-2 (-4.0), 0-2 (3.0).

    Because edges are undirected, the 1-2 edge can be walked back and
    forth (1->2->1->2->...) for an unbounded decrease: a genuine negative
    cycle reachable from node 0, so there is no well-defined shortest
    distance from 0 to 2.

    Dijkstra doesn't know that: after relaxing from node 0 it has
    dist[1]=5 and dist[2]=3 queued, so its min-heap pops (3, node 2)
    *before* ever relaxing the -4.0 edge, sees `u == dst`, and breaks —
    settling on the direct edge and returning a confident, wrong,
    finite answer. This is exactly "settles early" from the task brief:
    verified below by asserting dijkstra's own (wrong) output.

    spfa keeps relaxing across both directions of the 1-2 edge until its
    relaxation-count guard trips (shortest_path.py:63-65) and correctly
    returns None instead of fabricating a distance.
    """
    edge_index = _edges([(0, 1), (1, 2), (0, 2)])
    edge_weights = np.array([5.0, -4.0, 3.0])

    # Dijkstra is genuinely misled: it settles node 2 via the direct edge
    # before the negative edge is ever considered, and returns that as if
    # it were correct.
    dijkstra_path = dijkstra(edge_weights, edge_index, src=0, dst=2, num_nodes=3)
    assert dijkstra_path is not None
    np.testing.assert_array_equal(
        dijkstra_path, np.array([0.0, 0.0, 1.0], dtype=np.float32)
    )

    # spfa recognises the negative cycle and refuses to answer, rather
    # than reproducing Dijkstra's wrong answer.
    assert spfa(edge_weights, edge_index, src=0, dst=2, num_nodes=3) is None


def test_spfa_detects_a_negative_cycle_even_when_dst_does_not_touch_it():
    """
    The negative cycle lives on edge 1-2 (weight -3.0), reachable from src
    (node 0) but off the direct path to dst (node 3, reached only via the
    separate, all-positive 0-3 edge).

    NOTE: dijkstra is deliberately NOT called on this graph. Manually
    verified (outside the test suite, via a subprocess with a hard
    timeout) that dijkstra genuinely infinite-loops here: node 3's
    distance is fixed early, but the 1-2 pair's distances decrease
    without bound and always sort ahead of node 3's heap entries, so
    dijkstra's `while heap` loop never drains and node 3 is never popped.
    That is the literal "Dijkstra hangs on negative weights" CLAUDE.md
    warns about — calling it here would hang the test suite, not just
    this test.

    spfa must not hang: its relaxation-count guard (shortest_path.py:
    63-65) is global, not scoped to edges on the src->dst path, so it
    still trips and returns None even though the cycle never touches dst.
    """
    edge_index = _edges([(0, 1), (1, 2), (0, 3)])
    edge_weights = np.array([1.0, -3.0, 10.0])

    assert spfa(edge_weights, edge_index, src=0, dst=3, num_nodes=4) is None
