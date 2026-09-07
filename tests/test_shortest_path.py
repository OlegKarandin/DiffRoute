"""
Direct tests for spfa — the reason Dijkstra was replaced in the Vlastelica
backward pass: perturbed weights
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
    That is the literal "Dijkstra hangs on negative weights"
    docs/architecture/invariants.md warns about — calling it here would
    hang the test suite, not just this test.

    spfa must not hang: its relaxation-count guard (shortest_path.py:
    63-65) is global, not scoped to edges on the src->dst path, so it
    still trips and returns None even though the cycle never touches dst.
    """
    edge_index = _edges([(0, 1), (1, 2), (0, 3)])
    edge_weights = np.array([1.0, -3.0, 10.0])

    assert spfa(edge_weights, edge_index, src=0, dst=3, num_nodes=4) is None


def test_adjacency_structure_is_cached_across_calls():
    """The structure is a pure function of (edge_index, num_nodes). Callers
    hand in a FRESH numpy array every time -- surrogate.py materialises one
    per forward and one per backward via .detach().cpu().numpy() -- so the
    cache must key on the array's CONTENT, not its identity."""
    from diffopt.routing import shortest_path as sp

    sp._adjacency_cache.clear()
    edge_index = _edges([(0, 1), (1, 2), (0, 2)])

    first = sp._adjacency(edge_index, 3)
    second = sp._adjacency(np.array(edge_index, copy=True), 3)

    assert first is second
    assert len(sp._adjacency_cache) == 1


def test_adjacency_is_in_ascending_edge_id_order():
    """Load-bearing, and the reason this is not a free refactor. The old
    code filled adj[u] inside `for eid in range(E)`, interleaving the u->v
    and v->u directions in edge-id order. Relaxation uses strict `<`, so a
    tie between two equal-cost predecessors resolves to whichever edge id
    was visited first. Reordering silently changes which path is returned
    on a tied graph."""
    from diffopt.routing import shortest_path as sp

    sp._adjacency_cache.clear()
    adj = sp._adjacency(_edges([(0, 1), (0, 2), (1, 2)]), 3)

    assert adj[0] == [(1, 0), (2, 1)]
    assert adj[1] == [(0, 0), (2, 2)]
    assert adj[2] == [(0, 1), (1, 2)]


def test_tied_shortest_paths_resolve_to_the_lowest_edge_id():
    """The concrete regression guard for the ordering above. A square with
    two equal-cost routes 0->1->3 and 0->2->3: Dijkstra pops node 1 before
    node 2 (heap tie broken by node id) and settles node 3 through edge 2,
    so edges 0 and 2 are on the path and edges 1 and 3 are not. Pinning the
    literal indicator is the point -- an assertion that 'some 2-hop path'
    was found would not catch a reordering."""
    edge_index = _edges([(0, 1), (0, 2), (1, 3), (2, 3)])
    weights = np.array([1.0, 1.0, 1.0, 1.0])

    path = dijkstra(weights, edge_index, 0, 3, 4)

    np.testing.assert_array_equal(path, np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32))


def test_cached_structure_does_not_freeze_the_weights():
    """The structure is cached; the weights are not. Two calls on the same
    edge_index with different weights must return different paths."""
    edge_index = _edges([(0, 1), (0, 2), (1, 3), (2, 3)])

    cheap_top = dijkstra(np.array([1.0, 9.0, 1.0, 9.0]), edge_index, 0, 3, 4)
    cheap_bottom = dijkstra(np.array([9.0, 1.0, 9.0, 1.0]), edge_index, 0, 3, 4)

    np.testing.assert_array_equal(cheap_top, np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(cheap_bottom, np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# return_order — Dijkstra's own
# prev-chain already has the src->dst edge order, so a caller should not
# need a second graph walk to recover it from the unordered indicator.
# ---------------------------------------------------------------------------

def test_return_order_false_is_the_default_and_unchanged():
    """Every existing caller relies on the plain (E,) indicator return —
    return_order defaults to False and must not change that contract."""
    edge_index = _edges([(0, 1), (1, 2), (0, 2)])
    edge_weights = np.array([1.0, 1.0, 5.0])

    path = dijkstra(edge_weights, edge_index, src=0, dst=2, num_nodes=3)

    assert isinstance(path, np.ndarray)
    np.testing.assert_array_equal(path, np.array([1.0, 1.0, 0.0], dtype=np.float32))


def test_return_order_true_gives_the_src_to_dst_edge_order():
    """0-1-2, edges e0 (0-1) then e1 (1-2) — ordered_edges must read [0, 1],
    not just {0, 1} in some order, and path_indicator must be unchanged."""
    edge_index = _edges([(0, 1), (1, 2), (0, 2)])
    edge_weights = np.array([1.0, 1.0, 5.0])

    path, ordered_edges = dijkstra(
        edge_weights, edge_index, src=0, dst=2, num_nodes=3, return_order=True
    )

    np.testing.assert_array_equal(path, np.array([1.0, 1.0, 0.0], dtype=np.float32))
    assert ordered_edges == [0, 1]


def test_return_order_true_matches_a_manual_walk_of_the_indicator():
    """On a real multi-hop path (a tied 4-node square from the test above,
    forced onto the top route), ordered_edges must be exactly the same
    src->dst walk a caller would get by manually tracing the indicator's
    active edges from src — which is what the deleted
    diffopt.pipeline._reconstruct_path used to do as a second graph walk
    (verified equivalent on all 346 real ind_132 demands)."""
    edge_index = _edges([(0, 1), (0, 2), (1, 3), (2, 3)])
    weights = np.array([1.0, 9.0, 1.0, 9.0])

    path, ordered_edges = dijkstra(weights, edge_index, 0, 3, 4, return_order=True)

    np.testing.assert_array_equal(path, np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32))

    # Manual walk: from src (0), follow active edges until dst (3).
    edges_by_id = {i: (int(edge_index[0, i]), int(edge_index[1, i]))
                   for i in range(edge_index.shape[1])}
    adj = {}
    for eid, active in enumerate(path):
        if active > 0.5:
            u, v = edges_by_id[eid]
            adj.setdefault(u, []).append((v, eid))
            adj.setdefault(v, []).append((u, eid))
    manual_order = []
    node, dst = 0, 3
    while node != dst:
        for nxt, eid in adj[node]:
            if eid not in manual_order:
                manual_order.append(eid)
                node = nxt
                break

    assert ordered_edges == manual_order


def test_return_order_true_no_path_returns_none_none():
    edge_index = _edges([(0, 1)])
    edge_weights = np.array([1.0])

    result = dijkstra(
        edge_weights, edge_index, src=0, dst=2, num_nodes=3, return_order=True
    )

    assert result == (None, None)


def test_adjacency_cache_is_bounded():
    """A diagnostic sweeping many synthetic graphs must not grow the cache
    without limit. Real use has one topology per process."""
    from diffopt.routing import shortest_path as sp

    sp._adjacency_cache.clear()
    for n in range(sp._MAX_CACHED_TOPOLOGIES + 3):
        sp._adjacency(_edges([(0, 1), (1, 2)]) + n, 3 + n)

    assert len(sp._adjacency_cache) == sp._MAX_CACHED_TOPOLOGIES
