"""
Batched Dijkstra for WDM routing.

Edges are stored as src < dst (undirected). The algorithm treats each edge
as bidirectional and returns a binary indicator over the original edge IDs.
"""

from __future__ import annotations

import heapq
from collections import OrderedDict, deque
from typing import List, Optional, Tuple

import numpy as np


# Adjacency STRUCTURE cache. Only edge_weights change between calls: the
# node -> [(neighbour, edge_id)] mapping is a pure function of edge_index
# and num_nodes, and Topology is immutable after construction
# (Topology.undirected_edges is a cached_property with an explicit
# "nothing mutates after populate_optical" argument). Rebuilding it inside
# every dijkstra()/spfa() call cost 2E appends plus 3E int()/float()
# conversions -- 692 calls per epoch on ind_132's 346 demands, one Dijkstra
# forward and one SPFA backward each, doubled again by train.py's
# hard-evaluation pass.
#
# Keyed on edge_index's raw BYTES, not on the array object: surrogate.py
# materialises a fresh numpy array per call via .detach().cpu().numpy(), so
# an identity key would never hit. Bounded because diagnostics sweep
# synthetic graphs; real runs hold exactly one topology.
#
# ORDERING IS LOAD-BEARING. adj[u] must stay in ascending edge-id order,
# interleaving the u->v and v->u directions exactly as the original
# `for eid in range(E)` loop produced it. Relaxation uses strict `<`, so a
# tie between two equal-cost predecessors resolves to whichever edge id
# comes first; a differently-ordered CSR returns a different (still
# shortest) path and silently moves every downstream number.
_MAX_CACHED_TOPOLOGIES = 8
_adjacency_cache: "OrderedDict[Tuple[bytes, int], List[List[Tuple[int, int]]]]" = (
    OrderedDict()
)


def _adjacency(edge_index: np.ndarray, num_nodes: int) -> List[List[Tuple[int, int]]]:
    """node -> [(neighbour, edge_id), ...], ascending edge id. Cached.

    Returns the SHARED cached list. Callers must treat it as read-only —
    mutating it corrupts every later call on the same topology.
    """
    edge_index = np.asarray(edge_index)
    key = (edge_index.tobytes(), num_nodes)
    cached = _adjacency_cache.get(key)
    if cached is not None:
        return cached

    adj: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for eid in range(edge_index.shape[1]):
        u, v = int(edge_index[0, eid]), int(edge_index[1, eid])
        adj[u].append((v, eid))
        adj[v].append((u, eid))

    _adjacency_cache[key] = adj
    while len(_adjacency_cache) > _MAX_CACHED_TOPOLOGIES:
        _adjacency_cache.popitem(last=False)
    return adj


def spfa(
    edge_weights: np.ndarray,
    edge_index: np.ndarray,
    src: int,
    dst: int,
    num_nodes: int,
) -> Optional[np.ndarray]:
    """
    Shortest Path Faster Algorithm (SPFA / Bellman-Ford with queue).

    Handles negative edge weights. Used internally for the Vlastelica backward
    pass where perturbed weights can be negative.

    Returns (E,) binary float32 path indicator, or None if no path exists.
    """
    adj = _adjacency(edge_index, num_nodes)
    # One C-level conversion of the whole weight vector, so the relaxation
    # loop indexes Python floats rather than numpy scalars.
    weights = np.asarray(edge_weights, dtype=np.float64).tolist()
    num_edges = np.asarray(edge_index).shape[1]

    dist = np.full(num_nodes, np.inf)
    prev_node = np.full(num_nodes, -1, dtype=int)
    prev_edge = np.full(num_nodes, -1, dtype=int)
    in_queue = np.zeros(num_nodes, dtype=bool)

    dist[src] = 0.0
    queue: deque = deque([src])
    in_queue[src] = True
    relaxations = 0
    max_relaxations = num_nodes * num_edges  # cycle-detection guard

    while queue:
        u = queue.popleft()
        in_queue[u] = False
        for v, eid in adj[u]:
            nd = dist[u] + weights[eid]
            if nd < dist[v]:
                dist[v] = nd
                prev_node[v] = u
                prev_edge[v] = eid
                if not in_queue[v]:
                    queue.append(v)
                    in_queue[v] = True
                    relaxations += 1
                    if relaxations > max_relaxations:
                        # Negative cycle detected — return None
                        return None

    if dist[dst] == np.inf:
        return None

    path_indicator = np.zeros(num_edges, dtype=np.float32)
    node = dst
    while prev_edge[node] != -1:
        path_indicator[prev_edge[node]] = 1.0
        node = prev_node[node]

    return path_indicator


def dijkstra(
    edge_weights: np.ndarray,
    edge_index: np.ndarray,
    src: int,
    dst: int,
    num_nodes: int,
    return_order: bool = False,
):
    """
    Single-source single-target Dijkstra on an undirected graph.

    Parameters
    ----------
    edge_weights:
        (E,) float array of positive edge weights.
    edge_index:
        (2, E) int array; each edge stored once as (src < dst).
    src:
        Source node ID.
    dst:
        Destination node ID.
    num_nodes:
        Total number of nodes.
    return_order:
        If True, also return the path's edge IDs in src->dst traversal
        order. Dijkstra already builds `prev_node`/`prev_edge` while
        relaxing; walking that chain here is the SAME data a caller would
        otherwise recover by re-deriving traversal order from the
        unordered indicator (see `diffopt.pipeline`'s former
        `_reconstruct_path`, ~0.7 s/forward on `ind_132`/`constrained_stress`
        — docs/investigations/open_followups.md #7b / Finding 1 of
        pipeline_profile_and_restoration_scaling.md).

    Returns
    -------
    return_order=False (default): (E,) binary numpy array, 1 if the edge is
    on the shortest path, else 0. None if no path exists.

    return_order=True: (path_indicator, ordered_edges) where ordered_edges
    is a List[int] of edge IDs from src to dst. (None, None) if no path
    exists.
    """
    adj = _adjacency(edge_index, num_nodes)
    weights = np.asarray(edge_weights, dtype=np.float64).tolist()
    num_edges = np.asarray(edge_index).shape[1]

    # Dijkstra
    dist = np.full(num_nodes, np.inf)
    prev_node = np.full(num_nodes, -1, dtype=int)
    prev_edge = np.full(num_nodes, -1, dtype=int)
    dist[src] = 0.0

    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == dst:
            break
        for v, eid in adj[u]:
            nd = dist[u] + weights[eid]
            if nd < dist[v]:
                dist[v] = nd
                prev_node[v] = u
                prev_edge[v] = eid
                heapq.heappush(heap, (nd, v))

    if dist[dst] == np.inf:
        return (None, None) if return_order else None  # no path

    # Reconstruct path edges, walking dst -> src via the chain Dijkstra
    # already built.
    path_indicator = np.zeros(num_edges, dtype=np.float32)
    ordered_edges: List[int] = []
    node = dst
    while prev_edge[node] != -1:
        eid = int(prev_edge[node])
        path_indicator[eid] = 1.0
        ordered_edges.append(eid)
        node = int(prev_node[node])

    if not return_order:
        return path_indicator
    ordered_edges.reverse()  # src -> dst order
    return path_indicator, ordered_edges
