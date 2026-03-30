"""
Batched Dijkstra for WDM routing.

Edges are stored as src < dst (undirected). The algorithm treats each edge
as bidirectional and returns a binary indicator over the original edge IDs.
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import List, Optional, Tuple

import numpy as np


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
    adj: List[List[Tuple[float, int, int]]] = [[] for _ in range(num_nodes)]
    for eid in range(edge_index.shape[1]):
        u, v = int(edge_index[0, eid]), int(edge_index[1, eid])
        w = float(edge_weights[eid])
        adj[u].append((w, v, eid))
        adj[v].append((w, u, eid))

    dist = np.full(num_nodes, np.inf)
    prev_node = np.full(num_nodes, -1, dtype=int)
    prev_edge = np.full(num_nodes, -1, dtype=int)
    in_queue = np.zeros(num_nodes, dtype=bool)

    dist[src] = 0.0
    queue: deque = deque([src])
    in_queue[src] = True
    relaxations = 0
    max_relaxations = num_nodes * edge_index.shape[1]  # cycle-detection guard

    while queue:
        u = queue.popleft()
        in_queue[u] = False
        for w, v, eid in adj[u]:
            nd = dist[u] + w
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

    path_indicator = np.zeros(edge_index.shape[1], dtype=np.float32)
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
) -> Optional[np.ndarray]:
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

    Returns
    -------
    (E,) binary numpy array: 1 if edge is on the shortest path, else 0.
    Returns None if no path exists.
    """
    # Build adjacency list: node -> list of (neighbour, weight, edge_id)
    adj: List[List[Tuple[float, int, int]]] = [[] for _ in range(num_nodes)]
    for eid in range(edge_index.shape[1]):
        u, v = int(edge_index[0, eid]), int(edge_index[1, eid])
        w = float(edge_weights[eid])
        adj[u].append((w, v, eid))
        adj[v].append((w, u, eid))

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
        for w, v, eid in adj[u]:
            nd = dist[u] + w
            if nd < dist[v]:
                dist[v] = nd
                prev_node[v] = u
                prev_edge[v] = eid
                heapq.heappush(heap, (nd, v))

    if dist[dst] == np.inf:
        return None  # no path

    # Reconstruct path edges
    path_indicator = np.zeros(edge_index.shape[1], dtype=np.float32)
    node = dst
    while prev_edge[node] != -1:
        path_indicator[prev_edge[node]] = 1.0
        node = prev_node[node]

    return path_indicator


def batched_dijkstra(
    edge_weights: np.ndarray,
    edge_index: np.ndarray,
    demands: List[Tuple[int, int]],
    num_nodes: int,
) -> np.ndarray:
    """
    Run Dijkstra for each demand.

    Parameters
    ----------
    edge_weights:
        (E,) float array.
    edge_index:
        (2, E) int array.
    demands:
        List of (src, dst) pairs.
    num_nodes:
        Total number of nodes.

    Returns
    -------
    (num_demands, E) binary float32 array.
    """
    results = []
    for s, d in demands:
        indicator = dijkstra(edge_weights, edge_index, s, d, num_nodes)
        if indicator is None:
            raise ValueError(f"No path found from node {s} to node {d}")
        results.append(indicator)
    return np.stack(results, axis=0)
