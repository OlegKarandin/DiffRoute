"""Node positions for the trajectory viewer's map.

No topology JSON carries coordinates — docs/architecture/invariants.md
forbids `x`/`y` anywhere in topology data — so the layout is synthesized
here, once, and frozen into the frame file. The viewer never recomputes it:
spec decision 6 requires node positions to be identical in every frame, and
the cheapest way to guarantee that is to ship exactly one copy.

Kamada-Kawai over **kilometre** graph distance rather than hop count, so a
655 km eu_19 edge does not draw the same length as a 24 km ind_132 one.
`networkx` and `scipy` are already dependencies (pyproject.toml).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:  # pragma: no cover
    from diffopt.topology import Topology


def frozen_layout(topology: "Topology") -> List[List[float]]:
    """One [x, y] per node id, in node-id order, normalized to the unit box.

    Deterministic: Kamada-Kawai is a `scipy.optimize.minimize` run, and the
    initial positions are an explicit deterministic circular layout rather
    than networkx's default, so two calls in one environment agree exactly
    (tests/test_viz_layout.py::test_layout_is_reproducible).
    """
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(topology.num_nodes))
    for e in topology.undirected_edges:
        # A parallel-free undirected graph: the topology already
        # deduplicates on src < dst.
        g.add_edge(e.src, e.dst, km=float(e.length_km))

    # All-pairs shortest path in km. Disconnected pairs get the graph's
    # finite diameter rather than inf, which would make the KK stress
    # function non-finite.
    lengths = dict(nx.all_pairs_dijkstra_path_length(g, weight="km"))
    finite = [v for row in lengths.values() for v in row.values()]
    fallback = (max(finite) if finite else 1.0) * 2.0
    dist = {
        u: {v: lengths[u].get(v, fallback) for v in g.nodes}
        for u in g.nodes
    }

    pos = nx.kamada_kawai_layout(
        g, dist=dist, pos=nx.circular_layout(g),
    )
    xs = [float(pos[n][0]) for n in range(topology.num_nodes)]
    ys = [float(pos[n][1]) for n in range(topology.num_nodes)]
    return [
        [_unit(x, xs), _unit(y, ys)]
        for x, y in zip(xs, ys)
    ]


def _unit(v: float, vals: List[float]) -> float:
    """Map `v` into [0, 1] against `vals`' range. A degenerate (zero-span)
    axis maps to 0.5 rather than dividing by zero."""
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 0.0:
        return 0.5
    return round((v - lo) / span, 6)
