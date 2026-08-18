"""The one definition of the per-span feature ordering.

docs/architecture/invariants.md declares ``[span_length_km, fiber_type_idx, amp_nf_db,
channel_loading_fraction, accum_dist_km]`` a fixed architectural invariant,
and the on-disk parquet schema (``span_features_0`` .. ``span_features_
{max_spans*5-1}``) is laid out in that exact order. Before this module
existed, that ordering was written out three times independently
(``diffopt/pipeline.py``, ``data/generate_qot_dataset.py``, and a private
copy in ``scripts/diagnose_segmentation_identity.py``), so nothing enforced
that the three copies agreed. This module is now the single source all
three call.
"""
from __future__ import annotations

from typing import List, Sequence

from diffopt.topology import FIBER_TYPE_INDEX, Topology


SPAN_FEATURE_NAMES: tuple[str, ...] = (
    "span_length_km",
    "fiber_type_idx",
    "amp_nf_db",
    "channel_loading_fraction",
    "accum_dist_km",
)
SPAN_FEATURE_DIM = len(SPAN_FEATURE_NAMES)


def span_feature_rows(
    topology: Topology,
    edge_ids: Sequence[int],
    *,
    channel_loading_fraction: float,
    accum_start: float = 0.0,
) -> List[List[float]]:
    """Build raw (unpadded) per-span feature rows for one transparent segment.

    One row per span across `edge_ids` (in traversal order), each row in
    the canonical SPAN_FEATURE_NAMES order.

    `accum_dist_km` is the distance BEFORE each span: it starts at
    `accum_start` and is incremented by that span's length only *after*
    the row is recorded, so the first row's accum_dist_km is exactly
    `accum_start`. The live pipeline always passes the default
    (`accum_start=0.0` — accumulated distance resets at every segment
    boundary); `accum_start` exists so callers that want to carry
    accumulated distance across segment boundaries (e.g.
    scripts/diagnose_segmentation_identity.py's reset-vs-carry comparison)
    don't need a private copy of this function to do it.

    Args:
        topology: Topology whose `undirected_edges` is indexed by `edge_ids`.
        edge_ids: Edge IDs in traversal order (src -> dst), i.e. indices
            into `topology.undirected_edges`.
        channel_loading_fraction: Shared across every span in the segment —
            channel loading is sampled once per transparent segment, not
            per span (see docs/architecture/invariants.md's GNPy bridge constraints).
        accum_start: Starting value for `accum_dist_km`. Defaults to 0.0.

    Returns:
        List of rows, each `[span_length_km, fiber_type_idx, amp_nf_db,
        channel_loading_fraction, accum_dist_km]`.
    """
    edges = topology.undirected_edges
    rows: List[List[float]] = []
    accum_dist = accum_start

    for eid in edge_ids:
        edge = edges[eid]
        ftype_idx = float(FIBER_TYPE_INDEX.get(edge.fiber_type, 0))
        for span_idx in range(edge.num_spans):
            rows.append([
                edge.span_lengths_km[span_idx],
                ftype_idx,
                edge.amplifier_nf_db[span_idx],
                channel_loading_fraction,
                accum_dist,
            ])
            accum_dist += edge.span_lengths_km[span_idx]

    return rows
