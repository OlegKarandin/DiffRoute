"""Pins the canonical span-feature ordering.

CLAUDE.md declares [span_length_km, fiber_type_idx, amp_nf_db,
channel_loading_fraction, accum_dist_km] a fixed invariant, and the parquet
schema (span_features_0 .. span_features_{max_spans*5-1}) is laid out in that
order. A change here invalidates every existing dataset.
"""
from diffopt.qot.span_features import (
    SPAN_FEATURE_NAMES, SPAN_FEATURE_DIM, span_feature_rows,
)

from tests.test_pipeline import make_hub_topology


def test_ordering_is_the_documented_invariant():
    assert SPAN_FEATURE_NAMES == (
        "span_length_km",
        "fiber_type_idx",
        "amp_nf_db",
        "channel_loading_fraction",
        "accum_dist_km",
    )
    assert SPAN_FEATURE_DIM == 5


def test_accum_dist_starts_at_zero_and_trails_by_one_span():
    """accum_dist_km is the distance BEFORE each span, so the first row is 0.0
    and each subsequent row is the running sum of preceding span lengths."""
    topo = make_hub_topology()
    # Hub topology (see make_hub_topology's docstring): eid 0 is 0-1 (60km),
    # eid 2 is 1-3 (60km) -- the route via node 1, single span per edge.
    rows = span_feature_rows(topo, [0, 2], channel_loading_fraction=0.5)

    assert len(rows) == 2
    assert rows[0] == [60.0, 0.0, 5.0, 0.5, 0.0]
    assert rows[1] == [60.0, 0.0, 5.0, 0.5, 60.0]


def test_accum_start_offsets_every_accum_dist_km():
    """accum_start shifts every accum_dist_km by the same fixed amount --
    this is the parameter that lets diagnose_segmentation_identity.py drop
    its private copy (it needs to carry accumulated distance across a
    segment boundary instead of resetting to 0.0 at each segment)."""
    topo = make_hub_topology()

    reset_rows = span_feature_rows(topo, [0, 2], channel_loading_fraction=0.5)
    carried_rows = span_feature_rows(
        topo, [0, 2], channel_loading_fraction=0.5, accum_start=100.0
    )

    assert [r[4] for r in carried_rows] == [r[4] + 100.0 for r in reset_rows]
    assert [r[4] for r in carried_rows] == [100.0, 160.0]
    # Only accum_dist_km (column 4) is affected -- everything else is identical.
    for reset_row, carried_row in zip(reset_rows, carried_rows):
        assert reset_row[:4] == carried_row[:4]


def test_channel_loading_fraction_is_shared_across_every_span_in_segment():
    """channel_loading_fraction is sampled once per transparent segment, not
    per span -- every row in the segment carries the same value."""
    topo = make_hub_topology()
    rows = span_feature_rows(topo, [0, 2], channel_loading_fraction=0.7291)

    assert [r[3] for r in rows] == [0.7291, 0.7291]


def test_empty_edge_ids_returns_empty_rows():
    topo = make_hub_topology()
    assert span_feature_rows(topo, [], channel_loading_fraction=0.5) == []
