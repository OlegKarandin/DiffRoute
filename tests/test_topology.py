"""Tests for topology builder and topology loading."""
import math
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from diffopt.topology_builder import split_link_into_spans, parse_dat_file, build_topology_json
from diffopt.topology import load_topology


# ---- split_link_into_spans tests ----

def test_split_170km_gives_two_balanced_spans():
    spans = split_link_into_spans(170.0)
    assert len(spans) == 2, f"Expected 2 spans, got {spans}"
    # Each span should be close to 85 km, not 80+80+10
    assert all(s >= 20.0 for s in spans)
    assert abs(spans[0] - 85.0) < 1.0


def test_split_36km_single_span():
    spans = split_link_into_spans(36.0)
    assert len(spans) == 1
    assert abs(spans[0] - 36.0) < 0.01


def test_split_353km_four_spans():
    spans = split_link_into_spans(353.0)
    assert len(spans) == 4
    # Each span ~88.25 km
    assert all(s >= 20.0 for s in spans)


def test_all_german_spans_ge_20km():
    """All spans from German topology must be >= 20 km."""
    base = Path(__file__).parent.parent
    _, edges = parse_dat_file(str(base / "german_17.dat"))
    for src, dst, length in edges:
        spans = split_link_into_spans(length)
        assert all(s >= 20.0 for s in spans), (
            f"Edge {src}-{dst} ({length} km): span < 20 km: {spans}"
        )


def test_all_eu_spans_ge_20km():
    """All spans from EU topology must be >= 20 km."""
    base = Path(__file__).parent.parent
    _, edges = parse_dat_file(str(base / "EU_19.dat"))
    for src, dst, length in edges:
        spans = split_link_into_spans(length)
        assert all(s >= 20.0 for s in spans), (
            f"Edge {src}-{dst} ({length} km): span < 20 km: {spans}"
        )


def test_spans_sum_to_link_length():
    """Span lengths must sum to original link length within 0.01 km."""
    test_lengths = [36.0, 80.0, 100.0, 144.0, 170.0, 208.0, 278.0, 353.0,
                    200.0, 750.0, 1050.0, 1240.0, 1500.0, 1630.0]
    for length in test_lengths:
        spans = split_link_into_spans(length)
        total = sum(spans)
        assert abs(total - length) < 0.01, (
            f"Length {length}: spans sum to {total}, diff={abs(total - length)}"
        )


def test_load_german_topology():
    base = Path(__file__).parent.parent
    t = load_topology(str(base / "configs/topology/german_17.json"))
    assert t.num_nodes == 17
    assert t.num_edges == 26


def test_load_eu_topology():
    base = Path(__file__).parent.parent
    t = load_topology(str(base / "configs/topology/eu_19.json"))
    assert t.num_nodes == 19
    assert t.num_edges == 38


def test_edge_features_shape():
    base = Path(__file__).parent.parent
    t = load_topology(str(base / "configs/topology/german_17.json"))
    feats = t.get_edge_features()
    assert feats.shape == (t.num_edges, 5)


def test_regen_candidate_nodes():
    base = Path(__file__).parent.parent
    t = load_topology(str(base / "configs/topology/german_17.json"))
    regen = t.regen_candidate_nodes
    # All nodes should have degree info
    assert len(regen) > 0
    assert all(0 <= n < t.num_nodes for n in regen)
