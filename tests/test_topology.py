"""Tests for topology builder and topology loading."""
import json
import math
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from diffopt.topology_builder import split_link_into_spans, parse_dat_file, build_topology_json
from diffopt.topology import Topology, Edge, load_topology, FIBER_TYPE_INDEX

from multilayer_optical_network.model.optical_network import OpticalNetworkModel
from multilayer_optical_network.model.optical_topology_import import populate_optical
from multilayer_optical_network.model.modes import load_modulation_formats


BASE = Path(__file__).parent.parent
MODULATION_FORMATS_PATH = BASE / "configs/modulation_formats.yaml"
GERMAN_JSON = BASE / "configs/topology/german_17.json"
EU_JSON = BASE / "configs/topology/eu_19.json"


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


# ---- Topology construction / type tests ----

def test_load_german_topology_is_topology_and_optical_network_model():
    t = Topology.from_graph_json(GERMAN_JSON, MODULATION_FORMATS_PATH)
    assert isinstance(t, Topology)
    assert isinstance(t, OpticalNetworkModel)
    assert t.num_nodes == 17
    assert len(t.undirected_edges) == 26


def test_load_eu_topology_is_topology_and_optical_network_model():
    t = Topology.from_graph_json(EU_JSON, MODULATION_FORMATS_PATH)
    assert isinstance(t, Topology)
    assert isinstance(t, OpticalNetworkModel)
    assert t.num_nodes == 19
    assert len(t.undirected_edges) == 38


def test_load_topology_alias_matches_from_graph_json():
    t = load_topology(str(GERMAN_JSON), str(MODULATION_FORMATS_PATH))
    assert isinstance(t, Topology)
    assert t.num_nodes == 17


# ---- Topology-vs-JSON equivalence ----

def _hand_computed_regen_candidates(graph: dict) -> set:
    degree = {}
    for e in graph["edges"]:
        degree[e["src"]] = degree.get(e["src"], 0) + 1
        degree[e["dst"]] = degree.get(e["dst"], 0) + 1
    return {n for n, d in degree.items() if d >= 3}


@pytest.mark.parametrize("json_path", [GERMAN_JSON, EU_JSON])
def test_topology_matches_raw_json(json_path):
    t = Topology.from_graph_json(json_path, MODULATION_FORMATS_PATH)
    graph = json.loads(Path(json_path).read_text())

    raw_edges_by_pair = {(e["src"], e["dst"]): e for e in graph["edges"]}
    assert len(t.undirected_edges) == len(raw_edges_by_pair)

    for edge in t.undirected_edges:
        assert edge.src < edge.dst
        raw = raw_edges_by_pair[(edge.src, edge.dst)]
        assert edge.num_spans == raw["num_spans"]
        assert edge.fiber_type == raw["fiber_type"]
        assert edge.span_lengths_km == pytest.approx(raw["span_lengths_km"], abs=1e-6)
        assert edge.amplifier_nf_db == pytest.approx(raw["amplifier_nf_db"], abs=1e-6)
        assert edge.length_km == pytest.approx(raw["length_km"], abs=0.01)

    # edge_index shape and ordering
    edge_index = t.edge_index
    assert edge_index.shape == (2, len(raw_edges_by_pair))
    for col in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, col]), int(edge_index[1, col])
        assert src < dst

    # regen_candidate_nodes matches a hand-computed degree->=3 set from raw JSON
    expected_regen = _hand_computed_regen_candidates(graph)
    assert set(t.regen_candidate_nodes) == expected_regen


# ---- Booster exclusion ----

def test_amplifier_count_excludes_booster():
    """For a multi-span edge, amplifier_nf_db must have exactly num_spans
    entries — not num_spans + 1, which would mean the booster amp leaked in."""
    t = Topology.from_graph_json(GERMAN_JSON, MODULATION_FORMATS_PATH)
    multi_span_edges = [e for e in t.undirected_edges if e.num_spans > 1]
    assert len(multi_span_edges) > 0, "expected at least one multi-span edge"
    for edge in multi_span_edges:
        assert len(edge.amplifier_nf_db) == edge.num_spans
        assert len(edge.span_lengths_km) == edge.num_spans


# ---- Contiguity check ----

def test_num_nodes_raises_on_node_id_gap():
    """A graph whose node ids skip a value (e.g. [0, 1, 3]) must raise
    ValueError from num_nodes rather than silently under-counting nodes."""
    modes = load_modulation_formats(MODULATION_FORMATS_PATH)
    t = Topology(modes=modes)
    graph = {
        "nodes": [{"id": 0}, {"id": 1}, {"id": 3}],
        "edges": [
            {"src": 0, "dst": 1, "length_km": 80.0, "num_spans": 1,
             "span_lengths_km": [80.0], "fiber_type": "SSMF",
             "amplifier_nf_db": [5.5]},
            {"src": 1, "dst": 3, "length_km": 80.0, "num_spans": 1,
             "span_lengths_km": [80.0], "fiber_type": "SSMF",
             "amplifier_nf_db": [5.5]},
        ],
    }
    populate_optical(t, graph, 0.2)
    with pytest.raises(ValueError):
        t.num_nodes


# ---- get_edge_features ----

def test_edge_features_shape():
    t = Topology.from_graph_json(GERMAN_JSON, MODULATION_FORMATS_PATH)
    feats = t.get_edge_features()
    assert feats.shape == (len(t.undirected_edges), 5)


def test_edge_features_values_match_derivation():
    t = Topology.from_graph_json(GERMAN_JSON, MODULATION_FORMATS_PATH)
    feats = t.get_edge_features()
    for row, edge in zip(feats, t.undirected_edges):
        expected = [
            edge.mean_span_length_km,
            float(FIBER_TYPE_INDEX.get(edge.fiber_type, 0)),
            edge.mean_amp_nf_db,
            float(edge.num_spans),
            edge.length_km,
        ]
        assert row.tolist() == pytest.approx(expected, abs=1e-4)


def test_regen_candidate_nodes():
    t = Topology.from_graph_json(GERMAN_JSON, MODULATION_FORMATS_PATH)
    regen = t.regen_candidate_nodes
    # All nodes should have degree info
    assert len(regen) > 0
    assert all(0 <= n < t.num_nodes for n in regen)
