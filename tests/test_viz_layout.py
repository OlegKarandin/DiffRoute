"""Layout for the trajectory viewer. Pure graph maths — no torch, no
pipeline, no checkpoint, per tests/test_diagnose_scripts.py's style."""
from pathlib import Path

import pytest

from diffopt.topology import load_topology
from diffopt.viz.layout import frozen_layout

BASE = Path(__file__).parent.parent
MODULATION_FORMATS_PATH = BASE / "configs/modulation_formats.yaml"
TOPOLOGIES = {
    "german_17": BASE / "configs/topology/german_17.json",
    "eu_19": BASE / "configs/topology/eu_19.json",
    "jp_70": BASE / "configs/topology/jp_70.json",
    "ind_132": BASE / "configs/topology/ind_132.json",
}


@pytest.mark.parametrize("name", sorted(TOPOLOGIES))
def test_layout_is_one_unit_box_point_per_node(name):
    topo = load_topology(str(TOPOLOGIES[name]), str(MODULATION_FORMATS_PATH))
    pos = frozen_layout(topo)
    assert len(pos) == topo.num_nodes
    for p in pos:
        assert len(p) == 2
        assert 0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0
    # Both axes are actually used — a degenerate collapse would make every
    # ribbon overlap and the map unreadable.
    assert max(p[0] for p in pos) - min(p[0] for p in pos) == pytest.approx(1.0)
    assert max(p[1] for p in pos) - min(p[1] for p in pos) == pytest.approx(1.0)


@pytest.mark.parametrize("name", sorted(TOPOLOGIES))
def test_layout_is_reproducible(name):
    """Node positions must never move between frames (decision 6). The frame
    file freezes ONE layout, so two FrameWriters over the same topology must
    agree exactly."""
    topo = load_topology(str(TOPOLOGIES[name]), str(MODULATION_FORMATS_PATH))
    assert frozen_layout(topo) == frozen_layout(topo)


def test_layout_separates_a_long_edge_from_a_short_one():
    """Distance is graph km, not hop count: a 100 km edge must end up longer
    on the page than a 60 km one from the same node."""
    from tests.test_pipeline import make_hub_topology

    pos = frozen_layout(make_hub_topology())

    def d(a, b):
        return ((pos[a][0] - pos[b][0]) ** 2 + (pos[a][1] - pos[b][1]) ** 2) ** 0.5

    assert d(0, 2) > d(0, 1)   # eid 1 is 100 km, eid 0 is 60 km
