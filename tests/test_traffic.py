"""Tests for diffopt/traffic.py — the fixed traffic matrix.

The matrix is the constraint set's identity. If it drifts silently (upstream
gravity change, a different alpha default), every "N/M demands violated"
number in the investigation record becomes incomparable to every other.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from diffopt.demands import Demand
from diffopt.modulation import ModulationConfig
from diffopt.qot.segment_combiner import SegmentCombiner
from diffopt.traffic import (
    SCENARIO_ALPHA,
    build_traffic_matrix,
    preflight_filter,
    scenario_alpha,
    shortest_path_edges_by_km,
    traffic_matrix_checksum,
)
from diffopt.topology import load_topology

from tests.test_pipeline import make_hub_topology, make_linear_topology


REPO_ROOT = Path(__file__).parent.parent
IND132_PATH = REPO_ROOT / "configs/topology/ind_132.json"
MODULATION_FORMATS_PATH = REPO_ROOT / "configs/modulation_formats.yaml"

BITRATE_OPTIONS = [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]


@pytest.fixture(scope="module")
def ind132():
    return load_topology(str(IND132_PATH), str(MODULATION_FORMATS_PATH))


def test_traffic_matrix_deterministic(ind132):
    """Same (seed, scale, alpha) must give byte-identical output."""
    kwargs = dict(seed=0, scale=1.0e6, alpha=0.0, bitrate_options=BITRATE_OPTIONS)
    first = build_traffic_matrix(ind132, **kwargs)
    second = build_traffic_matrix(ind132, **kwargs)
    assert first == second
    assert len(first) > 0


def test_different_seed_gives_a_different_matrix(ind132):
    a = build_traffic_matrix(ind132, seed=0, scale=1.0e6, alpha=0.0,
                             bitrate_options=BITRATE_OPTIONS)
    b = build_traffic_matrix(ind132, seed=1, scale=1.0e6, alpha=0.0,
                             bitrate_options=BITRATE_OPTIONS)
    assert a != b, "held-out matrix must actually differ from the training one"


def test_traffic_matrix_bitrates_are_valid(ind132):
    """ModulationConfig.required_snr_threshold is an exact-float dict lookup
    with no interpolation (invariants.md) — a bitrate that is not an exact
    member of bitrate_options raises at training time, not here."""
    valid = {float(b) for b in BITRATE_OPTIONS}
    demands = build_traffic_matrix(ind132, seed=0, scale=1.0e6, alpha=0.0,
                                   bitrate_options=BITRATE_OPTIONS)
    for d in demands:
        assert d.bitrate_gbps in valid, f"demand {d.id} has bitrate {d.bitrate_gbps}"


def test_traffic_matrix_node_ids_are_ints(ind132):
    """Upstream generate_demands emits string node ids; invariants.md says
    nodes are integer IDs only."""
    demands = build_traffic_matrix(ind132, seed=0, scale=1.0e6, alpha=0.0,
                                   bitrate_options=BITRATE_OPTIONS)
    n = ind132.num_nodes
    for d in demands:
        assert type(d.src) is int and type(d.dst) is int
        assert 0 <= d.src < n and 0 <= d.dst < n
        assert d.src != d.dst


def test_demand_ids_are_contiguous_from_zero(ind132):
    """Demand.id indexes the per-demand dual vector in diffopt/loss.py.
    A gap or a repeat silently attaches a dual to the wrong demand."""
    demands = build_traffic_matrix(ind132, seed=0, scale=1.0e6, alpha=0.0,
                                   bitrate_options=BITRATE_OPTIONS)
    assert [d.id for d in demands] == list(range(len(demands)))


def test_pairs_are_unordered_and_unique(ind132):
    """undirected=True + aggregate=True means one record per unordered pair."""
    demands = build_traffic_matrix(ind132, seed=0, scale=1.0e6, alpha=0.0,
                                   bitrate_options=BITRATE_OPTIONS)
    keys = {frozenset((d.src, d.dst)) for d in demands}
    assert len(keys) == len(demands)


def test_alpha_changes_bitrate_distance_coupling(ind132):
    """alpha=1.0 (realistic) puts big pipes on short hops; alpha=0.0 (stress)
    decouples them. Spec §8.4 measures corr(bitrate, km) = -0.829 vs -0.008."""
    realistic = build_traffic_matrix(ind132, seed=0, scale=1.0e6, alpha=1.0,
                                     bitrate_options=BITRATE_OPTIONS)
    stress = build_traffic_matrix(ind132, seed=0, scale=1.0e6, alpha=0.0,
                                  bitrate_options=BITRATE_OPTIONS)
    assert [(d.src, d.dst) for d in realistic] != [(d.src, d.dst) for d in stress] or \
        [d.bitrate_gbps for d in realistic] != [d.bitrate_gbps for d in stress]


def test_scenario_alpha_maps_the_two_named_scenarios():
    assert scenario_alpha("realistic") == 1.0
    assert scenario_alpha("stress") == 0.0
    assert set(SCENARIO_ALPHA) == {"realistic", "stress"}
    with pytest.raises(ValueError, match="unknown traffic scenario"):
        scenario_alpha("gravity")


def test_small_topology_still_produces_a_matrix():
    """A 5-node hub topology must not crash the gravity wrapper."""
    demands = build_traffic_matrix(make_hub_topology(), seed=0, scale=5.0e3,
                                   alpha=1.0, bitrate_options=BITRATE_OPTIONS)
    assert len(demands) >= 1


def test_checksum_is_stable_and_sensitive(ind132):
    a = build_traffic_matrix(ind132, seed=0, scale=1.0e6, alpha=0.0,
                             bitrate_options=BITRATE_OPTIONS)
    b = build_traffic_matrix(ind132, seed=1, scale=1.0e6, alpha=0.0,
                             bitrate_options=BITRATE_OPTIONS)
    assert traffic_matrix_checksum(a) == traffic_matrix_checksum(a)
    assert traffic_matrix_checksum(a) != traffic_matrix_checksum(b)
    assert len(traffic_matrix_checksum(a)) == 16


# ---------------------------------------------------------------------------
# Pinned checksums — regenerate with the command in this test's docstring if
# an upstream bump legitimately changes the matrix, and say so in the commit.
# ---------------------------------------------------------------------------

IND132_PINNED = {
    # (alpha, expected_len, expected_checksum)
    "stress": (0.0, 172, "d754c606030311de"),
    "realistic": (1.0, 583, "0feed7f87eee4796"),
}


@pytest.mark.parametrize("scenario", sorted(IND132_PINNED))
def test_ind132_matrix_checksum_is_pinned(ind132, scenario):
    """Catches silent upstream gravity drift.

    Regenerate with:
        python -c "from diffopt.topology import load_topology; \
from diffopt.traffic import build_traffic_matrix, traffic_matrix_checksum; \
t = load_topology('configs/topology/ind_132.json', 'configs/modulation_formats.yaml'); \
print(traffic_matrix_checksum(build_traffic_matrix(t, seed=0, scale=1.0e6, alpha=0.0, bitrate_options=[300,350,400,450,500,550,600,650,700,750,800])))"

    Bumping the `multilayer-optical-network` pin requires re-verifying this,
    not just updating a version number (invariants.md, "Upstream dependency").
    """
    alpha, expected_len, expected_checksum = IND132_PINNED[scenario]
    demands = build_traffic_matrix(ind132, seed=0, scale=1.0e6, alpha=alpha,
                                   bitrate_options=BITRATE_OPTIONS)
    assert len(demands) == expected_len
    assert traffic_matrix_checksum(demands) == expected_checksum


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


class _ConstantQoT(torch.nn.Module):
    """Stand-in for SpanAttentionQoT returning a fixed per-segment GSNR.

    The preflight's job is a route/segment/threshold decision, not physics —
    a constant makes the arithmetic exact and the test independent of any
    trained checkpoint. Matches the real model's contract: forward(span_feats,
    padding_mask) -> (batch,).
    """

    def __init__(self, gsnr_db: float) -> None:
        super().__init__()
        self.gsnr_db = gsnr_db
        # A real parameter so `next(qot_model.parameters()).device` works.
        self._anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, span_feats, padding_mask):
        return torch.full((span_feats.shape[0],), self.gsnr_db,
                          device=self._anchor.device)


def _two_format_mod_config() -> ModulationConfig:
    """400 G needs 20 dB (easy); 800 G needs 40 dB (impossible here)."""
    return ModulationConfig(
        channel_spacing_ghz=100.0,
        symbol_rate_gbaud=64.0,
        num_channels_cband=48,
        cut_channel_index=24,
        formats=[
            {"bitrate_gbps": 400, "snr_threshold_db": 20.0},
            {"bitrate_gbps": 800, "snr_threshold_db": 40.0},
        ],
    )


def test_shortest_path_edges_by_km_returns_traversal_order():
    """make_hub_topology: 0-1 (eid 0, 60km), 0-2 (eid 1, 100km),
    1-3 (eid 2, 60km), 2-3 (eid 3, 100km), 3-4 (eid 4, 80km).
    0->4 by kilometres is 60+60+80=200 via node 1, not 100+100+80=280."""
    topo = make_hub_topology()
    assert shortest_path_edges_by_km(topo, 0, 4) == [0, 2, 4]
    assert shortest_path_edges_by_km(topo, 4, 0) == [4, 2, 0]
    assert shortest_path_edges_by_km(topo, 0, 0) == []


def test_shortest_path_edges_by_km_returns_none_when_disconnected():
    """A two-node topology with no edge between the components."""
    from tests.test_pipeline import _build_topology, _make_edge_dict
    topo = _build_topology(4, [_make_edge_dict(0, 1), _make_edge_dict(2, 3)])
    assert shortest_path_edges_by_km(topo, 0, 3) is None


def test_preflight_excludes_impossible_demand():
    """A demand whose threshold exceeds what full regeneration on the
    shortest-by-km route can deliver is excluded and reported.

    Constant 25 dB per segment. make_hub_topology has exactly one regen
    candidate (node 3), and the 0->4 route [0, 2, 4] crosses it, so with all
    candidates active the path is two transparent segments joined by a
    regenerator: end-to-end GSNR ~= 25 dB (worst segment). 400 G (20 dB + 0.5
    margin) clears; 800 G (40 dB) cannot.
    """
    topo = make_hub_topology()
    demands = [
        Demand(id=0, src=0, dst=4, bitrate_gbps=400.0),
        Demand(id=1, src=0, dst=4, bitrate_gbps=800.0),
    ]
    kept, excluded = preflight_filter(
        topo, demands,
        qot_model=_ConstantQoT(25.0),
        segment_combiner=SegmentCombiner(),
        modulation_config=_two_format_mod_config(),
        margin_db=0.5,
    )
    assert [d.bitrate_gbps for d in kept] == [400.0]
    assert len(excluded) == 1
    excluded_demand, shortfall = excluded[0]
    assert excluded_demand.bitrate_gbps == 800.0
    assert shortfall == pytest.approx(40.0 + 0.5 - 25.0, abs=0.2), (
        "shortfall must be reported in dB against threshold + margin"
    )


def test_preflight_renumbers_kept_demands_contiguously():
    """Duals are indexed by Demand.id — a gap left by an exclusion would
    attach every later dual to the wrong demand."""
    topo = make_hub_topology()
    demands = [
        Demand(id=0, src=0, dst=4, bitrate_gbps=800.0),   # excluded
        Demand(id=1, src=0, dst=4, bitrate_gbps=400.0),   # kept
        Demand(id=2, src=0, dst=3, bitrate_gbps=400.0),   # kept
    ]
    kept, excluded = preflight_filter(
        topo, demands,
        qot_model=_ConstantQoT(25.0),
        segment_combiner=SegmentCombiner(),
        modulation_config=_two_format_mod_config(),
        margin_db=0.5,
    )
    assert len(excluded) == 1
    assert [d.id for d in kept] == list(range(len(kept)))
    assert [d.dst for d in kept] == [4, 3], "kept order must be preserved"


def test_preflight_excludes_disconnected_demand():
    """A src/dst pair with no route at all is excluded, not crashed on."""
    from tests.test_pipeline import _build_topology, _make_edge_dict
    topo = _build_topology(4, [_make_edge_dict(0, 1), _make_edge_dict(2, 3)])
    kept, excluded = preflight_filter(
        topo, [Demand(id=0, src=0, dst=3, bitrate_gbps=400.0)],
        qot_model=_ConstantQoT(25.0),
        segment_combiner=SegmentCombiner(),
        modulation_config=_two_format_mod_config(),
        margin_db=0.5,
    )
    assert kept == []
    assert len(excluded) == 1
    assert excluded[0][1] == float("inf")


def test_preflight_keeps_everything_when_all_demands_clear():
    """A linear chain has no regen candidates (all degrees <= 2), so every
    path is a single transparent segment — the preflight must still work."""
    topo = make_linear_topology()
    demands = [Demand(id=i, src=0, dst=4, bitrate_gbps=400.0) for i in range(3)]
    kept, excluded = preflight_filter(
        topo, demands,
        qot_model=_ConstantQoT(25.0),
        segment_combiner=SegmentCombiner(),
        modulation_config=_two_format_mod_config(),
        margin_db=0.5,
    )
    assert len(kept) == 3
    assert excluded == []
