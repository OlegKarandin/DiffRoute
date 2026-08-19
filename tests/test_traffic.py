"""Tests for diffopt/traffic.py — the fixed traffic matrix.

The matrix is the constraint set's identity. If it drifts silently (upstream
gravity change, a different alpha default), every "N/M demands violated"
number in the investigation record becomes incomparable to every other.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from diffopt.traffic import (
    SCENARIO_ALPHA,
    build_traffic_matrix,
    scenario_alpha,
    traffic_matrix_checksum,
)
from diffopt.topology import load_topology

from tests.test_pipeline import make_hub_topology


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
