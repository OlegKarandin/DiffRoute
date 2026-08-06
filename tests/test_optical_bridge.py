"""Tests for ``diffopt.qot.optical_bridge`` — the first GNPy execution in this
project's history.

The old ``diffopt/qot/gnpy_bridge.py`` wrapped its GNPy call in
``except Exception`` and returned an analytical GN approximation on every
single call, forever, silently. These tests exist to make that failure mode
impossible to reintroduce:

* the ground-truth test reproduces upstream's own hand-built 2x80 km toy and
  lands on its published 18.85 dB figure;
* four independent "GNPy actually ran" guards run against diffopt's real
  ``german_17`` topology (pinned value, monotone-with-length chain, a spy on
  ``gnpy.core.elements.Fiber.__call__``, monotone-with-loading, and the exact
  per-element breakdown shape);
* one test proves the explicit ``center_freq_hz`` probe argument is
  load-bearing rather than decorative;
* one test proves the module contains no ``try``/``except`` at all.

Every numeric expectation below was measured by running this code (gnpy
2.14.0, multilayer-optical-mcp @ 2b64361) or copied from upstream's own test
source — none were typed from memory.
"""
from __future__ import annotations

import ast
import math
import subprocess
import sys
from pathlib import Path

import pytest

from multilayer_optical_mcp.gnpy_adapter.adapter import compute_qot
from multilayer_optical_mcp.gnpy_adapter.loading import Channel, LoadingState
from multilayer_optical_mcp.model.assets import (
    Amplifier,
    Direction,
    Fiber,
    FiberType,
    OMS,
    ROADM,
    Transceiver,
    TransceiverMode,
)
from multilayer_optical_mcp.model.modes import ModeRegistry
from multilayer_optical_mcp.model.qot_results import QoTResultStore

from diffopt.qot import optical_bridge as ob
from diffopt.topology import Topology

REPO_ROOT = Path(__file__).resolve().parents[1]
GERMAN_17 = REPO_ROOT / "configs" / "topology" / "german_17.json"
MOD_FORMATS = REPO_ROOT / "configs" / "modulation_formats.yaml"
BRIDGE_SRC = REPO_ROOT / "diffopt" / "qot" / "optical_bridge.py"

MODE = "400G@7.1dB"

# Shortest paths in german_17.json, verified by Dijkstra on the raw JSON:
#   0->1  =   144 km, 0->3->5      = 366 km,
#   0->3->5->10 = 548 km, 0->3->5->10->16 = 772 km.
PATH_144 = [0, 1]
PATH_366 = [0, 3, 5]
PATH_548 = [0, 3, 5, 10]
PATH_772 = [0, 3, 5, 10, 16]


# --------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def german17() -> Topology:
    return Topology.from_graph_json(GERMAN_17, MOD_FORMATS)


def _toy_modes() -> ModeRegistry:
    """Byte-for-byte the ``_modes()`` helper of upstream's
    ``tests/model/test_optical_network_model.py`` at commit 2b64361."""
    return ModeRegistry([
        TransceiverMode(
            id="400G@7.1dB",
            bitrate_gbps=400.0,
            required_gsnr_db=7.1,
            symbol_rate_baud=87.5e9,
            channel_spacing_hz=100e9,
        ),
    ])


def _toy_topology() -> Topology:
    """Upstream's ``_toy_optical()`` fixture, rebuilt as a ``diffopt.Topology``.

    Deliberately hand-built rather than routed through ``from_graph_json`` /
    ``populate_optical``: the importer derives each amplifier's gain as
    ``round(span_km * loss_coef, 2)`` (16.0 dB for an 80 km SSMF span), whereas
    upstream's fixture pins every amplifier at 20.0 dB. Those are physically
    different networks; only the hand-built one reproduces 18.85 dB.
    """
    n = Topology(modes=_toy_modes())
    n.register_fiber_type(FiberType(type_variety="SSMF", loss_coef_db_per_km=0.2))
    # East (A -> Z).
    n.add_roadm(ROADM(id="roadm_A", target_pch_out_db=-20.0))
    n.add_transceiver(Transceiver(id="trx_A", site="A"))
    n.add_amplifier(Amplifier(id="booster A", type_variety="advanced_toy",
                              gain_db=20.0, nf_db=5.5))
    n.add_fiber(Fiber(id="east fiber A to ILA", a_end="roadm_A",
                      z_end="east edfa in ILA", length_km=80.0, type_variety="SSMF"))
    n.add_amplifier(Amplifier(id="east edfa in ILA", type_variety="advanced_toy",
                              gain_db=20.0, nf_db=5.5))
    n.add_fiber(Fiber(id="east fiber ILA to Z", a_end="east edfa in ILA",
                      z_end="east edfa at Z", length_km=80.0, type_variety="SSMF"))
    n.add_amplifier(Amplifier(id="east edfa at Z", type_variety="advanced_toy",
                              gain_db=20.0, nf_db=5.5))
    n.add_oms(OMS(
        id="oms-AZ", src_node_id="A", dst_node_id="Z",
        elements=(
            "roadm_A",
            "booster A",
            "east fiber A to ILA",
            "east edfa in ILA",
            "east fiber ILA to Z",
            "east edfa at Z",
        ),
    ))
    # West (Z -> A): a physically separate reverse OMS with its own amp chain.
    n.add_roadm(ROADM(id="roadm_Z", target_pch_out_db=-20.0))
    n.add_transceiver(Transceiver(id="trx_Z", site="Z"))
    n.add_amplifier(Amplifier(id="booster Z", type_variety="advanced_toy",
                              gain_db=20.0, nf_db=5.5))
    n.add_fiber(Fiber(id="west fiber Z to ILA", a_end="roadm_Z",
                      z_end="west edfa in ILA", length_km=80.0, type_variety="SSMF"))
    n.add_amplifier(Amplifier(id="west edfa in ILA", type_variety="advanced_toy",
                              gain_db=20.0, nf_db=5.5))
    n.add_fiber(Fiber(id="west fiber ILA to A", a_end="west edfa in ILA",
                      z_end="west edfa at A", length_km=80.0, type_variety="SSMF"))
    n.add_amplifier(Amplifier(id="west edfa at A", type_variety="advanced_toy",
                              gain_db=20.0, nf_db=5.5))
    n.add_oms(OMS(
        id="oms-ZA", src_node_id="Z", dst_node_id="A",
        elements=(
            "roadm_Z",
            "booster Z",
            "west fiber Z to ILA",
            "west edfa in ILA",
            "west fiber ILA to A",
            "west edfa at A",
        ),
    ))
    return n


def _n_spans(topology: Topology, oms_id: str) -> int:
    """Span count of an OMS: elements are [roadm, booster, (fiber, amp) * n]."""
    return len(topology.get_oms(oms_id).elements[2::2])


# ------------------------------------------------------------------- grid / CUT


def test_grid_and_cut_constants():
    assert (ob.GRID.anchor_hz, ob.GRID.spacing_hz, ob.GRID.num_slots) == (
        191.4e12, 100e9, 48)
    assert ob.CUT_FREQ_HZ == 193.5e12
    # (193.5e12 - 191.4e12) / 100e9 == 21.0 exactly, no rounding involved.
    assert ob.CUT_SLOT == 21
    assert ob.GRID.freq(ob.CUT_SLOT) == pytest.approx(ob.CUT_FREQ_HZ, abs=1.0)


# ------------------------------------------------------------------ build_loading


@pytest.mark.parametrize("n_channels", [1, 2, 3, 12, 24, 47, 48])
def test_build_loading_is_cut_centered_exact_size_and_deterministic(n_channels):
    loading = ob.build_loading(n_channels, MODE)
    slots = [ob.GRID.slot_of(c.center_freq_hz) for c in loading.channels]

    assert len(slots) == n_channels                  # exactly n_channels active
    assert len(set(slots)) == n_channels             # no duplicates
    assert ob.CUT_SLOT in slots                      # CUT always lit
    assert slots == sorted(slots)                    # ascending frequency order
    assert all(0 <= s < ob.GRID.num_slots for s in slots)
    # Contiguous CUT-centered block (the documented outward-expansion policy).
    assert slots == list(range(min(slots), max(slots) + 1))
    # Deterministic: identical on a second call.
    again = ob.build_loading(n_channels, MODE)
    assert again.channels == loading.channels
    # power_dbm is always literal None (ROADMs re-equalise; no launch power here).
    assert all(c.power_dbm is None for c in loading.channels)
    assert all(c.mode_id == MODE for c in loading.channels)
    assert all(c.slot_width_hz == ob.GRID.spacing_hz for c in loading.channels)


def test_build_loading_full_grid_lights_every_slot():
    loading = ob.build_loading(48, MODE)
    slots = {ob.GRID.slot_of(c.center_freq_hz) for c in loading.channels}
    assert slots == set(range(48))


# ---------------------------------------------------- 1. ground truth: 18.85 dB


def test_toy_topology_reproduces_upstream_ground_truth_gsnr():
    """Load-bearing: proves GNPy actually propagates.

    18.85 dB is upstream's own published ground truth for this symmetric
    2x80 km toy at gnpy 2.14.0 (see the GROUND TRUTH header of
    ``tests/gnpy_adapter/test_compute_qot.py`` and the final assertion of
    ``test_compute_qot_matches_between_bare_optical_and_network_model``, which
    uses ``pytest.approx(18.85, abs=0.3)`` — the same tolerance used here).

    Measured through this bridge: 18.8445 dB in both directions. It is not
    bit-identical to upstream's own run because upstream probes at 193.4 THz
    (its grid slot 20) while this bridge's CUT is diffopt's 193.5 THz (slot
    21); both sit inside the same 0.3 dB band.
    """
    topology = _toy_topology()
    assert isinstance(topology, Topology)

    forward = ob.segment_gsnr_db(topology, ("oms-AZ",), MODE, 1)
    backward = ob.segment_gsnr_db(topology, ("oms-ZA",), MODE, 1)

    assert math.isfinite(forward) and math.isfinite(backward)
    assert forward == pytest.approx(18.85, abs=0.3)
    assert backward == pytest.approx(18.85, abs=0.3)
    # The toy is physically symmetric, so the two OMS must agree exactly.
    assert forward == pytest.approx(backward, abs=1e-9)
    # Tight regression pin on the value actually measured here.
    assert forward == pytest.approx(18.8445, abs=0.01)


# ------------------------------- 2a. GNPy-actually-ran: pinned + monotone chain


def test_pinned_value_and_monotone_decrease_with_path_length(german17):
    """GSNR must fall monotonically as the segment gets longer.

    Values measured through this bridge (gnpy 2.14.0, german_17, 400G mode,
    24 channels, explicit ``center_freq_hz``):

        0->1      144 km   19.1215 dB
        0->3->5   366 km   16.8178 dB
        0->..->10 548 km   15.1489 dB
        0->..->16 772 km   14.0908 dB

    The migration plan quotes 19.17 / 16.89 / 15.22 / 14.17. That ~0.05-0.07 dB
    offset is fully explained: those reference numbers were measured *without*
    an explicit ``center_freq_hz``, i.e. with the adapter probing the lowest
    channel in the loading rather than the CUT. Re-running these same paths
    with ``center_freq_hz=None`` reproduces the plan's figures to within
    rounding (19.1662 / 16.8828 / 15.2150 / 14.1639) — see
    ``test_explicit_center_freq_changes_the_answer``.
    """
    def gsnr(node_path):
        seq = ob.oms_sequence_for_node_path(german17, node_path)
        return ob.segment_gsnr_db(german17, seq, MODE, 24)

    g144 = gsnr(PATH_144)
    g366 = gsnr(PATH_366)
    g548 = gsnr(PATH_548)
    g772 = gsnr(PATH_772)

    # Load-bearing: strict monotonicity in path length.
    assert g144 > g366 > g548 > g772

    # Secondary sanity pins against the values measured through this bridge.
    assert g144 == pytest.approx(19.1215, abs=0.05)
    assert g366 == pytest.approx(16.8178, abs=0.05)
    assert g548 == pytest.approx(15.1489, abs=0.05)
    assert g772 == pytest.approx(14.0908, abs=0.05)


def test_oms_sequence_for_node_path_uses_traversal_direction(german17):
    assert ob.oms_sequence_for_node_path(german17, PATH_366) == (
        "oms_0_3", "oms_3_5")
    # Not canonicalised to src < dst: the reverse walk picks the reverse OMS.
    assert ob.oms_sequence_for_node_path(german17, [5, 3, 0]) == (
        "oms_5_3", "oms_3_0")


# --------------------------------------- 2b. GNPy-actually-ran: propagation spy


def test_gnpy_fiber_propagation_actually_executes(german17, monkeypatch):
    """Physics-independent proof that gnpy's own ``Fiber.__call__`` ran.

    Immune to any future change in the numbers: a fallback approximation (the
    old ``gnpy_bridge.py``'s behaviour) leaves this counter at zero.
    """
    import gnpy.core.elements as gnpy_elements

    real_call = gnpy_elements.Fiber.__call__
    calls = {"n": 0}

    def counting_call(self, *args, **kwargs):
        calls["n"] += 1
        return real_call(self, *args, **kwargs)

    monkeypatch.setattr(gnpy_elements.Fiber, "__call__", counting_call)

    seq = ob.oms_sequence_for_node_path(german17, PATH_366)
    expected_spans = sum(_n_spans(german17, oms_id) for oms_id in seq)
    assert expected_spans == 5  # 0->3 has 4 spans, 3->5 has 1

    ob.segment_gsnr_db(german17, seq, MODE, 4)

    assert calls["n"] >= expected_spans


# ------------------------------------ 2c. GNPy-actually-ran: loading sensitivity


def test_gsnr_decreases_monotonically_with_channel_loading(german17):
    """More interferers -> more NLI -> lower GSNR. The old bridge failed this.

    The old ``gnpy_bridge.py`` returned 25.75 dB at *both* 12 and 48 channels —
    a bit-identical number, because its analytical fallback dominated. Any
    strict inequality here is therefore the regression guard.

    Measured on 0->..->16 (772 km) through this bridge:
        n=1: 14.1933   n=12: 14.1179   n=24: 14.0908   n=48: 14.0642 dB

    NOTE (deviation from the task brief): the brief asked for
    ``gsnr(48) < gsnr(12) - 0.3``. That margin is not physically reachable on
    this topology — every ROADM re-equalises to ``target_pch_out_db=-20 dBm``,
    a launch power at which NLI is far below ASE, so the whole 1->48 channel
    sweep only spans 0.129 dB. The measured 12->48 gap is 0.0537 dB, so the
    threshold below is 0.03 dB; the four-point strict monotone chain is added
    to compensate for the smaller margin.
    """
    seq = ob.oms_sequence_for_node_path(german17, PATH_772)
    g1 = ob.segment_gsnr_db(german17, seq, MODE, 1)
    g12 = ob.segment_gsnr_db(german17, seq, MODE, 12)
    g24 = ob.segment_gsnr_db(german17, seq, MODE, 24)
    g48 = ob.segment_gsnr_db(german17, seq, MODE, 48)

    assert g1 > g12 > g24 > g48
    assert g48 < g12 - 0.03
    assert g1 == pytest.approx(14.1933, abs=0.05)
    assert g48 == pytest.approx(14.0642, abs=0.05)


# ---------------------------------------- 2d. GNPy-actually-ran: breakdown shape


def _breakdown(topology, oms_sequence, n_channels=24):
    """Replicate ``segment_gsnr_db``'s compute_qot call, keeping the store.

    Inline rather than changing ``segment_gsnr_db``'s public signature for one
    test's sake.
    """
    store = QoTResultStore(max_results=4, ttl_seconds=None)
    state, rid = compute_qot(
        model=topology,
        store=store,
        oms_sequence=oms_sequence,
        direction=Direction.FORWARD,
        mode_id=MODE,
        loading=ob.build_loading(n_channels, MODE),
        center_freq_hz=ob.CUT_FREQ_HZ,
    )
    return state, store.get(rid)


def test_breakdown_has_one_snapshot_per_propagated_element(german17):
    """Snapshot count == sum over legs of (2 + 2*n_spans) + 1.

    Each OMS contributes [roadm, booster, (fiber, amp) * n_spans] = 2 + 2n
    elements, and the adapter appends the terminal drop ROADM once for the
    whole path (S4-4) — hence the single trailing +1.

    NOTE (deviation from the task brief): the brief said a single-OMS segment
    yields ``2 + 2*n_spans`` snapshots and that the ``+1`` is only needed for
    multi-leg sequences. That is wrong — the terminal drop ROADM is appended
    for every path, single-leg included. Upstream's own
    ``test_breakdown_cached_in_store_with_one_snapshot_per_element`` asserts
    exactly 7 snapshots for its single-OMS 2-span toy ("Six OMS elements + the
    terminal drop ROADM appended by C2 Step B -> seven"), which is
    2 + 2*2 + 1, not 2 + 2*2. Measured here: 7 for 0->1 and 15 for 0->3->5.
    """
    # Single leg: oms_0_1 has 2 spans -> 2 + 2*2 + 1 = 7.
    seq = ob.oms_sequence_for_node_path(german17, PATH_144)
    n_spans = _n_spans(german17, seq[0])
    assert n_spans == 2
    _state, bd = _breakdown(german17, seq)
    assert len(bd.snapshots) == 2 + 2 * n_spans + 1
    assert bd.snapshots[0].element_id == "roadm_0"
    assert bd.snapshots[-1].element_id == "roadm_1"

    # Multi leg: (2 + 2*4) + (2 + 2*1) + 1 = 15.
    seq = ob.oms_sequence_for_node_path(german17, PATH_366)
    expected = sum(2 + 2 * _n_spans(german17, oms_id) for oms_id in seq) + 1
    _state, bd = _breakdown(german17, seq)
    assert expected == 15
    assert len(bd.snapshots) == expected
    assert bd.snapshots[0].element_id == "roadm_0"
    assert bd.snapshots[-1].element_id == "roadm_5"


# ------------------------------------------------ 3. probe selection has teeth


def test_cut_is_not_first_in_tuple_order_for_multichannel_loading():
    """Precondition for the test below: without this, omitting
    ``center_freq_hz`` would accidentally pick the right channel anyway."""
    loading = ob.build_loading(24, MODE)
    assert loading.channels[0].center_freq_hz != ob.CUT_FREQ_HZ
    assert any(c.center_freq_hz == ob.CUT_FREQ_HZ for c in loading.channels)


def test_explicit_center_freq_changes_the_answer(german17):
    """``center_freq_hz=CUT_FREQ_HZ`` is load-bearing, not decorative.

    With it omitted, ``compute_qot`` falls back to "first channel whose
    mode_id matches" — which, for a frequency-ascending CUT-centered comb, is
    the lowest channel, not the CUT. Measured on 0->1 at 24 channels:
    19.1215 dB (explicit, correct) vs 19.1662 dB (omitted, wrong carrier).
    """
    seq = ob.oms_sequence_for_node_path(german17, PATH_144)
    loading = ob.build_loading(24, MODE)

    explicit = ob.segment_gsnr_db(german17, seq, MODE, 24)

    state_default, _rid = compute_qot(
        model=german17,
        store=QoTResultStore(max_results=1, ttl_seconds=None),
        oms_sequence=seq,
        direction=Direction.FORWARD,
        mode_id=MODE,
        loading=loading,
        center_freq_hz=None,
    )
    implicit = state_default.gsnr_db

    assert explicit != pytest.approx(implicit, abs=1e-3)
    assert abs(explicit - implicit) > 0.01
    assert explicit == pytest.approx(19.1215, abs=0.05)
    assert implicit == pytest.approx(19.1662, abs=0.05)


# ------------------------------------------------------------ 4. mode invariance


def test_all_modes_share_baud_and_roll_off(german17):
    """Guards the GSNR-invariance assertion below from becoming vacuous."""
    modes = german17.modes.list()
    assert len(modes) == 11
    assert {m.symbol_rate_baud for m in modes} == {87.5e9}
    assert {m.roll_off for m in modes} == {0.15}
    assert {m.channel_spacing_hz for m in modes} == {100e9}


def test_gsnr_is_identical_across_all_eleven_modes(german17):
    """GSNR depends on baud rate and roll-off, not bitrate or SNR threshold.

    Measured: all 11 modes return 16.842452460440548 dB bit-for-bit on
    0->3->5 at 12 channels.
    """
    seq = ob.oms_sequence_for_node_path(german17, PATH_366)
    values = {
        m.id: ob.segment_gsnr_db(german17, seq, m.id, 12)
        for m in german17.modes.list()
    }
    assert len(values) == 11
    assert len(set(values.values())) == 1, values
    assert next(iter(values.values())) == pytest.approx(16.8425, abs=0.05)


# ------------------------------------------------------------ 5. failure-raises


def test_node_path_with_nonexistent_edge_raises_key_error(german17):
    # 0 and 16 are not directly connected in german_17.
    with pytest.raises(KeyError):
        ob.oms_sequence_for_node_path(german17, [0, 16])


def test_node_path_shorter_than_two_nodes_raises_value_error(german17):
    with pytest.raises(ValueError):
        ob.oms_sequence_for_node_path(german17, [0])
    with pytest.raises(ValueError):
        ob.oms_sequence_for_node_path(german17, [])


@pytest.mark.parametrize("n_channels", [0, -1, 49, 100])
def test_build_loading_rejects_out_of_range_channel_counts(n_channels):
    with pytest.raises(ValueError):
        ob.build_loading(n_channels, MODE)


def test_bridge_source_contains_no_try_except():
    """No fallback, ever. The whole point of replacing ``gnpy_bridge.py``.

    AST-based, so it is immune to formatting (line breaks, comments, a bare
    ``except:``, ``except (A, B)``, ...).
    """
    tree = ast.parse(BRIDGE_SRC.read_text(encoding="utf-8"))
    offenders = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Try, ast.ExceptHandler))
    ]
    assert not offenders, [
        f"{type(n).__name__} at line {n.lineno}" for n in offenders
    ]


# ------------------------------------------------------------ 6. import isolation


def test_import_pulls_in_no_mcp_pydantic_or_ip_layer():
    """Must run in a genuinely fresh subprocess: in-process, pytest has already
    imported half the world via the other test modules above."""
    code = (
        "import diffopt.qot.optical_bridge;"
        "import sys;"
        "bad=[m for m in sys.modules "
        "if m == 'mcp' or m.startswith('mcp.') "
        "or m == 'pydantic' or m.startswith('pydantic.') "
        "or m.endswith(('.ip_assets','.network','.ip_routing'))];"
        "assert not bad, bad"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
