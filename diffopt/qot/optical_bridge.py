"""GNPy bridge built on ``multilayer_optical_network``'s ``compute_qot``.

This module replaces the pre-migration wrapper module, which wrapped every GNPy call
in a blanket exception handler and silently returned an analytical GN-model
approximation instead — for the whole project's history, *every* label came
from that fallback and GNPy never once executed.

The rule here is therefore absolute: **no ``try``/``except`` anywhere in this
file**. A physics failure must surface as a traceback, never as a plausible
number.

Public surface
--------------
``GRID`` / ``CUT_FREQ_HZ`` / ``CUT_SLOT``
    The fixed WDM grid and the channel-under-test.
``build_loading(n_channels, mode_id)``
    Deterministic, CUT-centered ``LoadingState`` with exactly *n_channels*.
``oms_sequence_for_node_path(topology, node_path)``
    diffopt numeric node path -> validated upstream OMS id tuple.
``segment_gsnr_db(topology, oms_sequence, mode_id, n_channels, cache=None)``
    End-to-end GNPy GSNR (dB) at the CUT for one transparent segment.

This module works purely in terms of upstream ``mode_id`` strings; resolving a
diffopt bitrate to a mode id is the caller's job.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from multilayer_optical_network.gnpy_adapter.adapter import compute_qot
from multilayer_optical_network.gnpy_adapter.loading import Channel, LoadingState
from multilayer_optical_network.model.assets import Direction
from multilayer_optical_network.model.qot_results import QoTCache, QoTResultStore
from multilayer_optical_network.model.spectrum import SpectrumGrid

# Fixed C-band grid shared with upstream: anchor 191.4 THz, 100 GHz spacing,
# 48 slots (191.4-196.1 THz).
GRID = SpectrumGrid.default()

# diffopt's C-band-center convention (docs/architecture/invariants.md).
CUT_FREQ_HZ = 193.5e12

# Module-level guard: ``slot_of`` raises ValueError for an off-grid frequency,
# so if GRID or CUT_FREQ_HZ ever drift apart, importing this module fails
# loudly instead of quietly probing the wrong channel.
CUT_SLOT = GRID.slot_of(CUT_FREQ_HZ)


def _cut_centered_slots(n_channels: int) -> List[int]:
    """Deterministic slot set of size *n_channels*, expanding outward from CUT.

    Policy: start at ``CUT_SLOT`` and alternate upward/downward
    (``CUT, CUT+1, CUT-1, CUT+2, CUT-2, ...``), skipping candidates that fall
    outside ``[0, GRID.num_slots)``. Rationale: NLI from a neighbouring channel
    falls off with spectral distance, so a CUT-centered comb is the physically
    meaningful way to grow the load — an arbitrary block at one edge of the
    grid would leave the CUT's immediate neighbourhood empty at low loads and
    understate cross-phase effects.

    The returned list is sorted ascending by slot index.
    """
    slots = [CUT_SLOT]
    offset = 1
    while len(slots) < n_channels:
        for candidate in (CUT_SLOT + offset, CUT_SLOT - offset):
            if 0 <= candidate < GRID.num_slots and len(slots) < n_channels:
                slots.append(candidate)
        offset += 1
    return sorted(slots)


def build_loading(n_channels: int, mode_id: str) -> LoadingState:
    """Build a deterministic ``LoadingState`` of *n_channels* on the fixed grid.

    Guarantees:

    * ``CUT_SLOT`` is always active (it is the probe channel);
    * exactly *n_channels* slots are active;
    * the same *n_channels* always yields the same slot set (no randomness);
    * slots grow outward from the CUT (see ``_cut_centered_slots``).

    Channels are emitted in **ascending frequency order**, matching how gnpy
    sorts its ``SpectralInformation`` internally. A consequence worth naming:
    for ``n_channels >= 2`` the CUT is *not* the first channel in tuple order,
    so ``compute_qot`` must be told which channel to probe via
    ``center_freq_hz`` — otherwise it falls back to "first channel with a
    matching mode_id", which here is the wrong carrier.

    ``power_dbm`` is always ``None``: every ROADM re-equalises to its own
    ``target_pch_out_db``, so a per-channel launch power is inert; ``None``
    means "use the adapter's default" and keeps a nonexistent launch-power
    concept out of this layer.
    """
    if not 1 <= n_channels <= GRID.num_slots:
        raise ValueError(
            f"n_channels must be in [1, {GRID.num_slots}], got {n_channels}"
        )
    channels = tuple(
        Channel(GRID.freq(slot), GRID.spacing_hz, None, mode_id)
        for slot in _cut_centered_slots(n_channels)
    )
    return LoadingState(channels=channels)


def oms_sequence_for_node_path(topology, node_path: Sequence[int]) -> Tuple[str, ...]:
    """Convert a diffopt numeric node path into a validated OMS id tuple.

    ``populate_optical`` registers two independent directed OMS per undirected
    edge, ``oms_{u}_{v}`` and ``oms_{v}_{u}``, each with its own amplifier
    chain. The traversal direction of *node_path* therefore picks which of the
    two is used — the OMS id is built literally from consecutive pairs, **not**
    canonicalised to ``src < dst``.

    Each id is validated with ``topology.get_oms``, which raises ``KeyError``
    for a nonexistent edge rather than deferring the failure into GNPy.
    """
    if len(node_path) < 2:
        raise ValueError(
            f"node_path must contain at least 2 nodes to traverse an edge, "
            f"got {list(node_path)!r}"
        )
    oms_ids = tuple(
        f"oms_{node_path[i]}_{node_path[i + 1]}" for i in range(len(node_path) - 1)
    )
    for oms_id in oms_ids:
        topology.get_oms(oms_id)
    return oms_ids


def segment_gsnr_db(
    topology,
    oms_sequence: Tuple[str, ...],
    mode_id: str,
    n_channels: int,
    cache: Optional[QoTCache] = None,
) -> float:
    """GNPy GSNR (dB) at the CUT for one transparent segment.

    *oms_sequence* is taken directly rather than derived from a node path, so
    this works both for diffopt's numeric-node-id topologies (via
    ``oms_sequence_for_node_path``) and for any hand-built model whose OMS ids
    follow a different convention.

    A fresh minimal ``QoTResultStore`` is built per call: ``compute_qot``
    always stores a breakdown, and the default store (512 entries, 600 s TTL)
    would rescan hundreds of entries per call for breakdowns nobody reads here.

    *cache* is a pass-through to ``compute_qot`` and defaults to ``None`` (no
    caching); enabling one is only worthwhile after an A/B measurement, since a
    varied ``n_channels`` sampling has a near-zero hit rate while the
    fingerprinting still costs real work every call.
    """
    store = QoTResultStore(max_results=1, ttl_seconds=None)
    loading = build_loading(n_channels, mode_id)
    state, _result_id = compute_qot(
        model=topology,
        store=store,
        oms_sequence=oms_sequence,
        direction=Direction.FORWARD,
        mode_id=mode_id,
        loading=loading,
        center_freq_hz=CUT_FREQ_HZ,
        cache=cache,
    )
    if not math.isfinite(state.gsnr_db):
        # adapter.py returns +inf when ase + nli == 0 — physically impossible on
        # a real span, so fail loudly instead of propagating inf downstream.
        raise RuntimeError(
            f"non-finite GSNR for oms_sequence={oms_sequence}, "
            f"mode_id={mode_id!r}, n_channels={n_channels}"
        )
    return float(state.gsnr_db)
