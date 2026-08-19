"""Fixed traffic matrix — the constraint set the duals in diffopt/loss.py act on.

Before this module, diffopt/train.py called `generate_demands(..., seed=epoch)`
and drew 100 fresh random demands every epoch. "All demands feasible" cannot be
stated, let alone enforced, against a set that is replaced each epoch, and a
per-demand dual is meaningless without demand identity persisting across
epochs (spec §1 finding #4).

The matrix is NOT committed as a file. It regenerates deterministically from
(topology, seed, scale, alpha) and `test_traffic.py` pins a checksum of the
result. This catches silent upstream drift without a committed artifact that
can go stale — a deliberate choice given this project's history of a dependency
pin that became uninstallable (invariants.md, "Pin a tag, never a branch
commit").
"""
from __future__ import annotations

import hashlib
from typing import Dict, List

from multilayer_optical_network.model.traffic import (
    generate_demands as _gravity_demands,
)

from diffopt.demands import Demand
from diffopt.topology import Topology


# `alpha` is the distance exponent in the gravity kernel
# (weight ~ mass(u)*mass(v) / dist(u,v)^alpha). It is derived from a *named*
# scenario rather than set directly in configs, so the two settings stay
# reportable things rather than free-floating numbers.
#
#   realistic (alpha=1.0): gravity falls off with distance; big pipes land on
#       short hops. Measured corr(bitrate, km) = -0.829, 0/865 infeasible.
#   stress    (alpha=0.0): pure mass-product; bitrate uncorrelated with
#       distance. Measured corr(bitrate, km) = -0.008, 20/865 infeasible.
#
# Both are kept deliberately (spec §3): reporting only `realistic` would
# demonstrate the dual machinery on a problem where the constraint never binds;
# reporting only `stress` would overstate difficulty relative to real traffic.
SCENARIO_ALPHA: Dict[str, float] = {"realistic": 1.0, "stress": 0.0}

# A pair offering less than the smallest lightpath minus half a bitrate step
# (300 - 25 = 275 G) carries no lightpath at all and is dropped rather than
# rounded up to 300 G.
_MIN_VOLUME_MARGIN_GBPS = 25.0


def scenario_alpha(scenario: str) -> float:
    """Map a named traffic scenario to its gravity distance exponent."""
    if scenario not in SCENARIO_ALPHA:
        raise ValueError(
            f"unknown traffic scenario {scenario!r}; "
            f"valid values: {sorted(SCENARIO_ALPHA)}"
        )
    return SCENARIO_ALPHA[scenario]


def build_traffic_matrix(
    topology: Topology,
    *,
    seed: int,
    scale: float,
    alpha: float,
    bitrate_options: List[float],
) -> List[Demand]:
    """Build the fixed traffic matrix for a topology.

    Wraps upstream `generate_demands` with `aggregate=True` (one record per
    pair carrying its raw unquantized offered Gbps, an OD-matrix shape rather
    than 100 G grooming units), `undirected=True` (one record per unordered
    pair — gravity weight is symmetric, so both directions are exact
    duplicates), and `protected_fraction=0.0` (this project has no protection
    concept; a `protected` flag would be dead weight on every record).

    Each pair's offered volume is mapped to the NEAREST member of
    `bitrate_options`; pairs below `min(bitrate_options) - 25` are dropped.

    Args:
        topology:        Loaded Topology (a bare OpticalNetworkModel subclass —
                         upstream derives node ids from OMS endpoints when
                         `list_routers` is absent, which is why the v0.1.2 pin
                         is required).
        seed:            Matrix identity. Enters upstream only through a
                         deterministic per-node mass jitter, so a fixed seed is
                         byte-stable and a different seed gives a distinct
                         held-out matrix.
        scale:           Total offered load in Gbps, spread across pairs by
                         gravity weight. Calibrated per scenario to land in
                         roughly 500-900 demands.
        alpha:           Gravity distance exponent — use `scenario_alpha()`.
        bitrate_options: The 11 valid bitrates. Every emitted `bitrate_gbps` is
                         an exact member, because `ModulationConfig.
                         required_snr_threshold` is an exact-float dict lookup
                         with no interpolation (invariants.md).

    Returns:
        List of `Demand` with ids contiguous `0..N-1`. Contiguity is
        load-bearing: `Demand.id` indexes the per-demand dual vector in
        `diffopt.loss.compute_loss`.
    """
    options = sorted(float(b) for b in bitrate_options)
    volume_floor = options[0] - _MIN_VOLUME_MARGIN_GBPS

    raw = _gravity_demands(
        topology,
        seed=seed,
        scale=scale,
        alpha=alpha,
        aggregate=True,
        undirected=True,
        protected_fraction=0.0,
    )

    demands: List[Demand] = []
    for record in raw:
        volume = float(record["demand_gbps"])
        if volume < volume_floor:
            continue
        # Nearest option; ties broken toward the lower bitrate so the mapping
        # is a deterministic function of the volume alone.
        bitrate = min(options, key=lambda b: (abs(b - volume), b))
        demands.append(Demand(
            id=len(demands),
            # Upstream node ids are strings ("0", "1", ... "131"); nodes are
            # integer IDs only downstream (invariants.md).
            src=int(record["src"]),
            dst=int(record["dst"]),
            bitrate_gbps=bitrate,
        ))
    return demands


def traffic_matrix_checksum(demands: List[Demand]) -> str:
    """Stable 16-hex-char digest of a matrix's (src, dst, bitrate) content.

    Deliberately excludes `id`, which is a positional artifact — this way the
    digest is unchanged by a renumbering that preserves content, and changes
    when the actual demand set does.
    """
    payload = ";".join(
        f"{d.src}-{d.dst}-{d.bitrate_gbps:.6f}" for d in demands
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
