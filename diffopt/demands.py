"""Demand generation for optical network routing."""
from __future__ import annotations

import random
from typing import List, NamedTuple, Optional

from diffopt.topology import Topology


class Demand(NamedTuple):
    id: int
    src: int
    dst: int
    bitrate_gbps: float


def generate_demands(
    topology: Topology,
    num_demands: int,
    bitrate_options: List[float],
    seed: Optional[int] = None,
) -> List[Demand]:
    """Generate random demands.

    Args:
        topology: Network topology.
        num_demands: Number of demands to generate.
        bitrate_options: List of valid bitrate values to sample from.
        seed: Optional random seed.

    Returns:
        List of Demand namedtuples (id, src, dst, bitrate_gbps).
    """
    rng = random.Random(seed)
    n = topology.num_nodes
    demands = []
    for i in range(num_demands):
        src = rng.randrange(n)
        dst = rng.randrange(n)
        while dst == src:
            dst = rng.randrange(n)
        bitrate = rng.choice(bitrate_options)
        demands.append(Demand(id=i, src=src, dst=dst, bitrate_gbps=float(bitrate)))
    return demands
