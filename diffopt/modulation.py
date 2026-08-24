"""Modulation format config: bitrate -> SNR threshold lookup."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class ModulationConfig:
    channel_spacing_ghz: float
    symbol_rate_gbaud: float
    num_channels_cband: int
    cut_channel_index: int
    formats: List[dict]  # list of {bitrate_gbps, snr_threshold_db}
    _snr_table: Dict[float, float] = None

    def __post_init__(self):
        self._snr_table = {
            float(f["bitrate_gbps"]): float(f["snr_threshold_db"])
            for f in self.formats
        }

    @classmethod
    def from_yaml(cls, path: str) -> "ModulationConfig":
        data = yaml.safe_load(Path(path).read_text())
        return cls(
            channel_spacing_ghz=data["channel_spacing_ghz"],
            symbol_rate_gbaud=data["symbol_rate_gbaud"],
            num_channels_cband=data["num_channels_cband"],
            cut_channel_index=data["cut_channel_index"],
            formats=data["formats"],
        )

    def required_snr_threshold(self, bitrate_gbps: float) -> float:
        """Return SNR threshold (dB) for the given bitrate. Raises ValueError if not found."""
        key = float(bitrate_gbps)
        if key not in self._snr_table:
            raise ValueError(
                f"Bitrate {bitrate_gbps} Gbps not in modulation table. "
                f"Valid values: {sorted(self._snr_table.keys())}"
            )
        return self._snr_table[key]

    def max_feasible_bitrate(self, gsnr_db: float) -> Optional[float]:
        """Return highest bitrate (Gbps) whose SNR threshold <= gsnr_db, or None."""
        feasible = [
            br for br, thr in self._snr_table.items() if thr <= gsnr_db
        ]
        return max(feasible) if feasible else None

    @property
    def bitrate_options(self) -> List[float]:
        return sorted(self._snr_table.keys())


def bar_db_for_demands(demands, modulation_config, margin_db: float) -> "torch.Tensor":
    """(D,) tensor of `threshold(bitrate_d) + margin_db`, in demand order.

    One definition of "the bar", shared by the loss's hinge, the allocation
    head's features 1/2/4, the oracle's feasibility test and hard_rollout's
    violation count. Four places computing `threshold + margin` inline is
    four places to drift, and a drifted bar makes oracle_gap unreadable.
    """
    import torch

    return torch.tensor(
        [
            modulation_config.required_snr_threshold(d.bitrate_gbps) + margin_db
            for d in demands
        ],
        dtype=torch.float32,
    )
