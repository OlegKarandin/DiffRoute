"""GNPy bridge: simulate a transparent optical segment, return GSNR at CUT."""
from __future__ import annotations

import random
from typing import List, Optional

import numpy as np

# CUT center frequency (Hz)
CUT_FREQ_HZ = 193.5e12
# C-band channel grid: 48 channels at 100 GHz spacing centered around 193.5 THz
CHANNEL_SPACING_HZ = 100e9
NUM_CHANNELS = 48
# Build the full grid: channels centered at 193.5 THz ± k*100GHz
# Using ITU grid: f_start so that channel index 24 (0-based) = 193.5 THz
# f_0 = 193.5e12 - 24 * 100e9 = 191.1e12
GRID_START_HZ = CUT_FREQ_HZ - 24 * CHANNEL_SPACING_HZ  # 191.1 THz
ALL_CHANNEL_FREQS_HZ = [GRID_START_HZ + i * CHANNEL_SPACING_HZ for i in range(NUM_CHANNELS)]

# SSMF fiber parameters
SSMF_ALPHA_DB_KM = 0.2        # attenuation dB/km
SSMF_D_PS_NM_KM = 17.0        # dispersion ps/nm/km
SSMF_GAMMA_PER_W_KM = 1.3     # nonlinear coefficient 1/W/km

# Speed of light
C_M_S = 299792458.0


def build_channel_list(n_channels: int, seed: Optional[int] = None) -> List[float]:
    """Return sorted list of n_channels frequencies (Hz) always including CUT.

    Args:
        n_channels: Total number of channels (1-48).
        seed: Optional random seed.

    Returns:
        Sorted list of channel center frequencies in Hz.
    """
    if n_channels < 1 or n_channels > NUM_CHANNELS:
        raise ValueError(f"n_channels must be in [1, {NUM_CHANNELS}], got {n_channels}")

    others = [f for f in ALL_CHANNEL_FREQS_HZ if abs(f - CUT_FREQ_HZ) > 1e6]
    rng = random.Random(seed)
    selected = rng.sample(others, k=n_channels - 1)
    selected.append(CUT_FREQ_HZ)
    return sorted(selected)


def _try_gnpy_simulate(
    span_lengths_km: List[float],
    amplifier_nf_db: List[float],
    fiber_type: str,
    n_channels: int,
    launch_power_dbm: float,
    seed: Optional[int],
) -> float:
    """Attempt GNPy simulation. Returns GSNR (dB) at CUT."""
    from gnpy.core.elements import Fiber, Edfa, Roadm
    from gnpy.core.network import build_network
    from gnpy.core.info import create_input_spectral_information, SpectralInformation
    from gnpy.core.utils import db2lin, lin2db, automatic_nch
    from gnpy.tools.json_io import load_equipment
    import networkx as nx

    freqs = build_channel_list(n_channels, seed)
    baud_rate = 87.5e9
    roll_off = 0.1

    # Launch power per channel (linear, W)
    p_launch_w = db2lin(launch_power_dbm) * 1e-3

    n_spans = len(span_lengths_km)

    # Build networkx graph: alternating Fiber -> Edfa
    G = nx.DiGraph()

    nodes = []
    for i, (span_km, nf_db) in enumerate(zip(span_lengths_km, amplifier_nf_db)):
        fiber_id = f"fiber_{i}"
        amp_id = f"amp_{i}"

        span_loss_db = SSMF_ALPHA_DB_KM * span_km

        fiber = Fiber(
            uid=fiber_id,
            params={
                "length": span_km,
                "loss_coef": SSMF_ALPHA_DB_KM,
                "length_units": "km",
                "att_in": 0,
                "dispersion": SSMF_D_PS_NM_KM * 1e-3 * 1e-12,  # s/m/m -> convert
                "gamma": SSMF_GAMMA_PER_W_KM * 1e-3,
            }
        )
        amp = Edfa(
            uid=amp_id,
            params={
                "f_min": min(freqs) - CHANNEL_SPACING_HZ,
                "f_max": max(freqs) + CHANNEL_SPACING_HZ,
                "gain_flatmax": span_loss_db + 3,
                "gain_min": 0,
                "p_max": 23,
                "nf_min": nf_db,
                "nf_max": nf_db + 2,
                "out_voa_auto": False,
                "allowed_for_design": True,
            },
            operational={
                "gain_target": span_loss_db,
                "delta_p": 0,
                "out_voa": 0,
                "in_voa": 0,
            }
        )
        nodes.append((fiber, amp))

    # Create SI
    si = create_input_spectral_information(
        f_min=min(freqs),
        f_max=max(freqs),
        roll_off=roll_off,
        baud_rate=baud_rate,
        power=p_launch_w,
        spacing=CHANNEL_SPACING_HZ,
        tx_osnr=None,
    )

    # Propagate through chain
    for fiber, amp in nodes:
        si = fiber(si)
        si = amp(si)

    # Extract GSNR at CUT
    cut_idx = None
    for idx, ch in enumerate(si.carriers):
        if abs(ch.frequency - CUT_FREQ_HZ) < CHANNEL_SPACING_HZ / 2:
            cut_idx = idx
            break

    if cut_idx is None:
        raise RuntimeError("CUT channel not found in output SI")

    ch = si.carriers[cut_idx]
    # GSNR = signal power / (ASE + NLI)
    signal_power = ch.power.signal
    noise_power = ch.power.ase + ch.power.nli
    if noise_power <= 0:
        raise RuntimeError("Zero noise power encountered")

    gsnr_linear = signal_power / noise_power
    return float(10 * np.log10(gsnr_linear))


def analytical_gsnr_db(
    span_lengths_km: List[float],
    amp_nf_db: List[float],
    n_channels: int,
    launch_power_dbm: float = -1.0,
) -> float:
    """Simplified GN-model GSNR estimate (fallback when GNPy unavailable).

    Uses Poggiolini approximation for NLI noise.
    """
    # Physical constants
    h = 6.626e-34      # Planck constant J*s
    nu = CUT_FREQ_HZ   # optical frequency Hz

    alpha_lin = SSMF_ALPHA_DB_KM / (10 * np.log10(np.e)) / 1000  # 1/m
    L_eff_asymp = 1.0 / (2 * alpha_lin)  # m

    gamma = SSMF_GAMMA_PER_W_KM / 1000  # 1/W/m
    baud_rate = 87.5e9  # Hz

    p_ch_w = 10 ** (launch_power_dbm / 10) * 1e-3  # W per channel

    # Total ASE noise power over all spans
    G_ase_total = 0.0
    G_nli_total = 0.0

    for span_km, nf_db in zip(span_lengths_km, amp_nf_db):
        span_m = span_km * 1000

        # Span loss (linear)
        span_loss_lin = 10 ** (SSMF_ALPHA_DB_KM * span_km / 10)

        # Amp gain = span loss (lossless amplification)
        G_amp_lin = span_loss_lin

        # ASE PSD (one-sided, per polarization) from this amp
        nf_lin = 10 ** (nf_db / 10)
        G_ase = nf_lin * h * nu * (G_amp_lin - 1)  # W/Hz (one-sided)
        G_ase_total += G_ase

        # NLI power (GN model simplified)
        # P_NLI ≈ (8/27) * gamma^2 * L_eff^2 * P_ch^3 * N_ch^2 * asinh(pi^2 * |beta2| * L_eff_asymp * N_ch^2 * baud^2)
        # Use effective length
        L_eff = (1 - np.exp(-2 * alpha_lin * span_m)) / (2 * alpha_lin)

        beta2 = abs(SSMF_D_PS_NM_KM * 1e-3 * 1e-12 / (2 * np.pi * CUT_FREQ_HZ / C_M_S) ** 2 * C_M_S)
        # Simplified: beta2 ≈ D * lambda^2 / (2*pi*c)
        lam_m = C_M_S / CUT_FREQ_HZ
        beta2 = SSMF_D_PS_NM_KM * 1e-3 * 1e-12 * lam_m ** 2 / (2 * np.pi * C_M_S)

        bw_total = n_channels * baud_rate
        arg = np.pi ** 2 * abs(beta2) * L_eff_asymp * bw_total ** 2
        if arg > 0:
            nli_psd = (8.0 / 27.0) * gamma ** 2 * L_eff ** 2 * p_ch_w ** 3 / baud_rate ** 2 * np.arcsinh(arg)
        else:
            nli_psd = 0.0
        G_nli_total += nli_psd * baud_rate  # W

    # Signal power at output (after last amp restores to launch power)
    signal_w = p_ch_w
    noise_total_w = G_ase_total * baud_rate + G_nli_total

    if noise_total_w <= 0:
        return 30.0  # degenerate case

    gsnr_lin = signal_w / noise_total_w
    return float(10 * np.log10(gsnr_lin))


def simulate_segment(
    span_lengths_km: List[float],
    amplifier_nf_db: List[float],
    fiber_type: str = "SSMF",
    n_channels: int = 24,
    launch_power_dbm: float = -1.0,
    seed: Optional[int] = None,
) -> float:
    """Simulate a transparent optical segment and return GSNR (dB) at CUT.

    Tries GNPy first; falls back to analytical GN model if GNPy fails.

    Args:
        span_lengths_km: List of span lengths (one per span).
        amplifier_nf_db: List of amplifier NF values (one per span).
        fiber_type: Fiber type string (currently only "SSMF" supported).
        n_channels: Number of WDM channels loaded (1-48).
        launch_power_dbm: Per-channel launch power in dBm.
        seed: Optional random seed for channel selection.

    Returns:
        GSNR in dB at the center channel under test.
    """
    try:
        return _try_gnpy_simulate(
            span_lengths_km, amplifier_nf_db, fiber_type, n_channels, launch_power_dbm, seed
        )
    except Exception:
        return analytical_gsnr_db(span_lengths_km, amplifier_nf_db, n_channels, launch_power_dbm)
