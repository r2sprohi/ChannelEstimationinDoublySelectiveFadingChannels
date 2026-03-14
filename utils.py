"""
utils.py
Shared utilities: sine positional encoding, config dataclass, misc helpers.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# System configuration
# ---------------------------------------------------------------------------

@dataclass
class SystemConfig:
    """Holds all OFDM / channel / pilot parameters for one configuration."""
    n_rb: int           # number of resource blocks {6, 12, 18, 24, 32}
    dmrs_type: int      # DMRS type {1, 2}
    n_dmrs_sym: int     # number of DMRS symbols per slot {2, 3, 4}

    # Fixed 5G NR parameters
    n_sc_per_rb: int = 12          # subcarriers per RB
    n_t: int = 14                  # OFDM symbols per slot
    subcarrier_spacing: float = 30e3   # Hz
    carrier_freq: float = 2.5e9   # Hz
    max_speed_kmh: float = 120.0   # km/h
    max_delay_ns: float = 2510.0   # ns

    @property
    def n_f(self) -> int:
        return self.n_rb * self.n_sc_per_rb

    @property
    def n_fft(self) -> int:
        """FFT size: next power-of-2 >= n_f."""
        p = 1
        while p < self.n_f:
            p <<= 1
        return p

    @property
    def max_doppler_hz(self) -> float:
        v = self.max_speed_kmh / 3.6  # m/s
        lam = 3e8 / self.carrier_freq  # wavelength m
        return v / lam

    @property
    def symbol_duration_s(self) -> float:
        """OFDM symbol duration including normal CP (5G NR, 30 kHz SCS)."""
        return 1.0 / (14 * 1000)  # 1 slot = 1 ms / 14 symbols ≈ 71.43 µs

    @property
    def sample_rate_hz(self) -> float:
        return self.n_fft * self.subcarrier_spacing

    def max_delay_samples(self) -> int:
        tau_s = self.max_delay_ns * 1e-9
        return max(1, round(tau_s * self.sample_rate_hz))

    def pilot_freq_indices(self) -> np.ndarray:
        """Subcarrier indices used for DMRS in frequency domain."""
        if self.dmrs_type == 1:
            # Comb-2: every other subcarrier (port 0 = even)
            return np.arange(0, self.n_f, 2)
        elif self.dmrs_type == 2:
            # Groups of 2 per 6: {0,1,6,7,12,13,...}
            idx = []
            for rb in range(self.n_rb):
                base = rb * 12
                idx += [base, base + 1, base + 6, base + 7]
            return np.array(idx)
        else:
            raise ValueError(f"Unknown DMRS type {self.dmrs_type}")

    def pilot_time_indices(self) -> np.ndarray:
        """OFDM symbol indices for DMRS within a 14-symbol slot."""
        mapping = {2: [2, 11], 3: [2, 7, 11], 4: [2, 5, 8, 11]}
        if self.n_dmrs_sym not in mapping:
            raise ValueError(f"n_dmrs_sym must be in {list(mapping.keys())}")
        return np.array(mapping[self.n_dmrs_sym])

    def pilot_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (freq_idx, time_idx) arrays for all pilot positions.
        Each array has length n_pilots.
        """
        f_idx = self.pilot_freq_indices()  # (N_p_per_sym,)
        t_idx = self.pilot_time_indices()  # (N_dmrs_sym,)
        # Cartesian product: all (f, t) combinations
        F, T = np.meshgrid(f_idx, t_idx)  # each (N_dmrs_sym, N_p_per_sym)
        return F.ravel(), T.ravel()

    @property
    def n_pilots(self) -> int:
        f_idx = self.pilot_freq_indices()
        t_idx = self.pilot_time_indices()
        return len(f_idx) * len(t_idx)

    def __repr__(self):
        return (f"SystemConfig(n_rb={self.n_rb}, N_f={self.n_f}, N_t={self.n_t}, "
                f"dmrs_type={self.dmrs_type}, n_dmrs_sym={self.n_dmrs_sym}, "
                f"n_pilots={self.n_pilots})")


# ---------------------------------------------------------------------------
# Sine positional encoding (1D and 2D)
# ---------------------------------------------------------------------------

def sine_encode_1d(positions: torch.Tensor, d: int) -> torch.Tensor:
    """
    Standard sinusoidal encoding for 1D positions.
    positions : (..., )  float
    Returns   : (..., d)
    """
    assert d % 2 == 0
    half = d // 2
    freq = torch.arange(half, dtype=torch.float32, device=positions.device)
    freq = 1.0 / (10000.0 ** (freq / half))          # (half,)
    angles = positions.unsqueeze(-1) * freq           # (..., half)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (..., d)


def sine_encode_2d(rel_pos: torch.Tensor, d: int) -> torch.Tensor:
    """
    Sine encoding for 2D relative positions (Δt, Δf).
    rel_pos : (..., 2)   [Δt, Δf]
    d       : embedding dim, must be divisible by 4
    Returns : (..., d)
    """
    assert d % 4 == 0, "d must be divisible by 4 for 2D sine encoding"
    quarter = d // 4
    freq = torch.arange(quarter, dtype=torch.float32, device=rel_pos.device)
    freq = 1.0 / (10000.0 ** (freq / quarter))       # (quarter,)

    dt = rel_pos[..., 0:1]                            # (..., 1)
    df = rel_pos[..., 1:2]                            # (..., 1)

    t_angles = dt * freq                              # (..., quarter)
    f_angles = df * freq                              # (..., quarter)

    return torch.cat([
        torch.sin(t_angles), torch.cos(t_angles),
        torch.sin(f_angles), torch.cos(f_angles),
    ], dim=-1)                                        # (..., d)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 10.0)


def linear_to_db(x: float) -> float:
    return 10.0 * math.log10(max(x, 1e-15))


def mse_db(h_true: np.ndarray, h_est: np.ndarray) -> float:
    """MSE in dB between true and estimated channel matrices."""
    mse = np.mean(np.abs(h_true - h_est) ** 2)
    power = np.mean(np.abs(h_true) ** 2)
    return 10.0 * math.log10(max(mse / max(power, 1e-15), 1e-15))


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# All training configs  (paper Table I)
# ---------------------------------------------------------------------------

TRAIN_CONFIGS = [
    SystemConfig(n_rb=rb, dmrs_type=1, n_dmrs_sym=ns)
    for rb in [6, 12, 18, 24]
    for ns in [2, 3, 4]
]

# T1 = seen configs (same RBs/type as training but random realizations)
TEST_T1_CONFIGS = [
    SystemConfig(n_rb=rb, dmrs_type=1, n_dmrs_sym=ns)
    for rb in [6, 12, 18, 24]
    for ns in [2, 3, 4]
]

# T2 = unseen configs (32 RBs and/or DMRS type 2)
TEST_T2_CONFIGS = [
    SystemConfig(n_rb=32, dmrs_type=1, n_dmrs_sym=ns) for ns in [2, 3, 4]
] + [
    SystemConfig(n_rb=rb, dmrs_type=2, n_dmrs_sym=ns)
    for rb in [6, 12, 18, 24, 32]
    for ns in [2, 3, 4]
]

SNR_TRAIN_RANGE = (0.0, 30.0)   # continuous uniform during training
SNR_TEST_STEPS = list(range(0, 31, 5))  # 0, 5, 10, 15, 20, 25, 30 dB
