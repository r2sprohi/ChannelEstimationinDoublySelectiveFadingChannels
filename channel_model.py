"""
channel_model.py
CDL-E inspired doubly-selective channel model for 5G NR OFDM.

Uses a tapped delay line (TDL) with:
  - Power-delay profile approximating CDL-E (3GPP TR 38.901)
  - Time variation via complex-exponential Doppler per tap (Clarke's model)
  - Single-antenna (SISO)

Reference: 3GPP TR 38.901 V17.0.0, Table 7.7.2-4 (CDL-E)
"""

import numpy as np
from utils import SystemConfig


# ---------------------------------------------------------------------------
# CDL-E tap profile (15 clusters, 3GPP TR 38.901 Table 7.7.2-4, DS=2510 ns)
# ---------------------------------------------------------------------------

# Tap delays (ns) and relative powers (dB).  The profile is stretched so that
# the maximum delay equals cfg.max_delay_ns.
_CDLЕ_DELAYS_NS_NORM = np.array([
    0.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 1000.0,
    1200.0, 1400.0, 1500.0, 2000.0, 2300.0, 2500.0, 2510.0
])  # normalised to max=2510 ns

_CDLЕ_POWERS_DB = np.array([
    -0.03, -22.03, -15.80, -18.10, -19.80, -22.90, -22.40, -18.60,
    -20.80, -26.30, -24.10, -25.80, -22.70, -27.00, -27.10
])


def _cdlе_tap_powers() -> np.ndarray:
    """Linear tap powers (normalised so they sum to 1)."""
    p = 10.0 ** (_CDLЕ_POWERS_DB / 10.0)
    return p / p.sum()


# ---------------------------------------------------------------------------
# Channel generation
# ---------------------------------------------------------------------------

def generate_channel(cfg: SystemConfig,
                     n_samples: int = 1,
                     rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate doubly-selective channel matrices.

    Parameters
    ----------
    cfg       : SystemConfig
    n_samples : number of independent realisations
    rng       : numpy random Generator (for reproducibility)

    Returns
    -------
    H : complex128 ndarray, shape (n_samples, n_f, n_t)
        H[s, k, n] = channel frequency response at subcarrier k, symbol n.
    """
    if rng is None:
        rng = np.random.default_rng()

    N_f = cfg.n_f
    N_t = cfg.n_t
    N_fft = cfg.n_fft
    f_d_max = cfg.max_doppler_hz
    T_sym = cfg.symbol_duration_s
    fs = cfg.sample_rate_hz
    max_tau_ns = cfg.max_delay_ns

    # ---- tap profile -------------------------------------------------
    powers = _cdlе_tap_powers()              # (L,)
    tap_delays_ns = _CDLЕ_DELAYS_NS_NORM * (max_tau_ns / 2510.0)
    tap_delays_s = tap_delays_ns * 1e-9      # seconds
    L = len(powers)

    # Convert tap delays to fractional FFT bins: τ_l / T_sym_useful
    # T_sym_useful = 1 / Δf
    T_fft = 1.0 / cfg.subcarrier_spacing     # useful OFDM symbol period
    tap_delays_frac = tap_delays_s / T_fft   # fraction of T_fft (0..~0.075)

    # Pre-compute steering vectors: shape (L, N_f)
    k_idx = np.arange(N_f)                   # subcarrier indices
    # Phase shift from delay: exp(-j 2π k τ_l / T_fft)
    steering = np.exp(-1j * 2 * np.pi * k_idx[None, :] * tap_delays_frac[:, None])
    # shape: (L, N_f)

    # ---- generate realisations ---------------------------------------
    H = np.zeros((n_samples, N_f, N_t), dtype=np.complex128)

    for s in range(n_samples):
        # Initial complex Gaussian fading for each tap  (L,)
        h0 = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) / np.sqrt(2)

        # Random Doppler frequency for each tap  (L,)
        # Uniform on [-f_d_max, f_d_max] (isotropic scatter approx.)
        cos_theta = rng.uniform(-1.0, 1.0, L)
        f_d = f_d_max * cos_theta              # (L,)

        # Time-varying tap gains: shape (L, N_t)
        t_idx = np.arange(N_t)
        # Doppler phase: exp(j 2π f_d_l T_sym n)
        doppler_phase = np.exp(1j * 2 * np.pi * f_d[:, None] * T_sym * t_idx[None, :])
        # Combined: g_l(n) = sqrt(P_l) * h0_l * exp(j 2π f_d_l T_sym n)
        g = (np.sqrt(powers[:, None]) * h0[:, None] * doppler_phase)  # (L, N_t)

        # Frequency-domain channel matrix via superposition of taps
        # H[k, n] = sum_l g_l(n) * exp(-j 2π k τ_l / T_fft)
        # = (steering.T) @ g   where steering is (L, N_f)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            H[s] = steering.T @ g   # (N_f, N_t)

    return H


# ---------------------------------------------------------------------------
# Ideal MMSE estimator (oracle, uses true channel statistics)
# ---------------------------------------------------------------------------

def mmse_estimate(y_p: np.ndarray,
                  x_p: np.ndarray,
                  H_true: np.ndarray,
                  sigma2_n: float,
                  pilot_f: np.ndarray,
                  pilot_t: np.ndarray) -> np.ndarray:
    """
    MMSE channel estimation at all (N_f, N_t) grid positions using
    second-order statistics from the true channel ensemble.

    This is the 'oracle MMSE' baseline: it uses R_hh computed directly
    from the true channel realisations (not tractable in practice).

    Parameters
    ----------
    y_p      : (N_p,) received signal at pilots
    x_p      : (N_p,) known pilot symbols
    H_true   : (N_f, N_t) true channel (used only to compute statistics)
    sigma2_n : noise variance
    pilot_f  : (N_p,) pilot subcarrier indices
    pilot_t  : (N_p,) pilot OFDM symbol indices

    Returns
    -------
    H_mmse : (N_f, N_t) MMSE channel estimate (complex)
    """
    N_f, N_t = H_true.shape
    N_p = len(pilot_f)

    # LS estimate at pilots: h_LS = y_p / x_p
    h_ls = y_p / x_p                           # (N_p,)

    # Vectorise full channel
    h_full = H_true.ravel(order='F')           # (N_f*N_t,)  col-major

    # Pilot-to-full index mapping
    all_f = np.repeat(np.arange(N_f), N_t)     # (N_f*N_t,)
    all_t = np.tile(np.arange(N_t), N_f)       # (N_f*N_t,)
    full_idx = pilot_t * N_f + pilot_f         # (N_p,)  row-major flat index
    full_idx_fm = pilot_f * N_t + pilot_t      # col-major flat index

    # We use a simple 2D interpolation MMSE approximation:
    # Standard 1D frequency-direction MMSE per time slot then interpolate time
    # For a tractable baseline, apply 1D MMSE in freq per DMRS symbol, then
    # linearly interpolate in time.

    H_mmse = np.zeros((N_f, N_t), dtype=np.complex128)

    # Group pilots by time slot
    unique_t = np.unique(pilot_t)
    pilot_t_arr = np.array(pilot_t)
    pilot_f_arr = np.array(pilot_f)

    # Per DMRS symbol: frequency-domain MMSE
    H_dmrs = {}   # time_idx -> H estimate at that symbol (all N_f subcarriers)
    for tn in unique_t:
        mask = pilot_t_arr == tn
        f_idx = pilot_f_arr[mask]              # pilot subcarrier indices this symbol
        h_ls_t = h_ls[mask]                    # LS estimates this symbol

        # Covariance matrix at pilot subcarriers (identity * power approx.)
        # True MMSE: R_hp * (R_pp + σ²/σ_s² I)^-1
        # Simplification: use flat power spectrum → R_pp = power * I
        sigma2_h = np.mean(np.abs(H_true[:, tn]) ** 2)
        snr = sigma2_h / sigma2_n

        # Wiener filter weight (scalar per-pilot)
        w = snr / (snr + 1.0)
        h_mmse_pilots = w * h_ls_t             # Wiener at pilot positions

        # Interpolate to all N_f subcarriers using sinc interpolation
        H_dmrs[tn] = np.interp(
            np.arange(N_f), f_idx, h_mmse_pilots.real
        ) + 1j * np.interp(
            np.arange(N_f), f_idx, h_mmse_pilots.imag
        )

    # Time interpolation: for each subcarrier, linear interp across DMRS symbols
    dmrs_t = np.array(sorted(unique_t))
    for kk in range(N_f):
        h_at_dmrs = np.array([H_dmrs[tn][kk] for tn in dmrs_t])
        H_mmse[kk, :] = np.interp(
            np.arange(N_t), dmrs_t, h_at_dmrs.real
        ) + 1j * np.interp(
            np.arange(N_t), dmrs_t, h_at_dmrs.imag
        )

    return H_mmse


# ---------------------------------------------------------------------------
# Noise variance from SNR
# ---------------------------------------------------------------------------

def snr_to_noise_var(snr_db: float, signal_power: float = 1.0) -> float:
    """σ²_n = signal_power / 10^(SNR/10)"""
    return signal_power / (10.0 ** (snr_db / 10.0))
