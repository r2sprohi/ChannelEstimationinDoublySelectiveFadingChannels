"""
ofdm_system.py
OFDM signal processing: 256-QAM modulation, pilot insertion, LS estimation,
MMSE equalizer, and BER computation.
"""

import numpy as np
from utils import SystemConfig
from channel_model import snr_to_noise_var


# ---------------------------------------------------------------------------
# 256-QAM modulation / demodulation
# ---------------------------------------------------------------------------

class QAM256:
    """Gray-coded 256-QAM (16 × 16 constellation)."""

    BITS_PER_SYM = 8

    def __init__(self):
        # Build 16-PAM alphabet: {±1, ±3, ..., ±15} (unit spacing)
        self._levels = np.array([-15, -13, -11, -9, -7, -5, -3, -1,
                                   1,   3,   5,  7,  9, 11, 13, 15],
                                dtype=np.float64)
        # Gray code for 16-PAM (4 bits)
        self._gray4 = self._build_gray(4)
        # Normalisation factor so average power = 1
        avg_pow = np.mean(self._levels ** 2) * 2   # I + Q
        self._scale = np.sqrt(avg_pow)

    @staticmethod
    def _build_gray(n_bits: int) -> np.ndarray:
        """Build Gray code mapping: index → gray code integer."""
        size = 2 ** n_bits
        gray = np.arange(size, dtype=np.int32) ^ (np.arange(size, dtype=np.int32) >> 1)
        # Invert: gray code value → index
        inv = np.zeros(size, dtype=np.int32)
        for i, g in enumerate(gray):
            inv[g] = i
        return inv

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        bits : (N * 8,) uint8  with values {0, 1}
        Returns complex (N,) symbols with E[|x|²] = 1
        """
        N = len(bits) // self.BITS_PER_SYM
        bits = bits[:N * self.BITS_PER_SYM].reshape(N, self.BITS_PER_SYM)

        # MSB 4 bits → I, LSB 4 bits → Q
        i_bits = bits[:, :4]
        q_bits = bits[:, 4:]

        # packbits pads 4-bit rows on the RIGHT with zeros → high nibble holds data
        # e.g. bits [0,0,1,0] → byte 0b00100000 = 32; we need 32>>4 = 2
        i_idx = np.packbits(i_bits, axis=1, bitorder='big').ravel().astype(np.int32) >> 4
        q_idx = np.packbits(q_bits, axis=1, bitorder='big').ravel().astype(np.int32) >> 4

        # Gray decode → PAM level index → level value
        i_syms = self._levels[self._gray4[i_idx]].astype(np.float64)
        q_syms = self._levels[self._gray4[q_idx]].astype(np.float64)

        return (i_syms + 1j * q_syms) / self._scale

    def demodulate(self, syms: np.ndarray) -> np.ndarray:
        """
        syms  : (N,) complex (noisy, received)
        Returns (N * 8,) uint8 hard-decided bits
        """
        syms_scaled = syms * self._scale

        def _decide_pam(vals):
            return np.argmin(
                np.abs(vals[:, None] - self._levels[None, :]), axis=1
            ).astype(np.int32)

        i_idx = _decide_pam(syms_scaled.real)   # PAM level index n  (0-15)
        q_idx = _decide_pam(syms_scaled.imag)

        # Gray-encode n → g = n XOR (n>>1), which is the 4-bit input used at Tx.
        # Then unpack g into 4 bits via high-nibble approach (matches modulate).
        gray_i = ((i_idx ^ (i_idx >> 1)) << 4).astype(np.uint8)  # shift to high nibble
        gray_q = ((q_idx ^ (q_idx >> 1)) << 4).astype(np.uint8)

        i_bits = np.unpackbits(gray_i[:, None], axis=1, bitorder='big')[:, :4]
        q_bits = np.unpackbits(gray_q[:, None], axis=1, bitorder='big')[:, :4]

        return np.hstack([i_bits, q_bits]).ravel().astype(np.uint8)

    @property
    def avg_power(self) -> float:
        return 1.0   # by design


_qam256 = QAM256()


# ---------------------------------------------------------------------------
# OFDM frame building
# ---------------------------------------------------------------------------

def build_ofdm_frame(cfg: SystemConfig,
                     rng: np.random.Generator = None
                     ) -> tuple:
    """
    Build a full N_f × N_t OFDM resource grid.

    Returns
    -------
    X_tx   : (N_f, N_t) complex – transmitted symbols (data + pilots)
    bits   : (N_data * 8,) uint8 – transmitted data bits
    pilot_f, pilot_t : (N_p,) int – pilot subcarrier / symbol indices
    data_f,  data_t  : (N_d,) int – data subcarrier / symbol indices
    """
    if rng is None:
        rng = np.random.default_rng()

    N_f, N_t = cfg.n_f, cfg.n_t
    pilot_f, pilot_t = cfg.pilot_positions()        # (N_p,)
    pilot_set = set(zip(pilot_f.tolist(), pilot_t.tolist()))

    # Build data position lists
    data_f, data_t = [], []
    for t in range(N_t):
        for f in range(N_f):
            if (f, t) not in pilot_set:
                data_f.append(f)
                data_t.append(t)
    data_f = np.array(data_f, dtype=np.int32)
    data_t = np.array(data_t, dtype=np.int32)
    N_data = len(data_f)

    # Generate random bits and modulate
    bits = rng.integers(0, 2, N_data * QAM256.BITS_PER_SYM, dtype=np.uint8)
    data_syms = _qam256.modulate(bits)

    # Pilot symbols: known BPSK-like {+1} for simplicity (unit power)
    pilot_syms = np.ones(cfg.n_pilots, dtype=np.complex128)

    # Assemble grid
    X_tx = np.zeros((N_f, N_t), dtype=np.complex128)
    X_tx[pilot_f, pilot_t] = pilot_syms
    X_tx[data_f,  data_t]  = data_syms

    return X_tx, bits, pilot_f, pilot_t, data_f, data_t


# ---------------------------------------------------------------------------
# Channel application + noise
# ---------------------------------------------------------------------------

def apply_channel_and_noise(X_tx: np.ndarray,
                             H: np.ndarray,
                             snr_db: float,
                             rng: np.random.Generator = None
                             ) -> np.ndarray:
    """
    Y = H ⊙ X + N  (element-wise Hadamard product, frequency domain)

    Parameters
    ----------
    X_tx   : (N_f, N_t) transmitted grid
    H      : (N_f, N_t) channel frequency response
    snr_db : SNR in dB (signal power normalised to 1)

    Returns
    -------
    Y_rx : (N_f, N_t) received signal
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma2_n = snr_to_noise_var(snr_db, signal_power=1.0)
    noise_std = np.sqrt(sigma2_n / 2.0)

    noise = (rng.standard_normal(X_tx.shape) +
             1j * rng.standard_normal(X_tx.shape)) * noise_std

    return H * X_tx + noise


# ---------------------------------------------------------------------------
# LS channel estimation at pilot positions
# ---------------------------------------------------------------------------

def ls_estimate(Y_rx: np.ndarray,
                X_tx: np.ndarray,
                pilot_f: np.ndarray,
                pilot_t: np.ndarray
                ) -> np.ndarray:
    """
    Least-squares channel estimate at pilot positions.

    h_LS[p] = Y[f_p, t_p] / X[f_p, t_p]

    Returns
    -------
    h_ls : (N_p,) complex
    """
    y_p = Y_rx[pilot_f, pilot_t]   # (N_p,)
    x_p = X_tx[pilot_f, pilot_t]   # (N_p,)
    return y_p / x_p


# ---------------------------------------------------------------------------
# LS full-grid estimate (2D interpolation baseline)
# ---------------------------------------------------------------------------

def ls_interpolate(h_ls: np.ndarray,
                   pilot_f: np.ndarray,
                   pilot_t: np.ndarray,
                   cfg: SystemConfig
                   ) -> np.ndarray:
    """
    Simple 2D linear interpolation of LS estimates to the full grid.
    Used as the LS baseline for comparison.

    Returns
    -------
    H_ls : (N_f, N_t) complex
    """
    N_f, N_t = cfg.n_f, cfg.n_t
    from scipy.interpolate import griddata

    all_f = np.arange(N_f)
    all_t = np.arange(N_t)
    F, T = np.meshgrid(all_f, all_t, indexing='ij')  # (N_f, N_t)

    points = np.stack([pilot_f, pilot_t], axis=1)     # (N_p, 2)

    H_real = griddata(points, h_ls.real, (F, T),
                      method='linear', fill_value=0.0)
    H_imag = griddata(points, h_ls.imag, (F, T),
                      method='linear', fill_value=0.0)

    # Nearest-neighbour fill for boundary extrapolation
    H_real_nn = griddata(points, h_ls.real, (F, T), method='nearest')
    H_imag_nn = griddata(points, h_ls.imag, (F, T), method='nearest')

    mask = np.isnan(H_real)
    H_real[mask] = H_real_nn[mask]
    H_imag[mask] = H_imag_nn[mask]

    return H_real + 1j * H_imag


# ---------------------------------------------------------------------------
# MMSE equalizer (per-subcarrier scalar, frequency domain)
# ---------------------------------------------------------------------------

def mmse_equalize(Y_rx: np.ndarray,
                  H_est: np.ndarray,
                  snr_db: float
                  ) -> np.ndarray:
    """
    Per-subcarrier scalar MMSE equalizer.

    x̂[k,n] = H*[k,n] / (|H[k,n]|² + σ²) * Y[k,n]

    Parameters
    ----------
    Y_rx   : (N_f, N_t)
    H_est  : (N_f, N_t) estimated channel
    snr_db : SNR (used to compute σ²)

    Returns
    -------
    X_eq : (N_f, N_t) equalized symbols
    """
    sigma2_n = snr_to_noise_var(snr_db, signal_power=1.0)
    denom = np.abs(H_est) ** 2 + sigma2_n
    return (np.conj(H_est) / denom) * Y_rx


# ---------------------------------------------------------------------------
# BER computation
# ---------------------------------------------------------------------------

def compute_ber(X_eq: np.ndarray,
                bits_tx: np.ndarray,
                data_f: np.ndarray,
                data_t: np.ndarray
                ) -> float:
    """
    Compute BER by hard-demapping equalized 256-QAM symbols.

    Parameters
    ----------
    X_eq     : (N_f, N_t) equalized received grid
    bits_tx  : (N_data * 8,) transmitted bits
    data_f   : (N_data,) data subcarrier indices
    data_t   : (N_data,) data symbol indices

    Returns
    -------
    ber : float in [0, 1]
    """
    syms_eq = X_eq[data_f, data_t]          # (N_data,)
    bits_rx = _qam256.demodulate(syms_eq)   # (N_data * 8,)

    n_bits = min(len(bits_tx), len(bits_rx))
    n_errors = np.sum(bits_tx[:n_bits] != bits_rx[:n_bits])
    return float(n_errors) / n_bits if n_bits > 0 else 0.0
