"""
evaluate.py
Evaluate DSCE model and baselines (LS, MMSE) for MSE and BER vs. SNR.

Splits:
  T1 – seen configurations (same pilot patterns as training)
  T2 – unseen configurations (32 RBs, DMRS type 2)
"""

import os
import numpy as np
import torch
from typing import List, Optional, Dict

from utils import (SystemConfig, mse_db, set_seed,
                   TEST_T1_CONFIGS, TEST_T2_CONFIGS, SNR_TEST_STEPS)
from channel_model import generate_channel, snr_to_noise_var, mmse_estimate
from ofdm_system import (
    build_ofdm_frame, apply_channel_and_noise, ls_estimate,
    ls_interpolate, mmse_equalize, compute_ber
)
from dsce_model import DSCETransformer


# ---------------------------------------------------------------------------
# Single-sample DSCE inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def dsce_estimate(model: DSCETransformer,
                  h_ls: np.ndarray,
                  pilot_f: np.ndarray,
                  pilot_t: np.ndarray,
                  cfg: SystemConfig,
                  device: torch.device) -> np.ndarray:
    """
    Run DSCE on a single channel observation.

    Parameters
    ----------
    h_ls    : (N_p,) complex – LS estimates at pilots
    pilot_f : (N_p,) int
    pilot_t : (N_p,) int
    cfg     : SystemConfig

    Returns
    -------
    H_est : (N_f, N_t) complex – DSCE channel estimate
    """
    N_f, N_t = cfg.n_f, cfg.n_t
    N_total = N_f * N_t

    # Build full-grid index arrays
    all_t_arr = np.repeat(np.arange(N_t), N_f)
    all_f_arr = np.tile(np.arange(N_f), N_t)

    h_ls_ri = np.stack([h_ls.real, h_ls.imag], axis=-1).astype(np.float32)
    h_ls_t  = torch.from_numpy(h_ls_ri).unsqueeze(0).to(device)  # (1, N_p, 2)

    pf = torch.from_numpy(pilot_f.astype(np.int64)).to(device)
    pt = torch.from_numpy(pilot_t.astype(np.int64)).to(device)
    af = torch.from_numpy(all_f_arr.astype(np.int64)).to(device)
    at = torch.from_numpy(all_t_arr.astype(np.int64)).to(device)

    model.eval()
    H_pred_ri = model(h_ls_t, pf, pt, af, at, N_f, N_t, pilot_mask=None)
    # H_pred_ri : (1, N_f*N_t, 2)
    H_pred_ri = H_pred_ri.squeeze(0).cpu().numpy()  # (N_total, 2)

    H_flat = H_pred_ri[:, 0] + 1j * H_pred_ri[:, 1]
    # Reshape: the flat order is t-major (iterate t then f)
    H_est = H_flat.reshape(N_t, N_f).T              # (N_f, N_t)
    return H_est.astype(np.complex64)


# ---------------------------------------------------------------------------
# Evaluate one config at one SNR
# ---------------------------------------------------------------------------

def evaluate_config_snr(cfg: SystemConfig,
                         snr_db: float,
                         n_samples: int,
                         model: Optional[DSCETransformer],
                         device: Optional[torch.device],
                         seed: int = 0) -> dict:
    """
    Evaluate MSE (dB) and BER for LS, MMSE, and DSCE methods.

    Returns
    -------
    dict with keys: "ls_mse", "mmse_mse", "dsce_mse",
                    "ls_ber",  "mmse_ber",  "dsce_ber"
    Each value is the average over n_samples.
    """
    rng = np.random.default_rng(seed)
    N_f, N_t = cfg.n_f, cfg.n_t

    mse_ls, mse_mmse, mse_dsce = [], [], []
    ber_ls, ber_mmse, ber_dsce = [], [], []

    for _ in range(n_samples):
        # ---- Generate channel & frame --------------------------------
        H_true = generate_channel(cfg, n_samples=1, rng=rng)[0]  # (N_f, N_t)
        X_tx, bits, pilot_f, pilot_t, data_f, data_t = build_ofdm_frame(cfg, rng=rng)
        Y_rx = apply_channel_and_noise(X_tx, H_true, snr_db, rng=rng)

        # ---- LS estimate at pilots -----------------------------------
        h_ls = ls_estimate(Y_rx, X_tx, pilot_f, pilot_t)  # (N_p,)

        # ---- LS baseline: 2D interpolation ---------------------------
        H_ls_full = ls_interpolate(h_ls, pilot_f, pilot_t, cfg)  # (N_f, N_t)
        mse_ls.append(mse_db(H_true, H_ls_full))

        X_eq_ls = mmse_equalize(Y_rx, H_ls_full, snr_db)
        ber_ls.append(compute_ber(X_eq_ls, bits, data_f, data_t))

        # ---- MMSE baseline -------------------------------------------
        sigma2_n = snr_to_noise_var(snr_db)
        H_mmse = mmse_estimate(Y_rx[pilot_f, pilot_t],
                               X_tx[pilot_f, pilot_t],
                               H_true,
                               sigma2_n,
                               pilot_f, pilot_t)  # (N_f, N_t)
        mse_mmse.append(mse_db(H_true, H_mmse))

        X_eq_mmse = mmse_equalize(Y_rx, H_mmse, snr_db)
        ber_mmse.append(compute_ber(X_eq_mmse, bits, data_f, data_t))

        # ---- DSCE ----------------------------------------------------
        if model is not None:
            H_dsce = dsce_estimate(model, h_ls, pilot_f, pilot_t, cfg, device)
            mse_dsce.append(mse_db(H_true, H_dsce))
            X_eq_dsce = mmse_equalize(Y_rx, H_dsce, snr_db)
            ber_dsce.append(compute_ber(X_eq_dsce, bits, data_f, data_t))

    result = {
        "ls_mse":   float(np.mean(mse_ls)),
        "mmse_mse": float(np.mean(mse_mmse)),
        "ls_ber":   float(np.mean(ber_ls)),
        "mmse_ber": float(np.mean(ber_mmse)),
    }
    if model is not None:
        result["dsce_mse"] = float(np.mean(mse_dsce))
        result["dsce_ber"] = float(np.mean(ber_dsce))
    return result


# ---------------------------------------------------------------------------
# Full evaluation: MSE / BER vs SNR for a list of configs
# ---------------------------------------------------------------------------

def evaluate_all(configs: List[SystemConfig],
                 snr_list: List[float],
                 n_samples_per_snr: int,
                 model: Optional[DSCETransformer],
                 device: Optional[torch.device],
                 label: str = "T1",
                 seed: int = 100) -> Dict:
    """
    Evaluate all configs and all SNRs.  Returns a nested dict:
        results[config_str][snr] = {ls_mse, mmse_mse, dsce_mse, ls_ber, ...}
    Also returns per-SNR averages:
        avg[snr] = {ls_mse, mmse_mse, dsce_mse, ...}
    """
    all_results = {}
    snr_accumulator = {s: {k: [] for k in ["ls_mse","mmse_mse","dsce_mse",
                                            "ls_ber", "mmse_ber", "dsce_ber"]}
                       for s in snr_list}

    for ci, cfg in enumerate(configs):
        key = str(cfg)
        all_results[key] = {}
        print(f"\n[Eval {label}] Config {ci+1}/{len(configs)}: {cfg}")
        for snr in snr_list:
            res = evaluate_config_snr(cfg, snr, n_samples_per_snr, model, device,
                                      seed=seed + ci * 100 + int(snr))
            all_results[key][snr] = res
            for k, v in res.items():
                if k in snr_accumulator[snr]:
                    snr_accumulator[snr][k].append(v)
            mse_str = (f"LS={res['ls_mse']:.2f}dB  "
                       f"MMSE={res['mmse_mse']:.2f}dB"
                       + (f"  DSCE={res['dsce_mse']:.2f}dB"
                          if "dsce_mse" in res else ""))
            print(f"  SNR={snr:4.0f}dB | {mse_str}")

    # Average across configs per SNR
    avg = {}
    for snr in snr_list:
        avg[snr] = {}
        for k, vals in snr_accumulator[snr].items():
            if vals:
                avg[snr][k] = float(np.mean(vals))

    return {"per_config": all_results, "average": avg}


# ---------------------------------------------------------------------------
# Save / load results
# ---------------------------------------------------------------------------

def save_results(results: dict, path: str):
    np.save(path, results)
    print(f"[Eval] Results saved to {path}")


def load_results(path: str) -> dict:
    return np.load(path, allow_pickle=True).item()


# ---------------------------------------------------------------------------
# Quick sanity check (no model)
# ---------------------------------------------------------------------------

def baseline_quick_check(n_samples: int = 50, seed: int = 0):
    """Print LS and MMSE MSE at a few SNRs for a quick sanity check."""
    cfg = SystemConfig(n_rb=12, dmrs_type=1, n_dmrs_sym=4)
    print(f"\n[Sanity] Config: {cfg}")
    print(f"{'SNR':>6} | {'LS MSE (dB)':>12} | {'MMSE MSE (dB)':>14}")
    print("-" * 40)
    for snr in [0, 10, 20, 30]:
        res = evaluate_config_snr(cfg, snr, n_samples, None, None, seed)
        print(f"{snr:6.0f} | {res['ls_mse']:12.2f} | {res['mmse_mse']:14.2f}")
