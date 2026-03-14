"""
dataset.py
Dataset generation and PyTorch Dataset / DataLoader wrappers.

Each sample contains:
  - h_ls_ri    : (N_p, 2) float32 – [Re, Im] of LS estimates at pilots
  - H_true_ri  : (N_f*N_t, 2) float32 – [Re, Im] of true channel (label)
  - pilot_f    : (N_p,) int64  – pilot subcarrier indices
  - pilot_t    : (N_p,) int64  – pilot symbol indices
  - all_f      : (N_f*N_t,) int64
  - all_t      : (N_f*N_t,) int64
  - n_f        : int
  - n_t        : int
  - snr_db     : float

Because N_p and N_f vary across configurations, samples from the SAME config
are batched together (collate_fn handles per-config batching).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional

from utils import SystemConfig, TRAIN_CONFIGS, SNR_TRAIN_RANGE, SNR_TEST_STEPS
from channel_model import generate_channel, snr_to_noise_var
from ofdm_system import (
    build_ofdm_frame, apply_channel_and_noise, ls_estimate
)


# ---------------------------------------------------------------------------
# Core sample generator
# ---------------------------------------------------------------------------

def generate_sample(cfg: SystemConfig,
                    snr_db: float,
                    rng: np.random.Generator) -> dict:
    """
    Generate a single OFDM channel estimation sample.

    Returns a dict with numpy arrays (float32 / int64).
    """
    # Channel realisation: (1, N_f, N_t)
    H = generate_channel(cfg, n_samples=1, rng=rng)[0]  # (N_f, N_t)

    # Transmit frame
    X_tx, bits, pilot_f, pilot_t, data_f, data_t = build_ofdm_frame(cfg, rng=rng)

    # Noisy received signal
    Y_rx = apply_channel_and_noise(X_tx, H, snr_db, rng=rng)

    # LS estimate at pilots
    h_ls = ls_estimate(Y_rx, X_tx, pilot_f, pilot_t)  # (N_p,) complex

    # Full grid index arrays (row-major: iterate t then f)
    N_f, N_t = cfg.n_f, cfg.n_t
    all_t_arr = np.repeat(np.arange(N_t), N_f)   # [0,0,...,0, 1,1,...,1, ..., 13,...]
    all_f_arr = np.tile(np.arange(N_f), N_t)     # [0,1,...,N_f-1, 0,1,...] repeating

    H_flat = H[all_f_arr, all_t_arr]              # (N_f*N_t,) complex, row-major by t

    return {
        "h_ls_ri":   np.stack([h_ls.real, h_ls.imag], axis=-1).astype(np.float32),
        "H_true_ri": np.stack([H_flat.real, H_flat.imag], axis=-1).astype(np.float32),
        "pilot_f":   pilot_f.astype(np.int64),
        "pilot_t":   pilot_t.astype(np.int64),
        "all_f":     all_f_arr.astype(np.int64),
        "all_t":     all_t_arr.astype(np.int64),
        "n_f":       np.int64(N_f),
        "n_t":       np.int64(N_t),
        "snr_db":    np.float32(snr_db),
        # keep references for BER evaluation
        "H_complex": H.astype(np.complex64),
        "Y_rx":      Y_rx.astype(np.complex64),
        "X_tx":      X_tx.astype(np.complex64),
        "bits_tx":   bits,
        "data_f":    data_f.astype(np.int64),
        "data_t":    data_t.astype(np.int64),
    }


# ---------------------------------------------------------------------------
# Bulk dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(configs: List[SystemConfig],
                     n_total: int,
                     snr_mode: str = "continuous",
                     snr_range: tuple = (0.0, 30.0),
                     snr_fixed_list: Optional[List[float]] = None,
                     seed: int = 0,
                     verbose: bool = True) -> List[dict]:
    """
    Generate n_total samples distributed uniformly across configs.

    snr_mode:
        "continuous" – sample SNR uniformly from snr_range (for training)
        "fixed"      – round-robin from snr_fixed_list (for testing)
    """
    rng = np.random.default_rng(seed)
    n_configs = len(configs)
    samples_per_cfg = n_total // n_configs
    remainder = n_total - samples_per_cfg * n_configs

    samples = []
    for ci, cfg in enumerate(configs):
        n_this = samples_per_cfg + (1 if ci < remainder else 0)
        for si in range(n_this):
            if snr_mode == "continuous":
                snr = float(rng.uniform(*snr_range))
            else:
                snr = float(snr_fixed_list[si % len(snr_fixed_list)])
            samples.append(generate_sample(cfg, snr, rng))
        if verbose:
            print(f"  Config {ci+1}/{n_configs}: {cfg} – {n_this} samples")

    rng.shuffle(np.arange(len(samples)))   # shuffle order (no-op for list, done below)
    order = rng.permutation(len(samples))
    samples = [samples[i] for i in order]
    return samples


# ---------------------------------------------------------------------------
# PyTorch Dataset (per-config, so all items have same N_p and N_total)
# ---------------------------------------------------------------------------

class ChannelDataset(Dataset):
    """
    Dataset for a SINGLE SystemConfig so that tensors can be stacked in batch.
    """

    def __init__(self,
                 cfg: SystemConfig,
                 n_samples: int,
                 snr_mode: str = "continuous",
                 snr_range: tuple = (0.0, 30.0),
                 snr_fixed_list: Optional[List[float]] = None,
                 seed: int = 42):
        self.cfg = cfg
        self.samples = generate_dataset(
            [cfg], n_samples,
            snr_mode=snr_mode,
            snr_range=snr_range,
            snr_fixed_list=snr_fixed_list,
            seed=seed,
            verbose=False,
        )
        # Cache fixed index arrays (same for all samples of this config)
        s0 = self.samples[0]
        self.pilot_f = torch.from_numpy(s0["pilot_f"])
        self.pilot_t = torch.from_numpy(s0["pilot_t"])
        self.all_f   = torch.from_numpy(s0["all_f"])
        self.all_t   = torch.from_numpy(s0["all_t"])
        self.n_f     = int(s0["n_f"])
        self.n_t     = int(s0["n_t"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "h_ls_ri":   torch.from_numpy(s["h_ls_ri"]),    # (N_p, 2)
            "H_true_ri": torch.from_numpy(s["H_true_ri"]),  # (N_f*N_t, 2)
            "snr_db":    torch.tensor(s["snr_db"]),
        }


def get_dataloader(cfg: SystemConfig,
                   n_samples: int,
                   batch_size: int = 128,
                   snr_mode: str = "continuous",
                   snr_range: tuple = (0.0, 30.0),
                   snr_fixed_list=None,
                   seed: int = 42,
                   shuffle: bool = True,
                   num_workers: int = 0) -> tuple:
    """
    Returns (DataLoader, ChannelDataset).
    The DataLoader yields dicts of batched tensors.
    """
    ds = ChannelDataset(cfg, n_samples,
                        snr_mode=snr_mode,
                        snr_range=snr_range,
                        snr_fixed_list=snr_fixed_list,
                        seed=seed)
    dl = DataLoader(ds, batch_size=batch_size,
                    shuffle=shuffle, num_workers=num_workers,
                    pin_memory=True)
    return dl, ds


# ---------------------------------------------------------------------------
# Mixed-config DataLoader (for training across all configs in one pass)
# ---------------------------------------------------------------------------

class MixedConfigDataset(Dataset):
    """
    Wraps multiple ChannelDatasets for different configs.
    All samples must be from configs with the SAME N_p (pilot count) and
    N_f, N_t grid size for batching to work.

    For mixed N_p / N_f, use separate DataLoaders and alternate batches
    (see train.py).
    """

    def __init__(self, datasets: List[ChannelDataset]):
        self.datasets = datasets
        self.lengths  = [len(d) for d in datasets]
        self.cumlen   = np.cumsum([0] + self.lengths)

    def __len__(self):
        return int(self.cumlen[-1])

    def __getitem__(self, idx):
        # Find which sub-dataset
        di = np.searchsorted(self.cumlen[1:], idx, side='right')
        local_idx = idx - self.cumlen[di]
        return self.datasets[di][local_idx]
