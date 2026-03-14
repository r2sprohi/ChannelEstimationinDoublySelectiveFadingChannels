"""
train.py
Training loop for the DSCE Transformer.

Key settings (paper Section III-C):
  - L1 loss (mean absolute error), Eq. (10)
  - Adam optimiser, lr = 1e-3
  - Batch size 128
  - 100 epochs
  - Pilot augmentation: 20% random dropout per training step
  - Training configs: {6,12,18,24} RBs, DMRS type 1, {2,3,4} DMRS symbols
  - SNR: continuous uniform 0–30 dB
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Optional

from utils import SystemConfig, TRAIN_CONFIGS, set_seed
from dsce_model import DSCETransformer, build_dsce_model
from dataset import ChannelDataset


# ---------------------------------------------------------------------------
# L1 loss (Eq. 10 in paper: 1/(N_f N_t) ‖H − Ĥ‖_1)
# ---------------------------------------------------------------------------

def l1_channel_loss(H_pred_ri: torch.Tensor, H_true_ri: torch.Tensor) -> torch.Tensor:
    """
    H_pred_ri, H_true_ri : (B, N_total, 2)
    Returns mean L1 per element (scalar).
    """
    return torch.mean(torch.abs(H_pred_ri - H_true_ri))


# ---------------------------------------------------------------------------
# Pilot augmentation
# ---------------------------------------------------------------------------

def pilot_augmentation(batch_size: int, n_pilots: int,
                        dropout_rate: float = 0.2,
                        device: torch.device = None) -> torch.Tensor:
    """
    Randomly zero out dropout_rate fraction of pilot tokens per sample.

    Returns
    -------
    mask : (B, N_p) bool tensor – True = keep, False = drop
    """
    keep_prob = 1.0 - dropout_rate
    mask = torch.bernoulli(
        torch.full((batch_size, n_pilots), keep_prob)
    ).bool()
    if device is not None:
        mask = mask.to(device)
    return mask


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_epoch(model: DSCETransformer,
                loaders: List[DataLoader],
                datasets: List[ChannelDataset],
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                pilot_dropout: float = 0.2) -> float:
    """
    One training epoch across all config loaders (round-robin batch sampling).

    Returns mean L1 loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    # Interleave batches from all loaders
    loader_iters = [iter(ld) for ld in loaders]
    active = list(range(len(loaders)))

    while active:
        # Shuffle order each round to avoid bias
        np.random.shuffle(active)
        next_active = []
        for ci in active:
            try:
                batch = next(loader_iters[ci])
            except StopIteration:
                continue   # this loader is exhausted
            next_active.append(ci)

            ds = datasets[ci]
            loss = _train_step(model, batch, ds, optimizer, device, pilot_dropout)
            total_loss += loss
            n_batches  += 1

        active = next_active

    return total_loss / max(n_batches, 1)


def _train_step(model: DSCETransformer,
                batch: dict,
                ds: ChannelDataset,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                pilot_dropout: float) -> float:
    """Single gradient step. Returns scalar loss."""
    h_ls_ri   = batch["h_ls_ri"].to(device)    # (B, N_p, 2)
    H_true_ri = batch["H_true_ri"].to(device)  # (B, N_total, 2)
    B = h_ls_ri.shape[0]

    pilot_f = ds.pilot_f.to(device)
    pilot_t = ds.pilot_t.to(device)
    all_f   = ds.all_f.to(device)
    all_t   = ds.all_t.to(device)
    n_f, n_t = ds.n_f, ds.n_t

    # Pilot augmentation mask
    mask = pilot_augmentation(B, ds.pilot_f.shape[0], pilot_dropout, device)

    optimizer.zero_grad()
    H_pred_ri = model(
        h_ls_ri, pilot_f, pilot_t, all_f, all_t,
        n_f, n_t, pilot_mask=mask
    )
    loss = l1_channel_loss(H_pred_ri, H_true_ri)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model: DSCETransformer,
             loaders: List[DataLoader],
             datasets: List[ChannelDataset],
             device: torch.device) -> float:
    """Returns mean validation L1 loss across all config loaders."""
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    for ci, ld in enumerate(loaders):
        ds = datasets[ci]
        for batch in ld:
            h_ls_ri   = batch["h_ls_ri"].to(device)
            H_true_ri = batch["H_true_ri"].to(device)

            H_pred_ri = model(
                h_ls_ri,
                ds.pilot_f.to(device),
                ds.pilot_t.to(device),
                ds.all_f.to(device),
                ds.all_t.to(device),
                ds.n_f, ds.n_t,
                pilot_mask=None,
            )
            total_loss += l1_channel_loss(H_pred_ri, H_true_ri).item()
            n_batches  += 1
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(n_train_total: int = 150_000,
          n_val_total:   int = 10_000,
          batch_size:    int = 128,
          n_epochs:      int = 100,
          lr:            float = 1e-3,
          pilot_dropout: float = 0.20,
          d_model:       int = 32,
          n_encoder_blocks: int = 3,
          n_decoder_blocks: int = 1,
          n_heads:       int = 4,
          save_dir:      str = "checkpoints",
          device_str:    str = "auto",
          seed:          int = 42,
          configs:       Optional[List[SystemConfig]] = None) -> DSCETransformer:
    """
    Full training pipeline.

    Parameters
    ----------
    n_train_total : total training samples (distributed across all configs)
    n_val_total   : total validation samples
    configs       : list of SystemConfig to train on (default = TRAIN_CONFIGS)
    """
    set_seed(seed)

    if device_str == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
    else:
        device = torch.device(device_str)
    print(f"[Train] Using device: {device}")

    if configs is None:
        configs = TRAIN_CONFIGS

    os.makedirs(save_dir, exist_ok=True)

    # ---- Build per-config datasets -----------------------------------
    print(f"\n[Train] Generating {n_train_total} training samples "
          f"across {len(configs)} configs…")
    n_per_cfg_train = n_train_total // len(configs)
    n_per_cfg_val   = max(n_val_total // len(configs), 10)

    train_datasets, train_loaders = [], []
    val_datasets,   val_loaders   = [], []

    for ci, cfg in enumerate(configs):
        print(f"  [{ci+1}/{len(configs)}] {cfg}")
        tr_ds = ChannelDataset(cfg, n_per_cfg_train,
                               snr_mode="continuous",
                               snr_range=(0.0, 30.0),
                               seed=seed + ci)
        va_ds = ChannelDataset(cfg, n_per_cfg_val,
                               snr_mode="continuous",
                               snr_range=(0.0, 30.0),
                               seed=seed + 1000 + ci)
        train_datasets.append(tr_ds)
        val_datasets.append(va_ds)
        train_loaders.append(DataLoader(tr_ds, batch_size=batch_size,
                                        shuffle=True, num_workers=0,
                                        pin_memory=(device.type == "cuda")))
        val_loaders.append(DataLoader(va_ds, batch_size=batch_size,
                                      shuffle=False, num_workers=0,
                                      pin_memory=(device.type == "cuda")))

    # ---- Model, optimiser, scheduler ---------------------------------
    model = build_dsce_model(d_model, n_encoder_blocks, n_decoder_blocks, n_heads)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    # ---- Training loop -----------------------------------------------
    best_val_loss = float('inf')
    best_ckpt = os.path.join(save_dir, "dsce_best.pt")
    history = {"train_loss": [], "val_loss": []}

    print(f"\n[Train] Starting training for {n_epochs} epochs\n" + "="*60)
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        tr_loss = train_epoch(model, train_loaders, train_datasets,
                              optimizer, device, pilot_dropout)
        va_loss = validate(model, val_loaders, val_datasets, device)

        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{n_epochs} | "
              f"train_L1={tr_loss:.4f}  val_L1={va_loss:.4f} | "
              f"{elapsed:.1f}s")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": va_loss,
                "config": {
                    "d_model": d_model,
                    "n_encoder_blocks": n_encoder_blocks,
                    "n_decoder_blocks": n_decoder_blocks,
                    "n_heads": n_heads,
                },
            }, best_ckpt)
            print(f"  *** New best val_L1={va_loss:.4f} → saved to {best_ckpt}")

    # ---- Save final model and history --------------------------------
    final_ckpt = os.path.join(save_dir, "dsce_final.pt")
    torch.save(model.state_dict(), final_ckpt)
    np.save(os.path.join(save_dir, "training_history.npy"), history)
    print(f"\n[Train] Training complete. Best val_L1={best_val_loss:.4f}")
    print(f"  Best model: {best_ckpt}")

    # Reload best weights
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model


# ---------------------------------------------------------------------------
# Checkpoint loading utility
# ---------------------------------------------------------------------------

def load_checkpoint(path: str,
                    device_str: str = "auto") -> tuple:
    """
    Load a saved DSCE model.

    Returns
    -------
    model  : DSCETransformer
    device : torch.device
    """
    if device_str == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(device_str)

    ckpt = torch.load(path, map_location=device)
    cfg  = ckpt.get("config", {})
    model = build_dsce_model(
        d_model          = cfg.get("d_model", 32),
        n_encoder_blocks = cfg.get("n_encoder_blocks", 3),
        n_decoder_blocks = cfg.get("n_decoder_blocks", 1),
        n_heads          = cfg.get("n_heads", 4),
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    print(f"[Load] Loaded model from {path} "
          f"(epoch {ckpt.get('epoch','?')}, "
          f"val_L1={ckpt.get('val_loss',float('nan')):.4f})")
    return model, device
