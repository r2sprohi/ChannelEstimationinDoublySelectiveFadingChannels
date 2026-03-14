"""
main.py
Full end-to-end pipeline for the DSCE Transformer paper
(Shi et al., ICC 2025 – "Doubly-Selective Channel Estimation Transformer
Network with Varying Pilot Pattern").

Steps:
  1. (Optional) Quick sanity-check of baselines only
  2. Train the DSCE model
  3. Evaluate on T1 (seen) and T2 (unseen) test configs
  4. Plot MSE / BER curves

Usage examples:
  # Full pipeline
  python main.py

  # Baseline-only quick check (no model training)
  python main.py --mode baseline

  # Training only
  python main.py --mode train

  # Evaluate with existing checkpoint
  python main.py --mode eval --checkpoint checkpoints/dsce_best.pt

  # Small-scale debug run
  python main.py --quick

Run `python main.py --help` for all options.
"""

import argparse
import os

import numpy as np
import torch

from utils import (
    SystemConfig, TRAIN_CONFIGS, TEST_T1_CONFIGS, TEST_T2_CONFIGS,
    SNR_TEST_STEPS, set_seed
)
from train import train, load_checkpoint
from evaluate import (
    evaluate_all, baseline_quick_check, save_results, load_results
)
from plot_results import (
    make_figure_4, plot_single_set, plot_training_history
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DSCE Transformer Pipeline")
    p.add_argument("--mode", choices=["full", "train", "eval", "baseline"],
                   default="full",
                   help="Pipeline mode (default: full)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to existing checkpoint for eval mode")
    p.add_argument("--quick", action="store_true",
                   help="Debug run with small dataset / few epochs")
    p.add_argument("--n_train", type=int, default=150_000,
                   help="Total training samples (default: 150k)")
    p.add_argument("--n_test",  type=int, default=30_000,
                   help="Total test samples per set (default: 30k)")
    p.add_argument("--epochs",  type=int, default=100)
    p.add_argument("--batch",   type=int, default=128)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--enc",     type=int, default=3,
                   help="Encoder transformer blocks (default: 3)")
    p.add_argument("--dec",     type=int, default=1,
                   help="Decoder transformer blocks (default: 1)")
    p.add_argument("--heads",   type=int, default=4)
    p.add_argument("--save_dir",  type=str, default="checkpoints")
    p.add_argument("--out_dir",   type=str, default="results")
    p.add_argument("--device",    type=str, default="auto",
                   help="Device: auto / cpu / cuda / mps")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Quick / debug overrides
# ---------------------------------------------------------------------------

QUICK_OVERRIDES = dict(
    n_train=2_000,
    n_test=500,
    epochs=3,
    batch=32,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.quick:
        print("[main] *** QUICK / DEBUG mode – small dataset ***")
        for k, v in QUICK_OVERRIDES.items():
            setattr(args, k, v)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir,  exist_ok=True)

    history_path = os.path.join(args.save_dir, "training_history.npy")
    t1_path      = os.path.join(args.out_dir,  "eval_t1.npy")
    t2_path      = os.path.join(args.out_dir,  "eval_t2.npy")

    # ------------------------------------------------------------------
    # 1. Baseline sanity check
    # ------------------------------------------------------------------
    if args.mode in ("full", "baseline"):
        print("\n" + "="*60)
        print("STEP 0: Baseline sanity check (LS / MMSE)")
        print("="*60)
        baseline_quick_check(n_samples=100, seed=args.seed)

    if args.mode == "baseline":
        return

    # ------------------------------------------------------------------
    # 2. Training
    # ------------------------------------------------------------------
    model = None

    if args.mode in ("full", "train"):
        print("\n" + "="*60)
        print("STEP 1: Training DSCE Transformer")
        print("="*60)
        print(f"  n_train   = {args.n_train:,}")
        print(f"  epochs    = {args.epochs}")
        print(f"  batch     = {args.batch}")
        print(f"  d_model   = {args.d_model}, enc={args.enc}, "
              f"dec={args.dec}, heads={args.heads}")
        print(f"  device    = {args.device}")

        model = train(
            n_train_total    = args.n_train,
            n_val_total      = max(args.n_train // 10, 500),
            batch_size       = args.batch,
            n_epochs         = args.epochs,
            lr               = args.lr,
            pilot_dropout    = 0.20,
            d_model          = args.d_model,
            n_encoder_blocks = args.enc,
            n_decoder_blocks = args.dec,
            n_heads          = args.heads,
            save_dir         = args.save_dir,
            device_str       = args.device,
            seed             = args.seed,
        )

    # ------------------------------------------------------------------
    # 3. Load checkpoint for eval-only mode
    # ------------------------------------------------------------------
    device = get_device(args.device)

    if args.mode == "eval":
        if args.checkpoint is None:
            args.checkpoint = os.path.join(args.save_dir, "dsce_best.pt")
        print(f"\n[main] Loading checkpoint: {args.checkpoint}")
        model, device = load_checkpoint(args.checkpoint, args.device)

    # ------------------------------------------------------------------
    # 4. Evaluation
    # ------------------------------------------------------------------
    if args.mode in ("full", "train", "eval"):
        n_samp = max(args.n_test // (len(SNR_TEST_STEPS) * len(TEST_T1_CONFIGS)), 5)
        print("\n" + "="*60)
        print(f"STEP 2: Evaluating  ({n_samp} samples/SNR/config)")
        print("="*60)

        # T1 – seen configs
        print("\n--- T1 (seen configs) ---")
        res_t1 = evaluate_all(
            configs=TEST_T1_CONFIGS,
            snr_list=SNR_TEST_STEPS,
            n_samples_per_snr=n_samp,
            model=model,
            device=device,
            label="T1",
            seed=args.seed + 999,
        )
        save_results(res_t1, t1_path)

        # T2 – unseen configs
        print("\n--- T2 (unseen configs) ---")
        res_t2 = evaluate_all(
            configs=TEST_T2_CONFIGS[:6],  # limit for speed (first 6 unseen)
            snr_list=SNR_TEST_STEPS,
            n_samples_per_snr=n_samp,
            model=model,
            device=device,
            label="T2",
            seed=args.seed + 1999,
        )
        save_results(res_t2, t2_path)

        # ------------------------------------------------------------------
        # 5. Plotting
        # ------------------------------------------------------------------
        print("\n" + "="*60)
        print("STEP 3: Plotting results")
        print("="*60)
        fig_dir = os.path.join(args.out_dir, "figures")

        make_figure_4(res_t1, res_t2, fig_dir)
        plot_single_set(res_t1, fig_dir, "t1")
        plot_single_set(res_t2, fig_dir, "t2")

        if os.path.exists(history_path):
            plot_training_history(history_path, fig_dir)

        # ------------------------------------------------------------------
        # 6. Summary table
        # ------------------------------------------------------------------
        print("\n" + "="*60)
        print("SUMMARY – Average MSE (dB) and BER vs SNR")
        print("="*60)
        for split, res in [("T1", res_t1), ("T2", res_t2)]:
            print(f"\n{split}:")
            avg = res["average"]
            hdr = f"{'SNR':>5} | {'LS MSE':>9} | {'MMSE MSE':>10}"
            if any("dsce_mse" in v for v in avg.values()):
                hdr += f" | {'DSCE MSE':>10}"
            print(hdr)
            print("-" * len(hdr))
            for snr in sorted(avg.keys()):
                v = avg[snr]
                row = (f"{snr:5.0f} | {v.get('ls_mse',np.nan):9.2f} | "
                       f"{v.get('mmse_mse',np.nan):10.2f}")
                if "dsce_mse" in v:
                    row += f" | {v['dsce_mse']:10.2f}"
                print(row)

    print("\n[main] Done.")


if __name__ == "__main__":
    main()
