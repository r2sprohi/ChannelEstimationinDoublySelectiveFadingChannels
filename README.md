# Doubly-Selective Channel Estimation Transformer (DSCE)

PyTorch implementation of the paper:

> **Doubly-Selective Channel Estimation Transformer Network with Varying Pilot Pattern**
> Wanting Shi, Bin Zhang, Yili Xia, Wenjiang Pei — IEEE ICC 2025

---

## Overview

This repo contains two channel estimation implementations:

| Directory / File | Description |
|---|---|
| `main.py` | DSCE Transformer pipeline (this work) |
| `DNN.py` | Legacy FCN baseline (TensorFlow, 802.11p) |
| `*.m` | MATLAB simulation scripts (legacy baseline) |

### DSCE Transformer

The DSCE network treats channel estimation as a **masked autoencoding** task:
- **Pilot symbols** = unmasked tokens (encoder input)
- **Data symbols** = masked tokens (decoder output)

Key contributions implemented:
- Encoder–decoder Transformer (3 + 1 blocks, d = 32)
- **Relative Positional Encoding (RPE)** — 2-D sine-encoded lookup table, makes the model invariant to pilot translation
- **Pilot augmentation** — 20 % random pilot dropout per training step for generalisation
- Cross-attention decoder (memory-efficient: O(N_total × N_p) vs O(N_total²))

---

## System Parameters (5G NR, paper Table I)

| Parameter | Value |
|---|---|
| Carrier frequency | 2.5 GHz |
| Subcarrier spacing | 30 kHz |
| OFDM symbols per slot | 14 |
| Channel model | CDL-E (TDL approx., 15 taps) |
| Max delay spread | 2510 ns |
| Max receiver speed | 120 km/h |
| Modulation | 256-QAM |
| Training RBs | {6, 12, 18, 24} |
| Test RBs | {6, 12, 18, 24} (T1) + {32} (T2) |
| DMRS type | 1 (train), {1, 2} (test) |
| DMRS symbols | {2, 3, 4} |

---

## Installation

```bash
pip install torch numpy scipy matplotlib
```

---

## Usage

```bash
# Quick debug run (~5 min, 2k samples, 3 epochs)
python main.py --quick

# Full training (150k samples, 100 epochs) — ~4–8 GPU hours on RTX 3090
python main.py

# Evaluate a saved checkpoint
python main.py --mode eval --checkpoint checkpoints/dsce_best.pt

# Baseline only (LS / MMSE, no model training)
python main.py --mode baseline

# Reproduce plots from saved results
python plot_results.py
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--mode` | `full` | `full` / `train` / `eval` / `baseline` |
| `--quick` | off | Debug run (2k samples, 3 epochs) |
| `--n_train` | 150000 | Training samples |
| `--n_test` | 30000 | Test samples |
| `--epochs` | 100 | Training epochs |
| `--batch` | 128 | Batch size (use 32 on MPS) |
| `--d_model` | 32 | Transformer feature dim |
| `--enc` | 3 | Encoder blocks |
| `--dec` | 1 | Decoder blocks |
| `--checkpoint` | — | Path to `.pt` file for eval |

---

## File Structure

```
├── main.py            # Pipeline entry point
├── utils.py           # SystemConfig, sine encoding, config lists
├── channel_model.py   # CDL-E TDL channel + MMSE baseline
├── ofdm_system.py     # 256-QAM, DMRS patterns, LS/MMSE equalizer, BER
├── dsce_model.py      # DSCE Transformer (encoder + cross-attn decoder + RPE)
├── dataset.py         # Dataset generation and DataLoader
├── train.py           # Training loop (L1 loss, Adam, pilot augmentation)
├── evaluate.py        # MSE (dB) / BER evaluation, T1/T2 split
├── plot_results.py    # Fig. 4-style MSE/BER curves
└── DNN.py             # Legacy TensorFlow FCN baseline (kept for reference)
```

Outputs are written to:
```
checkpoints/dsce_best.pt     # best model weights
results/figures/             # MSE/BER plots
```

---

## Expected Performance (paper Table III)

| Method | T1 MSE @ 25 dB | T2 MSE @ 25 dB |
|---|---|---|
| LS | ~−5 dB | ~−5 dB |
| MMSE | ~−23 dB | ~−23 dB |
| **DSCE (ours)** | **~−26 dB** | **~−26 dB** |
