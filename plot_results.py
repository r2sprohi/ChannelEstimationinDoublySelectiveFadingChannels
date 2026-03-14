"""
plot_results.py
Reproduces Fig. 4 style MSE (dB) and BER vs. SNR curves.

Usage:
    python plot_results.py --results_t1 results/eval_t1.npy \
                           --results_t2 results/eval_t2.npy \
                           --out_dir    results/figures
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

METHOD_STYLES = {
    "LS":      dict(color="#1f77b4", marker="o", linestyle="--",  linewidth=1.5),
    "MMSE":    dict(color="#ff7f0e", marker="s", linestyle="--",  linewidth=1.5),
    "DSCE":    dict(color="#2ca02c", marker="D", linestyle="-",   linewidth=2.0),
    "Perfect": dict(color="#7f7f7f", marker="",  linestyle=":",   linewidth=1.2),
}


def _extract_curve(results: dict, metric: str, methods: list):
    """
    results : dict with key "average" -> {snr: {method_key: value}}
    metric  : one of "mse" or "ber"
    Returns snr_arr (sorted), and {method: values} dicts
    """
    avg = results["average"]
    snr_arr = sorted(avg.keys())
    curves = {}
    for m in methods:
        key = f"{m.lower()}_{metric}"
        curves[m] = [avg[snr].get(key, np.nan) for snr in snr_arr]
    return np.array(snr_arr), curves


# ---------------------------------------------------------------------------
# MSE plot
# ---------------------------------------------------------------------------

def plot_mse(ax: plt.Axes,
             snr: np.ndarray,
             curves: dict,
             title: str = ""):
    for name, vals in curves.items():
        style = METHOD_STYLES.get(name, {})
        ax.plot(snr, vals, label=name, **style)
    ax.set_xlabel("SNR (dB)", fontsize=11)
    ax.set_ylabel("MSE (dB)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))


# ---------------------------------------------------------------------------
# BER plot (log scale)
# ---------------------------------------------------------------------------

def plot_ber(ax: plt.Axes,
             snr: np.ndarray,
             curves: dict,
             title: str = ""):
    for name, vals in curves.items():
        style = METHOD_STYLES.get(name, {})
        vals_clipped = np.clip(vals, 1e-5, 1.0)
        ax.semilogy(snr, vals_clipped, label=name, **style)
    ax.set_xlabel("SNR (dB)", fontsize=11)
    ax.set_ylabel("BER", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which='both', linestyle=":", alpha=0.6)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_ylim([1e-4, 1.0])


# ---------------------------------------------------------------------------
# Main figure: 2 rows × 2 cols  (T1 MSE | T2 MSE / T1 BER | T2 BER)
# ---------------------------------------------------------------------------

def make_figure_4(results_t1: dict,
                  results_t2: dict,
                  out_dir: str,
                  dmrs_label: str = "4 DMRS syms"):
    """Reproduce Fig. 4 style plot."""
    methods = ["LS", "MMSE"]
    if any("dsce_mse" in v for v in results_t1["average"].values()):
        methods.append("DSCE")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Channel Estimation Performance – {dmrs_label}", fontsize=13)

    # T1 MSE
    snr1, c1 = _extract_curve(results_t1, "mse", methods)
    plot_mse(axes[0, 0], snr1, c1, title="T1 (seen configs) – MSE")

    # T2 MSE
    snr2, c2 = _extract_curve(results_t2, "mse", methods)
    plot_mse(axes[0, 1], snr2, c2, title="T2 (unseen configs) – MSE")

    # T1 BER
    _, cb1 = _extract_curve(results_t1, "ber", methods)
    plot_ber(axes[1, 0], snr1, cb1, title="T1 – BER")

    # T2 BER
    _, cb2 = _extract_curve(results_t2, "ber", methods)
    plot_ber(axes[1, 1], snr2, cb2, title="T2 – BER")

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig4_mse_ber.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Per-SNR MSE and BER curves (single test set)
# ---------------------------------------------------------------------------

def plot_single_set(results: dict,
                    out_dir: str,
                    tag: str = "t1"):
    methods = ["LS", "MMSE"]
    if any("dsce_mse" in v for v in results["average"].values()):
        methods.append("DSCE")

    fig, (ax_mse, ax_ber) = plt.subplots(1, 2, figsize=(11, 4.5))

    snr, c_mse = _extract_curve(results, "mse", methods)
    plot_mse(ax_mse, snr, c_mse, title=f"{tag.upper()} – MSE vs SNR")

    _, c_ber = _extract_curve(results, "ber", methods)
    plot_ber(ax_ber, snr, c_ber, title=f"{tag.upper()} – BER vs SNR")

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"results_{tag}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Training loss curve
# ---------------------------------------------------------------------------

def plot_training_history(history_path: str, out_dir: str):
    history = np.load(history_path, allow_pickle=True).item()
    train_loss = history.get("train_loss", [])
    val_loss   = history.get("val_loss",   [])
    if not train_loss:
        print("[Plot] Empty training history, skipping.")
        return

    epochs = np.arange(1, len(train_loss) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_loss, label="Train L1", color="#1f77b4")
    ax.plot(epochs, val_loss,   label="Val L1",   color="#ff7f0e", linestyle="--")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("L1 Loss", fontsize=11)
    ax.set_title("Training History", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "training_history.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot DSCE evaluation results")
    parser.add_argument("--results_t1",   type=str, default="results/eval_t1.npy")
    parser.add_argument("--results_t2",   type=str, default="results/eval_t2.npy")
    parser.add_argument("--history",      type=str, default="checkpoints/training_history.npy")
    parser.add_argument("--out_dir",      type=str, default="results/figures")
    parser.add_argument("--dmrs_label",   type=str, default="4 DMRS symbols")
    args = parser.parse_args()

    from evaluate import load_results

    if os.path.exists(args.results_t1) and os.path.exists(args.results_t2):
        r1 = load_results(args.results_t1)
        r2 = load_results(args.results_t2)
        make_figure_4(r1, r2, args.out_dir, args.dmrs_label)
        plot_single_set(r1, args.out_dir, "t1")
        plot_single_set(r2, args.out_dir, "t2")
    else:
        print("[Plot] Result files not found. Run main.py first.")

    if os.path.exists(args.history):
        plot_training_history(args.history, args.out_dir)
