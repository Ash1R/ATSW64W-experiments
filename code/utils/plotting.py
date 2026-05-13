"""Plotting helpers for training curves and prediction samples."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # safe default for Colab/headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_training_curve(log_path: str | Path, out_path: str | Path) -> None:
    import csv

    epochs, train_losses, val_losses = [], [], []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("epoch") is None or row.get("train_loss") is None:
                continue
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            if row.get("val_loss") not in (None, ""):
                val_losses.append(float(row["val_loss"]))
            else:
                val_losses.append(np.nan)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_losses, label="train", marker="o")
    ax.plot(epochs, val_losses, label="val", marker="s")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (MSE on normalized data)")
    ax.set_title(Path(log_path).stem)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_prediction_sample(
    history: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    out_path: str | Path,
    channel: int = 0,
    title: str = "prediction sample",
) -> None:
    """``history`` shape [seq_len, M], ``truth``/``pred`` shape [pred_len, M]."""
    seq_len = history.shape[0]
    pred_len = truth.shape[0]
    h_x = np.arange(seq_len)
    f_x = np.arange(seq_len, seq_len + pred_len)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(h_x, history[:, channel], color="tab:gray", label="history")
    ax.plot(f_x, truth[:, channel], color="tab:blue", label="ground truth")
    ax.plot(f_x, pred[:, channel], color="tab:red", linestyle="--", label="prediction")
    ax.axvline(seq_len - 0.5, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("time step")
    ax.set_ylabel(f"channel {channel}")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
