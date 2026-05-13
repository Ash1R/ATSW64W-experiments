"""Train a forecasting model on a long-term-forecasting benchmark.

Designed to be runnable both from the CLI and from a Colab notebook cell:

    # CLI
    python code/train.py --model patchtst --dataset weather --seq_len 336 ...

    # Notebook
    from code.train import train_from_config
    metrics = train_from_config({"model": "patchtst", ...})

Outputs (all under ``output_dir``):
    logs/<run_name>.csv         per-epoch train/val loss + timing
    metrics/<run_name>.json     final test metrics + run config
    metrics/<run_name>.csv      one-row CSV mirror of the JSON
    checkpoints/<run_name>.pt   best model by val MSE (resumable)
    figures/<run_name>_training_curve.png
    figures/<run_name>_prediction_plot.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# When invoked as ``python code/train.py``, ``code/`` is on sys.path; when
# imported as ``code.train``, package-relative imports work. This shim makes
# both work.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data.dataset import build_data_bundle
    from models import build_model
    from utils.device import get_device, gpu_mem_summary, reset_gpu_mem
    from utils.logging import CSVLogger, print_run_header
    from utils.metrics import compute_metrics
    from utils.plotting import plot_prediction_sample, plot_training_curve
    from utils.seed import set_seed
else:
    from .data.dataset import build_data_bundle
    from .models import build_model
    from .utils.device import get_device, gpu_mem_summary, reset_gpu_mem
    from .utils.logging import CSVLogger, print_run_header
    from .utils.metrics import compute_metrics
    from .utils.plotting import plot_prediction_sample, plot_training_curve
    from .utils.seed import set_seed

import numpy as np
import torch
import torch.nn as nn


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["naive", "patchtst", "channel_mixing"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--data_path", default=None, help="optional explicit path to CSV")
    p.add_argument("--project_root", default=".", help="root containing data/ and results/")
    p.add_argument("--output_dir", default=None, help="overrides results/ under project_root")
    p.add_argument("--run_name", default=None)

    p.add_argument("--seq_len", type=int, required=True)
    p.add_argument("--pred_len", type=int, required=True)
    p.add_argument("--patch_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=16)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=5, help="early stop patience on val MSE")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="resume from existing checkpoint if present")
    return p.parse_args(argv)


def args_to_config(args: argparse.Namespace) -> dict:
    return {k: v for k, v in vars(args).items()}


def _default_run_name(cfg: dict) -> str:
    parts = [cfg["dataset"], cfg["model"], f"L{cfg['seq_len']}", f"T{cfg['pred_len']}"]
    if cfg["model"] in ("patchtst", "channel_mixing"):
        parts.append(f"P{cfg['patch_len']}S{cfg['stride']}")
    parts.append(f"seed{cfg['seed']}")
    return "_".join(str(p) for p in parts)


def _resolve_paths(cfg: dict) -> dict[str, Path]:
    project_root = Path(cfg["project_root"]).resolve()
    output_root = Path(cfg["output_dir"]) if cfg.get("output_dir") else project_root / "results"
    run_name = cfg.get("run_name") or _default_run_name(cfg)
    cfg["run_name"] = run_name
    cfg["output_dir"] = str(output_root)
    paths = {
        "logs": output_root / "logs" / f"{run_name}.csv",
        "metrics_json": output_root / "metrics" / f"{run_name}.json",
        "metrics_csv": output_root / "metrics" / f"{run_name}.csv",
        "checkpoint": output_root / "checkpoints" / f"{run_name}.pt",
        "fig_curve": output_root / "figures" / f"{run_name}_training_curve.png",
        "fig_pred": output_root / "figures" / f"{run_name}_prediction_plot.png",
    }
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    return paths


def run_split(model: nn.Module, loader, device, scaler=None) -> dict[str, float]:
    """Compute metrics on a full split. Returns metrics on inverse-scaled values when scaler given."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yhat = model(x)
            preds.append(yhat.cpu().numpy())
            trues.append(y.cpu().numpy())
    pred = np.concatenate(preds, axis=0)  # [N, T, M]
    true = np.concatenate(trues, axis=0)
    if scaler is not None:
        pred = scaler.inverse(pred)
        true = scaler.inverse(true)
    return compute_metrics(pred, true)


def train_from_config(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = get_device(verbose=True)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    paths = _resolve_paths(cfg)
    print_run_header(cfg, str(device), gpu_name)

    bundle = build_data_bundle(
        project_root=cfg["project_root"],
        dataset=cfg["dataset"],
        seq_len=cfg["seq_len"],
        pred_len=cfg["pred_len"],
        csv_override=cfg.get("data_path"),
    )
    print(f"[data] num_channels={bundle.num_channels}  "
          f"train={len(bundle.train)} val={len(bundle.val)} test={len(bundle.test)}")
    train_loader, val_loader, test_loader = bundle.loaders(
        cfg["batch_size"], num_workers=cfg.get("num_workers", 0)
    )

    model_kwargs = dict(
        seq_len=cfg["seq_len"],
        pred_len=cfg["pred_len"],
        num_channels=bundle.num_channels,
        patch_len=cfg.get("patch_len", 16),
        stride=cfg.get("stride", 8),
        d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 16),
        num_layers=cfg.get("num_layers", 3),
        d_ff=cfg.get("d_ff", 256),
        dropout=cfg.get("dropout", 0.2),
    )
    # extension models may need extra constructor kwargs (e.g. `groups` for
    # channel-grouping). Forward any keys whose names match expected model
    # arguments — anything else in cfg (lr, batch_size, …) is ignored.
    for extra in ("groups",):
        if extra in cfg:
            model_kwargs[extra] = cfg[extra]
    model = build_model(cfg["model"], **model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {cfg['model']} params={n_params:,}")

    is_naive = cfg["model"] == "naive"
    optimizer = (
        None if is_naive else torch.optim.Adam(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0)
        )
    )
    criterion = nn.MSELoss()
    logger = CSVLogger(paths["logs"])
    reset_gpu_mem()

    start_epoch = 0
    best_val = float("inf")
    epochs_since_improve = 0
    if cfg.get("resume") and paths["checkpoint"].exists():
        ckpt = torch.load(paths["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"[resume] from epoch {start_epoch} best_val={best_val:.6f}")

    epochs = 1 if is_naive else cfg["epochs"]
    epoch_times = []
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        train_loss_sum, n_batches = 0.0, 0
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yhat = model(x)
            loss = criterion(yhat, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1
        train_loss = train_loss_sum / max(n_batches, 1)

        val_metrics_norm = run_split(model, val_loader, device, scaler=None)
        val_loss = val_metrics_norm["mse"]
        elapsed = time.time() - epoch_start
        epoch_times.append(elapsed)

        logger.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_metrics_norm["mae"],
            "epoch_time_s": elapsed,
        })
        print(f"[epoch {epoch:>3}/{epochs}] train={train_loss:.6f}  val_mse={val_loss:.6f}  "
              f"val_mae={val_metrics_norm['mae']:.6f}  ({elapsed:.1f}s)")

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            epochs_since_improve = 0
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer else None,
                "epoch": epoch,
                "best_val": best_val,
                "config": cfg,
                "scaler_mean": bundle.scaler.mean,
                "scaler_std": bundle.scaler.std,
            }, paths["checkpoint"])
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= cfg.get("patience", 5):
                print(f"[early stop] no improvement for {epochs_since_improve} epochs")
                break

    if paths["checkpoint"].exists() and not is_naive:
        ckpt = torch.load(paths["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    test_inv = run_split(model, test_loader, device, scaler=bundle.scaler)
    test_norm = run_split(model, test_loader, device, scaler=None)
    print(f"[test/inverse] mse={test_inv['mse']:.6f}  mae={test_inv['mae']:.6f}")
    print(f"[test/normalized] mse={test_norm['mse']:.6f}  mae={test_norm['mae']:.6f}")

    avg_epoch_s = float(np.mean(epoch_times)) if epoch_times else 0.0
    summary = {
        "run_name": cfg["run_name"],
        "config": cfg,
        "num_channels": bundle.num_channels,
        "num_params": n_params,
        "test_mse_inverse": test_inv["mse"],
        "test_mae_inverse": test_inv["mae"],
        "test_mse_normalized": test_norm["mse"],
        "test_mae_normalized": test_norm["mae"],
        "best_val_mse_normalized": best_val,
        "avg_epoch_seconds": avg_epoch_s,
        "gpu_memory_mb": gpu_mem_summary(),
        "device": str(device),
        "gpu_name": gpu_name,
    }
    if hasattr(model, "num_patches"):
        summary["num_patches"] = int(model.num_patches)
        summary["attention_pairs"] = int(model.num_patches ** 2)

    paths["metrics_json"].write_text(json.dumps(summary, indent=2))
    flat = {**summary, **summary.pop("config", {})}
    flat.pop("gpu_memory_mb", None)
    import csv
    with paths["metrics_csv"].open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)

    if not is_naive:
        try:
            plot_training_curve(paths["logs"], paths["fig_curve"])
        except Exception as e:
            print(f"[warn] training curve plot failed: {e}")
    try:
        x_b, y_b = next(iter(test_loader))
        x_b = x_b.to(device)
        with torch.no_grad():
            yhat_b = model(x_b).cpu().numpy()
        history = bundle.scaler.inverse(x_b[0].cpu().numpy())
        truth = bundle.scaler.inverse(y_b[0].numpy())
        pred = bundle.scaler.inverse(yhat_b[0])
        plot_prediction_sample(history, truth, pred, paths["fig_pred"], channel=0,
                               title=cfg["run_name"])
    except Exception as e:
        print(f"[warn] prediction plot failed: {e}")

    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train_from_config(args_to_config(args))


if __name__ == "__main__":
    main()
