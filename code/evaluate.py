"""Load a saved checkpoint and report test MSE/MAE."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data.dataset import build_data_bundle
    from data.preprocessing import StandardScaler
    from models import build_model
    from train import run_split
    from utils.device import get_device
else:
    from .data.dataset import build_data_bundle
    from .data.preprocessing import StandardScaler
    from .models import build_model
    from .train import run_split
    from .utils.device import get_device

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--project_root", default=".")
    p.add_argument("--data_path", default=None)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    bundle = build_data_bundle(
        project_root=args.project_root,
        dataset=cfg["dataset"],
        seq_len=cfg["seq_len"],
        pred_len=cfg["pred_len"],
        csv_override=args.data_path,
    )
    # restore the original training-time scaler exactly to keep inverse-scale metrics comparable
    bundle.scaler = StandardScaler(
        mean=np.asarray(ckpt["scaler_mean"]),
        std=np.asarray(ckpt["scaler_std"]),
    )
    _, _, test_loader = bundle.loaders(args.batch_size)

    model = build_model(
        cfg["model"],
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
    ).to(device)
    model.load_state_dict(ckpt["model"])

    inv = run_split(model, test_loader, device, scaler=bundle.scaler)
    norm = run_split(model, test_loader, device, scaler=None)
    print(json.dumps({
        "run_name": cfg.get("run_name"),
        "test_mse_inverse": inv["mse"],
        "test_mae_inverse": inv["mae"],
        "test_mse_normalized": norm["mse"],
        "test_mae_normalized": norm["mae"],
    }, indent=2))


if __name__ == "__main__":
    main()
