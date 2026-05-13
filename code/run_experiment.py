"""Run a single experiment described by a YAML config file.

Usage:
    python code/run_experiment.py configs/weather_patchtst_96.yaml \
        --project_root /content/drive/MyDrive/cs4782_patchtst_project

The YAML is merged on top of the train.py defaults; CLI overrides win last.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train import train_from_config
else:
    from .train import train_from_config


DEFAULTS = {
    "patch_len": 16,
    "stride": 8,
    "d_model": 128,
    "n_heads": 16,
    "num_layers": 3,
    "d_ff": 256,
    "dropout": 0.2,
    "batch_size": 32,
    "epochs": 20,
    "lr": 1e-4,
    "weight_decay": 0.0,
    "patience": 5,
    "num_workers": 0,
    "seed": 42,
    "project_root": ".",
    "output_dir": None,
    "run_name": None,
    "data_path": None,
    "resume": False,
}


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return {**DEFAULTS, **cfg}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", help="path to YAML config")
    p.add_argument("--project_root", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--data_path", default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    for k in ("project_root", "output_dir", "run_name", "data_path"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.resume:
        cfg["resume"] = True

    train_from_config(cfg)


if __name__ == "__main__":
    main()
