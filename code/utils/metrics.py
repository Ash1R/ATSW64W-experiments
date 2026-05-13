"""Forecasting metrics. All inputs assumed to be aligned, same shape."""
from __future__ import annotations

import numpy as np
import torch

ArrayLike = np.ndarray | torch.Tensor


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mse(pred: ArrayLike, true: ArrayLike) -> float:
    p, t = _to_numpy(pred), _to_numpy(true)
    return float(np.mean((p - t) ** 2))


def mae(pred: ArrayLike, true: ArrayLike) -> float:
    p, t = _to_numpy(pred), _to_numpy(true)
    return float(np.mean(np.abs(p - t)))


def rmse(pred: ArrayLike, true: ArrayLike) -> float:
    return float(np.sqrt(mse(pred, true)))


def compute_metrics(pred: ArrayLike, true: ArrayLike) -> dict[str, float]:
    return {"mse": mse(pred, true), "mae": mae(pred, true), "rmse": rmse(pred, true)}
