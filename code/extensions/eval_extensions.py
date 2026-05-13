"""Inference-time experiment: patch-content shuffling.

Beyond the paper: PatchTST never tests how strongly the model relies on
temporal placement at the patch level. We randomly permute a fraction of patch
contents before adding positional embeddings, then measure test MSE as a
function of that fraction. Positional slots stay fixed while local subseries
contents move to the wrong time positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


def _mse_mae(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    err = pred - true
    return {"mse": float(np.mean(err ** 2)), "mae": float(np.mean(np.abs(err)))}


def _gather_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            yhat = model(x)
            preds.append(yhat.cpu().numpy())
            trues.append(y.numpy())
    return np.concatenate(preds, 0), np.concatenate(trues, 0)


def _forward_with_pre_pos_content_shuffle(
    model: nn.Module,
    x: torch.Tensor,
    frac: float,
    rng: np.random.RandomState,
) -> torch.Tensor:
    B, L, M = x.shape
    x_t = x.transpose(1, 2)
    x_norm, mean, std = model._instance_norm(x_t)
    patches = model.embed(x_norm)  # [B, M, N, d_model]
    N = patches.shape[2]

    if frac > 0:
        n_shuffle = max(2, int(round(frac * N)))
        idx = np.arange(N)
        shuf_pos = np.sort(rng.choice(N, size=n_shuffle, replace=False))
        perm = idx.copy()
        perm[shuf_pos] = rng.permutation(perm[shuf_pos])
        perm_t = torch.as_tensor(perm, device=patches.device, dtype=torch.long)
        patches = patches.index_select(dim=2, index=perm_t)

    tokens = patches.reshape(B * M, N, model.d_model) + model.pos_embed
    tokens = model.encoder_dropout(tokens)
    encoded = model.encoder(tokens)
    flat = encoded.reshape(B * M, N * model.d_model)
    pred = model.head(flat).reshape(B, M, model.pred_len)
    pred = pred * std + mean
    return pred.transpose(1, 2)


@dataclass
class PatchShuffleResult:
    type: str
    fractions: list[float]
    mse: list[float]
    mae: list[float]


@dataclass
class NoPositionalEmbeddingResult:
    type: str
    mse: float
    mae: float


def patch_shuffle_curve(
    model: nn.Module,
    loader,
    device: torch.device,
    fractions: Iterable[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    seed: int = 0,
) -> PatchShuffleResult:
    """Permute patch contents before positional embeddings; sweep over fracs."""
    fracs = list(fractions)
    mses, maes = [], []
    N = model.num_patches

    for frac in fracs:
        rng = np.random.RandomState(seed + int(frac * 1000))

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                preds.append(_forward_with_pre_pos_content_shuffle(model, x, frac, rng).cpu().numpy())
                trues.append(y.numpy())
        pred, true = np.concatenate(preds, 0), np.concatenate(trues, 0)

        m = _mse_mae(pred, true)
        mses.append(m["mse"])
        maes.append(m["mae"])

    return PatchShuffleResult(
        type="pre_positional_patch_content_shuffle",
        fractions=fracs,
        mse=mses,
        mae=maes,
    )


def evaluate_no_positional_embedding(
    model: nn.Module,
    loader,
    device: torch.device,
) -> NoPositionalEmbeddingResult:
    """Evaluate a trained PatchTST checkpoint with learned positions zeroed."""
    model.eval()
    saved_pos = model.pos_embed.detach().clone()
    preds, trues = [], []
    try:
        model.pos_embed.data.zero_()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                preds.append(model(x).cpu().numpy())
                trues.append(y.numpy())
    finally:
        model.pos_embed.data.copy_(saved_pos)

    m = _mse_mae(np.concatenate(preds, 0), np.concatenate(trues, 0))
    return NoPositionalEmbeddingResult(
        type="zero_positional_embedding",
        mse=m["mse"],
        mae=m["mae"],
    )
