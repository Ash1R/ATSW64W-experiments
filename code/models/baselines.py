"""Naive last-value baseline.

Predicts the most recent observed value for every future step. This is the
sanity check every forecasting paper expects — if a learned model can't beat
it on at least most settings, the metric pipeline is wrong.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NaiveLastValue(nn.Module):
    """Repeats x[:, -1, :] for ``pred_len`` steps. No trainable parameters."""

    def __init__(self, pred_len: int):
        super().__init__()
        self.pred_len = pred_len
        # one zero-buffer parameter so .to(device) and optimizer no-ops behave nicely
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, M]
        last = x[:, -1:, :]               # [B, 1, M]
        return last.repeat(1, self.pred_len, 1)  # [B, pred_len, M]
