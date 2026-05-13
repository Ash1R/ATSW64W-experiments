"""Train/val/test splits and per-feature standard scaling.

Splits follow the Informer/Autoformer benchmark conventions used by PatchTST:
the ratios depend on the dataset because the original benchmark partitions the
data by absolute row counts. We use the standard 0.7 / 0.1 / 0.2 fractions for
ETT-style data and the 12/4/4-month split for Electricity/Weather/Traffic
approximated as 0.7 / 0.1 / 0.2 — the precise paper splits differ slightly per
dataset, which is one of the discrepancies documented in the README.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StandardScaler:
    """Per-feature z-score, fit on train only."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "StandardScaler":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)


SPLIT_FRACTIONS = (0.7, 0.1, 0.2)


def load_csv(path: str) -> tuple[np.ndarray, list[str]]:
    """Returns (values [T, M], feature_names). Drops a 'date' column if present."""
    df = pd.read_csv(path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    feature_names = list(df.columns)
    values = df.values.astype(np.float32)
    return values, feature_names


def make_splits(
    values: np.ndarray,
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = values.shape[0]
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    train = values[:n_train]
    val = values[n_train : n_train + n_val]
    test = values[n_train + n_val :]
    return train, val, test
