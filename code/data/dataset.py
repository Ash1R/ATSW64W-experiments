"""Sliding-window dataset for long-term forecasting benchmarks.

Each sample is (x, y) where:
    x: [seq_len, num_channels]  -- the input window
    y: [pred_len, num_channels] -- the immediately-following target window

Scaling is done with a StandardScaler fit only on the train split. We expose
that scaler so evaluation can inverse-transform predictions to original units.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .download import find_dataset_csv
from .preprocessing import SPLIT_FRACTIONS, StandardScaler, load_csv, make_splits


Split = Literal["train", "val", "test"]


class SlidingWindowDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        if len(data) < seq_len + pred_len:
            raise ValueError(
                f"split has {len(data)} rows but seq_len+pred_len={seq_len + pred_len}"
            )
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


@dataclass
class DataBundle:
    train: SlidingWindowDataset
    val: SlidingWindowDataset
    test: SlidingWindowDataset
    scaler: StandardScaler
    num_channels: int
    feature_names: list[str]

    def loaders(self, batch_size: int, num_workers: int = 0, shuffle_train: bool = True):
        train = DataLoader(
            self.train, batch_size=batch_size, shuffle=shuffle_train,
            num_workers=num_workers, drop_last=True, pin_memory=True,
        )
        val = DataLoader(
            self.val, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False, pin_memory=True,
        )
        test = DataLoader(
            self.test, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False, pin_memory=True,
        )
        return train, val, test


def build_data_bundle(
    project_root: str | Path,
    dataset: str,
    seq_len: int,
    pred_len: int,
    split_fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
    csv_override: str | Path | None = None,
) -> DataBundle:
    csv_path = Path(csv_override) if csv_override else find_dataset_csv(project_root, dataset)
    values, feature_names = load_csv(str(csv_path))
    train_raw, val_raw, test_raw = make_splits(values, split_fractions)

    scaler = StandardScaler.fit(train_raw)
    train_s = scaler.transform(train_raw)
    val_s = scaler.transform(val_raw)
    test_s = scaler.transform(test_raw)

    return DataBundle(
        train=SlidingWindowDataset(train_s, seq_len, pred_len),
        val=SlidingWindowDataset(val_s, seq_len, pred_len),
        test=SlidingWindowDataset(test_s, seq_len, pred_len),
        scaler=scaler,
        num_channels=values.shape[1],
        feature_names=feature_names,
    )
