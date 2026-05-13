"""CSV training logger."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CSVLogger:
    """Append-only CSV logger. Writes header on first row."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields: list[str] | None = None
        # If the file already exists with a header, reuse those fields so we
        # don't duplicate or reorder columns on resume.
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    self._fields = header

    def log(self, row: dict[str, Any]) -> None:
        if self._fields is None:
            self._fields = list(row.keys())
            with self.path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fields)
                writer.writeheader()
                writer.writerow({k: row.get(k) for k in self._fields})
            return
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fields)
            writer.writerow({k: row.get(k) for k in self._fields})


def print_run_header(cfg: dict, device: str, gpu_name: str | None) -> None:
    """Standardized run banner per PLAN.md §3."""
    rows = [
        ("device", device),
        ("gpu_name", gpu_name or "n/a"),
        ("dataset", cfg.get("dataset")),
        ("model", cfg.get("model")),
        ("seq_len", cfg.get("seq_len")),
        ("pred_len", cfg.get("pred_len")),
        ("patch_len", cfg.get("patch_len")),
        ("stride", cfg.get("stride")),
        ("batch_size", cfg.get("batch_size")),
        ("lr", cfg.get("lr")),
        ("epochs", cfg.get("epochs")),
        ("output_dir", cfg.get("output_dir")),
    ]
    width = max(len(k) for k, _ in rows)
    print("=" * 60)
    print("Run configuration")
    print("=" * 60)
    for k, v in rows:
        print(f"  {k:<{width}}  {v}")
    print("=" * 60)
