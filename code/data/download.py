"""Dataset download / verification.

We do not redistribute the benchmark CSVs. The Informer/Autoformer authors host
mirrors of the standard long-term-forecasting datasets at:

    https://github.com/thuml/Autoformer  (see README "Datasets" section)
    https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy

Expected on-disk layout under ``<root>/data``:

    weather/weather.csv          # ~52k rows, 21 features
    electricity/electricity.csv  # ~26k rows, 321 features
    traffic/traffic.csv          # optional, ~17k rows, 862 features

If a CSV exists at ``<root>/data/<name>.csv`` (flat) we accept that too.
"""
from __future__ import annotations

from pathlib import Path

DATASET_FILENAMES = {
    "weather": ["weather/weather.csv", "weather.csv"],
    "electricity": ["electricity/electricity.csv", "electricity.csv", "ECL.csv"],
    "traffic": ["traffic/traffic.csv", "traffic.csv"],
}


def find_dataset_csv(root: str | Path, name: str) -> Path:
    """Return the first matching CSV path under ``root``. Raises FileNotFoundError."""
    name = name.lower()
    if name not in DATASET_FILENAMES:
        raise ValueError(f"unknown dataset {name!r}; expected one of {list(DATASET_FILENAMES)}")
    root = Path(root)
    candidates = [root / "data" / rel for rel in DATASET_FILENAMES[name]]
    for cand in candidates:
        if cand.exists():
            return cand
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find {name} CSV. Looked at:\n  {tried}\n"
        "Download from the Autoformer datasets release and place the file at one of those paths."
    )


def verify_datasets(root: str | Path, names: list[str]) -> dict[str, Path]:
    """Resolve all requested datasets up-front; raise if any are missing."""
    return {n: find_dataset_csv(root, n) for n in names}
