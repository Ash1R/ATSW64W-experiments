"""Helpers that abstract over Colab vs. local execution.

On Colab, the project root is on Drive so artifacts persist between sessions.
Locally we fall back to the repo directory.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/cs4782_patchtst_project"


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def mount_drive() -> bool:
    """Mount Google Drive when on Colab. Returns True if mounted."""
    if not is_colab():
        return False
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")
    return True


def get_project_root(default_local: str | os.PathLike = ".") -> Path:
    """Return Drive path on Colab, else the local repo directory."""
    if is_colab():
        return Path(DEFAULT_DRIVE_ROOT)
    # default_local may be the repo root
    return Path(default_local).resolve()


def ensure_dirs(root: str | os.PathLike) -> dict[str, Path]:
    """Create the canonical results subdirs under ``root`` and return them."""
    root = Path(root)
    subdirs = {
        "data": root / "data",
        "results": root / "results",
        "logs": root / "results" / "logs",
        "metrics": root / "results" / "metrics",
        "checkpoints": root / "results" / "checkpoints",
        "figures": root / "results" / "figures",
        "tables": root / "results" / "tables",
        "extensions": root / "results" / "extensions",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs
