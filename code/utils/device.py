"""Device selection and reporting."""
from __future__ import annotations

import torch


def get_device(verbose: bool = True, allow_mps: bool = False) -> torch.device:
    """Pick a compute device.

    The target deployment is Colab GPU (CUDA). MPS support in PyTorch still has
    NaN-producing bugs on the ops PatchTST relies on (notably ``torch.std`` and
    ``nn.Linear`` chains in fp32), so by default we fall back to CPU on Apple
    silicon. Pass ``allow_mps=True`` to opt in if a future PyTorch fixes them.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"[device] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"[device] CUDA capability: {torch.cuda.get_device_capability(0)}")
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[device] Total GPU memory: {total_gb:.2f} GB")
        return device
    if allow_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        if verbose:
            print("[device] Using Apple MPS (allow_mps=True)")
        return torch.device("mps")
    if verbose:
        print("[device] No CUDA detected, using CPU. Training will be slow but correct.")
    return torch.device("cpu")


def gpu_mem_summary() -> dict:
    """Returns peak GPU memory in MB if CUDA, else empty dict."""
    if not torch.cuda.is_available():
        return {}
    return {
        "peak_alloc_mb": torch.cuda.max_memory_allocated() / 1e6,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
    }


def reset_gpu_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
