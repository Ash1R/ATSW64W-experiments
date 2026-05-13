"""Model factory."""
from __future__ import annotations

from .baselines import NaiveLastValue
from .patchtst import PatchTST
from .channel_mixing_transformer import ChannelMixingPatchTransformer


def build_model(name: str, **kwargs):
    name = name.lower()
    if name == "naive":
        return NaiveLastValue(pred_len=kwargs["pred_len"])
    if name in ("patchtst", "patchtst_no_patch"):
        # The "no_patch" variant just uses patch_len=1, stride=1 — same code path.
        return PatchTST(
            seq_len=kwargs["seq_len"],
            pred_len=kwargs["pred_len"],
            patch_len=kwargs["patch_len"],
            stride=kwargs["stride"],
            d_model=kwargs.get("d_model", 128),
            n_heads=kwargs.get("n_heads", 16),
            num_layers=kwargs.get("num_layers", 3),
            d_ff=kwargs.get("d_ff", 256),
            dropout=kwargs.get("dropout", 0.2),
        )
    if name == "channel_mixing":
        return ChannelMixingPatchTransformer(
            seq_len=kwargs["seq_len"],
            pred_len=kwargs["pred_len"],
            num_channels=kwargs["num_channels"],
            patch_len=kwargs["patch_len"],
            stride=kwargs["stride"],
            d_model=kwargs.get("d_model", 128),
            n_heads=kwargs.get("n_heads", 8),
            num_layers=kwargs.get("num_layers", 3),
            d_ff=kwargs.get("d_ff", 256),
            dropout=kwargs.get("dropout", 0.2),
        )
    if name in ("patchtst_grouped", "channel_grouped"):
        # Beyond-the-paper extension: explicit channel groups (interpolation
        # between channel-independent and channel-mixing). Requires `groups`
        # (a list[list[int]]) in kwargs; see extensions/channel_grouping.py.
        # Top-level import: notebooks add `code/` to sys.path (the stdlib
        # already owns the name `code`).
        from extensions.channel_grouping import ChannelGroupedPatchTST
        return ChannelGroupedPatchTST(
            seq_len=kwargs["seq_len"],
            pred_len=kwargs["pred_len"],
            groups=kwargs["groups"],
            patch_len=kwargs.get("patch_len", 16),
            stride=kwargs.get("stride", 8),
            d_model=kwargs.get("d_model", 128),
            n_heads=kwargs.get("n_heads", 16),
            num_layers=kwargs.get("num_layers", 3),
            d_ff=kwargs.get("d_ff", 256),
            dropout=kwargs.get("dropout", 0.2),
        )
    raise ValueError(f"unknown model {name!r}")
