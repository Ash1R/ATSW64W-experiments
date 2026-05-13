"""Patch-based Transformer that mixes channels inside each patch token.

This is the channel-independence ablation from PLAN.md §5.5: instead of
folding M into the batch dim, we concatenate the M channels of each patch into
one fat token. Token count drops to N (vs N*M for channel-independent), but
each token mixes information across all channels — which the PatchTST paper
argues hurts because it lets the model overfit to spurious cross-channel
correlations on small datasets.

Use only on Weather (M=21). For Electricity (M=321) the P*M projection blows up.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChannelMixingPatchTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_channels: int,
        patch_len: int,
        stride: int,
        d_model: int = 128,
        n_heads: int = 8,
        num_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_channels = num_channels
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        eff_len = seq_len + stride
        self.num_patches = (eff_len - patch_len) // stride + 1

        self.embed = nn.Linear(patch_len * num_channels, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_dropout = nn.Dropout(dropout)

        self.head = nn.Linear(self.num_patches * d_model, pred_len * num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, M = x.shape
        assert L == self.seq_len and M == self.num_channels

        x_t = x.transpose(1, 2)                            # [B, M, L]
        # instance norm shared across channels for this variant
        mean = x_t.mean(dim=-1, keepdim=True)
        std = x_t.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
        x_norm = (x_t - mean) / std

        pad = x_norm[..., -1:].expand(B, M, self.stride)
        x_padded = torch.cat([x_norm, pad], dim=-1)
        patches = x_padded.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # patches: [B, M, N, P] -> [B, N, M, P] -> [B, N, M*P]
        patches = patches.permute(0, 2, 1, 3).reshape(B, self.num_patches, M * self.patch_len)

        tokens = self.embed(patches) + self.pos_embed
        tokens = self.encoder_dropout(tokens)
        encoded = self.encoder(tokens)                     # [B, N, d_model]

        flat = encoded.reshape(B, self.num_patches * self.d_model)
        pred = self.head(flat).reshape(B, self.pred_len, M)

        # reverse instance norm: pred is [B, T, M], stats are [B, M, 1]
        pred = pred * std.transpose(1, 2) + mean.transpose(1, 2)
        return pred
