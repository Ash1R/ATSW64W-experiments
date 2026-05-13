"""PatchTST: subseries patches as Transformer tokens, channel-independent.

Reference: Nie et al., "A Time Series is Worth 64 Words" (ICLR 2023).

Pipeline (PLAN.md §5.3):
    [B, L, M]
      -> transpose                   [B, M, L]
      -> per-(sample, channel) instance norm; remember mean/std
      -> right-pad by `stride` (replicate last value)
      -> unfold into patches         [B, M, N, P]
      -> fold channel into batch     [B*M, N, P]
      -> linear patch embedding      [B*M, N, d_model]
      -> add learnable pos embed
      -> Transformer encoder
      -> flatten + linear head       [B*M, pred_len]
      -> reshape + transpose         [B, pred_len, M]
      -> reverse instance norm using stored stats

Setting patch_len=1, stride=1 reproduces a "no-patching" point-token Transformer
ablation through the same code path.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _num_patches(seq_len: int, patch_len: int, stride: int) -> int:
    """Token count after PatchTST's right-pad-by-stride policy.

    The paper pads the right end of the sequence with ``stride`` copies of the
    last value before unfolding, giving an effective length of L + stride.
    Number of windows of size P with step S is then floor((L + stride - P)/S) + 1.
    """
    eff_len = seq_len + stride
    return (eff_len - patch_len) // stride + 1


class _PatchEmbedding(nn.Module):
    def __init__(self, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, M, L]  ->  patches: [B, M, N, d_model]."""
        B, M, L = x.shape
        # right-pad by `stride` copies of the last time step (replicate)
        pad = x[..., -1:].expand(B, M, self.stride)
        x_padded = torch.cat([x, pad], dim=-1)            # [B, M, L+stride]
        patches = x_padded.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # patches: [B, M, N, P]
        return self.proj(patches)                          # [B, M, N, d_model]


class PatchTST(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        stride: int,
        d_model: int = 128,
        n_heads: int = 16,
        num_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.num_patches = _num_patches(seq_len, patch_len, stride)

        self.embed = _PatchEmbedding(patch_len, stride, d_model)
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

        self.head = nn.Linear(self.num_patches * d_model, pred_len)

    @staticmethod
    def _instance_norm(x: torch.Tensor, eps: float = 1e-5):
        # x: [B, M, L]
        # unbiased=False matches the RevIN / paper convention and avoids an
        # MPS instability where the unbiased estimator can return NaN.
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False) + eps
        return (x - mean) / std, mean, std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, M = x.shape
        assert L == self.seq_len, f"expected seq_len={self.seq_len}, got {L}"

        x = x.transpose(1, 2)                              # [B, M, L]
        x_norm, mean, std = self._instance_norm(x)         # each [B, M, *]

        patches = self.embed(x_norm)                       # [B, M, N, d_model]
        N = patches.shape[2]
        assert N == self.num_patches, f"patch count {N} != expected {self.num_patches}"

        tokens = patches.reshape(B * M, N, self.d_model)   # channel-independence
        tokens = tokens + self.pos_embed                   # broadcast over B*M
        tokens = self.encoder_dropout(tokens)
        encoded = self.encoder(tokens)                     # [B*M, N, d_model]

        flat = encoded.reshape(B * M, N * self.d_model)
        pred = self.head(flat)                             # [B*M, pred_len]
        pred = pred.reshape(B, M, self.pred_len)           # [B, M, pred_len]

        # reverse instance norm using stored stats
        pred = pred * std + mean

        return pred.transpose(1, 2)                        # [B, pred_len, M]
