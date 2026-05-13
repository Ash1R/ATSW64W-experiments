"""PatchTST with channel grouping: an interpolation between channel-
independence (k = num_channels) and channel-mixing (k = 1).

The paper presents the channel-axis treatment as a binary choice:
    - channel-independent: shared Transformer, M channels folded into batch
    - channel-mixing: each token is the concatenation of all M channels

This module implements a third option not in the paper: cluster the M
channels into k groups, and within each group use the channel-mixing strategy
(concatenate channels into each token); across groups share weights but treat
groups as independent. k=M recovers the paper's channel-independent design;
k=1 recovers the paper's channel-mixing baseline. Intermediate k explores
whether tightly-correlated channels benefit from being mixed together while
weakly-correlated channels should stay independent.

Channel grouping uses a Pearson-correlation clustering on the train split
(computed once at construction time). For Weather (21 channels) we typically
sweep k in {1, 3, 7, 21} which gives a clean four-point sweep across the
spectrum.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# Notebooks add `code/` itself to sys.path (the stdlib owns the name `code`),
# so use top-level absolute imports here.
from models.patchtst import _num_patches


def cluster_channels_by_correlation(
    train_data: np.ndarray, k: int, seed: int = 0
) -> list[list[int]]:
    """Greedy correlation clustering: sort by absolute mean correlation, then
    assign in round-robin fashion to k groups. Deterministic given seed."""
    M = train_data.shape[1]
    if k <= 1:
        return [list(range(M))]
    if k >= M:
        return [[i] for i in range(M)]
    corr = np.corrcoef(train_data.T)                 # [M, M]
    np.fill_diagonal(corr, 0.0)
    score = np.abs(corr).mean(axis=1)
    order = np.argsort(-score)
    rng = np.random.RandomState(seed)
    rng.shuffle(order)                               # break ties reproducibly
    groups: list[list[int]] = [[] for _ in range(k)]
    for i, ch in enumerate(order):
        groups[i % k].append(int(ch))
    return [sorted(g) for g in groups]


class ChannelGroupedPatchTST(nn.Module):
    """PatchTST with explicit channel groups.

    Topology per group:
      [B, L, m_g] (channels in group g)
        -> instance norm
        -> patch embedding linear takes patch_len*m_g -> d_model
           (ie tokens carry all channels in the group)
        -> shared Transformer encoder (one across groups)
        -> linear head -> [B, pred_len, m_g]

    All groups share the same encoder weights, so total parameter count
    grows only via the per-group embedding/head.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        groups: list[list[int]],
        patch_len: int = 16,
        stride: int = 8,
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
        self.groups = [sorted(g) for g in groups]
        self.num_patches = _num_patches(seq_len, patch_len, stride)

        self.group_embed = nn.ModuleList(
            nn.Linear(patch_len * len(g), d_model) for g in self.groups
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_dropout = nn.Dropout(dropout)

        self.group_head = nn.ModuleList(
            nn.Linear(self.num_patches * d_model, pred_len * len(g)) for g in self.groups
        )

    @staticmethod
    def _instance_norm(x: torch.Tensor, eps: float = 1e-5):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False) + eps
        return (x - mean) / std, mean, std

    def _patchify_group(self, x_g: torch.Tensor) -> torch.Tensor:
        # x_g: [B, m_g, L]
        B, m_g, L = x_g.shape
        pad = x_g[..., -1:].expand(B, m_g, self.stride)
        x_padded = torch.cat([x_g, pad], dim=-1)
        patches = x_padded.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # patches: [B, m_g, N, P]; flatten m_g into the patch dim
        patches = patches.permute(0, 2, 1, 3).reshape(B, self.num_patches, m_g * self.patch_len)
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, M = x.shape
        assert L == self.seq_len, f"expected seq_len={self.seq_len}, got {L}"

        x = x.transpose(1, 2)                           # [B, M, L]
        x_norm, mean, std = self._instance_norm(x)      # [B, M, L]

        out = torch.zeros(B, M, self.pred_len, device=x.device, dtype=x.dtype)

        for g, embed, head in zip(self.groups, self.group_embed, self.group_head):
            idx = torch.as_tensor(g, device=x.device, dtype=torch.long)
            x_g = x_norm.index_select(dim=1, index=idx)                 # [B, m_g, L]
            patches = self._patchify_group(x_g)                          # [B, N, m_g*P]
            tokens = embed(patches) + self.pos_embed                     # [B, N, d_model]
            tokens = self.encoder_dropout(tokens)
            encoded = self.encoder(tokens)                               # [B, N, d_model]
            flat = encoded.reshape(B, self.num_patches * self.d_model)
            pred_g = head(flat).reshape(B, len(g), self.pred_len)        # [B, m_g, pred_len]
            out.index_copy_(1, idx, pred_g)

        # reverse instance norm
        out = out * std + mean
        return out.transpose(1, 2)                                       # [B, pred_len, M]
