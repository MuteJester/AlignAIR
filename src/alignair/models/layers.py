"""Faithful PyTorch port of the load-bearing TF ``Models/Layers/Layers.py`` layers.

Only the layers actually wired into ``SingleChainAlignAIR``/``MultiChainAlignAIR`` are ported:
``TokenAndPositionEmbedding``, ``Conv1DBatchNorm``, ``ConvResidualFeatureExtractionBlock``,
``SoftCutoutLayer``. Kendall task-weighting reuses :class:`alignair.nn.weighting.UncertaintyWeight`
(already proper-Kendall). TF-isms handled here: channels-first conv, ``same`` padding for even
kernels (asymmetric, matching TF), BatchNorm ``eps=0.8``/``momentum=0.9``, ``LeakyReLU(0.3)``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def same_pad1d(kernel: int, dilation: int = 1) -> tuple[int, int]:
    """(left, right) padding reproducing TF Conv1D ``padding='same'`` for stride 1.

    TF splits the total ``dilation*(kernel-1)`` padding with the extra pixel on the RIGHT for
    even kernels — replicating this exactly matters for the segmentation heads (a 1-position
    shift moves predicted boundaries)."""
    total = dilation * (kernel - 1)
    left = total // 2
    return left, total - left


class TokenAndPositionEmbedding(nn.Module):
    """Learned token embedding + learned absolute position embedding, added (TF Layers.py:626)."""

    def __init__(self, vocab_size: int, embed_dim: int, maxlen: int):
        super().__init__()
        self.maxlen = maxlen
        self.token = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(maxlen, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L) int
        positions = torch.arange(self.maxlen, device=x.device)
        return self.token(x.long()) + self.pos(positions)


_ACT = {"tanh": torch.tanh, "gelu": F.gelu, "swish": F.silu}


class Conv1DBatchNorm(nn.Module):
    """Three stacked same-padded Conv1d -> BatchNorm -> activation -> MaxPool (TF Layers.py:96).

    Channels-first ``(B, C, L)``. TF-faithful: no activation *between* the three convs; single BN
    with ``eps=0.8`` and TF ``momentum=0.1`` ported as PyTorch ``momentum=0.9``; ``MaxPool1d(pool)``.
    """

    def __init__(self, in_channels: int, filters: int, kernel: int, pool: int = 2, act: str = "tanh"):
        super().__init__()
        self.pad = same_pad1d(kernel)
        self.conv1 = nn.Conv1d(in_channels, filters, kernel)
        self.conv2 = nn.Conv1d(filters, filters, kernel)
        self.conv3 = nn.Conv1d(filters, filters, kernel)
        self.bn = nn.BatchNorm1d(filters, eps=0.8, momentum=0.9)
        self.act = _ACT[act]
        self.pool = nn.MaxPool1d(pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in (self.conv1, self.conv2, self.conv3):
            x = conv(F.pad(x, self.pad))
        return self.pool(self.act(self.bn(x)))


class ConvResidualFeatureExtractionBlock(nn.Module):
    """Residual conv feature tower (TF Layers.py:144). Input channels-last ``(B, L, C)``; output a
    fixed ``(B, out)`` global vector. Internal sequence length is halved ``N+1`` times.

    Exact TF structure: a residual channel (single same-padded Conv1d, no activation) that is pooled
    and added into the main stream; the index-0 block pools twice (once before, once after its add),
    every later block pools once; ``LeakyReLU(0.3)`` after each residual add; ``tanh`` inside each
    ``Conv1DBatchNorm``; final Flatten -> Linear(out).
    """

    def __init__(self, in_channels: int, N: int, kernels: list[int], max_len: int,
                 filters: int = 128, out: int = 576, conv_act: str = "tanh"):
        super().__init__()
        assert len(kernels) == N + 1, "kernels must have N+1 entries (N convs + 1 residual)"
        self.residual_channel = nn.Conv1d(in_channels, filters, kernels[-1])
        self.res_pad = same_pad1d(kernels[-1])
        self.conv_layers = nn.ModuleList(
            [Conv1DBatchNorm(in_channels if i == 0 else filters, filters, kernels[i], pool=2, act=conv_act)
             for i in range(N)])
        self.pool = nn.MaxPool1d(2)
        self.act = nn.LeakyReLU(0.3)
        l_final = max_len >> (N + 1)                    # length halved N+1 times (MaxPool floor)
        assert l_final >= 1, f"max_len={max_len} too short for N={N} (needs >= {1 << (N + 1)})"
        self.proj = nn.Linear(filters * l_final, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, C)
        x = x.transpose(1, 2)                                          # (B, C, L)
        res = self.pool(self.residual_channel(F.pad(x, self.res_pad)))  # pool #1
        res = self.pool(self.act(self.conv_layers[0](x) + res))         # pool #2 (index-0 double pool)
        for conv in self.conv_layers[1:]:
            res = self.act(conv(res) + self.pool(res))
        return self.proj(torch.flatten(res, 1))


class EmbeddingOrientationHead(nn.Module):
    """4-class orientation logits from the model's shared initial embeddings.

    Order-sensitive (a masked depthwise-ish conv) so forward vs reverse is distinguishable — a plain
    mean-pool could not tell a read from its reverse. Feeds the in-model correct-and-re-embed step.
    """

    def __init__(self, embed_dim: int, num_orientations: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(embed_dim, embed_dim, 7, padding="same")
        self.fc = nn.Linear(embed_dim, num_orientations)

    def forward(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # (B,L,C), (B,L)
        m = mask.unsqueeze(-1).to(emb.dtype)
        h = F.gelu(self.conv((emb * m).transpose(1, 2))).transpose(1, 2) * m
        pooled = h.sum(1) / m.sum(1).clamp(min=1.0)
        return self.fc(pooled)


class SoftCutoutLayer(nn.Module):
    """Differentiable soft interval mask (TF Layers.py:404). ``start``/``end`` are ``(B,1)``
    soft-argmax expectations; returns ``(B, max_size)`` = ``sigmoid((i-start)/k)·sigmoid((end-i)/k)``,
    a smooth indicator of ``[start, end)`` with ramp width ``k``. Enforces ``end >= start + 1``.
    """

    def __init__(self, max_size: int, k: float = 3.0):
        super().__init__()
        self.max_size = max_size
        self.k = k

    def forward(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        start = start.clamp(0, self.max_size)
        end = torch.maximum(end.clamp(0, self.max_size), start + 1.0)
        idx = torch.arange(self.max_size, device=start.device, dtype=start.dtype)   # (L,)
        left = torch.sigmoid((idx - start) / self.k)                                # (B, L)
        right = torch.sigmoid((end - idx) / self.k)
        return left * right
