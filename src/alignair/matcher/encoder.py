import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MatcherConfig


class DilatedConvBlock(nn.Module):
    """Length-preserving depthwise-separable dilated residual block. (B, C, L) -> (B, C, L).

    Uses LayerNorm over channels (per position, per sample) so padding and batch
    composition never pollute the normalization statistics — important because D
    alleles are short and heavily padded within a gene batch.
    """

    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        assert kernel % 2 == 1, "odd kernel keeps length exact"
        pad = dilation * (kernel - 1) // 2          # exact 'same' for odd kernel
        self.depthwise = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation, groups=ch)
        self.pointwise = nn.Conv1d(ch, ch, 1)
        self.norm = nn.LayerNorm(ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, L)
        y = self.pointwise(self.depthwise(x))            # (B, C, L)
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)  # LayerNorm over C
        y = self.drop(self.act(y))
        return x + y


class MatcherEncoder(nn.Module):
    """Char embed -> dilated-conv tower -> per-position match embedding + salience.

    No positional encoding: convolutions are translation-equivariant, which is what
    late-interaction matching needs (an allele must match wherever it appears in s).
    """

    def __init__(self, cfg: MatcherConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=0)
        self.blocks = nn.ModuleList([
            DilatedConvBlock(cfg.d_model, cfg.kernel,
                             cfg.dilations[i % len(cfg.dilations)], cfg.dropout)
            for i in range(cfg.n_layers)
        ])
        self.match_proj = nn.Linear(cfg.d_model, cfg.d_match)
        self.sal_proj = nn.Linear(cfg.d_model, 1)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor):
        x = self.embed(ids).transpose(1, 2)      # (B, d_model, L)
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2)                     # (B, L, d_model)
        emb = F.normalize(self.match_proj(x), dim=-1)      # (B, L, d_match)
        sal = F.softplus(self.sal_proj(x)).squeeze(-1)     # (B, L)
        return emb, sal
