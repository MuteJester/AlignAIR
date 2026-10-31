"""The bank scan: every bank row scored by how well its germline explains the read.

For a gene, each read base i carries a weight ``w_i`` (how much it belongs to the gene: the router's soft region
weight, times 1 - P(inserted)), its SHM probability ``m_i``, and its top-k columns ``c_ik`` with probabilities
``p_ik``. A row r scores

    Σ_i w_i Σ_k p_ik · [ row r has base read_i at c_ik : log(1 - m_i),
                         row r has another base at c_ik : log(m_i / 3),
                         row r has no base at c_ik       : κ ]

with κ = -softplus(a learned scalar), then divided by a learned temperature. The sum never visits rows one at a time:
the read's terms are scattered into one vector per read over (column, base) slots and one over columns, and the
bank's one-hot table and cover turn them into every row's score by a matrix product, so the cost is L·k to scatter
plus B·W·R to multiply. (The per-read constant Σ w·p·κ, the same for every row, is left out.) N bases and pads have no
weight. Rows the genotype leaves out (``row_mask``) are out of the softmax.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .bank import BASES

OUT = -1e9          # a row or allele out of the bank: finite, so the softmax's gradients stay finite
_M_MIN = 1e-4


class BankScan(nn.Module):
    def __init__(self):
        super().__init__()
        self.off_row = nn.Parameter(torch.tensor(1.0))           # κ = -softplus(off_row)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def forward(self, weights, m, tokens, columns, probs, table, cover, row_mask) -> torch.Tensor:
        """(B, L), (B, L), (B, L), (B, L, k), (B, L, k), (R, W·4), (R, W), (B, R) bool -> row logits (B, R)."""
        width = cover.shape[1]
        valid = (tokens >= 1) & (tokens <= len(BASES))
        base = (tokens - 1).clamp(0, len(BASES) - 1)
        m = m.clamp(_M_MIN, 1 - _M_MIN)
        match, mismatch = torch.log1p(-m), torch.log(m / 3)
        kappa = -F.softplus(self.off_row)
        weight = (weights * valid).unsqueeze(-1) * probs                                       # (B, L, k)
        slots = columns * len(BASES) + base.unsqueeze(-1)
        on_match = torch.zeros(tokens.shape[0], width * len(BASES), device=probs.device, dtype=probs.dtype)
        on_match = on_match.scatter_add(1, slots.flatten(1), (weight * (match - mismatch).unsqueeze(-1)).flatten(1))
        on_cover = torch.zeros(tokens.shape[0], width, device=probs.device, dtype=probs.dtype)
        on_cover = on_cover.scatter_add(1, columns.flatten(1), (weight * (mismatch.unsqueeze(-1) - kappa)).flatten(1))
        score = on_match @ table.t() + on_cover @ cover.t()
        return (score * torch.exp(-self.log_temperature)).masked_fill(~row_mask, OUT)


def allele_log_posterior(row_logits: torch.Tensor, row_allele: torch.Tensor, n_alleles: int) -> torch.Tensor:
    """(B, R) row logits -> (B, n_alleles) log posterior over alleles: an allele's rows (D's forward and reverse
    complement) add their probabilities; an allele with no row in the bank is ``OUT``."""
    owner = torch.full((row_logits.shape[1], n_alleles), OUT, device=row_logits.device, dtype=row_logits.dtype)
    owner[torch.arange(row_logits.shape[1], device=row_logits.device), row_allele] = 0.
    allele = torch.logsumexp(row_logits.unsqueeze(-1) + owner, dim=1)
    return (allele - torch.logsumexp(allele, -1, keepdim=True)).masked_fill(allele <= OUT / 2, OUT)
