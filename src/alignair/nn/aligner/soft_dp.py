"""Differentiable soft-DP germline recurrence (semi-global, affine-gap).

Shared building block for the seed-and-extend aligner (banded_dp.SeedExtendAligner).
A pairwise score matrix is reduced by a semi-global soft dynamic program (sum-product /
"differentiable Smith-Waterman", temperature 1) that marginalises over alignment paths
INCLUDING germline deletions and read insertions, returning germline end posteriors.

Vectorisation: the affine-gap recurrences normally couple within a row (deletions)
and within a column (insertions). We avoid an O(S*Lg) inner scan by
  - deletions: a discounted ``logcumsumexp`` over the previous row gives, in closed
    form, the log-sum over all numbers of skipped germline columns (linear gap);
  - insertions: a per-row carry state updated in O(1) per row.
So the DP is S sequential row-steps, each a handful of (B, Lg) ops — GPU-friendly.
``soft_dp_end_logits`` operates on a given score matrix (easy to unit-test);
``_reverse_valid_2d`` flips a valid prefix so the START posterior is the reversed-frame END.
"""
import torch

NEG = -1e4


def _reverse_valid_2d(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Reverse each row's valid prefix [0:len) along dim=1; pad region untouched."""
    B, L = x.shape[0], x.shape[1]
    ar = torch.arange(L, device=x.device).unsqueeze(0)
    rev = (lengths.unsqueeze(1) - 1 - ar).clamp(min=0)
    valid = ar < lengths.unsqueeze(1)
    out = torch.gather(x, 1, rev.unsqueeze(-1).expand_as(x) if x.dim() == 3 else rev)
    return torch.where(valid if x.dim() == 2 else valid.unsqueeze(-1), out, x)


def soft_dp_end_logits(M: torch.Tensor, seg_mask: torch.Tensor, germ_mask: torch.Tensor,
                       gap_open: torch.Tensor, gap_extend: torch.Tensor,
                       del_gap: torch.Tensor) -> torch.Tensor:
    """Semi-global soft-DP forward over score matrix M (B,S,Lg). The read (rows) is
    consumed fully; germline (columns) prefix/suffix are free (fit alignment).
    Returns end_logits (B,Lg): log-mass of alignments whose LAST matched germline
    column is j, read fully consumed. gap_open/gap_extend govern read insertions;
    del_gap is the per-column germline-deletion (skip) cost. All gap costs <= 0."""
    B, S, Lg = M.shape
    M = M.masked_fill(~germ_mask.unsqueeze(1), NEG)
    colp = torch.arange(Lg, device=M.device, dtype=M.dtype)
    seg_len = seg_mask.sum(dim=1)                              # (B,)
    Hm_rows = M.new_full((B, S, Lg), NEG)
    Hm_prev = M.new_full((B, Lg), NEG)
    Ins_prev = M.new_full((B, Lg), NEG)
    for i in range(S):
        Mi = M[:, i, :]
        if i == 0:
            Hm = Mi                                            # free start at any column
            Ins = M.new_full((B, Lg), NEG)
        else:
            prev = torch.logsumexp(torch.stack([Hm_prev, Ins_prev], 0), dim=0)  # (B,Lg)
            # deletions: sum over k<=j-1 of prev[k] + (j-1-k)*del_gap, via discounted cumsum
            A = prev - colp.unsqueeze(0) * del_gap
            P = torch.logcumsumexp(A, dim=1)                   # P[j] = lse_{k<=j} A[k]
            P_shift = torch.cat([P.new_full((B, 1), NEG), P[:, :-1]], dim=1)  # P[j-1]
            diag_in = (colp - 1).unsqueeze(0) * del_gap + P_shift
            diag_in = diag_in.clone()
            diag_in[:, 0] = NEG                                # column 0 has no predecessor
            Hm = Mi + diag_in
            Ins = torch.logsumexp(torch.stack([Hm_prev + gap_open,
                                               Ins_prev + gap_extend], 0), dim=0)
        Hm = Hm.masked_fill(~germ_mask, NEG)
        Hm_rows[:, i, :] = Hm
        Hm_prev, Ins_prev = Hm, Ins
    end_idx = (seg_len - 1).clamp(min=0)
    end_logits = Hm_rows[torch.arange(B, device=M.device), end_idx]
    return end_logits.masked_fill(~germ_mask, NEG)
