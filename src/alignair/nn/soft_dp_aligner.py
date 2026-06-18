"""Differentiable soft-DP germline aligner (learned, gap-aware).

Replaces the diagonal-cosine correlation in germline_aligner.py, whose contiguous
``end = start + len`` assumption cannot represent indel'd alignments. Here a learned
pairwise score matrix is reduced by a semi-global soft dynamic program (sum-product /
"differentiable Smith-Waterman", temperature 1) that marginalises over alignment
paths INCLUDING germline deletions and read insertions, and returns germline
start/end posteriors.

Vectorisation: the affine-gap recurrences normally couple within a row (deletions)
and within a column (insertions). We avoid an O(S*Lg) inner scan by
  - deletions: a discounted ``logcumsumexp`` over the previous row gives, in closed
    form, the log-sum over all numbers of skipped germline columns (linear gap);
  - insertions: a per-row carry state updated in O(1) per row.
So the DP is S sequential row-steps, each a handful of (B, Lg) ops — GPU-friendly.
``soft_dp_end_logits`` operates on a given score matrix (easy to unit-test); the
nn.Module wraps projection + learned gap costs and returns (start_logits, end_logits)
matching the GermlineAligner interface.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SoftDPAligner(nn.Module):
    """Learned, gap-aware germline aligner. Drop-in for GermlineAligner: returns
    (start_logits, end_logits) over germline positions."""

    def __init__(self, d_model: int):
        super().__init__()
        self.seg_proj = nn.Linear(d_model, d_model)
        self.germ_proj = nn.Linear(d_model, d_model)
        # sharp match scale + strong gap priors so even the UNTRAINED aligner localizes
        # (sum-product soft-DP counts paths; a weak per-step signal lets the marginal
        # drift, so we start peaked and let training relax it if useful).
        self.log_scale = nn.Parameter(torch.tensor(1.6))       # scale ~5
        self._gap_open = nn.Parameter(torch.tensor(3.0))       # -softplus -> ~ -3.0
        self._gap_extend = nn.Parameter(torch.tensor(2.0))     # ~ -2.1
        self._del_gap = nn.Parameter(torch.tensor(3.0))        # ~ -3.0

    def _scores(self, seg_reps, germ_reps):
        S = F.normalize(self.seg_proj(seg_reps), dim=-1)
        G = F.normalize(self.germ_proj(germ_reps), dim=-1)
        scale = self.log_scale.clamp(-2.0, 3.0).exp()
        return scale * torch.einsum("bid,bjd->bij", S, G)      # (B,S,Lg) cosine*scale

    def forward(self, seg_reps, seg_mask, germ_reps, germ_mask):
        go = -F.softplus(self._gap_open)
        ge = -F.softplus(self._gap_extend)
        dg = -F.softplus(self._del_gap)
        M = self._scores(seg_reps, germ_reps)
        end_logits = soft_dp_end_logits(M, seg_mask, germ_mask, go, ge, dg)
        # start = reversed-frame end: reverse read rows and germline cols, re-run, flip back
        seg_len = seg_mask.sum(dim=1)
        germ_len = germ_mask.sum(dim=1)
        Mr = _reverse_valid_2d(M.transpose(1, 2), germ_len).transpose(1, 2)  # reverse germ cols
        Mr = _reverse_valid_2d(Mr, seg_len)                                  # reverse read rows
        end_rev = soft_dp_end_logits(Mr, seg_mask, germ_mask, go, ge, dg)
        start_logits = _reverse_valid_2d(end_rev, germ_len)
        return start_logits.masked_fill(~germ_mask, NEG), end_logits

    def alignment_score(self, seg_reps, seg_mask, germ_reps, germ_mask):
        """Candidate-allele log-likelihood: the soft-DP log-partition over end
        positions = total alignment mass of the observed segment against this
        candidate germline. Higher = better alignment = more likely the true allele.
        This is the learned, differentiable replacement for a classical SW score
        (the allele-reader scorer). Returns (B,)."""
        go = -F.softplus(self._gap_open)
        ge = -F.softplus(self._gap_extend)
        dg = -F.softplus(self._del_gap)
        M = self._scores(seg_reps, germ_reps)
        end_logits = soft_dp_end_logits(M, seg_mask, germ_mask, go, ge, dg)
        return torch.logsumexp(end_logits, dim=-1)
