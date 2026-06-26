"""BandedPointerAligner: a fully-parallel, reference-conditioned germline pointer head.

Start/end logits are weighted leading/reverse diagonals of the score matrix M, extracted in
single CUDA launches (no S-length recurrence). REJECTED as the soft-DP coordinate replacement
(structurally ~0.1 lower coord competence — a rigid diagonal cannot marginalise over alignment
paths), but RETAINED as the fast READER candidate / A-B baseline; its diagonal score is also
the seed-and-extend band predictor's pattern. See the redesign memory + spec §4.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_match import base_match_channel
from .diagonal_ops import weighted_leading_diag, weighted_reverse_diag, banded_start_end

NEG = -1e4


class BandedPointerAligner(nn.Module):
    """Returns (start_logits, end_logits) over germline positions and a fast diagonal
    alignment_score. Single-diagonal core (band_half_width extends it with an indel band).
    seg_tok/germ_tok/seg_reliability enable the base-match channel + reliability gating."""

    def __init__(self, d_model: int, match_floor: float = 1.0, max_len: int = 1024,
                 band_half_width: int = 0):
        super().__init__()
        self.seg_proj = nn.Linear(d_model, d_model)
        self.germ_proj = nn.Linear(d_model, d_model)
        self.log_scale = nn.Parameter(torch.tensor(1.6))       # scale ~5
        # High temp init so the diagonal posterior is PEAKED (a flat bump near a germline edge
        # biases the soft-argmax expected-position inward); sharp posterior -> soft-argmax ~ argmax.
        self.log_temp = nn.Parameter(torch.tensor(1.6))        # temp ~5
        self._match_weight = nn.Parameter(torch.tensor(1.0))
        self.match_floor = float(match_floor)
        self.diag_bias = nn.Parameter(torch.zeros(max_len))    # per-position weight, init uniform
        self.band_half_width = int(band_half_width)
        self.band_gamma = nn.Parameter(torch.zeros(2 * int(band_half_width) + 1))

    def _M(self, seg_reps, germ_reps, germ_mask, seg_tok, germ_tok, seg_reliability):
        S = F.normalize(self.seg_proj(seg_reps).float(), dim=-1)
        G = F.normalize(self.germ_proj(germ_reps).float(), dim=-1)
        scale = self.log_scale.clamp(-2.0, 3.0).exp()
        M = scale * torch.einsum("bid,bjd->bij", S, G)         # (B,S,Lg) fp32
        M = base_match_channel(M, seg_tok, germ_tok, seg_reliability,
                               self._match_weight, self.match_floor)
        return M.masked_fill(~germ_mask.unsqueeze(1), NEG)

    def _weights(self, seg_mask, seg_reliability):
        B, S = seg_mask.shape
        w = F.softmax(self.diag_bias[:S], dim=0)[None, :, None].expand(B, S, 1).clone()
        if seg_reliability is not None:
            w = w * seg_reliability.detach().clamp(0.0, 1.0).unsqueeze(-1)
        return w.masked_fill(~seg_mask.unsqueeze(-1), 0.0)     # mask padded read positions

    def forward(self, seg_reps, seg_mask, germ_reps, germ_mask,
                seg_tok=None, germ_tok=None, seg_reliability=None):
        M = self._M(seg_reps, germ_reps, germ_mask, seg_tok, germ_tok, seg_reliability)
        w = self._weights(seg_mask, seg_reliability)
        temp = self.log_temp.clamp(0.0, 4.5).exp()             # up to ~90 so the posterior can sharpen
        if self.band_half_width > 0:
            start, end = banded_start_end(M, w, self.band_gamma, self.band_half_width)
            start, end = temp * start, temp * end
        else:
            start = temp * weighted_leading_diag(M, w)
            end = temp * weighted_reverse_diag(M, w)
        start = start.masked_fill(~germ_mask, NEG)
        end = end.masked_fill(~germ_mask, NEG)
        return start, end

    def alignment_score(self, seg_reps, seg_mask, germ_reps, germ_mask,
                        seg_tok=None, germ_tok=None, seg_reliability=None):
        """Fast diagonal seed score (B,): length-normalized logsumexp over start-offset
        diagonal scores."""
        start, _ = self.forward(seg_reps, seg_mask, germ_reps, germ_mask,
                                seg_tok, germ_tok, seg_reliability)
        n_valid = germ_mask.sum(dim=-1).clamp(min=1).to(start.dtype)
        return torch.logsumexp(start, dim=-1) - torch.log(n_valid)
