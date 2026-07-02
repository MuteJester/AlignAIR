"""Sequential banded exact soft-DP (the "extend" of seed-and-extend). Reuses the existing,
correct soft_dp_end_logits recurrence (Hm + Ins + logcumsumexp germline-skip) on a band-masked
score matrix, so it is the SAME math as the full soft-DP, restricted to a ±w window the seed
places. Emits start/end coordinate posteriors AND the log-partition as the final allele reader
score. This is the reference the fused kernel (build step 5) will validate against."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .soft_dp import soft_dp_end_logits, _reverse_valid_2d, NEG
from .base_match import base_match_channel


def band_mask_scores(M: torch.Tensor, band_center: torch.Tensor, w: int) -> torch.Tensor:
    """Mask the score matrix to a ±w band: keep column j for read row i iff
    |j - (band_center[b] + i)| <= w; else set to NEG. band_center is the predicted germline
    start offset per read (an INPUT here; oracle in tests, the seed head at deployment)."""
    B, S, Lg = M.shape
    i = torch.arange(S, device=M.device)[None, :, None]            # (1,S,1)
    j = torch.arange(Lg, device=M.device)[None, None, :]           # (1,1,Lg)
    center = band_center.view(-1, 1, 1) + i                        # (B,S,1)
    in_band = (j - center).abs() <= w
    return M.masked_fill(~in_band, NEG)


class SeedExtendAligner(nn.Module):
    """Banded exact soft-DP over the shared soft_dp_end_logits recurrence. Base-match and SHM
    reliability are first-class inputs; the only difference from the full DP is the band mask.
    The band center is an INPUT (oracle in tests; the structural band head at deployment)."""

    def __init__(self, d_model: int, match_floor: float = 1.0):
        super().__init__()
        self.seg_proj = nn.Linear(d_model, d_model)
        self.germ_proj = nn.Linear(d_model, d_model)
        self.log_scale = nn.Parameter(torch.tensor(1.6))
        self._gap_open = nn.Parameter(torch.tensor(3.0))
        self._gap_extend = nn.Parameter(torch.tensor(2.0))
        self._del_gap = nn.Parameter(torch.tensor(3.0))
        self._match_weight = nn.Parameter(torch.tensor(1.0))
        self.match_floor = float(match_floor)

    def _scores(self, seg_reps, germ_reps, seg_tok, germ_tok, seg_reliability):
        S = F.normalize(self.seg_proj(seg_reps), dim=-1)
        G = F.normalize(self.germ_proj(germ_reps), dim=-1)
        M = self.log_scale.clamp(-2.0, 3.0).exp() * torch.einsum("bid,bjd->bij", S, G)
        return base_match_channel(M, seg_tok, germ_tok, seg_reliability,
                                  self._match_weight, self.match_floor)

    def _gaps(self):
        return (-F.softplus(self._gap_open), -F.softplus(self._gap_extend),
                -F.softplus(self._del_gap))

    def forward(self, seg_reps, seg_mask, germ_reps, germ_mask, band_center, w,
                seg_tok=None, germ_tok=None, seg_reliability=None):
        go, ge, dg = self._gaps()
        M = self._scores(seg_reps, germ_reps, seg_tok, germ_tok, seg_reliability)
        M = band_mask_scores(M, band_center, w)
        end = soft_dp_end_logits(M, seg_mask, germ_mask, go, ge, dg)
        seg_len = seg_mask.sum(dim=1); germ_len = germ_mask.sum(dim=1)
        Mr = _reverse_valid_2d(M.transpose(1, 2), germ_len).transpose(1, 2)
        Mr = _reverse_valid_2d(Mr, seg_len)
        end_rev = soft_dp_end_logits(Mr, seg_mask, germ_mask, go, ge, dg)
        start = _reverse_valid_2d(end_rev, germ_len)
        return start.masked_fill(~germ_mask, NEG), end

    def alignment_score(self, seg_reps, seg_mask, germ_reps, germ_mask, band_center, w,
                        seg_tok=None, germ_tok=None, seg_reliability=None):
        """Final allele reader = banded soft-DP log-partition (length-normalized). Rule 1:
        the allele score is the exact DP log-partition, never MaxSim."""
        go, ge, dg = self._gaps()
        M = self._scores(seg_reps, germ_reps, seg_tok, germ_tok, seg_reliability)
        M = band_mask_scores(M, band_center, w)
        end = soft_dp_end_logits(M, seg_mask, germ_mask, go, ge, dg)
        n_valid = germ_mask.sum(dim=-1).clamp(min=1).to(end.dtype)
        return torch.logsumexp(end, dim=-1) - torch.log(n_valid)
