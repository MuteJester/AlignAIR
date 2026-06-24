"""BandedPointerAligner: a fully-parallel, reference-conditioned germline pointer
head that replaces the sequential soft-DP. Start/end logits are weighted leading /
reverse diagonals of the score matrix M, extracted in single CUDA launches via
as_strided (no S-length recurrence). See the design spec §4."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_match import base_match_channel

NEG = -1e4


def weighted_leading_diag(M: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """out[b,o] = (Σ_i w[b,i]·M[b,i,o+i]) / Σ_i w[b,i], for o+i < Lg. (B,Lg)."""
    B, S, Lg = M.shape
    Mp = F.pad(M, (0, S))                                  # (B,S,Lg+S)
    bs, ss, es = Mp.stride()
    diag = Mp.as_strided((B, S, Lg), (bs, ss + es, es))    # diag[b,i,o] = Mp[b,i,o+i]
    num = (w * diag).sum(dim=1)                            # (B,Lg)
    return num / w.sum(dim=1).clamp(min=1e-6)


def weighted_reverse_diag(M: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """out[b,o] = (Σ_i w[b,S-1-i]·M[b,S-1-i,o-i]) / Σ_i w[b,i], for 0<=o-i<Lg. (B,Lg).
    CRITICAL: w is flipped into the flipped-row frame (wf = flip(w)), else w[i] pairs with
    row S-1-i and the head trains to the wrong coordinate once w is non-uniform (spec §4.2 B1)."""
    B, S, Lg = M.shape
    Mf = torch.flip(M, (1, 2))                             # reverse read rows AND germ cols
    Mfp = F.pad(Mf, (0, S))
    bs, ss, es = Mfp.stride()
    led = Mfp.as_strided((B, S, Lg), (bs, ss + es, es))    # led[b,i,o] = Mf[b,i,o+i]
    wf = torch.flip(w, (1,))                               # weights into led's frame
    num = torch.flip((wf * led).sum(dim=1), (1,))          # (B,Lg)
    return num / w.sum(dim=1).clamp(min=1e-6)


class BandedPointerAligner(nn.Module):
    """Drop-in for SoftDPAligner: returns (start_logits, end_logits) over germline
    positions and a fast diagonal alignment_score. Single-diagonal core (band_half_width
    extends it; see Task 8). seg_tok/germ_tok/seg_reliability enable the base-match
    channel + reliability gating (spec §4.1/§4.3)."""

    def __init__(self, d_model: int, match_floor: float = 1.0, max_len: int = 1024,
                 band_half_width: int = 0):
        super().__init__()
        self.seg_proj = nn.Linear(d_model, d_model)
        self.germ_proj = nn.Linear(d_model, d_model)
        self.log_scale = nn.Parameter(torch.tensor(1.6))       # scale ~5
        self.log_temp = nn.Parameter(torch.zeros(()))          # learned softmax temperature
        self._match_weight = nn.Parameter(torch.tensor(1.0))
        self.match_floor = float(match_floor)
        self.diag_bias = nn.Parameter(torch.zeros(max_len))    # per-position weight, init uniform
        self.band_half_width = int(band_half_width)

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
        temp = self.log_temp.clamp(0.0, 3.4).exp()
        start = temp * weighted_leading_diag(M, w)
        end = temp * weighted_reverse_diag(M, w)
        start = start.masked_fill(~germ_mask, NEG)
        end = end.masked_fill(~germ_mask, NEG)
        return start, end

    def alignment_score(self, seg_reps, seg_mask, germ_reps, germ_mask,
                        seg_tok=None, germ_tok=None, seg_reliability=None):
        """Fast diagonal seed score (B,): length-normalized logsumexp over start-offset
        diagonal scores. Replaces the soft-DP alignment_score on the training+seed path."""
        start, _ = self.forward(seg_reps, seg_mask, germ_reps, germ_mask,
                                seg_tok, germ_tok, seg_reliability)
        n_valid = germ_mask.sum(dim=-1).clamp(min=1).to(start.dtype)
        return torch.logsumexp(start, dim=-1) - torch.log(n_valid)
