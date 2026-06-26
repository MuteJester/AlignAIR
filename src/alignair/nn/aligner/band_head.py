"""Structural diagonal-offset band head (the "seed" of seed-and-extend).

Predicts a distribution over germline START offsets for a read segment, from
representation-INDEPENDENT raw base-match (dominant) + learned token cosine
(additive), so it survives the later encoder refactor. Trained with offset
cross-entropy; band recall is the decision metric, not the loss (spec §4.4).

Fail-open uses `peak_evidence` (the OVERLAP FRACTION at the predicted offset): a real
alignment fits the germline (~1.0), a spurious low-overlap peak does not (~0.03), so
signal-absent reads route to the full DP instead of committing a wrong narrow band.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_match import base_match_matrix
from .diagonal_ops import weighted_leading_diag

NEG = -1e4


class BandHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj_s = nn.Linear(d_model, d_model)
        self.proj_g = nn.Linear(d_model, d_model)
        self.log_scale = nn.Parameter(torch.tensor(1.6))           # cosine scale ~5
        # base-match (representation-INDEPENDENT) DOMINATES at init; cosine is a small
        # additive correction so the head survives the encoder refactor.
        self.w_bm = nn.Parameter(torch.tensor(1.0))
        self.w_cos = nn.Parameter(torch.tensor(0.1))
        self.log_temp = nn.Parameter(torch.tensor(1.6))            # sharpen the offset posterior

    def forward(self, seg_reps, seg_mask, germ_reps, germ_mask, seg_tok, germ_tok):
        w = seg_mask.float().unsqueeze(-1)                          # (B,S,1) mask pad rows
        bm = weighted_leading_diag(base_match_matrix(seg_tok, germ_tok).float(), w)  # (B,Lg)
        Sn = F.normalize(self.proj_s(seg_reps).float(), dim=-1)
        Gn = F.normalize(self.proj_g(germ_reps).float(), dim=-1)
        cos_M = self.log_scale.clamp(-2.0, 3.0).exp() * torch.einsum("bid,bjd->bij", Sn, Gn)
        cos = weighted_leading_diag(cos_M, w)                       # (B,Lg)
        temp = self.log_temp.clamp(0.0, 4.5).exp()
        logit = temp * (F.softplus(self.w_bm) * bm + self.w_cos * cos)
        return logit.masked_fill(~germ_mask, NEG)


def band_offset_loss(offset_logits: torch.Tensor, true_start: torch.Tensor) -> torch.Tensor:
    """Offset cross-entropy of the start-offset posterior against the true germline_start."""
    Lg = offset_logits.shape[-1]
    tgt = true_start.clamp(min=0, max=Lg - 1)
    return F.cross_entropy(offset_logits, tgt)


def peak_evidence(offset_logits, seg_tok, germ_tok, seg_mask) -> torch.Tensor:
    """OVERLAP FRACTION at the predicted (argmax) offset (B,): how much of the read segment
    actually lands on the germline along that diagonal, as a fraction of the segment length.
    A real alignment fits (~1.0); a SPURIOUS low-overlap peak near the germline end covers only
    a few positions (~0.03). The physical separator of signal-present (commit) vs signal-absent
    (fail open) reads — overlap-NORMALIZED mean does NOT separate them (a spurious 8/8-match peak
    also has mean ~1.0), so we use the overlap COUNT relative to the segment length."""
    bm = base_match_matrix(seg_tok, germ_tok)                       # (B,S,Lg) +1/-1/0 (0 = pad/out)
    B, S, Lg = bm.shape
    Mp = F.pad(bm, (0, S)); bs, ss, es = Mp.stride()
    diag = Mp.as_strided((B, S, Lg), (bs, ss + es, es))            # diag[b,i,o]=bm[b,i,o+i]
    o = offset_logits.argmax(dim=-1)                                # (B,)
    ar = torch.arange(B, device=bm.device)
    col = diag[ar, :, o]                                            # (B,S) base-match along pred diag
    vm = seg_mask.float()
    overlap = (vm * (col != 0).float()).sum(dim=1)                 # read positions landing on germline
    return overlap / vm.sum(dim=1).clamp(min=1.0)                  # (B,) overlap fraction in [0,1]
