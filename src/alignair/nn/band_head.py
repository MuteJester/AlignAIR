"""Structural diagonal-offset band head (the "seed" of seed-and-extend).

Predicts a distribution over germline START offsets for a read segment, from
representation-INDEPENDENT raw base-match (dominant) + learned token cosine
(additive), so it survives the later encoder refactor. Trained with offset
cross-entropy; band recall is the decision metric, not the loss (spec §4.4)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointer_aligner import weighted_leading_diag

NEG = -1e4


def base_match_matrix(seg_tok: torch.Tensor, germ_tok: torch.Tensor) -> torch.Tensor:
    """Raw +1 match / -1 mismatch / 0 non-ACGT base-match grid (B,S,Lg)."""
    st, gt = seg_tok.unsqueeze(2), germ_tok.unsqueeze(1)            # (B,S,1),(B,1,Lg)
    real = (st >= 1) & (st <= 4) & (gt >= 1) & (gt <= 4)
    return real.float() * (2.0 * (st == gt).float() - 1.0)


class BandHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj_s = nn.Linear(d_model, d_model)
        self.proj_g = nn.Linear(d_model, d_model)
        self.log_scale = nn.Parameter(torch.tensor(1.6))           # cosine scale ~5
        # base-match weight DOMINATES at init (representation-independent); cosine is a
        # small additive correction so the head survives the encoder refactor.
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
