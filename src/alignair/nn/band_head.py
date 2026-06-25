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
        # confidence head over posterior-shape features + ABSOLUTE base-match peak evidence
        # (4th feature) -> P(band covers truth). The absolute evidence is what catches the
        # SIGNAL-ABSENT reads (spurious low-overlap peaks have a tiny absolute base-match sum
        # even when their normalized mean is the argmax) so they FAIL OPEN to full DP.
        self.conf = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 1))

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


def _nms_top2_margin(offset_logits: torch.Tensor) -> torch.Tensor:
    """logit gap between the two DISTINCT (NMS-spaced) peaks. A confident localization has a
    dominant single peak (large margin); a confused read has competing peaks (small margin)."""
    top = offset_logits.argmax(dim=-1)                                  # (B,)
    Lg = offset_logits.shape[-1]
    pos = torch.arange(Lg, device=offset_logits.device)
    near = (pos.unsqueeze(0) - top.unsqueeze(1)).abs() <= 4             # suppress around the peak
    second = offset_logits.masked_fill(near, -1e9).max(dim=-1).values
    return offset_logits.max(dim=-1).values - second


def peak_evidence(offset_logits, seg_tok, germ_tok, seg_mask) -> torch.Tensor:
    """ABSOLUTE base-match evidence at the predicted (argmax) offset (B,): the mean +1/-1
    base-match over the OVERLAPPING read positions of that diagonal. A real alignment scores
    high (~0.5-1.0 over a long overlap); a spurious low-overlap peak scores near 0. This is the
    feature that separates signal-present (commit) from signal-absent (fail open) reads."""
    bm = base_match_matrix(seg_tok, germ_tok)                       # (B,S,Lg)
    B, S, Lg = bm.shape
    Mp = torch.nn.functional.pad(bm, (0, S)); bs, ss, es = Mp.stride()
    diag = Mp.as_strided((B, S, Lg), (bs, ss + es, es))            # diag[b,i,o]=bm[b,i,o+i]
    valid = seg_mask.float().unsqueeze(-1)                          # (B,S,1)
    o = offset_logits.argmax(dim=-1)                                # (B,)
    ar = torch.arange(B, device=bm.device)
    col = diag[ar, :, o]                                            # (B,S) base-match along pred diag
    vm = valid[ar, :, 0]
    overlap = (vm * (col != 0).float())                            # positions that actually overlap germline
    num = (col * vm).sum(dim=1)
    return num / overlap.sum(dim=1).clamp(min=1.0)                  # (B,) mean base-match over overlap


def _conf_features(offset_logits, evidence) -> torch.Tensor:
    """(B,4): max logit, NMS top-2 margin, entropy, ABSOLUTE base-match evidence at the peak."""
    p = torch.softmax(offset_logits.float(), dim=-1)
    ent = -(p.clamp_min(1e-9) * p.clamp_min(1e-9).log()).sum(dim=-1)
    return torch.stack([offset_logits.max(dim=-1).values, _nms_top2_margin(offset_logits),
                        ent, evidence], dim=-1)


# attach confidence_logit as a method (posterior-shape features + absolute peak evidence)
def _confidence_logit(self, offset_logits, seg_tok, germ_tok, seg_mask) -> torch.Tensor:
    """P(band covers truth) logit per read (B,). sigmoid(.) >= threshold -> commit, else fail open.
    Uses the absolute base-match evidence so signal-absent (spurious-peak) reads fail open."""
    ev = peak_evidence(offset_logits, seg_tok, germ_tok, seg_mask)
    return self.conf(_conf_features(offset_logits, ev)).squeeze(-1)


BandHead.confidence_logit = _confidence_logit


def band_calibration_loss(conf_logit: torch.Tensor, offset_logits: torch.Tensor,
                          true_start: torch.Tensor, w: int, m: int = 2) -> torch.Tensor:
    """BCE training the confidence to predict whether the top-m band actually COVERS the true
    start (within w). Detaches the offset logits used for the target so calibration trains the
    confidence head, not the offset posterior."""
    from ..gym.instrument.band_metrics import _topm_centers
    centers = _topm_centers(offset_logits.detach(), w, m)               # (B,m)
    covered = ((centers - true_start.unsqueeze(1)).abs() <= w).any(dim=1).float()
    return F.binary_cross_entropy_with_logits(conf_logit, covered)
