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


def kmer_seed_counts(seg_tok: torch.Tensor, germ_tok: torch.Tensor,
                     seg_mask: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Per-offset count of length-k EXACT-match windows along the o-diagonal (B,Lg). A
    contiguity signal: robust where scattered base-match is ambiguous (heavy-SHM / 3'-end
    label noise), because a k-mer seed anchors the true diagonal between mutations."""
    bm = base_match_matrix(seg_tok, germ_tok)                       # (B,S,Lg) +1/-1/0
    B, S, Lg = bm.shape
    Mp = torch.nn.functional.pad(bm, (0, S))
    bs, ss, es = Mp.stride()
    diag = Mp.as_strided((B, S, Lg), (bs, ss + es, es))             # diag[b,i,o] = bm[b,i,o+i]
    match = (diag >= 0.5).float() * seg_mask.float().unsqueeze(-1)  # (B,S,Lg) masked exact-match
    if S < k:
        return match.sum(dim=1) * 0.0
    m = match.permute(0, 2, 1).reshape(B * Lg, 1, S)               # conv over the read axis
    kernel = torch.ones(1, 1, k, device=bm.device, dtype=m.dtype)
    run = torch.nn.functional.conv1d(m, kernel)                    # (B*Lg,1,S-k+1) window sums
    return (run >= k - 0.5).float().sum(dim=-1).reshape(B, Lg)     # count of full-k windows


class BandHead(nn.Module):
    def __init__(self, d_model: int, kmer_k: int = 5):
        super().__init__()
        self.proj_s = nn.Linear(d_model, d_model)
        self.proj_g = nn.Linear(d_model, d_model)
        self.log_scale = nn.Parameter(torch.tensor(1.6))           # cosine scale ~5
        self.kmer_k = int(kmer_k)
        # base-match + k-mer (representation-INDEPENDENT) DOMINATE at init; cosine is a
        # small additive correction so the head survives the encoder refactor. The k-mer
        # contiguity feature anchors the diagonal where scattered base-match is ambiguous.
        self.w_bm = nn.Parameter(torch.tensor(1.0))
        self.w_kmer = nn.Parameter(torch.tensor(0.5))
        self.w_cos = nn.Parameter(torch.tensor(0.1))
        self.log_temp = nn.Parameter(torch.tensor(1.6))            # sharpen the offset posterior
        # confidence head over posterior shape features -> P(band covers truth); enables the
        # fail-open safety net (spec rule 3) so confidently-WRONG reads route to full DP.
        self.conf = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, seg_reps, seg_mask, germ_reps, germ_mask, seg_tok, germ_tok):
        w = seg_mask.float().unsqueeze(-1)                          # (B,S,1) mask pad rows
        bm = weighted_leading_diag(base_match_matrix(seg_tok, germ_tok).float(), w)  # (B,Lg)
        Sn = F.normalize(self.proj_s(seg_reps).float(), dim=-1)
        Gn = F.normalize(self.proj_g(germ_reps).float(), dim=-1)
        cos_M = self.log_scale.clamp(-2.0, 3.0).exp() * torch.einsum("bid,bjd->bij", Sn, Gn)
        cos = weighted_leading_diag(cos_M, w)                       # (B,Lg)
        km = kmer_seed_counts(seg_tok, germ_tok, seg_mask, self.kmer_k)
        seg_len = seg_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        km = km / seg_len                                           # length-normalize the count
        temp = self.log_temp.clamp(0.0, 4.5).exp()
        logit = temp * (F.softplus(self.w_bm) * bm + F.softplus(self.w_kmer) * km
                        + self.w_cos * cos)
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


def _conf_features(offset_logits: torch.Tensor) -> torch.Tensor:
    """(B,3) posterior-shape features: max logit, NMS top-2 margin, entropy."""
    p = torch.softmax(offset_logits.float(), dim=-1)
    ent = -(p.clamp_min(1e-9) * p.clamp_min(1e-9).log()).sum(dim=-1)
    return torch.stack([offset_logits.max(dim=-1).values, _nms_top2_margin(offset_logits), ent], dim=-1)


# attach confidence_logit as a method (uses self.conf over the posterior-shape features)
def _confidence_logit(self, offset_logits: torch.Tensor) -> torch.Tensor:
    """P(band covers truth) logit per read (B,). sigmoid(.) >= threshold -> commit, else fail open."""
    return self.conf(_conf_features(offset_logits)).squeeze(-1)


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
