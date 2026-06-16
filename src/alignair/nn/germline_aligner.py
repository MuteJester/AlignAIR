"""Germline coordinate aligner.

Aligns the observed segment to the matched allele's per-position germline reps by
**diagonal correlation**: a candidate germline start offset ``o`` places segment
position ``i`` against germline position ``o+i``; its score is the summed
similarity along that diagonal. The start distribution is the per-offset score;
the end distribution is the symmetric correlation anchored at the segment's last
position. Argmax gives exact germline start/end. Aggregating over the whole
segment (interior positions with full context) is robust where single-endpoint
matching is ambiguous.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GermlineAligner(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.seg_proj = nn.Linear(d_model, d_model)
        self.germ_proj = nn.Linear(d_model, d_model)
        # Learnable, clamped temperature on a *cosine* correlation. Cosine bounds
        # each per-position term to [-1, 1] so the diagonal sum cannot run away as
        # the projections grow (raw dot products summed over a ~300bp segment
        # otherwise drive a positive-feedback explosion to 1e37 -> inf in the CE).
        self.log_temp = nn.Parameter(torch.zeros(()))

    def forward(self, seg_reps: torch.Tensor, seg_mask: torch.Tensor,
                germ_reps: torch.Tensor, germ_mask: torch.Tensor):
        """seg_reps (B,Ls,d) right-padded; germ_reps (B,Lg,d). Returns
        (start_logits (B,Lg), end_logits (B,Lg))."""
        S = F.normalize(self.seg_proj(seg_reps), dim=-1)
        G = F.normalize(self.germ_proj(germ_reps), dim=-1)
        M = torch.einsum("bid,bjd->bij", S, G)                  # (B, Ls, Lg), cosine in [-1,1]
        valid = seg_mask.unsqueeze(2) & germ_mask.unsqueeze(1)
        M = M.masked_fill(~valid, 0.0)

        B, Ls, Lg = M.shape
        device = M.device
        i = torch.arange(Ls, device=device)
        neg = -1e4  # moderate mask: a masked target keeps CE bounded (finfo.min -> ~1e37)
        sums = M.new_zeros((B, Lg))

        for o in range(Lg):
            # start offset o: segment pos i aligns to germline o+i
            j = o + i
            ok = j < Lg
            if ok.any():
                sums[:, o] = M[:, i[ok], j[ok]].sum(dim=1)
        # Mean cosine along each diagonal (divide by the segment length, a constant
        # per sample) keeps the logit scale ~[-temp, temp] regardless of segment
        # length, so fragments and full reads are comparable and CE never spikes to
        # ~temp*Ls. Partial-overlap offsets keep their smaller raw sum -> downweighted.
        seg_len_d = seg_mask.sum(dim=1).clamp(min=1).unsqueeze(1)
        temp = self.log_temp.clamp(0.0, 3.4).exp()              # temp in [1, ~30]
        start_logits = (temp * sums / seg_len_d).masked_fill(~germ_mask, neg)

        # Contiguous alignment (no indels here): the last aligned germline position
        # is start + (seg_len - 1), so end_logits is the start distribution shifted
        # right per-sample by (seg_len - 1). Independent end prediction (for indels)
        # is a later refinement.
        seg_len = seg_mask.sum(dim=1)
        shift = (seg_len - 1).clamp(min=0)
        ar = torch.arange(Lg, device=device).unsqueeze(0)        # (1,Lg)
        src = ar - shift.unsqueeze(1)                            # (B,Lg) source offset
        valid_src = (src >= 0) & (src < Lg)
        end_logits = torch.gather(start_logits, 1, src.clamp(0, Lg - 1))
        end_logits = end_logits.masked_fill(~valid_src, neg).masked_fill(~germ_mask, neg)
        return start_logits, end_logits


def decode_germline_coords(start_logits: torch.Tensor, end_logits: torch.Tensor):
    """Exact germline_start (argmax) and germline_end (argmax+1, end-exclusive)."""
    gs = start_logits.argmax(dim=-1)
    ge = end_logits.argmax(dim=-1) + 1
    return gs, ge
