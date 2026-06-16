"""Germline coordinate aligner: cross-attention from the observed segment's
endpoints to the matched allele's per-position germline reps -> exact trims."""
import torch
import torch.nn as nn


class GermlineAligner(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q_start = nn.Linear(d_model, d_model)
        self.q_end = nn.Linear(d_model, d_model)

    def forward(self, seg_reps: torch.Tensor, seg_mask: torch.Tensor,
                germ_reps: torch.Tensor, germ_mask: torch.Tensor):
        """seg_reps (B,Ls,d) right-padded; germ_reps (B,Lg,d). Returns
        (start_logits (B,Lg), end_logits (B,Lg))."""
        B = seg_reps.shape[0]
        lengths = seg_mask.sum(dim=1)
        first = seg_reps[:, 0]                                  # first valid position
        last = seg_reps[torch.arange(B, device=seg_reps.device),
                        (lengths - 1).clamp(min=0)]             # last valid position
        qs = self.q_start(first)                                # (B,d)
        qe = self.q_end(last)
        start_logits = torch.einsum("bd,bld->bl", qs, germ_reps)  # (B,Lg)
        end_logits = torch.einsum("bd,bld->bl", qe, germ_reps)
        neg = torch.finfo(start_logits.dtype).min
        start_logits = start_logits.masked_fill(~germ_mask, neg)
        end_logits = end_logits.masked_fill(~germ_mask, neg)
        return start_logits, end_logits


def decode_germline_coords(start_logits: torch.Tensor, end_logits: torch.Tensor):
    """Exact germline_start (argmax) and germline_end (argmax+1, end-exclusive)."""
    gs = start_logits.argmax(dim=-1)
    ge = end_logits.argmax(dim=-1) + 1
    return gs, ge
