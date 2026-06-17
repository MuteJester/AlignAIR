"""Query-based region decoder (DETR/Mask2Former-style, 1D).

Replaces the per-position Linear RegionTagger + argmax-contiguous-run boundary rule
with learned TYPED queries (one per region) that cross-attend to the backbone reps
and produce, per position, region mask logits, and per gene, soft start/end
distributions over positions. The boundary distributions give calibrated POSTERIORS
(and uncertainty) instead of a brittle point decode, and the query features are a
richer, context-pooled representation than a linear projection.

Region order is fixed and known for AIRR (pre<V<N1<D<N2<J<post), so we keep typed
queries (no Hungarian matching). Per-position region logits stay an 8-way score
(drop-in for RegionTagger), and gene start/end heads add the boundary posteriors.
"""
import torch
import torch.nn as nn

from .region_head import REGIONS, REGION_INDEX  # ("pad","pre","V","N1","D","N2","J","post")

GENES = ("V", "D", "J")


class RegionMaskSpanDecoder(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4):
        super().__init__()
        self.n_regions = len(REGIONS)
        self.queries = nn.Parameter(torch.randn(self.n_regions, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.q_norm = nn.LayerNorm(d_model)
        self.scale = d_model ** -0.5
        # per-gene start/end projections of the query feature, scored against reps
        self.start_proj = nn.Linear(d_model, d_model)
        self.end_proj = nn.Linear(d_model, d_model)

    def forward(self, reps: torch.Tensor, mask: torch.Tensor) -> dict:
        """reps (B,L,d), mask (B,L) bool. Returns region_logits (B,L,8) and per-gene
        start_logits/end_logits (B,L) over positions (pad positions -> -inf)."""
        B, L, d = reps.shape
        q = self.queries.unsqueeze(0).expand(B, -1, -1)            # (B, R, d)
        attended, _ = self.cross_attn(q, reps, reps, key_padding_mask=~mask)
        qf = self.q_norm(attended)                                 # (B, R, d) context-pooled per region
        # per-position region mask logits: reps . query_feat
        region_logits = torch.einsum("bld,brd->blr", reps, qf) * self.scale  # (B, L, R)
        neg = torch.finfo(region_logits.dtype).min
        region_logits = region_logits.masked_fill(~mask.unsqueeze(-1), neg)

        start_logits, end_logits = {}, {}
        for g in GENES:
            r = REGION_INDEX[g]
            sf = self.start_proj(qf[:, r])                         # (B, d)
            ef = self.end_proj(qf[:, r])
            sl = torch.einsum("bld,bd->bl", reps, sf) * self.scale  # (B, L)
            el = torch.einsum("bld,bd->bl", reps, ef) * self.scale
            start_logits[g] = sl.masked_fill(~mask, neg)
            end_logits[g] = el.masked_fill(~mask, neg)
        return {"region_logits": region_logits,
                "start_logits": start_logits, "end_logits": end_logits}
