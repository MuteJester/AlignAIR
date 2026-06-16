"""Unified DNAlignAIR model: assembles orientation + backbone + region/state/scalar
heads + allele matching + germline alignment over a ReferenceSet."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.dnalignair_config import DNAlignAIRConfig
from ..nn.orientation import OrientationHead
from ..nn.backbone import SequenceBackbone
from ..nn.region_head import RegionTagger, REGION_INDEX
from ..nn.state_head import PerPositionStateHead
from ..nn.germline_encoder import GermlineEncoder
from ..nn.matching import AlleleMatchingHead
from ..nn.germline_aligner import GermlineAligner


def _masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def extract_segment(reps: torch.Tensor, mask: torch.Tensor,
                    region_labels: torch.Tensor, gene: str):
    """Gather each sample's positions tagged `gene` into a left-aligned padded
    (B, Smax, d) tensor + (B, Smax) bool mask. Smax is the batch's longest gene run."""
    gid = REGION_INDEX[gene]
    sel = (region_labels == gid) & mask                 # (B, L)
    counts = sel.sum(dim=1)                              # (B,)
    smax = int(counts.max().item()) if counts.numel() else 0
    B, L, d = reps.shape
    smax = max(smax, 1)
    seg = reps.new_zeros(B, smax, d)
    seg_mask = torch.zeros(B, smax, dtype=torch.bool, device=reps.device)
    for b in range(B):
        idx = sel[b].nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n:
            seg[b, :n] = reps[b, idx]
            seg_mask[b, :n] = True
    return seg, seg_mask


def extract_segment_tokens(tokens: torch.Tensor, mask: torch.Tensor,
                           region_labels: torch.Tensor, gene: str):
    """Gather each sample's `gene`-tagged token ids into a left-aligned padded
    (B, Smax) long tensor + (B, Smax) bool mask (for re-encoding the segment)."""
    gid = REGION_INDEX[gene]
    sel = (region_labels == gid) & mask
    counts = sel.sum(dim=1)
    smax = max(int(counts.max().item()) if counts.numel() else 0, 1)
    B = tokens.shape[0]
    seg = tokens.new_zeros(B, smax)
    seg_mask = torch.zeros(B, smax, dtype=torch.bool, device=tokens.device)
    for b in range(B):
        idx = sel[b].nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n:
            seg[b, :n] = tokens[b, idx]
            seg_mask[b, :n] = True
    return seg, seg_mask


class DNAlignAIR(nn.Module):
    def __init__(self, config: DNAlignAIRConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.orientation_head = OrientationHead(d=config.orientation_dim)
        self.backbone = SequenceBackbone(
            d_model=d, n_layers=config.n_layers, nhead=config.nhead,
            dim_feedforward=config.dim_feedforward, max_len=config.max_len)
        self.region_tagger = RegionTagger(d_model=d)
        self.state_head = PerPositionStateHead(d_model=d)
        self.germline_encoder = GermlineEncoder(embed_dim=d)
        self.matching = AlleleMatchingHead()
        self.aligner = GermlineAligner(d_model=d)
        self.noise_head = nn.Linear(d, 1)
        self.mutation_head = nn.Linear(d, 1)
        self.indel_head = nn.Linear(d, 1)
        self.productive_head = nn.Linear(d, 1)

    def forward_dense(self, tokens: torch.Tensor, mask: torch.Tensor) -> dict:
        orientation_logits = self.orientation_head(tokens, mask)
        reps = self.backbone(tokens, mask)
        region_logits = self.region_tagger(reps)
        state_logits = self.state_head(reps)
        pooled = _masked_mean(reps, mask)
        return {
            "orientation_logits": orientation_logits,
            "region_logits": region_logits,
            "state_logits": state_logits,
            "noise_count": F.relu(self.noise_head(pooled)),
            "mutation_rate": torch.sigmoid(self.mutation_head(pooled)),
            "indel_count": F.relu(self.indel_head(pooled)),
            "productive": torch.sigmoid(self.productive_head(pooled)),
            "reps": reps,
        }

    def encode_reference(self, reference_set) -> dict:
        """Encode each gene's germline sequences -> pooled embeddings + per-position reps (cached)."""
        from ..data.tokenizer import pad_tokenize
        device = next(self.parameters()).device
        out = {}
        for gene, ref in reference_set.genes.items():
            tok, msk = pad_tokenize(ref.sequences)
            tok, msk = tok.to(device), msk.to(device)
            out[gene] = {
                "embeddings": self.germline_encoder(tok, msk),                 # (K, d) normalized
                "pos_reps": self.germline_encoder.forward_positions(tok, msk),  # (K, Lg, d)
                "pos_mask": msk,                                               # (K, Lg)
            }
        return out

    def match_alleles(self, tokens: torch.Tensor, mask: torch.Tensor,
                      region_labels: torch.Tensor, ref_emb: dict) -> dict:
        """Per-gene allele match scores: re-encode each gene's observed segment with
        the (shared) germline encoder and score against the reference embeddings in the
        same space. Inference passes predicted region labels; training teacher-forces
        the true labels."""
        match = {}
        for gene, emb in ref_emb.items():
            seg_tok, seg_mask = extract_segment_tokens(tokens, mask, region_labels, gene)
            query = self.germline_encoder(seg_tok, seg_mask)        # (B, d) normalized
            match[gene] = self.matching(query, emb["embeddings"])
        return match

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, ref_emb: dict) -> dict:
        out = self.forward_dense(tokens, mask)
        region_labels = out["region_logits"].argmax(dim=-1)
        out["match"] = self.match_alleles(tokens, mask, region_labels, ref_emb)
        return out

    def germline_coords(self, seg_reps, seg_mask, germ_reps, germ_mask):
        """Align a gene's segment reps to a chosen allele's per-position germline reps."""
        return self.aligner(seg_reps, seg_mask, germ_reps, germ_mask)
