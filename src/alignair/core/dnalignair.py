"""Unified DNAlignAIR model (single architecture path).

Assembles orientation + the shared nucleotide encoder + region/state/scalar heads +
retrieval allele matching + seed-and-extend germline alignment over a ReferenceSet.

There is exactly ONE encoder (SharedNucleotideEncoder) for both reads and germline
references (token-type embedding distinguishes them), so there is no separate germline
encoder and no double-encode. Allele identity is NEVER memorised in weights: calls are
made by retrieval (siamese cosine of the pooled read segment vs germline embeddings), and
germline coordinates come from the structural band head (center) + the exact banded DP
(SeedExtendAligner) — the dynamic-genotype property.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.dnalignair_config import DNAlignAIRConfig
from ..nn.heads.orientation import OrientationHead, apply_orientation
from ..nn.encoder.shared import SharedNucleotideEncoder
from ..nn.heads.region import RegionTagger, REGION_INDEX
from ..nn.heads.state import PerPositionStateHead
from ..nn.heads.matching import AlleleMatchingHead
from ..nn.aligner.banded_dp import SeedExtendAligner
from ..nn.aligner.band_head import BandHead


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
        # ONE shared encoder for reads AND references (token_type embedding distinguishes them).
        self.backbone = SharedNucleotideEncoder(
            d_model=d, n_layers=config.n_layers, nhead=config.nhead, max_len=config.max_len)
        self.region_tagger = RegionTagger(d_model=d)
        self.state_head = PerPositionStateHead(d_model=d)
        # retrieval caller: siamese cosine of the pooled read segment vs germline embeddings
        self.matching = AlleleMatchingHead()
        # seed-and-extend germline aligner: BandHead predicts the band center, the exact banded
        # soft-DP (SeedExtendAligner) extends within the +-band_width window.
        self.aligner = SeedExtendAligner(d_model=d)
        self.band_head = BandHead(d_model=d)
        self.noise_head = nn.Linear(d, 1)
        self.mutation_head = nn.Linear(d, 1)
        self.indel_head = nn.Linear(d, 1)
        self.productive_head = nn.Linear(d, 1)

    def forward_dense(self, tokens: torch.Tensor, mask: torch.Tensor,
                      orientation_ids: torch.Tensor | None = None) -> dict:
        # Detect orientation on the observed (possibly reverse/complemented) tokens,
        # then CANONICALIZE to forward before the backbone (transforms are involutions
        # so re-applying recovers forward). Teacher-force the true orientation when
        # given (training); otherwise use the predicted argmax (inference).
        orientation_logits = self.orientation_head(tokens, mask)
        t = orientation_ids if orientation_ids is not None else orientation_logits.argmax(dim=-1)
        canon = apply_orientation(tokens, mask, t)
        reps = self.backbone.forward_positions(canon, mask)
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
            "canon_tokens": canon,
        }

    def _germ_encode_pooled(self, tok, msk):
        """Pooled normalized germline embedding via the SHARED encoder (token_type=GERMLINE)."""
        return self.backbone(tok, msk, token_type=SharedNucleotideEncoder.GERMLINE)

    def _germ_encode_positions(self, tok, msk):
        """Per-position germline reps via the SHARED encoder (token_type=GERMLINE)."""
        return self.backbone.forward_positions(tok, msk, token_type=SharedNucleotideEncoder.GERMLINE)

    def encode_reference(self, reference_set) -> dict:
        """Encode each gene's germline sequences -> pooled embeddings + per-position reps (cached)."""
        from ..data.tokenizer import pad_tokenize
        device = next(self.parameters()).device
        out = {}
        for gene, ref in reference_set.genes.items():
            tok, msk = pad_tokenize(ref.sequences)
            tok, msk = tok.to(device), msk.to(device)
            out[gene] = {
                "embeddings": self._germ_encode_pooled(tok, msk),               # (K, d) normalized
                "pos_reps": self._germ_encode_positions(tok, msk),              # (K, Lg, d)
                "pos_mask": msk,                                               # (K, Lg)
                "pos_tok": tok,                                                # (K, Lg) for base-match
            }
        return out

    def match_alleles(self, tokens: torch.Tensor, mask: torch.Tensor,
                      region_labels: torch.Tensor, ref_emb: dict, reps: torch.Tensor | None = None,
                      candidate_masks: dict | None = None) -> dict:
        """Per-gene retrieval allele match scores. Read the observed segment OFF the backbone
        reps (no re-encode) and project it through the SAME pooling head as the germline
        embeddings (siamese cosine) — allele identity is never memorised in weights.
        Inference passes predicted region labels; training teacher-forces the true ones.

        `candidate_masks` is the dynamic GENOTYPE restriction: {gene: (K,) bool} keeping only
        alleles in the caller's genotype. Disallowed alleles are scored -inf so they can never
        be selected — this is how a subset/novel genotype conditions each call."""
        match = {}
        for gene, emb in ref_emb.items():
            cmask = candidate_masks.get(gene) if candidate_masks else None
            seg, seg_mask = extract_segment(reps, mask, region_labels, gene)
            m = seg_mask.unsqueeze(-1).to(seg.dtype)
            pooled = (seg * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            query = F.normalize(self.backbone.proj(pooled), dim=-1)       # (B, d) normalized
            cm = cmask.to(query.device) if cmask is not None else None
            match[gene] = self.matching(query, emb["embeddings"], candidate_mask=cm)
        return match

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, ref_emb: dict,
                orientation_ids: torch.Tensor | None = None,
                candidate_masks: dict | None = None) -> dict:
        out = self.forward_dense(tokens, mask, orientation_ids)
        region_labels = out["region_logits"].argmax(dim=-1)
        # match on the canonicalized tokens, same frame as the region prediction
        out["match"] = self.match_alleles(out["canon_tokens"], mask, region_labels, ref_emb,
                                          reps=out["reps"], candidate_masks=candidate_masks)
        return out

    def band_logits(self, seg_reps, seg_mask, germ_reps, germ_mask, seg_tok, germ_tok):
        """The structural band head's start-offset posterior (B,Lg), used both to place the DP
        band (argmax center) AND to train the band head (band_offset_loss against the true start)."""
        return self.band_head(seg_reps, seg_mask, germ_reps, germ_mask, seg_tok, germ_tok)

    def germline_coords(self, seg_reps, seg_mask, germ_reps, germ_mask,
                        seg_tok=None, germ_tok=None, seg_reliability=None):
        """Align a gene's segment reps to a chosen allele's per-position germline reps: the band
        head places a +-band_width band, and the exact banded DP decodes germline start/end within
        it. seg_tok/germ_tok drive the base-match channel; seg_reliability gates SHM positions."""
        center = self.band_logits(seg_reps, seg_mask, germ_reps, germ_mask, seg_tok, germ_tok).argmax(dim=-1)
        w = getattr(self.config, "band_width", 16)
        return self.aligner(seg_reps, seg_mask, germ_reps, germ_mask, center, w,
                            seg_tok=seg_tok, germ_tok=germ_tok, seg_reliability=seg_reliability)
