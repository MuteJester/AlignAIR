"""Unified DNAlignAIR model: assembles orientation + backbone + region/state/scalar
heads + allele matching + germline alignment over a ReferenceSet."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.dnalignair_config import DNAlignAIRConfig
from ..nn.orientation import OrientationHead, apply_orientation
from ..nn.backbone import SequenceBackbone
from ..nn.encoder import SharedNucleotideEncoder
from ..nn.region_head import RegionTagger, REGION_INDEX
from ..nn.region_decoder import RegionMaskSpanDecoder
from ..nn.state_head import PerPositionStateHead
from ..nn.germline_encoder import GermlineEncoder
from ..nn.matching import AlleleMatchingHead
from ..nn.aligner.germline_aligner import GermlineAligner
from ..nn.aligner.soft_dp import SoftDPAligner
from ..nn.aligner.pointer import BandedPointerAligner


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
        self.shared_backbone = getattr(config, "backbone", "conv") == "shared"
        if self.shared_backbone:
            self.backbone = SharedNucleotideEncoder(
                d_model=d, n_layers=config.n_layers, nhead=config.nhead, max_len=config.max_len)
        else:
            self.backbone = SequenceBackbone(
                d_model=d, n_layers=config.n_layers, nhead=config.nhead,
                dim_feedforward=config.dim_feedforward, max_len=config.max_len)
        self.query_regions = getattr(config, "region_decoder", "linear") == "query"
        self.region_tagger = (RegionMaskSpanDecoder(d_model=d, nhead=config.nhead)
                              if self.query_regions else RegionTagger(d_model=d))
        self.state_head = PerPositionStateHead(d_model=d)
        self.germline_encoder = GermlineEncoder(embed_dim=d)
        self.matching = AlleleMatchingHead()
        # allele caller: retrieval (default) or a masked per-allele classifier head
        self.caller = getattr(config, "caller", "retrieval")
        if self.caller == "classifier":
            counts = config.allele_counts or {}
            self.classifier = nn.ModuleDict(
                {g: nn.Linear(d, counts[g]) for g in counts})
        _aligner = getattr(config, "aligner", "diagonal")
        if _aligner == "softdp":
            self.aligner = SoftDPAligner(d_model=d)
        elif _aligner == "pointer":
            self.aligner = BandedPointerAligner(
                d_model=d, max_len=config.max_len,
                band_half_width=getattr(config, "band_half_width", 0))
        else:
            self.aligner = GermlineAligner(d_model=d)
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
        reps = (self.backbone.forward_positions(canon, mask) if self.shared_backbone
                else self.backbone(canon, mask))
        if self.query_regions:
            rdec = self.region_tagger(reps, mask)
            region_logits = rdec["region_logits"]
            boundary = {"start": rdec["start_logits"], "end": rdec["end_logits"]}
        else:
            region_logits = self.region_tagger(reps)
            boundary = None
        state_logits = self.state_head(reps)
        pooled = _masked_mean(reps, mask)
        return {
            "orientation_logits": orientation_logits,
            "region_logits": region_logits,
            "boundary": boundary,
            "state_logits": state_logits,
            "noise_count": F.relu(self.noise_head(pooled)),
            "mutation_rate": torch.sigmoid(self.mutation_head(pooled)),
            "indel_count": F.relu(self.indel_head(pooled)),
            "productive": torch.sigmoid(self.productive_head(pooled)),
            "reps": reps,
            "canon_tokens": canon,
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
                "pos_tok": tok,                                                # (K, Lg) for base-match
            }
        return out

    def match_alleles(self, tokens: torch.Tensor, mask: torch.Tensor,
                      region_labels: torch.Tensor, ref_emb: dict, reps: torch.Tensor | None = None,
                      candidate_masks: dict | None = None) -> dict:
        """Per-gene allele match scores. Retrieval (default): re-encode the observed
        segment with the shared germline encoder and score (cosine) against reference
        embeddings. Classifier: masked-mean-pool the segment's BACKBONE reps and apply
        a learned per-allele head (free prototypes, sees mutated examples in training).
        Inference passes predicted region labels; training teacher-forces the true ones.

        `candidate_masks` is the dynamic GENOTYPE restriction: {gene: (K,) bool} keeping
        only alleles in the caller's genotype. Disallowed alleles are scored -inf so they
        can never be selected — this is how a subset/novel genotype conditions each call."""
        match = {}
        for gene, emb in ref_emb.items():
            cmask = candidate_masks.get(gene) if candidate_masks else None
            if self.caller == "classifier":
                seg, seg_mask = extract_segment(reps, mask, region_labels, gene)
                m = seg_mask.unsqueeze(-1).to(seg.dtype)
                pooled = (seg * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
                logits = self.classifier[gene](pooled)              # (B, K) logits
                if cmask is not None:
                    logits = logits.masked_fill(~cmask.to(logits.device).unsqueeze(0), float("-inf"))
                match[gene] = logits
            else:
                seg_tok, seg_mask = extract_segment_tokens(tokens, mask, region_labels, gene)
                query = self.germline_encoder(seg_tok, seg_mask)    # (B, d) normalized
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

    def germline_coords(self, seg_reps, seg_mask, germ_reps, germ_mask,
                        seg_tok=None, germ_tok=None, seg_reliability=None):
        """Align a gene's segment reps to a chosen allele's per-position germline reps.
        seg_tok/germ_tok/seg_reliability enable the pointer aligner's base-match + SHM
        reliability gating; aligners that don't accept them fall back transparently."""
        try:
            return self.aligner(seg_reps, seg_mask, germ_reps, germ_mask,
                                seg_tok=seg_tok, germ_tok=germ_tok,
                                seg_reliability=seg_reliability)
        except TypeError:
            return self.aligner(seg_reps, seg_mask, germ_reps, germ_mask)
