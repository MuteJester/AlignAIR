from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.encoder.shared import SharedNucleotideEncoder
from ..nn.heads.orientation import OrientationHead, apply_orientation
from ..nn.heads.region_decoder import RegionMaskSpanDecoder
from ..nn.heads.matching import AlleleMatchingHead
from ..nn.heads.cross_attn_matcher import CrossAttnMatcher, xattn_match
from ..core.dnalignair import extract_segment
from ..data.tokenizer import pad_tokenize


def _masked_mean(reps, mask):
    m = mask.unsqueeze(-1).to(reps.dtype)
    return (reps * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class XAttnAligner(nn.Module):
    """LLM-encoder aligner: shared encoder + orientation/region heads + retrieval-top-k candidate
    pool + token-level cross-attention matcher. Reference is an input (encoded + cached); the four
    head outputs (orientation, query span, germline span, allele) come from one forward pass."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model
        self.orientation_head = OrientationHead(d=getattr(config, "orientation_dim", 64))
        self.backbone = SharedNucleotideEncoder(d_model=d, n_layers=config.n_layers, nhead=config.nhead)
        self.region_tagger = RegionMaskSpanDecoder(d_model=d, nhead=config.nhead)
        self.matching = AlleleMatchingHead()
        self.matcher = CrossAttnMatcher(d_model=d, nhead=config.nhead)
        self._seed = None                                          # lazily-built SeedPrefilter

    def encode_reference(self, reference_set) -> dict:
        device = next(self.parameters()).device
        out = {}
        for gene, ref in reference_set.genes.items():
            tok, msk = pad_tokenize(ref.sequences)
            tok, msk = tok.to(device), msk.to(device)
            pooled = self.backbone(tok, msk, token_type=SharedNucleotideEncoder.GERMLINE)   # (K,d) norm
            pos = self.backbone.forward_positions(tok, msk, SharedNucleotideEncoder.GERMLINE)  # (K,Lg,d)
            out[gene] = {"embeddings": pooled, "pos_reps": pos, "pos_mask": msk, "pos_tok": tok}
        return out

    def _union_seed(self, cand_idx, canon, mask, region_labels, gene, cm, seed_m):
        """Union non-learned k-mer seed candidates into the retrieval pool so a divergent/novel
        allele the encoder misranks can still reach the matcher. Pads each row to k+seed_m."""
        from ..core.dnalignair import extract_segment_tokens
        from ..data.tokenizer import TOKEN_DICT
        inv = {v: kk for kk, v in TOKEN_DICT.items()}
        seg_tok, seg_tm = extract_segment_tokens(canon, mask, region_labels, gene)
        allowed = set(int(i) for i in cm.nonzero().flatten().tolist()) if cm is not None else None
        width = cand_idx.shape[1] + seed_m
        rows = []
        for b in range(cand_idx.shape[0]):
            s = "".join(inv.get(int(x), "N") for x, ok in zip(seg_tok[b].tolist(), seg_tm[b].tolist()) if ok)
            seeds = self._seed.candidates(s, gene, seed_m, allowed=allowed)
            merged = list(dict.fromkeys(cand_idx[b].tolist() + seeds))[:width]
            while len(merged) < width:
                merged.append(merged[-1] if merged else 0)
            rows.append(merged)
        return torch.tensor(rows, dtype=torch.long, device=cand_idx.device)

    def forward(self, tokens, mask, ref_emb, orientation_ids=None,
                candidate_masks=None, topk: int = 8, seed_m: int = 0, reference_set=None) -> dict:
        orientation_logits = self.orientation_head(tokens, mask)
        t = orientation_ids if orientation_ids is not None else orientation_logits.argmax(dim=-1)
        canon = apply_orientation(tokens, mask, t)
        reps = self.backbone.forward_positions(canon, mask)
        rdec = self.region_tagger(reps, mask)
        region_labels = rdec["region_logits"].argmax(dim=-1)
        if reference_set is not None and seed_m > 0 and self._seed is None:
            from ..align.seed_prefilter import SeedPrefilter
            self._seed = SeedPrefilter(reference_set, k=11)
        genes = ["V", "J"] + (["D"] if "D" in ref_emb else [])
        match = {}
        for g in genes:
            emb = ref_emb[g]
            seg, seg_mask = extract_segment(reps, mask, region_labels, g)
            query = F.normalize(self.backbone.proj(_masked_mean(seg, seg_mask)), dim=-1)    # (B,d)
            cm = candidate_masks.get(g) if candidate_masks else None
            cm = cm.to(query.device) if cm is not None else None
            scores = self.matching(query, emb["embeddings"], candidate_mask=cm)             # (B,K)
            k = min(topk, scores.shape[-1])
            cand_idx = scores.topk(k, dim=-1).indices                                       # (B,k)
            if self._seed is not None:
                cand_idx = self._union_seed(cand_idx, canon, mask, region_labels, g, cm, seed_m)
            match[g] = xattn_match(self.matcher, seg, seg_mask,
                                   emb["pos_reps"], emb["pos_mask"], cand_idx)
        return {"orientation_logits": orientation_logits,
                "region_logits": rdec["region_logits"],
                "boundary": {"start": rdec["start_logits"], "end": rdec["end_logits"]},
                "reps": reps, "canon_tokens": canon, "match": match}
