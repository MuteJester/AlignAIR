"""PyTorch port of SingleChainAlignAIR (originally TensorFlow/Keras).

The proven conv-based AlignAIR: token+position embedding → conv-residual feature blocks →
per-position softmax boundary heads (segmentation-first localization) → soft-cutout masking →
conv classification blocks → multi-label allele heads, plus mutation/indel/productivity heads.
Trained with a Kendall-weighted multitask loss (soft-label boundary CE + IoU/len/hinge aux +
allele BCE + meta losses). This is the fixed-reference AlignAIR; the dynamic-reference matcher is
a later patch. See legacy `Models/SingleChainAlignAIR/SingleChainAlignAIR.py`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (TokenAndPositionEmbedding, ConvResidualFeatureExtractionBlock, SoftCutout,
                     RegularizedConstrainedLogVar)

_SEG = ["v_start", "v_end", "j_start", "j_end"]
_SEG_D = ["d_start", "d_end"]


class SingleChainAlignAIR(nn.Module):
    def __init__(self, max_seq_length: int, v_allele_count: int, j_allele_count: int,
                 d_allele_count: int | None = None, embed_dim: int = 32, filter_size: int = 128,
                 feature_dim: int = 576, latent_factor: int = 2,
                 allele_mode: str = "fixed", match_dim: int = 256):
        super().__init__()
        self.L = max_seq_length
        self.has_d = d_allele_count is not None
        self.allele_mode = allele_mode                       # "fixed" (memorized) or "dynamic" (match)
        self.embedding = TokenAndPositionEmbedding(max_seq_length, 6, embed_dim)
        act = nn.Tanh()                                     # fblock_activation

        def block(n, ks):
            return ConvResidualFeatureExtractionBlock(filter_size, n, ks, 2, act, feature_dim)

        self.meta_fblock = block(4, [3, 3, 3, 2, 5])
        self.v_seg_fblock = block(4, [3, 3, 3, 2, 5])
        self.j_seg_fblock = block(4, [3, 3, 3, 2, 5])
        self.v_cls_fblock = block(6, [3, 3, 3, 2, 2, 2, 5])
        self.j_cls_fblock = block(6, [3, 3, 3, 2, 2, 2, 5])

        self.seg_heads = nn.ModuleDict({s: nn.Linear(feature_dim, max_seq_length) for s in _SEG})
        self.v_mask = SoftCutout(max_seq_length)
        self.j_mask = SoftCutout(max_seq_length)

        self.mutation_mid = nn.Linear(feature_dim, feature_dim)
        self.mutation_head = nn.Linear(feature_dim, 1)
        self.indel_mid = nn.Linear(feature_dim, feature_dim)
        self.indel_head = nn.Linear(feature_dim, 1)
        self.productivity_head = nn.Linear(feature_dim, 1)
        self.drop = nn.Dropout(0.05)

        if self.has_d:
            self.d_seg_fblock = block(4, [3, 3, 3, 2, 5])
            self.d_cls_fblock = block(4, [3, 3, 2, 2, 5])
            self.d_mask = SoftCutout(max_seq_length)
            for s in _SEG_D:
                self.seg_heads[s] = nn.Linear(feature_dim, max_seq_length)

        genes = ["v", "j"] + (["d"] if self.has_d else [])
        if allele_mode == "fixed":                           # memorized Dense head per gene
            counts = {"v": v_allele_count, "j": j_allele_count, "d": d_allele_count}
            self.allele_mid = nn.ModuleDict(
                {g: nn.Linear(feature_dim, counts[g] * latent_factor) for g in genes})
            self.allele_head = nn.ModuleDict(
                {g: nn.Linear(counts[g] * latent_factor, counts[g]) for g in genes})
        elif allele_mode == "dynamic":                       # reference-agnostic matcher
            from .dynamic import DynamicAlleleMatcher
            self.matcher = DynamicAlleleMatcher(feature_dim, tuple(genes), match_dim)
        else:
            raise ValueError(f"allele_mode must be fixed|dynamic, got {allele_mode}")

        keys = _SEG + (_SEG_D if self.has_d else []) + \
            ["v_clf", "j_clf"] + (["d_clf"] if self.has_d else []) + \
            ["mutation", "indel", "productivity"]
        self.log_vars = nn.ModuleDict({k: RegularizedConstrainedLogVar() for k in keys})

    def _expectation(self, logits):                         # (B, L) -> (B, 1) expected position
        pos = torch.arange(self.L, device=logits.device, dtype=logits.dtype)
        return (logits.softmax(-1) * pos[None]).sum(-1, keepdim=True)

    def _segment_feature(self, emb, start_exp, end_exp, mask_mod, cls_block):
        """Soft-cutout the read to the [start,end) segment and encode it -> (B, feature_dim)."""
        masked = emb * mask_mod(start_exp, end_exp).unsqueeze(-1)
        return cls_block(masked)

    def encode_alleles(self, gene: str, allele_tokens):
        """Encode reference germline alleles through the SAME embedding + gene cls block, so read
        segments and reference alleles live in one space (the dynamic-reference matcher). Returns
        (K, feature_dim). `allele_tokens` (K, L) are germline sequences padded to max_seq_length."""
        block = {"v": self.v_cls_fblock, "j": self.j_cls_fblock}
        if self.has_d:
            block["d"] = self.d_cls_fblock
        return block[gene.lower()](self.embedding(allele_tokens))

    def _classify(self, gene, feat):
        return torch.sigmoid(self.allele_head[gene](F.silu(self.allele_mid[gene](feat))))

    def forward(self, tokens, reference: dict | None = None, candidate_mask: dict | None = None):
        """tokens (B, L). In dynamic mode, `reference[G]` = (Kg, L) germline tokens the read is
        matched against; `candidate_mask[G]` = (Kg,) bool restricts which alleles are scorable."""
        emb = self.embedding(tokens)
        meta = self.meta_fblock(emb)
        vseg, jseg = self.v_seg_fblock(emb), self.j_seg_fblock(emb)

        out = {}
        seg_feat = {"v_start": vseg, "v_end": vseg, "j_start": jseg, "j_end": jseg}
        cls_blocks = {"v": self.v_cls_fblock, "j": self.j_cls_fblock}
        masks = {"v": self.v_mask, "j": self.j_mask}
        if self.has_d:
            dseg = self.d_seg_fblock(emb)
            seg_feat["d_start"] = seg_feat["d_end"] = dseg
            cls_blocks["d"], masks["d"] = self.d_cls_fblock, self.d_mask
        for s, feat in seg_feat.items():
            out[f"{s}_logits"] = self.seg_heads[s](feat)
            out[s] = self._expectation(out[f"{s}_logits"])

        out["mutation_rate"] = F.relu(self.mutation_head(self.drop(F.gelu(self.mutation_mid(meta)))))
        out["indel_count"] = F.relu(self.indel_head(self.drop(F.gelu(self.indel_mid(meta)))))
        out["productive"] = torch.sigmoid(self.productivity_head(self.drop(meta)))

        for g in (["v", "j"] + (["d"] if self.has_d else [])):
            feat = self._segment_feature(emb, out[f"{g}_start"], out[f"{g}_end"], masks[g], cls_blocks[g])
            out[f"{g}_feature"] = feat
            if self.allele_mode == "fixed":
                out[f"{g}_allele"] = self._classify(g, feat)
            else:
                G = g.upper()
                cm = candidate_mask.get(G) if candidate_mask else None
                out[f"{g}_allele_scores"] = self.matcher.match(
                    g, feat, self.encode_alleles(g, reference[G]), cm)
        return out
