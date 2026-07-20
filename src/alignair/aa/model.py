"""Thin amino-acid AlignAIR model.

Reuses the core embedding + conv towers + per-gene heads, with two AA-specific corrections:
1. SEPARATE boundary-slot masks. Coordinates are half-open ``[start, end)``, so an exclusive end
   may equal the residue count ``n``. Start slots are ``0..n-1``; end slots are ``1..n``. Sharing
   one mask (as the nt model does) makes ``end == n`` undecodable AND under-penalized.
2. SOFTMAX single-label class head: ``classify`` returns raw class logits (CE applied by the loss).

No orientation head, no meta heads, no meta tower, no D branch (Phase 1 is V/J only).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.config import AlignAIRConfig
from ..core.gene_branch import GeneBranch
from ..core.layers import TokenAndPositionEmbedding


def boundary_masks(tokens: torch.Tensor):
    """(start_mask, end_mask) for half-open coords: starts occupy 0..n-1, ends occupy 1..n."""
    valid = tokens != 0                                   # (B, L) residue positions 0..n-1
    end = torch.zeros_like(valid)
    end[:, 1:] = valid[:, :-1]                            # slot k allowed iff k-1 is a residue
    return valid, end


class AAGeneBranch(GeneBranch):
    """GeneBranch with mask-correct segmentation and a softmax single-label class head."""

    def segment_with_masks(self, emb: torch.Tensor, start_mask: torch.Tensor,
                           end_mask: torch.Tensor):
        f = self.seg_tower(emb)
        s_log, e_log = self.start_head(f), self.end_head(f)
        neg = torch.finfo(s_log.dtype).min
        s_dec = s_log.masked_fill(~start_mask, neg)
        e_dec = e_log.masked_fill(~end_mask, neg)
        return s_log, e_log, self._soft_argmax(s_dec), self._soft_argmax(e_dec)

    def classify(self, emb: torch.Tensor, s_exp: torch.Tensor, e_exp: torch.Tensor) -> torch.Tensor:
        mask = self.cutout(s_exp, e_exp).unsqueeze(-1)                # (B, L, 1)
        return self.cls_head(F.silu(self.cls_mid(self.cls_tower(emb * mask))))   # raw logits


class AAAlignAIR(nn.Module):
    def __init__(self, cfg: AlignAIRConfig):
        super().__init__()
        # V/J or V/D/J (has_d): the D branch is a softmax head over D alleles + D segmentation. D has no
        # clean protein-equivalence class, so its predictions are low-confidence by nature (calibrated set).
        self.cfg = cfg
        self.embedding = TokenAndPositionEmbedding(cfg.vocab_size, cfg.embed_dim, cfg.max_seq_length)
        self.branches = nn.ModuleDict({s.name: AAGeneBranch(s, cfg) for s in cfg.gene_specs})

    def forward(self, batch: dict) -> dict:
        tokens = batch["tokenized_sequence"]
        start_mask, end_mask = boundary_masks(tokens)
        if not start_mask.any(-1).all():
            raise ValueError("empty (all-pad) sequence in batch; reject empty reads upstream")
        emb = self.embedding(tokens)                                  # no orientation correction
        out: dict = {"start_mask": start_mask, "end_mask": end_mask}
        exp: dict = {}
        for name, branch in self.branches.items():
            s_log, e_log, s_exp, e_exp = branch.segment_with_masks(emb, start_mask, end_mask)
            out[f"{name}_start_logits"], out[f"{name}_end_logits"] = s_log, e_log
            out[f"{name}_start"], out[f"{name}_end"] = s_exp, e_exp
            exp[name] = (s_exp, e_exp)
        for name, branch in self.branches.items():
            out[f"{name}_class_logits"] = branch.classify(emb, *exp[name])
        return out
