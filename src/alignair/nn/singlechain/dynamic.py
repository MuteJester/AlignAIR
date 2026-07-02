"""Dynamic-reference allele matcher for SingleChainAlignAIR.

Replaces the fixed `Dense(allele_count)` head: the read's V/D/J segment feature and every reference
allele are encoded by the SAME shared gene encoder (`model.encode_alleles`), projected, L2-normalized,
and matched by cosine similarity × a learnable temperature (CLIP-style). Identity therefore comes
from sequence-embedding relationship, not memorized allele weights — so novel / renamed alleles in
the reference the caller supplies at inference need no new parameters.

Trained with multi-positive InfoNCE (reused from the detector) over the provided reference.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data.tokenizer import TOKEN_DICT
from ..sota.matching import contrastive_match_loss

__all__ = ["DynamicAlleleMatcher", "AlleleBank", "contrastive_match_loss"]


class DynamicAlleleMatcher(nn.Module):
    def __init__(self, feature_dim: int, genes=("v", "d", "j"), match_dim: int = 256,
                 init_temp: float = 0.07):
        super().__init__()
        self.proj = nn.ModuleDict({g: nn.Linear(feature_dim, match_dim) for g in genes})
        self.log_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temp)))

    def match(self, gene: str, read_feat, allele_feat, candidate_mask=None):
        """read_feat (B, feature_dim), allele_feat (K, feature_dim) -> scores (B, K)."""
        p = self.proj[gene.lower()]
        r = F.normalize(p(read_feat), dim=-1)
        a = F.normalize(p(allele_feat), dim=-1)
        scores = (r @ a.t()) * self.log_scale.exp().clamp(max=100.0)
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask[None].to(scores.device), float("-inf"))
        return scores


class AlleleBank:
    """Reference germline alleles tokenized + padded to `max_seq_length` (per gene), for encoding
    through `model.encode_alleles`. Fixed token ids; the model re-embeds them each step."""

    def __init__(self, reference_set, max_seq_length: int = 576, genes=("V", "D", "J")):
        self.genes = tuple(genes)
        self.tokens, self.sizes = {}, {}
        n = TOKEN_DICT["N"]
        for G in genes:
            gref = reference_set.gene(G)
            arr = torch.zeros(len(gref.names), max_seq_length, dtype=torch.long)
            for i, seq in enumerate(gref.sequences):
                t = [TOKEN_DICT.get(c, n) for c in seq.upper()[:max_seq_length]]
                arr[i, :len(t)] = torch.tensor(t, dtype=torch.long)
            self.tokens[G] = arr
            self.sizes[G] = len(gref.names)

    def to(self, device):
        self.tokens = {G: t.to(device) for G, t in self.tokens.items()}
        return self
