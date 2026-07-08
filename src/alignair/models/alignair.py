"""The one AlignAIR model — a faithful PyTorch port of TF SingleChain/MultiChain, unified.

Single vs multi chain is not two classes here, it is data: ``cfg.gene_specs`` says which genes exist
and ``cfg.num_chain_types`` says whether a chain_type/locus head is present. Build the config from one
GenAIRR dataconfig for the old single-chain behavior, or several for the old multi-chain behavior.

Flow (unchanged from the TF model): token+position embedding -> in-model orientation detect/correct
/re-embed -> shared meta tower -> per-gene :class:`GeneBranch` segmentation -> meta :class:`MetaHead`
predictions -> per-gene soft-cutout classification.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.alignair_config import _SEG_KERNELS, AlignAIRConfig
from ..nn.heads.orientation import apply_orientation
from .gene_branch import GeneBranch, MetaHead, build_tower
from .layers import EmbeddingOrientationHead, TokenAndPositionEmbedding


class AlignAIR(nn.Module):
    def __init__(self, cfg: AlignAIRConfig):
        super().__init__()
        self.cfg = cfg
        C = cfg.embed_dim

        self.embedding = TokenAndPositionEmbedding(cfg.vocab_size, C, cfg.max_seq_length)
        self.orientation_head = EmbeddingOrientationHead(C)
        self.meta_tower = build_tower(cfg, _SEG_KERNELS)
        self.branches = nn.ModuleDict({s.name: GeneBranch(s, cfg) for s in cfg.gene_specs})
        self.meta_heads = nn.ModuleDict(self._build_meta_heads(cfg))

    @staticmethod
    def _build_meta_heads(cfg: AlignAIRConfig) -> dict:
        L = cfg.max_seq_length
        heads = {
            "mutation_rate": MetaHead(cfg.block_out, 1, mid_dim=L, out_act=F.relu, clamp=(0.0, 1.0)),
            "indel_count": MetaHead(cfg.block_out, 1, mid_dim=L, out_act=F.relu, clamp=(0.0, 50.0)),
            "productive": MetaHead(cfg.block_out, 1, out_act=torch.sigmoid),
        }
        if cfg.num_chain_types > 1:                        # multi-chain: chain_type (locus) head
            heads["chain_type_logits"] = MetaHead(cfg.block_out, cfg.num_chain_types, mid_dim=L)
        return heads

    def forward(self, batch: dict) -> dict:
        tokens = batch["tokenized_sequence"]
        mask = tokens != 0                                 # non-pad
        out: dict = {}

        # orientation: predict from the shared initial embeddings, correct the input via an
        # involution transform, then re-embed the canonicalized read for the rest of the model.
        emb0 = self.embedding(tokens)
        orient_logits = self.orientation_head(emb0, mask)
        out["orientation_logits"] = orient_logits
        out["orientation"] = orient_logits.argmax(-1)
        correct = (batch["orientation"] if "orientation" in batch   # teacher-forced during training
                   else orient_logits.argmax(-1).detach())          # self-corrected at inference
        emb = self.embedding(apply_orientation(tokens, mask, correct))

        meta = self.meta_tower(emb)
        out["position_mask"] = mask                            # valid read positions (for pad-masked seg loss)

        # per-gene segmentation (soft-argmax boundary expectations; masked to the read, not the pad)
        exp: dict = {}
        for name, branch in self.branches.items():
            s_log, e_log, s_exp, e_exp = branch.segment(emb, mask)
            out[f"{name}_start_logits"], out[f"{name}_end_logits"] = s_log, e_log
            out[f"{name}_start"], out[f"{name}_end"] = s_exp, e_exp
            exp[name] = (s_exp, e_exp)

        # meta-tower heads (mutation / indel / productivity / chain_type)
        for key, head in self.meta_heads.items():
            out[key] = head(meta)

        # per-gene soft-cutout classification (multi-label allele probs)
        for name, branch in self.branches.items():
            out[f"{name}_allele"] = branch.classify(emb, *exp[name])
        return out

    @torch.no_grad()
    def clamp_params(self) -> None:
        """TF MinMaxValueConstraint on the analysis-head kernels (applied after each optimizer step)."""
        for head in self.meta_heads.values():
            head.clamp_()

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_config(cls, cfg: AlignAIRConfig) -> "AlignAIR":
        return cls(cfg)

    @classmethod
    def from_dataconfigs(cls, *dataconfigs, **kwargs) -> "AlignAIR":
        """One GenAIRR dataconfig -> single-chain model; several -> multi-chain model."""
        return cls(AlignAIRConfig.from_dataconfigs(*dataconfigs, **kwargs))
