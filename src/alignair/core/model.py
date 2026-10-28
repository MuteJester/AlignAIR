"""Shared AlignAIR assembly and execution for nucleotide and amino-acid models.

Single vs multi chain is not two classes here, it is data: ``cfg.gene_specs`` says which genes exist
and ``cfg.num_chain_types`` says whether a chain_type/locus head is present. Build the config from one
GenAIRR dataconfig for the old single-chain behavior, or several for the old multi-chain behavior.

Construction is selected by the serialized architecture name. Configuration independently selects
classification score semantics, boundary slots, orientation correction, auxiliary heads and edit states.
Every configuration follows the same per-gene segmentation and soft-cutout classification path.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from .config import AlignAIRConfig
from .boundaries import boundary_masks
from .components import architecture_components
from ..errors import InputError
from .runtime import restore_model_state
from ..neural.heads.orientation import apply_orientation


class AlignAIR(nn.Module):
    def __init__(self, cfg: AlignAIRConfig):
        super().__init__()
        self.cfg = cfg
        components = architecture_components(cfg.architecture)
        self.embedding = components.embedding(cfg)
        self.orientation_head = components.orientation_head(cfg) if cfg.use_orientation else None
        self.meta_tower = components.meta_tower(cfg) if cfg.use_meta_heads else None
        self.branches = nn.ModuleDict({s.name: components.gene_branch(s, cfg) for s in cfg.gene_specs})
        self.meta_heads = nn.ModuleDict(components.meta_heads(cfg) if cfg.use_meta_heads else {})
        self.state_branch = components.state_head(cfg) if cfg.state_head else None

    @classmethod
    def from_state_dict(cls, cfg: AlignAIRConfig, state_dict: Mapping[str, object], *,
                        device: str | torch.device = "cpu", source=None) -> "AlignAIR":
        """Restore decoded weights strictly, then place the model in evaluation mode.

        Construction uses the configured architecture on CPU before device placement. Artifact
        decoders own configuration migrations, reference/label checks, and calibration binding;
        this factory neither reads files nor relaxes key/shape matching. ``source`` supplies a
        diagnostic path only. Parameters retain gradients for subsequent fine-tuning.
        """
        with torch.device("cpu"):
            model = cls(cfg)
        restore_model_state(model, state_dict, source=source)
        return model.to(device).eval()

    def forward(self, batch: dict) -> dict:
        tokens = batch["tokenized_sequence"]
        mask, end_mask = boundary_masks(tokens, self.cfg.boundary_mode)
        if self.cfg.reject_empty_sequences and not mask.any(-1).all():
            raise InputError("empty (all-pad) sequence in batch; reject empty reads upstream")
        out: dict = {}

        # orientation: predict from the shared initial embeddings, correct the input via an
        # involution transform, then re-embed the canonicalized read for the rest of the model.
        # Disabled (orientation_head is None) for amino-acid models: reverse-complement is undefined
        # for protein and the complement table is nucleotide-only.
        emb = emb0 = self.embedding(tokens)
        if self.orientation_head is not None:
            orient_logits = self.orientation_head(emb0, mask)
            out["orientation_logits"] = orient_logits
            out["orientation"] = orient_logits.argmax(-1)
            correct = (batch["orientation"] if "orientation" in batch   # teacher-forced during training
                       else orient_logits.argmax(-1).detach())          # self-corrected at inference
            emb = self.embedding(apply_orientation(tokens, mask, correct))

        meta = self.meta_tower(emb) if self.meta_tower is not None else None
        if self.cfg.boundary_mode == "half_open":
            out["start_mask"], out["end_mask"] = mask, end_mask
        else:
            out["position_mask"] = mask

        # per-position edit states, from the canonicalized embedding -> forward-frame, aligns with the
        # forward-frame `state_labels` target (indel positions the postprocessing can consume directly)
        if self.state_branch is not None:
            out["state_logits"] = self.state_branch(emb)

        # per-gene segmentation (soft-argmax boundary expectations; masked to the read, not the pad)
        exp: dict = {}
        for name, branch in self.branches.items():
            s_log, e_log, s_exp, e_exp = branch.segment(emb, mask, end_mask)
            out[f"{name}_start_logits"], out[f"{name}_end_logits"] = s_log, e_log
            out[f"{name}_start"], out[f"{name}_end"] = s_exp, e_exp
            exp[name] = (s_exp, e_exp)

        # meta-tower heads (mutation / indel / productivity / chain_type)
        for key, head in self.meta_heads.items():
            out[key] = head(meta)

        # Per-gene scores retain their explicit probability/logit semantics in the output key.
        for name, branch in self.branches.items():
            out[self.cfg.classification_key(name)] = branch.classify(emb, *exp[name])
        return out

    def component_modules(self) -> dict[str, nn.Module]:
        """Named inspection points shared by profilers and model diagnostics.

        Values are the registered modules themselves, with no duplicate ownership or hooks.
        A custom gene branch declares its own inspection points through the same method.
        """
        modules = {"embedding": self.embedding}
        for name in ("orientation_head", "meta_tower", "state_branch"):
            module = getattr(self, name)
            if module is not None:
                modules[name] = module
        modules.update({f"meta_heads.{name}": head for name, head in self.meta_heads.items()})
        for gene, branch in self.branches.items():
            modules.update({f"{gene}.{name}": module for name, module in branch.component_modules().items()})
        return modules

    @torch.no_grad()
    def clamp_params(self) -> None:
        """TF MinMaxValueConstraint on the analysis-head kernels (applied after each optimizer step)."""
        for head in self.meta_heads.values():
            if hasattr(head, "clamp_"):
                head.clamp_()

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_config(cls, cfg: AlignAIRConfig) -> "AlignAIR":
        return cls(cfg)

    @classmethod
    def from_dataconfigs(cls, *dataconfigs, **kwargs) -> "AlignAIR":
        """One GenAIRR dataconfig -> single-chain model; several -> multi-chain model."""
        return cls(AlignAIRConfig.from_dataconfigs(*dataconfigs, **kwargs))
