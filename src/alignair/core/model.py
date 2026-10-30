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
from copy import deepcopy

import torch
import torch.nn as nn

from .config import AlignAIRConfig
from .boundaries import boundary_masks
from .components import _build_component, architecture_components
from ..errors import ConfigurationError, InputError, error_context
from .runtime import restore_model_state
from ..neural.heads.orientation import apply_orientation

# The configuration flags whose heads read the trunk's per-position features (``ModelComponents.trunk_heads``): an
# architecture without a trunk refuses each of them.
TRUNK_HEAD_FLAGS = ("region_head",)


class AlignAIR(nn.Module):
    def __init__(self, cfg: AlignAIRConfig):
        super().__init__()
        if not isinstance(cfg, AlignAIRConfig):
            raise ConfigurationError("model construction requires an AlignAIRConfig", context={"field": "config"})
        cfg = self.cfg = deepcopy(cfg)
        components = architecture_components(cfg.architecture)
        flagged = [flag for flag in TRUNK_HEAD_FLAGS if getattr(cfg, flag)]
        if flagged and components.trunk is None:
            raise ConfigurationError(f"{', '.join(flagged)} needs an architecture with a trunk: the head reads the "
                                     "trunk's per-position features",
                                     context={"stage": "model_construction", "component": "trunk_heads",
                                              "architecture": cfg.architecture, "fields": flagged},
                                     hint="Select an architecture with a trunk, or leave the flag off.")
        self.embedding = _build_component(components.embedding, cfg, component="embedding")
        self.orientation_head = (_build_component(components.orientation_head, cfg, component="orientation_head")
                                 if cfg.use_orientation else None)
        self.meta_tower = (_build_component(components.meta_tower, cfg, component="meta_tower")
                           if cfg.use_meta_heads else None)
        branches = {}
        for spec in cfg.gene_specs:
            name = f"branches.{spec.name}"
            branch = _build_component(components.gene_branch, spec, cfg, component=name)
            missing = [method for method in ("segment", "classify", "component_modules")
                       if not callable(getattr(branch, method, None))]
            if missing:
                raise ConfigurationError("gene branch does not implement the component contract",
                                         context={"stage": "model_construction", "component": name, "methods": missing})
            branches[spec.name] = branch
        self.branches = nn.ModuleDict(branches)
        with error_context(stage="model_construction", component="meta_heads"):
            heads = components.meta_heads(cfg) if cfg.use_meta_heads else {}
            if (not isinstance(heads, (Mapping, nn.ModuleDict))
                    or any(not isinstance(head, nn.Module) for head in heads.values())):
                raise ConfigurationError("meta-head factory must return a mapping of torch modules")
            try:
                self.meta_heads = nn.ModuleDict(heads)
            except (KeyError, TypeError) as exc:
                raise ConfigurationError("meta-head names must be valid module registration names") from exc
        self.state_branch = (_build_component(components.state_head, cfg, component="state_branch")
                             if cfg.state_head else None)
        self.trunk = (_build_component(components.trunk, cfg, component="trunk")
                      if components.trunk is not None else None)
        with error_context(stage="model_construction", component="trunk_heads"):
            trunk_heads = components.trunk_heads(cfg)
            if (not isinstance(trunk_heads, (Mapping, nn.ModuleDict))
                    or any(not isinstance(head, nn.Module) for head in trunk_heads.values())):
                raise ConfigurationError("trunk-head factory must return a mapping of torch modules")
            if trunk_heads and self.trunk is None:
                raise ConfigurationError("trunk heads require an architecture with a trunk")
            try:
                self.trunk_heads = nn.ModuleDict(trunk_heads)
            except (KeyError, TypeError) as exc:
                raise ConfigurationError("trunk-head names must be valid module registration names") from exc

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

        # shared per-position features: computed once from the corrected embedding when the architecture has a trunk
        features = self.trunk(emb, mask) if self.trunk is not None else None
        extra = {} if features is None else {"features": features}
        meta = self.meta_tower(emb, **extra) if self.meta_tower is not None else None
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
            s_log, e_log, s_exp, e_exp = branch.segment(emb, mask, end_mask, **extra)
            out[f"{name}_start_logits"], out[f"{name}_end_logits"] = s_log, e_log
            out[f"{name}_start"], out[f"{name}_end"] = s_exp, e_exp
            exp[name] = (s_exp, e_exp)

        for key, head in self.trunk_heads.items():
            out[key] = head(features, mask)

        # meta-tower heads (mutation / indel / productivity / chain_type)
        for key, head in self.meta_heads.items():
            out[key] = head(meta)

        # Per-gene scores retain their explicit probability/logit semantics in the output key.
        for name, branch in self.branches.items():
            out[self.cfg.classification_key(name)] = branch.classify(emb, *exp[name], **extra)
        return out

    def component_modules(self) -> dict[str, nn.Module]:
        """Named inspection points shared by profilers and model diagnostics.

        Values are the registered modules themselves, with no duplicate ownership or hooks.
        A custom gene branch declares its own inspection points through the same method.
        """
        modules = {"embedding": self.embedding}
        for name in ("orientation_head", "meta_tower", "state_branch", "trunk"):
            module = getattr(self, name)
            if module is not None:
                modules[name] = module
        modules.update({f"meta_heads.{name}": head for name, head in self.meta_heads.items()})
        modules.update({f"trunk_heads.{name}": head for name, head in self.trunk_heads.items()})
        owned = {id(module) for module in self.modules()}
        for gene, branch in self.branches.items():
            with error_context(stage="component_inspection", component=f"branches.{gene}"):
                points = branch.component_modules()
                if not isinstance(points, Mapping):
                    raise ConfigurationError("inspection points must name registered model modules")
                for name, module in points.items():
                    if (not isinstance(name, str) or not name or not isinstance(module, nn.Module)
                            or id(module) not in owned):
                        raise ConfigurationError("inspection points must name registered model modules",
                                                 context={"field": name})
                    modules[f"{gene}.{name}"] = module
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
