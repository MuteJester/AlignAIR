"""The AlignAIR model: the embedding, the orientation correction and the architecture's core.

The configuration's architecture name selects the components (``core.components``). The model embeds the read,
predicts and corrects its orientation (re-embedding the corrected read), and hands the corrected embedding to the
core, which does the rest (the chip: ``network.chip.ChipCore``).
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import torch
import torch.nn as nn

from .config import AlignAIRConfig
from .boundaries import boundary_masks
from .components import _build_component, architecture_components
from ..errors import ConfigurationError
from .runtime import restore_model_state
from ..network.orientation import BankOrientation, apply_orientation


class AlignAIR(nn.Module):
    def __init__(self, cfg: AlignAIRConfig):
        super().__init__()
        if not isinstance(cfg, AlignAIRConfig):
            raise ConfigurationError("model construction requires an AlignAIRConfig", context={"field": "config"})
        cfg = self.cfg = deepcopy(cfg)
        components = architecture_components(cfg.architecture)
        self.embedding = _build_component(components.embedding, cfg, component="embedding")
        self.orientation_head = _build_component(components.orientation_head, cfg, component="orientation_head")
        self.bank_orientation = BankOrientation() if cfg.bank_orientation else None
        self.core = _build_component(components.core, cfg, component="core")

    @classmethod
    def from_state_dict(cls, cfg: AlignAIRConfig, state_dict: Mapping[str, object], *,
                        device: str | torch.device = "cpu", source=None) -> "AlignAIR":
        """Restore decoded weights strictly, then place the model in evaluation mode.

        Construction uses the configured architecture on CPU before device placement. Artifact
        decoders own configuration migrations and reference/label checks; this factory neither reads files nor relaxes key/shape matching. ``source`` supplies a
        diagnostic path only. Parameters retain gradients for subsequent fine-tuning.
        """
        with torch.device("cpu"):
            model = cls(cfg)
        restore_model_state(model, state_dict, source=source)
        return model.to(device).eval()

    def forward(self, batch: dict) -> dict:
        tokens = batch["tokenized_sequence"]
        mask, end_mask = boundary_masks(tokens)
        out: dict = {}

        # orientation: predict from the shared initial embeddings, correct the input via an
        # involution transform, then re-embed the canonicalized read for the core.
        orient_logits = self.orientation_head(self.embedding(tokens), mask)
        out["orientation_logits"] = orient_logits
        called = orient_logits.argmax(-1)
        if self.bank_orientation is not None and "orientation" not in batch:
            voted, won = self.bank_orientation.vote(tokens, mask)        # the reference's reading, where it wins
            called = torch.where(won, voted, called)
        out["orientation"] = called
        correct = (batch["orientation"] if "orientation" in batch   # teacher-forced during training
                   else called.detach())                            # self-corrected at inference
        corrected = apply_orientation(tokens, mask, correct)
        emb = self.embedding(corrected)

        out["start_mask"], out["end_mask"] = mask, end_mask
        out.update(self.core(emb, mask, end_mask, tokens=corrected, batch=batch))
        return out

    def component_modules(self) -> dict[str, nn.Module]:
        """Named inspection points shared by profilers and model diagnostics: the registered modules themselves."""
        return {"embedding": self.embedding, "orientation_head": self.orientation_head, "core": self.core}

    def bind_reference(self, reference) -> None:
        """Give the parts that read the reference their bank (an input, not weights): the bank orientation's k-mer
        index and the core's bank. The model's allele outputs are then the bound reference's alleles, in its order."""
        if self.bank_orientation is not None:
            self.bank_orientation.bind(reference)
        bind = getattr(self.core, "bind_reference", None)
        if bind is not None:
            bind(reference)

