"""Shared training-task contract and built-in alignment objectives."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import torch
from torch import nn

from ..core.config import AlignAIRConfig
from ..errors import ConfigurationError, TrainingDataError
from .losses import auxiliary_losses, categorical_loss, hierarchical_loss, loss_weight_names, make_loss_weights
from .metrics import MetricReport, alignment_metrics

LossResult = tuple[torch.Tensor, dict[str, torch.Tensor]]


class TrainingTasks(Protocol):
    """A trainer consumes task behavior without knowing the input alphabet or score family."""
    def loss(self, outputs: dict, targets: dict, weights=None) -> LossResult: ...
    def make_weights(self) -> nn.ModuleDict: ...
    def metrics(self, outputs: dict, targets: dict) -> dict[str, float]: ...
    def metric_report(self, outputs: dict, targets: dict) -> MetricReport: ...
    def selection_score(self, metrics: dict[str, float]) -> float: ...


@dataclass(frozen=True)
class AlignmentTasks:
    """Configured gene objectives plus optional auxiliary tasks; target preparation stays separate."""
    config: AlignAIRConfig
    genes: tuple[str, ...]

    def make_weights(self) -> nn.ModuleDict:
        return make_loss_weights(self.config)

    def loss(self, outputs: dict, targets: dict, weights=None) -> LossResult:
        required_out, required_target = [], []
        for gene in self.genes:
            required_out.append(self.config.classification_key(gene))
            for boundary in ("start", "end"):
                required_out += [f"{gene}_{boundary}", f"{gene}_{boundary}_logits"]
                required_target.append(f"{gene}_{boundary}")
            if self.config.classification_mode == "multilabel":
                required_target.append(f"{gene}_allele")
            elif f"{gene}_compatible" not in targets:
                required_target.append(f"{gene}_class")
        if self.config.use_meta_heads:
            required_out += ["mutation_rate", "indel_count", "productive"]
            required_target += ["mutation_rate", "indel_count", "productive"]
        for mapping, required, kind in ((outputs, required_out, "output"), (targets, required_target, "target")):
            missing = [key for key in required if key not in mapping]
            if missing:
                raise TrainingDataError(f"missing task {kind} fields: {missing}",
                                        context={"kind": kind, "fields": missing})
        if self.config.classification_mode == "categorical" and not (
                "position_mask" in outputs or {"start_mask", "end_mask"} <= outputs.keys()):
            raise TrainingDataError("categorical boundary loss requires start/end masks or a position mask",
                                    context={"kind": "output", "fields": ["start_mask", "end_mask"]})
        needed = loss_weight_names(self.config)
        if needed and (weights is None or any(key not in weights for key in needed)):
            raise ConfigurationError("task loss requires its configured uncertainty weights",
                                     hint="Create weights with tasks.make_weights() and include them in the optimizer.")
        weights = {} if weights is None else weights
        if self.config.classification_mode == "multilabel":
            return hierarchical_loss(outputs, targets, self.config, weights)
        total, parts = categorical_loss(outputs, targets, self.genes)
        for key, value in auxiliary_losses(outputs, targets, self.config, weights).items():
            parts[key] = value
            total = total + value
        return total, parts

    def metric_report(self, outputs: dict, targets: dict) -> MetricReport:
        return alignment_metrics(outputs, targets, self.genes, self.config.classification_mode)

    def metrics(self, outputs: dict, targets: dict) -> dict[str, float]:
        return self.metric_report(outputs, targets).means()

    def selection_score(self, metrics: dict[str, float]) -> float:
        suffix = "allele_top1" if self.config.classification_mode == "multilabel" else "class_acc"
        values = [metrics[f"{gene}_{suffix}"] for gene in self.genes if f"{gene}_{suffix}" in metrics]
        return sum(values) / len(values) if values and all(math.isfinite(v) for v in values) else math.nan


def build_tasks(config: AlignAIRConfig, *, genes=None) -> AlignmentTasks:
    """Select built-in objectives by score semantics, with explicit categorical gene subsets if needed."""
    canonical = tuple(spec.name for spec in config.gene_specs)
    selected = canonical if genes is None else tuple(genes)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(canonical):
        raise ConfigurationError("task genes must be a nonempty, unique subset of the model genes")
    if config.classification_mode == "multilabel" and selected != canonical:
        raise ConfigurationError("the hierarchical objective supervises the configured gene set in canonical order")
    return AlignmentTasks(config, selected)
