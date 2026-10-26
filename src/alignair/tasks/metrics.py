"""Alignment metrics with explicit support counts for aggregation across batches."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from .supervision import categorical_rows, supervised_rows


@dataclass
class MetricReport:
    """Metric numerators and their actual supervised row/token counts."""
    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, values: torch.Tensor, mask=None, *, keep_empty=False) -> None:
        values = values.detach().float()
        if mask is not None:
            values = values[mask]
        count = values.numel()
        if count or keep_empty:
            self.totals[name] = float(values.sum())
            self.counts[name] = count

    def means(self) -> dict[str, float]:
        return {key: total / self.counts[key] if self.counts[key] else math.nan
                for key, total in self.totals.items()}


class MetricAccumulator:
    """Combine numerators and counts, so small batches and sparse genes get their correct weight."""
    def __init__(self):
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, report: MetricReport) -> None:
        for name, total in report.totals.items():
            self.totals[name] = self.totals.get(name, 0.0) + total
            self.counts[name] = self.counts.get(name, 0) + report.counts[name]

    def means(self) -> dict[str, float]:
        return MetricReport(self.totals, self.counts).means()


@torch.no_grad()
def alignment_metrics(out: dict, targets: dict, genes, classification_mode: str,
                      *, auxiliary: bool = True) -> MetricReport:
    report = MetricReport()
    categorical = classification_mode == "categorical"
    for gene in genes:
        key = f"{gene}_class_logits" if categorical else f"{gene}_allele"
        scores = out[key]
        valid = supervised_rows(targets, gene, scores)
        if not valid.any():
            continue
        prediction = scores.argmax(-1)
        if categorical:
            classified = categorical_rows(targets, gene, scores)
            if f"{gene}_compatible" in targets:
                correct = targets[f"{gene}_compatible"].gather(1, prediction[:, None]).squeeze(1) != 0
            else:
                correct = prediction == targets[f"{gene}_class"]
            report.add(f"{gene}_class_acc", correct, classified)
        else:
            correct = targets[f"{gene}_allele"].gather(1, prediction[:, None]).squeeze(1) > 0.5
            report.add(f"{gene}_allele_top1", correct, valid)
        errors = {}
        for boundary in ("start", "end"):
            errors[boundary] = (out[f"{gene}_{boundary}"].reshape(-1)
                                - targets[f"{gene}_{boundary}"].reshape(-1)).abs()
            if categorical:
                report.add(f"{gene}_{boundary}_mae", errors[boundary], valid)
        if not categorical:
            report.add(f"{gene}_seg_mae", (errors["start"] + errors["end"]) / 2, valid)
    if auxiliary:
        _auxiliary_metrics(report, out, targets)
    return report


def _auxiliary_metrics(report, out, targets):
    for output, target, name in (("orientation_logits", "orientation", "orientation_acc"),
                                 ("chain_type_logits", "chain_type", "chain_type_acc")):
        if output in out and target in targets:
            report.add(name, out[output].argmax(-1) == targets[target])
    for key, name in (("mutation_rate", "mutation_mae"), ("indel_count", "indel_mae")):
        if key in out and key in targets:
            report.add(name, (out[key] - targets[key]).abs())
    if "productive" in out and "productive" in targets:
        report.add("productive_acc", (out["productive"] > 0.5) == (targets["productive"] > 0.5))
    if "state_logits" in out and "state_labels" in targets:
        prediction, labels = out["state_logits"].argmax(-1), targets["state_labels"]
        valid = labels != -100
        report.add("state_acc", prediction == labels, valid, keep_empty=True)
        for name, index in (("sub", 1), ("ins", 2), ("del", 3)):
            report.add(f"state_{name}_recall", prediction == labels, valid & (labels == index), keep_empty=True)


def categorical_metrics(out: dict, targets: dict, genes=("v", "j")) -> dict:
    """Compatibility surface for the original protein-class metric names."""
    return alignment_metrics(out, targets, genes, "categorical", auxiliary=False).means()
