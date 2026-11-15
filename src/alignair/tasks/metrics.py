"""Alignment metrics with explicit support counts for aggregation across batches."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from .supervision import supervised_rows


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

    def merge(self, other: "MetricReport") -> None:
        """Add ``other``'s numerators and counts to this report's, so that across batches small batches and sparse
        genes keep their correct weight."""
        for name, total in other.totals.items():
            self.totals[name] = self.totals.get(name, 0.0) + total
            self.counts[name] = self.counts.get(name, 0) + other.counts[name]

    def means(self) -> dict[str, float]:
        return {key: total / self.counts[key] if self.counts[key] else math.nan
                for key, total in self.totals.items()}


@torch.no_grad()
def alignment_metrics(out: dict, targets: dict, genes) -> MetricReport:
    report = MetricReport()
    for gene in genes:
        scores = out[f"{gene}_allele"]
        valid = supervised_rows(targets, gene, scores)
        if not valid.any():
            continue
        prediction = scores.argmax(-1)
        correct = targets[f"{gene}_allele"].gather(1, prediction[:, None]).squeeze(1) > 0.5
        report.add(f"{gene}_allele_top1", correct, valid)
        errors = {boundary: (out[f"{gene}_{boundary}"].reshape(-1) - targets[f"{gene}_{boundary}"].reshape(-1)).abs()
                  for boundary in ("start", "end")}
        report.add(f"{gene}_seg_mae", (errors["start"] + errors["end"]) / 2, valid)
    _auxiliary_metrics(report, out, targets)
    return report


def _auxiliary_metrics(report, out, targets):
    if "orientation_logits" in out and "orientation" in targets:
        report.add("orientation_acc", out["orientation_logits"].argmax(-1) == targets["orientation"])
    for key, name in (("mutation_rate", "mutation_mae"), ("indel_count", "indel_mae")):
        if key in out and key in targets:
            report.add(name, (out[key] - targets[key]).abs())
