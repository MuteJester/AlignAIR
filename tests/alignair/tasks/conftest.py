"""Small hand-controlled predictions for the alignment metric contracts."""
import math

import pytest
import torch

from alignair.tasks.metrics import alignment_metrics


class AlleleTasks:
    """A minimal training objective of the shared metrics: the allele top-1 and segment errors of ``genes``, and the
    mean allele top-1 as the selection score."""

    def __init__(self, genes=("v", "j")):
        self.genes = genes

    def metric_report(self, outputs, targets):
        return alignment_metrics(outputs, targets, self.genes)

    def metrics(self, outputs, targets):
        return self.metric_report(outputs, targets).means()

    def selection_score(self, metrics):
        values = [metrics[f"{gene}_allele_top1"] for gene in self.genes if f"{gene}_allele_top1" in metrics]
        return sum(values) / len(values) if values and all(math.isfinite(v) for v in values) else math.nan


@pytest.fixture
def task_batch():
    """``make(batch_size, auxiliary)`` -> (outputs, targets) of the V and J genes: boundaries at 1 and 4 (exact), allele
    scores .5/.5 with allele 0 true; ``auxiliary`` adds the orientation, mutation and indel outputs and truths."""
    def make(batch_size=2, auxiliary=False):
        out, targets = {}, {}
        for gene in ("v", "j"):
            for boundary, value in (("start", 1.), ("end", 4.)):
                out[f"{gene}_{boundary}"] = torch.full((batch_size, 1), value)
                targets[f"{gene}_{boundary}"] = torch.full((batch_size, 1), value)
            out[f"{gene}_allele"] = torch.full((batch_size, 2), .5)
            targets[f"{gene}_allele"] = torch.tensor([[1., 0.]]).repeat(batch_size, 1)
        if auxiliary:
            out["orientation_logits"] = torch.zeros(batch_size, 4)
            targets["orientation"] = torch.zeros(batch_size, dtype=torch.long)
            for key, value in (("mutation_rate", .2), ("indel_count", 1.)):
                out[key] = torch.full((batch_size, 1), value)
                targets[key] = torch.zeros(batch_size, 1)
        return out, targets
    return make
