"""Metric denominators and production validation, checked against hand-counted examples."""
import pytest
import torch

from alignair.tasks import MetricAccumulator
from alignair.tasks.metrics import alignment_metrics
from alignair.train.trainer import validate

from .conftest import AlleleTasks

pytestmark = pytest.mark.component


def test_completely_unobserved_gene_has_no_fabricated_metrics(task_batch):
    out, targets = task_batch()
    targets["j_supervise"] = torch.zeros(2, dtype=torch.bool)
    metrics = alignment_metrics(out, targets, ("v", "j")).means()
    assert not any(key.startswith("j_") for key in metrics)
    assert metrics["v_allele_top1"] == 1 and metrics["v_seg_mae"] == 0


def test_auxiliary_metrics_omit_a_prediction_without_observed_truth(task_batch):
    out, targets = task_batch(auxiliary=True)
    del targets["mutation_rate"]
    metrics = alignment_metrics(out, targets, ("v", "j")).means()
    assert "mutation_mae" not in metrics
    assert metrics["indel_mae"] == 1 and metrics["orientation_acc"] == 1


def test_validation_weights_by_supervised_counts_instead_of_batch_means(task_batch):
    first, first_targets = task_batch(batch_size=2)
    second, second_targets = task_batch(batch_size=1)
    first["v_allele"] = torch.tensor([[.9, .1], [.9, .1]])
    first_targets["v_allele"] = torch.tensor([[1., 0.], [0., 1.]])
    second["v_allele"] = torch.tensor([[.1, .9]])
    second_targets["v_allele"] = torch.tensor([[0., 1.]])

    class Echo(torch.nn.Module):
        def forward(self, batch):
            return batch

    model = Echo().train()
    metrics, score = validate(model, [(first, first_targets), (second, second_targets)], None, tasks=AlleleTasks())
    assert metrics["v_allele_top1"] == pytest.approx(2 / 3)
    assert metrics["j_allele_top1"] == 1
    assert score == pytest.approx(5 / 6)
    assert model.training


def test_metric_accumulator_keeps_each_genes_own_denominator(task_batch):
    first, targets = task_batch()
    targets["v_supervise"] = torch.tensor([True, False])
    accumulator = MetricAccumulator()
    accumulator.update(alignment_metrics(first, targets, ("v", "j")))
    assert accumulator.counts["v_allele_top1"] == 1
    assert accumulator.counts["j_allele_top1"] == 2


def test_validation_restores_model_mode_after_task_failure(task_batch):
    out, targets = task_batch()

    class Broken(torch.nn.Module):
        def forward(self, batch):
            raise RuntimeError("broken task input")

    model = Broken().train()
    with pytest.raises(RuntimeError, match="broken task"):
        validate(model, [(out, targets)], None, tasks=AlleleTasks())
    assert model.training


@pytest.mark.parametrize("mask", [torch.tensor([.5, 1.]), torch.ones(2, 1), [True, False]],
                         ids=["not_binary", "not_a_row", "not_a_tensor"])
def test_a_row_mask_must_be_a_binary_row_vector(task_batch, mask):
    from alignair.errors import TrainingDataError
    out, targets = task_batch()
    targets["v_supervise"] = mask
    with pytest.raises(TrainingDataError, match="binary row mask") as caught:
        alignment_metrics(out, targets, ("v", "j"))
    assert caught.value.context == {"gene": "v", "field": "v_supervise"}
