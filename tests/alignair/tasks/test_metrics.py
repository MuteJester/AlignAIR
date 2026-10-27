"""Metric denominators and production validation, checked against hand-counted examples."""
import math

import pytest
import torch

from alignair.tasks import MetricAccumulator, build_tasks
from alignair.train.trainer import validate
from alignair.tasks.metrics import categorical_metrics

pytestmark = pytest.mark.component


def test_categorical_metrics_use_compatible_truth_and_skip_invalid_rows(task_batch):
    cfg, out, targets = task_batch()
    out["v_class_logits"] = torch.tensor([[0., 2.], [0., 2.]])
    targets["v_compatible"] = torch.tensor([[1., 1.], [0., 0.]])
    targets["v_valid"] = torch.tensor([1., 0.])
    out["v_start"] = torch.tensor([[1.], [99.]])
    report = build_tasks(cfg).metric_report(out, targets)
    assert report.means()["v_class_acc"] == 1
    assert report.means()["v_start_mae"] == 0
    assert report.counts["v_class_acc"] == report.counts["v_start_mae"] == 1


def test_completely_unobserved_gene_has_no_fabricated_metrics(task_batch):
    cfg, out, targets = task_batch()
    targets["j_valid"] = torch.zeros(2)
    metrics = build_tasks(cfg).metrics(out, targets)
    assert not any(key.startswith("j_") for key in metrics)


def test_categorical_metric_entry_delegates_to_shared_contract(task_batch):
    cfg, out, targets = task_batch()
    assert categorical_metrics(out, targets) == build_tasks(cfg).metrics(out, targets)


def test_auxiliary_metrics_omit_a_prediction_without_observed_truth(task_batch):
    cfg, out, targets = task_batch(auxiliary=True)
    del targets["productive"]
    metrics = build_tasks(cfg).metrics(out, targets)
    assert "productive_acc" not in metrics
    assert "mutation_mae" in metrics


def test_unlabeled_classes_keep_boundary_metrics_without_fabricating_accuracy(task_batch):
    cfg, out, targets = task_batch()
    targets["v_class"].fill_(-1)
    metrics = build_tasks(cfg).metrics(out, targets)
    assert "v_class_acc" not in metrics
    assert metrics["v_start_mae"] == metrics["v_end_mae"] == 0


def test_validation_weights_by_supervised_counts_instead_of_batch_means(task_batch):
    cfg, first, first_targets = task_batch(protein=False, batch_size=2)
    _, second, second_targets = task_batch(protein=False, batch_size=1)
    first["v_allele"] = torch.tensor([[.9, .1], [.9, .1]])
    first_targets["v_allele"] = torch.tensor([[1., 0.], [0., 1.]])
    second["v_allele"] = torch.tensor([[.1, .9]])
    second_targets["v_allele"] = torch.tensor([[0., 1.]])

    class Echo(torch.nn.Module):
        def forward(self, batch):
            return batch

    model = Echo().train()
    metrics, score = validate(model, [(first, first_targets), (second, second_targets)], cfg)
    assert metrics["v_allele_top1"] == pytest.approx(2 / 3)
    assert metrics["j_allele_top1"] == 1
    assert score == pytest.approx(5 / 6)
    assert model.training


def test_metric_accumulator_keeps_each_genes_own_denominator(task_batch):
    cfg, first, targets = task_batch(protein=False)
    targets["v_supervise"] = torch.tensor([True, False])
    tasks = build_tasks(cfg)
    accumulator = MetricAccumulator()
    accumulator.update(tasks.metric_report(first, targets))
    assert accumulator.counts["v_allele_top1"] == 1
    assert accumulator.counts["j_allele_top1"] == 2


def test_state_metrics_count_labeled_tokens_and_report_missing_evidence(task_batch):
    cfg, out, targets = task_batch(auxiliary=True, batch_size=1)
    targets["state_labels"].fill_(-100)
    targets["state_labels"][0, :3] = torch.tensor([0, 1, 2])
    logits = torch.zeros(1, 16, 4)
    logits[0, 0, 0] = logits[0, 1, 1] = logits[0, 2, 0] = 1
    out["state_logits"] = logits
    report = build_tasks(cfg).metric_report(out, targets)
    assert report.means()["state_acc"] == pytest.approx(2 / 3)
    assert report.means()["state_sub_recall"] == 1
    assert report.means()["state_ins_recall"] == 0
    assert math.isnan(report.means()["state_del_recall"])
    assert report.counts["state_acc"] == 3 and report.counts["state_sub_recall"] == 1


def test_validation_restores_model_mode_after_task_failure(task_batch):
    cfg, out, targets = task_batch()

    class Broken(torch.nn.Module):
        def forward(self, batch):
            raise RuntimeError("broken task input")

    model = Broken().train()
    with pytest.raises(RuntimeError, match="broken task"):
        validate(model, [(out, targets)], cfg)
    assert model.training


def test_eval_metrics_on_perfect_predictions():
    B = 3
    out = {}
    for g in ("v", "j"):
        lg = torch.full((B, 4), -10.0)
        lg[range(B), 2] = 10.0
        out[f"{g}_class_logits"] = lg
        out[f"{g}_start"] = torch.full((B, 1), 5.0)
        out[f"{g}_end"] = torch.full((B, 1), 40.0)
    tg = {"v_class": torch.full((B,), 2), "j_class": torch.full((B,), 2),
          "v_start": torch.full((B, 1), 5.0), "v_end": torch.full((B, 1), 40.0),
          "j_start": torch.full((B, 1), 5.0), "j_end": torch.full((B, 1), 40.0)}
    m = categorical_metrics(out, tg)
    assert m["v_class_acc"] == 1.0 and m["j_class_acc"] == 1.0
    for k in ("v_start_mae", "v_end_mae", "j_start_mae", "j_end_mae"):
        assert m[k] < 1e-4
