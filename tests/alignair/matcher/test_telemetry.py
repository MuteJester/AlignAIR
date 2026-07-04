import torch
from alignair.matcher.telemetry import batch_metrics


def test_accuracy_and_margin_per_gene():
    gs = {"V": [
        (torch.tensor([2.0, 0.5, 0.0]), torch.tensor([True, False, False])),   # correct; margin 2.0-0.5=+1.5
        (torch.tensor([2.0, 0.5, 0.0]), torch.tensor([False, True, False])),   # wrong;   margin 0.5-2.0=-1.5
    ]}
    m = batch_metrics(gs, tau=0.5, margin=0.0)
    assert m["V"]["n"] == 2
    assert m["V"]["acc"] == 0.5
    assert abs(m["V"]["margin"]) < 1e-6          # mean(+1.5, -1.5) == 0
    assert m["V"]["loss"] > 0


def test_skips_reads_without_a_true_target():
    gs = {"D": [(torch.tensor([1.0, 0.0]), torch.tensor([False, False]))]}  # e.g. no D truth
    m = batch_metrics(gs, tau=0.5, margin=0.0)
    assert "D" not in m                           # no valid reads -> gene omitted


def test_set_aware_true_group_counts_as_correct():
    gs = {"J": [
        (torch.tensor([3.0, 2.9, 0.0]), torch.tensor([True, True, False])),   # argmax=0 is a true group
    ]}
    m = batch_metrics(gs, tau=0.5, margin=0.0)
    assert m["J"]["acc"] == 1.0
    # margin = true_max(3.0) - best_wrong_max(0.0) = 3.0
    assert abs(m["J"]["margin"] - 3.0) < 1e-6
