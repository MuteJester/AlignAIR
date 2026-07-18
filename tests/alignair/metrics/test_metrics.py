import torch
from alignair.metrics.accumulator import MeanAccumulator
from alignair.metrics.boundary import BoundaryMetrics
from alignair.metrics.entropy import AlleleEntropy
from alignair.metrics.allele_auc import MultiLabelAUC
from alignair.metrics.average_last_label import AverageLastLabel


def test_mean_accumulator():
    m = MeanAccumulator()
    m.update(torch.tensor(2.0))
    m.update(torch.tensor(4.0))
    assert abs(m.compute().item() - 3.0) < 1e-6
    m.reset()
    assert m.count == 0


def test_boundary_exact_match():
    bm = BoundaryMetrics()
    # logits argmax at index 3 for both rows; gt = 3 -> exact acc 1, mae 0
    logits = torch.full((2, 8), -10.0)
    logits[:, 3] = 10.0
    gt = torch.tensor([[3.0], [3.0]])
    bm.update(gt, logits)
    res = bm.compute()
    assert abs(res["mae"]) < 1e-6
    assert abs(res["acc"] - 1.0) < 1e-6
    assert abs(res["acc_1nt"] - 1.0) < 1e-6


def test_boundary_within_1nt():
    bm = BoundaryMetrics()
    logits = torch.full((1, 8), -10.0)
    logits[:, 4] = 10.0  # predicts 4
    gt = torch.tensor([[3.0]])  # off by 1
    bm.update(gt, logits)
    res = bm.compute()
    assert abs(res["acc"]) < 1e-6        # not exact
    assert abs(res["acc_1nt"] - 1.0) < 1e-6  # within 1


def test_allele_entropy_uniform_is_max():
    ent = AlleleEntropy()
    probs = torch.full((1, 4), 0.25)
    ent.update(probs)
    import math
    assert abs(ent.compute().item() - math.log(4)) < 1e-4


def test_multilabel_auc_perfect():
    auc = MultiLabelAUC()
    y_true = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y_pred = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    auc.update(y_true, y_pred)
    assert auc.compute().item() > 0.99


def test_average_last_label():
    all_ = AverageLastLabel()
    d_allele = torch.tensor([[0.1, 0.2, 0.7], [0.0, 0.0, 0.5]])
    all_.update(d_allele)
    assert abs(all_.compute().item() - 0.6) < 1e-6
