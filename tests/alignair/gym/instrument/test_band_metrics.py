import torch
from alignair.gym.instrument.band_metrics import (
    top1_recall, topm_union_recall, fail_open_rate, cell_budget)


def _two_peaks(Lg=40, a=5, b=25):
    logits = torch.full((1, Lg), -1e4)
    logits[0, a] = 4.0; logits[0, b] = 5.0      # higher peak at b
    return logits


def test_top1_recall_within_tol():
    logits = _two_peaks()
    assert top1_recall(logits, torch.tensor([25]), w=1) == 1.0   # argmax=25
    assert top1_recall(logits, torch.tensor([5]), w=1) == 0.0    # argmax!=5


def test_topm_union_recovers_secondary_peak():
    logits = _two_peaks()
    # true=5 is the SECONDARY peak; top-1 misses, top-2 union recovers it
    assert top1_recall(logits, torch.tensor([5]), w=1) == 0.0
    assert topm_union_recall(logits, torch.tensor([5]), w=1, m=2) == 1.0


def test_fail_open_rate_thresholds_low_confidence():
    flat = torch.zeros(1, 50)                    # uniform -> max prob 0.02, low confidence
    peaked = _two_peaks()                        # confident
    assert fail_open_rate(flat, threshold=0.1) == 1.0
    assert fail_open_rate(peaked, threshold=0.1) == 0.0


def test_committed_recall_excludes_failopen():
    from alignair.gym.instrument.band_metrics import committed_recall, conf_fail_open_rate
    # 3 reads: row0 covered+confident, row1 NOT covered but LOW confidence (fails open),
    # row2 covered+confident. committed-recall should ignore row1 -> 1.0; fail-open = 1/3.
    Lg = 40
    pos = torch.arange(Lg).float()
    true = torch.tensor([5, 5, 30])
    cov0 = -(pos - 5.0) ** 2
    bad1 = -(pos - 20.0) ** 2                     # peak 15 off truth=5 -> not covered at w=2
    cov2 = -(pos - 30.0) ** 2
    logits = torch.stack([cov0, bad1, cov2])
    conf = torch.tensor([3.0, -3.0, 3.0])         # row1 below threshold -> fails open
    assert committed_recall(logits, conf, true, w=2, m=2, conf_thresh=0.5) == 1.0
    assert abs(conf_fail_open_rate(conf, conf_thresh=0.5) - 1/3) < 1e-6


def test_cell_budget_counts_band_vs_fullopen():
    logits = _two_peaks()                        # confident -> banded
    seg_len = torch.tensor([100])
    # confident: (2*4+1)*100 = 900 cells
    assert cell_budget(logits, w=4, threshold=0.1, seg_len=seg_len) == 900.0
    flat = torch.zeros(1, 40)                     # fail-open -> Lg*seg_len = 40*100 = 4000
    assert cell_budget(flat, w=4, threshold=0.1, seg_len=seg_len) == 4000.0
