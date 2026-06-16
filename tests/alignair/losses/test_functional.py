import torch
from alignair.losses.functional import soft_targets, expectation_from_logits, interval_iou_loss


def test_soft_targets_peak_at_gt():
    probs = soft_targets(torch.tensor([[4.0]]), L=10, sigma=1.5)
    assert probs.shape == (1, 10)
    assert torch.argmax(probs, dim=-1).item() == 4
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_expectation_recovers_peak():
    logits = torch.full((1, 10), -10.0)
    logits[0, 6] = 10.0
    exp = expectation_from_logits(logits, max_seq_length=10)
    assert abs(exp.item() - 6.0) < 1e-3


def test_iou_loss_zero_for_perfect_overlap():
    s = torch.tensor([[2.0]]); e = torch.tensor([[8.0]])
    loss = interval_iou_loss(s, e, torch.tensor([2.0]), torch.tensor([8.0]))
    assert loss.item() < 1e-4
