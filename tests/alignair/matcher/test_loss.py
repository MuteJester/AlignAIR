import torch
from alignair.matcher.loss import selection_ce


def test_lower_loss_when_true_scores_higher():
    tgt = torch.tensor([True, False, False])
    good = selection_ce(torch.tensor([2.0, 0.0, 0.0]), tgt, tau=0.5, margin=0.0)
    bad = selection_ce(torch.tensor([0.0, 2.0, 0.0]), tgt, tau=0.5, margin=0.0)
    assert good < bad


def test_margin_acts_in_score_space_not_scaled_by_tau():
    # A margin only bites near a tie; use a close sibling (0.9) against the true (1.0).
    # With margin applied BEFORE the tau divide, the true logit drops by margin/tau.
    scores = torch.tensor([1.0, 0.9, 0.0])
    tgt = torch.tensor([True, False, False])
    tau, m = 0.05, 0.2
    loss_m = selection_ce(scores, tgt, tau=tau, margin=m)
    logits = (scores - m * tgt.float()) / tau
    expected = torch.logsumexp(logits, 0) - logits[0]
    assert torch.allclose(loss_m, expected, atol=1e-5)
    # sanity: the effective margin is m (=0.2 in score space), NOT m*tau (=0.01). At tau=0.05
    # a 0.2 score-space margin flips a near-tie from ~0.13 loss to ~2.1 loss.
    loss_no_m = selection_ce(scores, tgt, tau=tau, margin=0.0)
    assert (loss_m - loss_no_m) > 1.0


def test_set_aware_any_true_target_is_low_loss():
    tgt = torch.tensor([True, True, False])
    loss = selection_ce(torch.tensor([3.0, 2.9, 0.0]), tgt, tau=0.5, margin=0.0)
    assert loss < 0.1


def test_gradient_flows():
    scores = torch.tensor([1.0, 0.5, 0.2], requires_grad=True)
    loss = selection_ce(scores, torch.tensor([True, False, False]), tau=0.5, margin=0.1)
    loss.backward()
    assert scores.grad is not None and scores.grad.abs().sum() > 0
