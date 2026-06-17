import math
import torch
from alignair.nn.matching import AlleleMatchingHead, multilabel_match_loss


def test_scores_shape_and_self_match_is_high():
    head = AlleleMatchingHead(init_temp=0.1)
    # 3 orthonormal candidate embeddings; queries equal to candidates
    E = torch.eye(3)                    # (K=3, d=3), unit-norm rows
    Q = torch.eye(3)                    # (B=3, d=3)
    scores = head(Q, E)
    assert scores.shape == (3, 3)
    # each query's own candidate scores highest
    assert (scores.argmax(dim=1) == torch.arange(3)).all()


def test_genotype_mask_excludes_candidates():
    head = AlleleMatchingHead(init_temp=0.1)
    E = torch.eye(4)
    Q = torch.eye(4)
    allowed = torch.tensor([True, False, True, False])
    scores = head(Q, E, candidate_mask=allowed)
    assert torch.isneginf(scores[:, 1]).all() and torch.isneginf(scores[:, 3]).all()
    assert torch.isfinite(scores[:, 0]).all() and torch.isfinite(scores[:, 2]).all()


def test_multilabel_loss_finite_and_backprops():
    head = AlleleMatchingHead()
    E = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    Q = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1).requires_grad_(True)
    scores = head(Q, E)
    target = torch.zeros(2, 5)
    target[0, 1] = 1.0
    target[1, 3] = 1.0
    target[1, 4] = 1.0  # multi-label row (two true alleles)
    loss = multilabel_match_loss(scores, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert Q.grad is not None and torch.isfinite(Q.grad).all()


def test_contrastive_set_loss_is_equivalence_class():
    from alignair.nn.matching import contrastive_match_loss
    # two indistinguishable positives; mass on EITHER one should give low loss
    scores = torch.tensor([[5.0, -5.0, 0.0, 0.0]])
    two_pos = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    assert contrastive_match_loss(scores, two_pos).item() < 0.1
    # mass on a negative -> high loss
    wrong = torch.tensor([[0.0, 0.0, 5.0, 0.0]])
    assert contrastive_match_loss(wrong, two_pos).item() > 2.0
    # no-positive (masked) row contributes exactly zero
    assert contrastive_match_loss(scores, torch.zeros_like(two_pos)).item() == 0.0
