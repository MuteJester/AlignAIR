import torch
import torch.nn.functional as F
from alignair.matcher.scoring import BandedAlignmentScorer, masked_maxsim, pick_strand


def test_identical_candidate_scores_highest():
    # read positions = 3 orthonormal directions in R^4.
    d = 4
    read = torch.eye(3, d)                        # rows e0, e1, e2
    read_mask = torch.ones(3, dtype=torch.bool)
    cand_A = torch.stack([read[0], read[1]])      # matches read pos 0 and 1 (sim 1 each)
    cand_B = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)  # orthogonal to all reads
    cand = torch.stack([cand_A, cand_B])          # [2, 2, 4]
    sal = torch.ones(2, 2)
    cmask = torch.ones(2, 2, dtype=torch.bool)
    scores = masked_maxsim(read, read_mask, cand, sal, cmask)
    assert scores[0] > scores[1]                  # matching allele wins


def test_padding_positions_are_ignored():
    read = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    read_mask = torch.tensor([True, False])       # 2nd read pos is padding
    cand = torch.tensor([[[0.0, 1.0]]])           # [1,1,2] best-matches the MASKED pos
    sal = torch.ones(1, 1)
    cmask = torch.ones(1, 1, dtype=torch.bool)
    s = masked_maxsim(read, read_mask, cand, sal, cmask)
    assert torch.allclose(s, torch.tensor([0.0]), atol=1e-3)  # forced to match pos0 -> sim 0


def test_salience_weights_the_sum():
    read = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    read_mask = torch.ones(2, dtype=torch.bool)
    # cand pos0 matches read (sim 1); cand pos1 is orthogonal to both reads (sim 0).
    cand = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    cmask = torch.ones(1, 2, dtype=torch.bool)
    hi_first = masked_maxsim(read, read_mask, cand, torch.tensor([[10.0, 0.0]]), cmask)
    hi_second = masked_maxsim(read, read_mask, cand, torch.tensor([[0.0, 10.0]]), cmask)
    assert torch.allclose(hi_first, torch.tensor([1.0]), atol=1e-3)   # weight on the sim-1 pos
    assert torch.allclose(hi_second, torch.tensor([0.0]), atol=1e-3)  # weight on the sim-0 pos


def test_pick_strand_prefers_better_v_match():
    assert pick_strand(torch.tensor([0.9, 0.1]), torch.tensor([0.2, 0.3])) == 0
    assert pick_strand(torch.tensor([0.2, 0.3]), torch.tensor([0.9, 0.1])) == 1


def _ids(seq: str):
    vocab = {"A": 1, "C": 2, "G": 3, "T": 4}
    return torch.tensor([vocab[c] for c in seq], dtype=torch.long)


def _onehot(ids: torch.Tensor):
    return F.one_hot(ids - 1, num_classes=4).float()


def test_banded_scorer_keeps_single_snp_colinear():
    # Wrong allele has G at the SNP position. MaxSim can hide that by matching the
    # wrong G to the read's terminal G; the banded scorer is forced to compare the
    # SNP colinearly at offset 0.
    read_ids = _ids("AAAACAAAAG").unsqueeze(0)
    true_ids = _ids("AAAACAAAAG")
    wrong_ids = _ids("AAAAGAAAAG")
    read_emb = _onehot(read_ids[0]).unsqueeze(0)
    cand_emb = torch.stack([_onehot(true_ids), _onehot(wrong_ids)])
    read_mask = torch.ones_like(read_ids, dtype=torch.bool)
    cand_mask = torch.ones(2, true_ids.numel(), dtype=torch.bool)
    sal = torch.ones(2, true_ids.numel())

    maxsim = masked_maxsim(read_emb[0], read_mask[0], cand_emb, sal, cand_mask)
    assert torch.allclose(maxsim[0], maxsim[1])

    scorer = BandedAlignmentScorer(radius=0, aggregation="diag", coverage_weight=0.0)
    scores = scorer(read_emb, read_mask, cand_emb, sal, cand_mask,
                    offsets=torch.zeros(1, 2, dtype=torch.long),
                    read_ids=read_ids, cand_ids=torch.stack([true_ids, wrong_ids]))
    assert scores[0, 0] > scores[0, 1] + 0.1


def test_banded_scorer_is_differentiable():
    read_ids = _ids("AAAACAAAAG").unsqueeze(0)
    cand_ids = torch.stack([_ids("AAAACAAAAG"), _ids("AAAAGAAAAG")])
    read_emb = _onehot(read_ids[0]).unsqueeze(0).requires_grad_()
    cand_emb = torch.stack([_onehot(cand_ids[0]), _onehot(cand_ids[1])]).requires_grad_()
    read_mask = torch.ones_like(read_ids, dtype=torch.bool)
    cand_mask = torch.ones(2, cand_ids.shape[1], dtype=torch.bool)
    sal = torch.ones(2, cand_ids.shape[1], requires_grad=True)
    scorer = BandedAlignmentScorer(radius=1, aggregation="softpath")
    loss = -scorer(read_emb, read_mask, cand_emb, sal, cand_mask,
                   offsets=torch.zeros(1, 2, dtype=torch.long),
                   read_ids=read_ids, cand_ids=cand_ids).sum()
    loss.backward()
    assert read_emb.grad is not None and read_emb.grad.abs().sum() > 0
    assert cand_emb.grad is not None and cand_emb.grad.abs().sum() > 0
    assert sal.grad is not None and sal.grad.abs().sum() > 0
