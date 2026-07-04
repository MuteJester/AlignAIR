import torch
from alignair.matcher.scoring import masked_maxsim, pick_strand


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
