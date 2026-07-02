import random
import torch

from alignair.training.reader import (build_sibling_index, build_candidates, reader_set_nce)


class _Gene:
    def __init__(self, names):
        self.names = names


class _RS:
    def __init__(self):
        # gene A: a*01,a*02 (siblings); gene B: b*01; gene C: c*01
        self.genes = {"V": _Gene(["IGHVA*01", "IGHVA*02", "IGHVB*01", "IGHVC*01"])}

    def gene(self, g):
        return self.genes[g.upper()]


def test_build_candidates_includes_primary_and_siblings():
    sib = build_sibling_index(_RS())["V"]
    primary = torch.tensor([0, 2])                       # a*01, b*01
    multihot = torch.zeros(2, 4); multihot[0, 0] = 1; multihot[1, 2] = 1
    cand, pos = build_candidates(primary, multihot, sib, random.Random(0), n_sib=2, n_rand=1)
    assert cand.shape == (2, 4) and pos.shape == (2, 4)
    assert cand[0, 0].item() == 0 and pos[0, 0].item() == 1.0    # primary positive
    assert 1 in cand[0].tolist()                                  # sibling a*02 included for a*01


def test_reader_set_nce_ranks_true_set_over_negatives():
    # multi-positive set-NCE: higher score on the positive column -> lower loss
    pos_mask = torch.zeros(3, 4); pos_mask[:, 0] = 1.0
    good = torch.tensor([[5.0, 0.0, 0.0, 0.0]] * 3)              # true (col 0) scored highest
    bad = torch.tensor([[0.0, 5.0, 5.0, 5.0]] * 3)              # negatives scored highest
    assert reader_set_nce(good, pos_mask) < reader_set_nce(bad, pos_mask)


def test_reader_set_nce_zero_when_no_positive():
    scores = torch.randn(2, 4)
    pos_mask = torch.zeros(2, 4)
    assert float(reader_set_nce(scores, pos_mask)) == 0.0
