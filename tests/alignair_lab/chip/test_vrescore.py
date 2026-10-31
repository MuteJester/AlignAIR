"""The chip's V re-score: the read's V span aligned with gaps to each of its top V rows, in a band around the diagonal
the conserved Cys fixes, so an indel does not throw the allele call."""
import math

import pytest
import torch

from alignair.data.tokenizer import TOKEN_DICT

from alignair_lab.architectures.chip.vrescore import GAP, banded_scores

pytestmark = pytest.mark.component

V1 = "CAGGTGCAGCTGGTGCAGTCTGGGGCTGAGGTGAAGAAGCCTGGGGCCTCAGTGAAGGTCTCCTGCAAGGCTTCTGGATACACCTTC"
V2 = V1[:20] + "A" + V1[21:40] + "T" + V1[41:]          # two substitutions from V1
ANCHOR = 70                                              # the row base the (toy) conserved codon starts at
M = 0.05
MATCH, MISMATCH = math.log(4 * (1 - M)), math.log(4 * M / 3)


def _tokens(sequence):
    return torch.tensor([[TOKEN_DICT[base] for base in sequence]])


def _score(read, start, read_anchor):
    rows = torch.cat([_tokens(V1), _tokens(V2)])[None]                               # (1, K=2, row length)
    return banded_scores(_tokens(read), torch.tensor([start]), torch.tensor([len(read)]),
                         torch.tensor([read_anchor]), rows, torch.tensor([[ANCHOR, ANCHOR]]), torch.tensor([[M]]))


def test_a_clean_read_scores_every_base_as_a_match_of_its_allele():
    read = V1[10:]
    scores = _score(read, 0, ANCHOR - 10)
    assert scores[0, 0].item() == pytest.approx(len(read) * MATCH, rel=1e-5)
    assert scores[0, 0] > scores[0, 1]


def test_a_deletion_costs_one_gap_and_keeps_the_allele():
    read = V1[10:35] + V1[36:]                            # germline base 35 deleted, upstream of the anchor
    scores = _score(read, 0, ANCHOR - 11)
    assert scores[0, 0].item() == pytest.approx(len(read) * MATCH - GAP, rel=1e-4)
    assert scores[0, 0] > scores[0, 1]


def test_an_insertion_costs_one_gap_and_keeps_the_allele():
    read = V1[10:35] + "G" + V1[35:]                      # one base inserted upstream of the anchor
    scores = _score(read, 0, ANCHOR - 9)
    assert scores[0, 0].item() == pytest.approx((len(read) - 1) * MATCH - GAP, rel=1e-4)
    assert scores[0, 0] > scores[0, 1]


def test_bases_outside_the_span_do_not_count():
    read = "TTTTT" + V1[10:]
    assert _score(read, 5, ANCHOR - 5)[0, 0].item() == pytest.approx((len(read) - 5) * MATCH, rel=1e-5)
