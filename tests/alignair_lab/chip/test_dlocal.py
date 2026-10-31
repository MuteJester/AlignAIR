"""The chip's local D aligner: every D row (forward and reverse complement) aligned without gaps to the read's
junction, each row scored by its best contiguous stretch, so a D trimmed at both ends still finds its allele."""
import math

import pytest
import torch

from alignair.data.tokenizer import TOKEN_DICT

from alignair_lab.architectures.chip.bank import reverse_complement
from alignair_lab.architectures.chip.dlocal import junction, local_scores

pytestmark = pytest.mark.component

D1, D2 = "GTATTACGATATTTTGACTGG", "GTATTACTATGGTTCGGGGAG"
M = torch.tensor([[0.05]])
MATCH = math.log(4 * 0.95)


def _tokens(sequence, length=None):
    tokens = torch.zeros(1, length or len(sequence), dtype=torch.long)
    tokens[0, :len(sequence)] = torch.tensor([TOKEN_DICT[base] for base in sequence])
    return tokens


def _rows(*sequences):
    width = max(len(s) for s in sequences)
    return torch.cat([_tokens(s, width) for s in sequences])


def test_a_d_trimmed_at_both_ends_scores_its_allele_by_the_bases_it_kept():
    crop = _tokens("CCCC" + D1[6:14] + "AAAA")
    scores = local_scores(crop, _rows(D1, D2), M)
    assert scores[0, 0] > scores[0, 1]
    assert scores[0, 0].item() == pytest.approx(8 * MATCH, rel=1e-5)        # the 8 kept bases, nothing overhanging


def test_an_inverted_d_finds_its_reverse_complement_row():
    crop = _tokens("CC" + reverse_complement(D1)[3:15] + "TT")
    scores = local_scores(crop, _rows(D1, D2, reverse_complement(D1), reverse_complement(D2)), M)
    assert scores[0].argmax().item() == 2


def test_the_junction_is_the_read_between_the_v_end_and_the_j_start_with_a_flank():
    read = "A" * 30 + "CCCCGGGG" + "T" * 30
    crop = junction(_tokens(read), _tokens(read) != 0, torch.tensor([30]), torch.tensor([38]))
    kept = "".join("?ATGCN"[t] for t in crop[0].tolist() if t)
    assert kept == "AAAAAA" + "CCCCGGGG" + "TTTTTT"                        # FLANK = 6 bases each side
