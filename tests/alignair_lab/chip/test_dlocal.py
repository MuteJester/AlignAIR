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


def _reference_local_scores(crop, rows, m):
    """Each row's best ungapped local alignment score as written first: Kadane along the row, base by base."""
    from torch.nn import functional as F
    width = rows.shape[1]
    windows = F.pad(crop, (width - 1, width - 1)).unfold(1, width, 1)
    match, mismatch = torch.log(4 * (1 - m))[..., None], torch.log(4 * m / 3)[..., None]
    running = torch.zeros(crop.shape[0], rows.shape[0], windows.shape[1])
    best = torch.zeros_like(running)
    for k in range(width):
        read, germline = windows[:, None, :, k], rows[None, :, k, None]
        both = (read > 0) & (germline > 0)
        ratio = torch.where(both & (read == germline), match, torch.where(both, mismatch, 0.))
        running = (running + ratio).clamp_min(0.)
        best = torch.maximum(best, running)
    return best.amax(-1)


def test_local_scores_are_the_base_by_base_definition():
    """Random crops and rows with pads (0) and N (5) on both sides, a mutation rate per read: the scores are the
    definition's."""
    generator = torch.Generator().manual_seed(0)
    crop = torch.randint(0, 6, (3, 30), generator=generator)
    rows = torch.randint(1, 6, (7, 12), generator=generator)
    rows[2, 9:] = 0
    m = torch.tensor([[0.02], [0.1], [0.25]])
    assert torch.allclose(local_scores(crop, rows, m), _reference_local_scores(crop, rows, m), atol=1e-4)
