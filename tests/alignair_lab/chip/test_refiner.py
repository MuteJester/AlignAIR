"""The chip's boundary refiner: the V end and the J start move to where the read stops following the called allele's
germline, counted from the conserved anchor, the parser's boundary distribution weighing in."""
import pytest
import torch

from alignair.data.tokenizer import TOKEN_DICT

from alignair_lab.architectures.chip.bank import BankCompiler
from alignair_lab.architectures.chip.refiner import refine_boundaries

from .conftest import toy_reference

pytestmark = pytest.mark.component

LENGTH = 24


def _tokens(read):
    tokens = torch.zeros(1, LENGTH, dtype=torch.long)
    tokens[0, :len(read)] = torch.tensor([TOKEN_DICT[base] for base in read])
    return tokens


def _outputs(read, *, v_anchor=None, j_anchor=None, v_end=None, j_start=None):
    """The chip outputs the refiner reads: flat boundary logits (or peaked at ``v_end``/``j_start``), anchor logits
    peaked at the given read positions (None: not in the read), V*01 and J*01 the calls, mutation rate 0.05."""
    def peaked(position, width):
        logits = torch.full((1, width), -10.)
        logits[0, LENGTH if position is None else position] = 10.
        return logits

    def boundary(position):
        logits = torch.zeros(1, LENGTH)
        if position is not None:
            logits[0, position] = 1.
        return logits
    return {"v_end_logits": boundary(v_end), "j_start_logits": boundary(j_start),
            "v_anchor_logits": peaked(v_anchor, LENGTH + 1), "j_anchor_logits": peaked(j_anchor, LENGTH + 1),
            "v_log_posterior": torch.log(torch.tensor([[0.9, 0.1]])),
            "j_log_posterior": torch.log(torch.tensor([[0.9, 0.1]])),
            "mutation_rate": torch.tensor([[0.05]])}


def _refined(read, **anchors_and_peaks):
    tokens = _tokens(read)
    mask = tokens != 0
    end_mask = torch.zeros_like(mask)
    end_mask[:, 1:] = mask[:, :-1]
    out = _outputs(read, **anchors_and_peaks)
    refine_boundaries(out, tokens, mask, end_mask, BankCompiler(toy_reference()))
    return out


def test_the_v_end_follows_the_germline_past_the_parsers_peak_and_stops_at_its_end():
    """V*01 is ACGTACGTAC: the read holds all 10 bases, then G's the germline does not have. The parser leans to 7."""
    out = _refined("ACGTACGTAC" + "GGGG", v_anchor=4, v_end=7)
    assert out["v_end_logits"].argmax().item() == 10


def test_the_v_end_moves_inward_over_a_mismatching_edge_base():
    """Base 7 is mutated (A for T) and base 8 is not the germline's A: the end lands before the mutated base."""
    out = _refined("ACGTACGA" + "CCCC", v_anchor=4)
    assert out["v_end_logits"].argmax().item() == 7


def test_the_j_start_follows_the_germline_back_to_its_first_base():
    """J*01 is TTTGGAA with its W/F codon at base 3: the read's J starts at 2, after two C's."""
    out = _refined("CC" + "TTTGGAA", j_anchor=5, j_start=4)
    assert out["j_start_logits"].argmax().item() == 2


def test_a_read_without_the_anchor_keeps_the_parsers_boundaries():
    out = _refined("ACGTACGTAC" + "GGGG", v_end=7, j_start=4)
    assert out["v_end_logits"].argmax().item() == 7 and out["j_start_logits"].argmax().item() == 4


def test_the_expected_boundaries_follow_the_refined_logits():
    out = _refined("ACGTACGTAC" + "GGGG", v_anchor=4, v_end=7)
    assert out["v_end"].shape == (1, 1) and abs(out["v_end"].item() - 10) < 0.5
