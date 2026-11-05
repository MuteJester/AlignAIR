"""The chip's boundary refiner: the V end and the J start move to where the read stops following the called allele's
germline, counted from the conserved anchor, the parser's boundary distribution weighing in."""
import pytest
import torch

from alignair.data.tokenizer import TOKEN_DICT
from alignair.reference.reference_set import GeneReference, ReferenceSet

from alignair.network.bank import BankCompiler
from alignair.refine.boundaries import refine_boundaries

from .conftest import toy_reference

pytestmark = pytest.mark.component

LENGTH = 24

# germlines long enough for an indel between the anchor and a boundary: a 40-base V with its Cys at 30 (where a V's
# sits, ~10 bases from its 3' end) and a 30-base J with its W/F at 22
_V = "GACTTCAGGATCCTAGCATGGTACAACGTTGCAGTCCAGA"
_J = "CTGGTACTTCGATCTCTGGGGCCAAGGAAC"


def _long_reference():
    v = GeneReference(["V*01", "V*02"], [_V, "T" * 40], {"V*01": 0, "V*02": 1}, anchors={"V*01": 30, "V*02": 30},
                      gapped={"V*01": _V, "V*02": "T" * 40})
    d = GeneReference(["D*01", "D*02"], ["GGGCCC", "TTTAAA"], {"D*01": 0, "D*02": 1})
    j = GeneReference(["J*01", "J*02"], [_J, "A" * 30], {"J*01": 0, "J*02": 1}, anchors={"J*01": 22, "J*02": 22})
    return ReferenceSet({"V": v, "D": d, "J": j}, has_d=True)


def _tokens(read, length=LENGTH):
    tokens = torch.zeros(1, length, dtype=torch.long)
    tokens[0, :len(read)] = torch.tensor([TOKEN_DICT[base] for base in read])
    return tokens


def _outputs(read, *, length=LENGTH, v_anchor=None, j_anchor=None, v_end=None, j_start=None):
    """The chip outputs the refiner reads: flat boundary logits (or peaked at ``v_end``/``j_start``), anchor logits
    peaked at the given read positions (None: not in the read), V*01 and J*01 the calls, mutation rate 0.05."""
    def peaked(position, width):
        logits = torch.full((1, width), -10.)
        logits[0, length if position is None else position] = 10.
        return logits

    def boundary(position):
        logits = torch.zeros(1, length)
        if position is not None:
            logits[0, position] = 1.
        return logits
    return {"v_end_logits": boundary(v_end), "j_start_logits": boundary(j_start),
            "v_anchor_logits": peaked(v_anchor, length + 1), "j_anchor_logits": peaked(j_anchor, length + 1),
            "v_log_posterior": torch.log(torch.tensor([[0.9, 0.1]])),
            "j_log_posterior": torch.log(torch.tensor([[0.9, 0.1]])),
            "mutation_rate": torch.tensor([[0.05]])}


def _refined(read, reference=None, length=LENGTH, **anchors_and_peaks):
    tokens = _tokens(read, length)
    mask = tokens != 0
    end_mask = torch.zeros_like(mask)
    end_mask[:, 1:] = mask[:, :-1]
    out = _outputs(read, length=length, **anchors_and_peaks)
    refine_boundaries(out, tokens, mask, end_mask, BankCompiler(reference or toy_reference()))
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


def test_the_v_end_follows_the_germline_over_a_deleted_base_after_the_anchor():
    """V*01 with base 33 deleted (3 bases after its Cys at 30), then T's: the V end is the read's last V base, 39, not
    the deletion, where an ungapped comparison stops."""
    read = _V[:33] + _V[34:] + "TTTTT"
    out = _refined(read, _long_reference(), 64, v_anchor=30)
    assert out["v_end_logits"].argmax().item() == 39


def test_an_insertion_just_before_the_anchor_does_not_pull_the_v_end_back():
    """An A inserted 4 bases before the Cys: ungapped, the shifted bases before it look like non-V and the V end jumps
    back past the anchor; the read's V still ends at 41."""
    read = _V[:26] + "A" + _V[26:] + "TTTTT"
    out = _refined(read, _long_reference(), 64, v_anchor=31)
    assert out["v_end_logits"].argmax().item() == 41


def test_the_j_start_follows_the_germline_over_a_deleted_base_before_the_anchor():
    """J*01 with base 8 deleted, after four C's: the J starts at 4, not at the deletion."""
    read = "CCCC" + _J[:8] + _J[9:]
    out = _refined(read, _long_reference(), 64, j_anchor=25)
    assert out["j_start_logits"].argmax().item() == 4


def test_a_read_without_the_anchor_keeps_the_parsers_boundaries():
    out = _refined("ACGTACGTAC" + "GGGG", v_end=7, j_start=4)
    assert out["v_end_logits"].argmax().item() == 7 and out["j_start_logits"].argmax().item() == 4


def test_the_expected_boundaries_follow_the_refined_logits():
    out = _refined("ACGTACGTAC" + "GGGG", v_anchor=4, v_end=7)
    assert out["v_end"].shape == (1, 1) and abs(out["v_end"].item() - 10) < 0.5
