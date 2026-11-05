"""The chip's D refiner: the D start and end move to where the read stops following the called D allele's germline,
on the row and diagonal the junction aligns best (an inverted D on its reverse complement row), the parser's boundary
distribution weighing in."""
import pytest
import torch

from alignair.data.tokenizer import TOKEN_DICT
from alignair.reference.reference_set import GeneReference, ReferenceSet

from alignair.network.bank import BankCompiler, reverse_complement
from alignair.refine.drefine import refine_d_boundaries

pytestmark = pytest.mark.component

LENGTH = 48
D1, D2 = "GTATTACGATATTTTGACTGG", "GTATTACTATGGTTCGGGGAG"
# V [0, 10), N1 [10, 12), D [12, 24), N2 [24, 26), J [26, 36); no N base continues the D's germline on its diagonal
V, N1, N2, J = "T" * 10, "GG", "CC", "T" * 10


def _reference():
    v = GeneReference(["V*01", "V*02"], ["ACGTACGTAC", "TTTTTTTTTT"], {"V*01": 0, "V*02": 1},
                      anchors={"V*01": 4, "V*02": 4}, gapped={"V*01": "ACGT..ACGTAC", "V*02": "TTTT..TTTTTT"})
    d = GeneReference(["D*01", "D*02"], [D1, D2], {"D*01": 0, "D*02": 1})
    j = GeneReference(["J*01", "J*02"], ["TTTGGAA", "TTAGGAA"], {"J*01": 0, "J*02": 1},
                      anchors={"J*01": 3, "J*02": 3})
    return ReferenceSet({"V": v, "D": d, "J": j}, has_d=True)


def _refined(read, *, d_end_peak=None, j_start=26):
    """The D logits flat over the read's slots (one more at ``d_end_peak``), the V end at 10, the J start at
    ``j_start``, D*01 the call, mutation rate 0.05; refined."""
    tokens = torch.zeros(1, LENGTH, dtype=torch.long)
    tokens[0, :len(read)] = torch.tensor([TOKEN_DICT[base] for base in read])
    mask = tokens != 0
    end_mask = torch.zeros_like(mask)
    end_mask[:, 1:] = mask[:, :-1]

    def peaked(position):
        logits = torch.zeros(1, LENGTH)
        if position is not None:
            logits[0, position] = 1.
        return logits
    out = {"v_end_logits": peaked(10), "j_start_logits": peaked(j_start), "d_start_logits": peaked(None),
           "d_end_logits": peaked(d_end_peak), "d_log_posterior": torch.log(torch.tensor([[0.9, 0.1]])),
           "mutation_rate": torch.tensor([[0.05]])}
    before = {key: value.clone() for key, value in out.items()}
    refine_d_boundaries(out, tokens, mask, end_mask, BankCompiler(_reference()))
    return out, before


def test_the_d_spans_the_bases_that_follow_its_germline():
    out, _ = _refined(V + N1 + D1[3:15] + N2 + J)
    assert out["d_start_logits"].argmax().item() == 12
    assert out["d_end_logits"].argmax().item() == 24
    assert out["d_start"].item() == pytest.approx(12, abs=0.5)               # the expectations follow the logits
    assert out["d_end"].item() == pytest.approx(24, abs=0.5)


def test_an_inverted_d_is_placed_by_its_reverse_complement_row():
    out, _ = _refined(V + N1 + reverse_complement(D1)[3:15] + N2 + J)
    assert out["d_start_logits"].argmax().item() == 12
    assert out["d_end_logits"].argmax().item() == 24


def test_an_n_base_says_nothing_so_the_parsers_choice_around_it_stands():
    """Base 24 is an N: ending the D before or after it is level on the germline's evidence; the parser leans to 25."""
    out, _ = _refined(V + N1 + D1[3:15] + "N" + N2[1:] + J, d_end_peak=25)
    assert out["d_end_logits"].argmax().item() == 25


def test_a_junction_without_a_base_of_the_called_allele_keeps_the_parsers_boundaries():
    out, before = _refined("N" * 36)
    assert torch.equal(out["d_start_logits"], before["d_start_logits"])
    assert torch.equal(out["d_end_logits"], before["d_end_logits"])


def test_the_d_ends_at_the_j_start_where_its_germline_runs_on_into_the_j():
    """N2 and the J's first two bases continue D*01's germline to base 28; the J starts at 26."""
    out, _ = _refined(V + N1 + D1[3:15] + D1[15:17] + D1[17:19] + "T" * 8)
    assert out["d_end_logits"].argmax().item() == 26


def test_the_d_starts_at_the_v_end_where_its_germline_runs_back_into_the_v():
    """N1 and the V's last base continue D*01's germline back to base 9; the V ends at 10."""
    out, _ = _refined("T" * 9 + D1[0] + D1[1:3] + D1[3:15] + N2 + J)
    assert out["d_start_logits"].argmax().item() == 10


def test_a_d_start_with_no_d_end_after_it_is_impossible():
    """No D end can follow a start at or past the J start (26): those slots are the parser's impossible logit, not one
    that depends on how many impossible slots the window has."""
    from alignair.network.parser import IMPOSSIBLE
    out, _ = _refined(V + N1 + D1[3:15] + N2 + J)
    assert (out["d_start_logits"][0, 26:] == IMPOSSIBLE).all()
    assert (out["d_end_logits"][0, :11] == IMPOSSIBLE).all()                # nor a D end at or before the V end


def test_a_heavy_weight_on_a_long_read_still_finds_the_d(monkeypatch):
    """At a weight where the evidence of the bases before the junction alone passes the impossible logit, the D is
    still the one its germline places: the evidence counts from the V end, and a slot no path uses never outweighs
    one a path does. N2 runs 10 bases, 4 of them past the germline's end; the J starts at 34."""
    from alignair.refine import drefine
    monkeypatch.setattr(drefine, "EVIDENCE_WEIGHT", 200.)
    out, _ = _refined(V + N1 + D1[3:15] + "C" * 10 + J, j_start=34)
    assert out["d_start_logits"].argmax().item() == 12
    assert out["d_end_logits"].argmax().item() == 24
