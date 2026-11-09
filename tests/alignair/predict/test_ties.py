"""The V alleles a read cannot separate from its call (``predict.ties``): an allele ties when, over the read bases the
call's alignment puts on germline bases, it matches at least as many as the call, compared at IMGT columns; only an
allele of the reference ties. ``v_call`` lists the call, then its ties. An allele matching strictly more of those read
bases than the call is better (``better_v_alleles``): it becomes the call."""
import random

import numpy as np
import pytest

from alignair.predict.state import GermlineAlignment
from alignair.predict.ties import better_v_alleles, tied_alleles

pytestmark = pytest.mark.component

A = "".join(random.Random(3).choice("ACGT") for _ in range(120))
_SWAP = {"A": "C", "C": "G", "G": "T", "T": "A"}


def _substituted(seq, position):
    return seq[:position] + _SWAP[seq[position]] + seq[position + 1:]


B = _substituted(A, 100)            # A's sibling: one SNP at germline position 100
C = _substituted(A, 20)             # one SNP at position 20


class _Gene:
    def __init__(self, alleles, gapped=None):
        self.names = list(alleles)
        self.sequences = list(alleles.values())
        self.index = {name: i for i, name in enumerate(self.names)}
        self.gapped = gapped


class _Reference:
    def __init__(self, alleles, gapped=None):
        self._v = _Gene(alleles, gapped)

    def gene(self, letter):
        assert letter == "V"
        return self._v


def _ties(read, aln, alleles, gapped=None):
    return tied_alleles([read], [aln], _Reference(alleles, gapped))[0]


def test_an_allele_equal_to_the_call_over_the_reads_v_bases_ties():
    read = A[:90]                                   # the read stops before the SNP at 100
    assert _ties(read, GermlineAlignment("A", 0, 90, 0, 90, "90M"), {"A": A, "B": B}) == ("A", "B")


def test_an_allele_differing_where_the_read_shows_the_calls_base_does_not_tie():
    read = A[60:]
    assert _ties(read, GermlineAlignment("A", 0, 60, 60, 120, "60M"), {"A": A, "B": B}) == ("A",)


def test_an_allele_matching_more_read_bases_than_the_call_ties():
    read = B[60:]                                   # the read shows B's base at 100, the call is A
    assert _ties(read, GermlineAlignment("A", 0, 60, 60, 120, "60M"), {"A": A, "B": B}) == ("A", "B")


def test_alleles_compare_at_imgt_columns():
    # D is A without its first 3 bases, its IMGT gapped form leading with 3 gap columns: its germline position p is
    # A's p + 3, the same IMGT column
    d = A[3:]
    gapped = {"A": A, "D": "..." + d}
    read = A[30:90]
    assert _ties(read, GermlineAlignment("A", 0, 60, 30, 90, "60M"), {"A": A, "D": d}, gapped) == ("A", "D")


def test_an_allele_without_a_base_at_an_aligned_column_matches_no_read_base_there():
    d = A[3:]
    gapped = {"A": A, "D": "..." + d}
    read = A[:60]                                   # the read shows A's first 3 bases, which D lacks
    assert _ties(read, GermlineAlignment("A", 0, 60, 0, 60, "60M"), {"A": A, "D": d}, gapped) == ("A",)


def test_an_inserted_read_base_is_compared_with_no_allele():
    read = A[:50] + "T" + A[50:90]
    aln = GermlineAlignment("A", 0, 91, 0, 90, "50M1I40M")
    assert _ties(read, aln, {"A": A, "B": B, "C": C}) == ("A", "B")


def test_the_ties_follow_the_call_by_matches_then_reference_order():
    read = B[60:]                                   # B matches every read base, A and E one fewer
    e = _substituted(A, 110)
    alleles = {"E": _substituted(e, 100), "A": A, "B": B}
    assert _ties(read, GermlineAlignment("A", 0, 60, 60, 120, "60M"), alleles) == ("A", "B", "E")
    read = A[:90]
    alleles = {"E": e, "A": A, "B": B}
    assert _ties(read, GermlineAlignment("A", 0, 90, 0, 90, "90M"), alleles) == ("A", "E", "B")


def test_a_read_without_a_v_alignment_has_no_ties():
    assert tied_alleles([A], [None], _Reference({"A": A})) == [()]


def test_reads_without_a_v_alignment_read_no_reference():
    class _NoReference:
        def gene(self, letter):
            raise AssertionError("read the reference")
    assert tied_alleles([A], [None], _NoReference()) == [()]


def _better(read, aln, alleles, candidates=(), gapped=None):
    return better_v_alleles([read], [aln], _Reference(alleles, gapped), [tuple(candidates)])[0]


def test_an_allele_matching_more_read_bases_than_the_call_is_better():
    read = B[60:]                                   # the read shows B's base at 100, the call is A
    assert _better(read, GermlineAlignment("A", 0, 60, 60, 120, "60M"), {"A": A, "B": B}) == "B"


def test_no_allele_is_better_than_a_call_matching_most():
    read = A[60:]                                   # the read shows A's base at 100
    assert _better(read, GermlineAlignment("A", 0, 60, 60, 120, "60M"), {"A": A, "B": B}) is None
    read = A[:90]                                   # B matches as many: a tie is not better
    assert _better(read, GermlineAlignment("A", 0, 90, 0, 90, "90M"), {"A": A, "B": B}) is None


def test_among_equal_better_alleles_the_models_candidates_come_first_then_reference_order():
    f = _substituted(B, 10)                         # B with another SNP where the read shows nothing
    read, aln = B[60:], GermlineAlignment("A", 0, 60, 60, 120, "60M")
    alleles = {"F": f, "A": A, "B": B}
    assert _better(read, aln, alleles) == "F"
    assert _better(read, aln, alleles, candidates=("A", "B")) == "B"


def test_a_better_allele_is_found_at_imgt_columns():
    d = _substituted(B, 3)[3:]                      # B without its first 3 bases, gapped on A's columns
    gapped = {"A": A, "D": "..." + d}
    read = B[30:110]
    assert _better(read, GermlineAlignment("A", 0, 80, 30, 110, "80M"), {"A": A, "D": d}, gapped=gapped) == "D"


def test_a_read_without_a_v_alignment_has_no_better_allele():
    assert better_v_alleles([A], [None], _Reference({"A": A}), [()]) == [None]


def test_an_allele_outside_the_reads_locus_is_never_better():
    read, aln = B[60:], GermlineAlignment("A", 0, 60, 60, 120, "60M")         # B is better, but of another locus
    assert better_v_alleles([read], [aln], _Reference({"A": A, "B": B}), [()],
                            allowed=[np.array([True, False])]) == [None]


def test_the_best_allele_of_the_reads_locus_is_better_when_the_best_of_all_is_outside_it():
    x = _substituted(B, 110)                        # the read is x: x matches every base, B all but 110, A two fewer
    read, aln = x[60:], GermlineAlignment("A", 0, 60, 60, 120, "60M")
    alleles = {"A": A, "B": B, "X": x}
    assert better_v_alleles([read], [aln], _Reference(alleles), [()]) == ["X"]
    assert better_v_alleles([read], [aln], _Reference(alleles), [()], allowed=[np.array([True, True, False])]) == ["B"]


def test_ties_outside_the_reads_locus_are_not_listed():
    read, aln = A[:90], GermlineAlignment("A", 0, 90, 0, 90, "90M")           # B ties: its SNP is past the read
    reference = _Reference({"A": A, "B": B})
    assert tied_alleles([read], [aln], reference) == [("A", "B")]
    assert tied_alleles([read], [aln], reference, allowed=[np.array([True, False])]) == [("A",)]
