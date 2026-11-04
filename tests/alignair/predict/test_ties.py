"""The V alleles a read cannot separate from its call (``predict.ties``): an allele ties when, over the read bases the
call's alignment puts on germline bases, it matches at least as many as the call, compared at IMGT columns; only an
allele the read may be called with ties. ``v_call`` lists the call, then its ties."""
import random

import numpy as np
import pytest

from alignair.predict.state import GermlineAlignment
from alignair.predict.ties import tied_alleles

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


def _ties(read, aln, alleles, gapped=None, allowed=None):
    return tied_alleles([read], [aln], _Reference(alleles, gapped), allowed=allowed)[0]


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


def test_only_alleles_the_read_may_be_called_with_tie():
    read = A[:90]
    aln = GermlineAlignment("A", 0, 90, 0, 90, "90M")
    alleles = {"A": A, "B": B}
    assert _ties(read, aln, alleles, allowed=np.array([True, False])) == ("A",)
    assert _ties(read, aln, alleles, allowed=np.array([[True, False]])) == ("A",)


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
