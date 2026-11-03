"""A read without a V (a D-J rearrangement, a read that starts inside the junction) gets no V call
(``predict.v_presence``): it holds no exact ``V_KMER``-mer of any reference V allele, and the V the germline reader
placed is no V either (under ``V_MIN_SPAN`` read bases or under ``V_MIN_IDENTITY`` identity along its CIGAR). Its row
has no V call or coordinates, carries ``v_presence`` ``"absent"`` and is a partial AIRR row (``no_v``); its D and J
are called as any read's. A heavily mutated V that keeps no exact k-mer keeps its call by its span and identity."""
import random

import pytest

from alignair.predict.state import GermlineAlignment
from alignair.predict.v_presence import V_KMER, V_MIN_IDENTITY, V_MIN_SPAN, v_absent

pytestmark = pytest.mark.component

# IGHV1-2*02 from its first base: a V germline long enough for every case below
V = ("CAGGTGCAGCTGGTGCAGTCTGGGGCTGAGGTGAAGAAGCCTGGGGCCTCAGTGAAGGTCTCCTGCAAGGCTTCTGGATACACCTTCACCGGCTACTATATGCAC"
     "TGGGTGCGACAGGCCCCTGGACAAGGGCTTGAGTGGATGGGA")
_SWAP = {"A": "C", "C": "G", "G": "T", "T": "A"}


class _Gene:
    def __init__(self, names, sequences):
        self.names, self.sequences = list(names), list(sequences)
        self.index = {name: i for i, name in enumerate(self.names)}


class _Reference:
    def __init__(self, v):
        self._v = _Gene(["V1"], [v])

    def gene(self, letter):
        assert letter == "V"
        return self._v


def _kmers(seq):
    return {seq[i:i + V_KMER] for i in range(len(seq) - V_KMER + 1)}


def _mutated(seq, every):
    """``seq`` with its base at every ``every``-th position (from ``every - 1``) substituted."""
    return "".join(_SWAP[b] if (i + 1) % every == 0 else b for i, b in enumerate(seq))


def _flank(n, seed=7):
    return "".join(random.Random(seed).choice("ACGT") for _ in range(n))


def _whole(read, germline):
    """The V placed over the whole read against the whole germline, base for base."""
    return GermlineAlignment("V1", 0, len(read), 0, len(germline), f"{len(read)}M")


def test_a_read_holding_an_exact_v_kmer_holds_a_v_however_short_its_v():
    read = "TTTT" + V[10:10 + V_KMER] + "AAAA"
    short = GermlineAlignment("V1", 4, 7, 10, 13, "3M")
    assert v_absent([read], [short], _Reference(V)) == [False]


def test_a_read_without_a_v_kmer_whose_v_is_under_the_minimum_span_has_no_v():
    # the read's V is ten germline bases matched exactly (identity 1.0), too few for any exact k-mer
    read = _flank(40) + V[50:60] + _flank(40, seed=8)
    assert not _kmers(read) & _kmers(V)
    stub = GermlineAlignment("V1", 40, 50, 50, 60, "10M")
    assert 10 < V_MIN_SPAN
    assert v_absent([read], [stub], _Reference(V)) == [True]


def test_a_mutated_v_without_an_exact_kmer_keeps_its_v():
    # a substitution every 10th base leaves no exact 15-mer, at identity 0.9
    read = _mutated(V, 10)
    assert not _kmers(read) & _kmers(V)
    assert len(read) >= V_MIN_SPAN and 0.9 >= V_MIN_IDENTITY
    assert v_absent([read], [_whole(read, V)], _Reference(V)) == [False]


def test_a_long_v_without_a_kmer_under_the_minimum_identity_has_no_v():
    read = _mutated(V, 3)                       # identity 2/3
    assert not _kmers(read) & _kmers(V)
    assert 2 / 3 < V_MIN_IDENTITY
    assert v_absent([read], [_whole(read, V)], _Reference(V)) == [True]


def test_the_identity_counts_an_indel_column_as_a_mismatch():
    # 0.9 identity on the bases, with indel columns enough to take it under the minimum
    read = _mutated(V, 10)
    n = len(read)
    gaps = n // 4                               # n - gaps read bases align, gaps read bases are insertions
    aln = GermlineAlignment("V1", 0, n, 0, n - gaps, f"{n - gaps}M{gaps}I")
    assert v_absent([read], [aln], _Reference(V)) == [True]


def test_a_read_without_a_v_call_has_no_v_to_drop():
    assert v_absent([_flank(60), V], [None, None], _Reference(V)) == [False, False]


def test_reads_without_a_v_call_read_no_reference():
    class _NoReference:
        def gene(self, letter):
            raise AssertionError("read the reference")
    assert v_absent([_flank(60)], [None], _NoReference()) == [False]
