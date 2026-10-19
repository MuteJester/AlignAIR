"""Tests for the assay Reference and CIGAR utilities."""
from __future__ import annotations

from alignair_assay.core.cigar import format_cigar, parse_cigar, walk
from alignair_assay.core.reference import Reference, gene_name


def test_parse_format_roundtrip():
    assert format_cigar(parse_cigar("3M0I2M1D4M")) == "5M1D4M"


def test_walk_columns():
    cols = walk(parse_cigar("2M1I1M1D1M"), 10, 20)
    assert [c.read_pos for c in cols] == [10, 11, 12, 13, None, 14]
    assert [c.germ_pos for c in cols] == [20, 21, None, 22, 23, 24]


def test_reference_from_dataconfig(reference):
    assert len(reference.alleles["v"]) == 198
    assert len(reference.alleles["d"]) == 33
    assert len(reference.alleles["j"]) == 7
    for letter in ("v", "j"):
        for name in reference.alleles[letter]:
            assert name in reference.anchors
    name = next(iter(reference.alleles["v"]))
    assert reference.has(name)
    assert reference.gene_letter(name) == "v"
    assert reference.seq(name) == reference.alleles["v"][name]
    assert not reference.has("NOPE*01")
    assert gene_name("IGHVF1-G2*01") == "IGHVF1-G2"


def test_reference_digest():
    ref = Reference(alleles={"v": {"A*01": "ACGT", "A*02": "ACGA"}, "j": {"J*01": "TTT"}}, anchors={"A*01": 1})
    shuffled = Reference(alleles={"j": {"J*01": "TTT"}, "v": {"A*02": "ACGA", "A*01": "ACGT"}}, anchors={})
    assert ref.digest() == shuffled.digest()  # order and anchors do not matter
    assert len(ref.digest()) == 64
    changed = Reference(alleles={"v": {"A*01": "ACGT", "A*02": "ACGG"}, "j": {"J*01": "TTT"}}, anchors={})
    renamed = Reference(alleles={"v": {"A*01": "ACGT", "A*03": "ACGA"}, "j": {"J*01": "TTT"}}, anchors={})
    moved = Reference(alleles={"v": {"A*01": "ACGT"}, "j": {"J*01": "TTT", "A*02": "ACGA"}}, anchors={})
    assert len({ref.digest(), changed.digest(), renamed.digest(), moved.digest()}) == 4
