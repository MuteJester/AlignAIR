"""Tests for the assay Reference and CIGAR utilities."""
from __future__ import annotations

import pytest

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


def test_reference_gapped_v_from_dataconfig(reference, tmp_path):
    # The dataconfig's IMGT-gapped V sequences, uppercase, each the allele's sequence once the gaps are removed.
    for name, sequence in reference.alleles["v"].items():
        gapped = reference.gapped(name)
        assert gapped == gapped.upper() and gapped.replace(".", "") == sequence and len(gapped) >= 318
    assert reference.gapped("IGHVF1-G1*01").startswith("CAGGTCACCTTGAAGGAGTCTGGTCCT...")
    assert reference.gapped(next(iter(reference.alleles["d"]))) is None
    with pytest.raises(KeyError):
        reference.gapped("NOPE*01")
    # Equality and the digest are the alleles' alone; a FASTA holds no gapped sequence.
    plain = Reference(alleles=reference.alleles, anchors=reference.anchors)
    assert plain == reference and plain.digest() == reference.digest()
    path = tmp_path / "germline.fasta"
    path.write_text(">IGHV1-2*02\nACGT\n", encoding="utf-8")
    assert Reference.from_fasta(path).gapped("IGHV1-2*02") is None


def test_reference_digest():
    ref = Reference(alleles={"v": {"A*01": "ACGT", "A*02": "ACGA"}, "j": {"J*01": "TTT"}}, anchors={"A*01": 1})
    shuffled = Reference(alleles={"j": {"J*01": "TTT"}, "v": {"A*02": "ACGA", "A*01": "ACGT"}}, anchors={})
    assert ref.digest() == shuffled.digest()  # order and anchors do not matter
    assert len(ref.digest()) == 64
    changed = Reference(alleles={"v": {"A*01": "ACGT", "A*02": "ACGG"}, "j": {"J*01": "TTT"}}, anchors={})
    renamed = Reference(alleles={"v": {"A*01": "ACGT", "A*03": "ACGA"}, "j": {"J*01": "TTT"}}, anchors={})
    moved = Reference(alleles={"v": {"A*01": "ACGT"}, "j": {"J*01": "TTT", "A*02": "ACGA"}}, anchors={})
    assert len({ref.digest(), changed.digest(), renamed.digest(), moved.digest()}) == 4


def test_reference_gapped_digest_follows_the_gapped_forms_and_leaves_the_digest_alone():
    # the IMGT-gapped forms decide Cys104 and the 318 rule, so what reads them is keyed on them; digest() stays the
    # alleles' alone, so every cache keyed on it stays valid
    alleles = {"v": {"A*01": "ACGT", "A*02": "ACGA"}}
    ref = Reference(alleles=alleles, anchors={}, gapped_v={"A*01": "AC.GT", "A*02": "ACG.A"})
    regapped = Reference(alleles=alleles, anchors={}, gapped_v={"A*01": "A.CGT", "A*02": "ACG.A"})
    shuffled = Reference(alleles=alleles, anchors={"A*01": 3}, gapped_v={"A*02": "ACG.A", "A*01": "AC.GT"})
    plain = Reference(alleles=alleles, anchors={})
    assert ref.digest() == regapped.digest() == plain.digest()
    assert ref.gapped_digest() == shuffled.gapped_digest() and len(ref.gapped_digest()) == 64
    assert len({ref.gapped_digest(), regapped.gapped_digest(), plain.gapped_digest()}) == 3


def test_the_dataconfigs_digest_is_unchanged(reference):
    # P1 and P2 caches are keyed on it
    assert reference.digest() == "4d70015675eb18e3af36cc94c516a83aeb7288cc8aa84aaf05d358c25489bf89"
    assert reference.gapped_digest() != reference.digest()
