"""Completing a reference against a known one: each allele gets an IMGT-gapped form and a conserved anchor from the
known reference (the same allele, or its nearest same-length variant: a SNP variant carries the gaps and the anchor
over), so a reference-agnostic model reads a user's FASTA as it reads the reference it trained on."""
import pytest

from alignair.reference.complete import complete_reference
from alignair.reference.reference_set import GeneReference, ReferenceSet

pytestmark = pytest.mark.component


def _known():
    v = GeneReference(["IGHV1-1*01", "IGHV2-1*01"], ["ACGTACGTAC", "TTTTGGGGCC"], {"IGHV1-1*01": 0, "IGHV2-1*01": 1},
                      anchors={"IGHV1-1*01": 4, "IGHV2-1*01": 5},
                      gapped={"IGHV1-1*01": "ACGT..ACGTAC", "IGHV2-1*01": "TTTTG..GGGCC"})
    d = GeneReference(["IGHD1-1*01"], ["GGGCCC"], {"IGHD1-1*01": 0})
    j = GeneReference(["IGHJ1*01"], ["TTTGGAA"], {"IGHJ1*01": 0}, anchors={"IGHJ1*01": 3})
    return ReferenceSet({"V": v, "D": d, "J": j}, has_d=True)


def _users(v, j=None):
    return ReferenceSet.from_genotype({"V": v, "D": {"IGHD1-1*01": "GGGCCC"}, "J": j or {"IGHJ1*01": "TTTGGAA"}})


def test_a_known_allele_keeps_its_gapped_form_and_anchor():
    done = complete_reference(_users({"IGHV1-1*01": "ACGTACGTAC"}), _known())
    assert done.gene("V").gapped == {"IGHV1-1*01": "ACGT..ACGTAC"}
    assert done.gene("V").anchors == {"IGHV1-1*01": 4} and done.gene("J").anchors == {"IGHJ1*01": 3}


def test_a_snp_variant_takes_its_nearest_alleles_gaps_with_its_own_bases():
    done = complete_reference(_users({"IGHV1-1*novel": "ACGAACGTAC"}), _known())     # one SNP from IGHV1-1*01
    assert done.gene("V").gapped == {"IGHV1-1*novel": "ACGA..ACGTAC"}
    assert done.gene("V").anchors == {"IGHV1-1*novel": 4}


def test_a_variant_differing_at_more_than_a_tenth_of_its_bases_takes_nothing():
    done = complete_reference(_users({"IGHV1-1*far": "ACGAACGTAA"}), _known())       # two SNPs in ten bases
    assert not done.gene("V").gapped and not done.gene("V").anchors


def test_a_gapped_sequence_keeps_its_gaps():
    done = complete_reference(_users({"IGHV9-9*01": "ACGT..ACGTCC"}), _known())
    assert done.gene("V").sequences == ["ACGTACGTCC"] and done.gene("V").gapped == {"IGHV9-9*01": "ACGT..ACGTCC"}


def test_an_allele_without_a_same_length_neighbour_stays_without_gaps_or_anchor():
    done = complete_reference(_users({"IGHV7-7*01": "ACGTACG"}), _known())
    assert done.gene("V").sequences == ["ACGTACG"]
    assert not done.gene("V").gapped and not done.gene("V").anchors
