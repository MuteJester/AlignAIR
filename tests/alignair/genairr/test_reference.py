"""A dataconfig's reference (``alignair.genairr.reference_from_dataconfigs``, which ``ReferenceSet.from_dataconfigs``
is) goes through ``annotate``: a stored anchor is supplied evidence, a stored 0 is resolved by GenAIRR and its verdict
recorded; each locus records its species, each mixed-species allele its own, the reference its dataconfigs; a union
naming one allele with two sequences is refused."""
import GenAIRR.data as gd
import pytest

from alignair.errors import ConfigurationError
from alignair.genairr import reference_from_dataconfigs
from alignair.reference import AnchorEvidence, SourceProvenance, anchor_report
from alignair.reference.reference_set import ReferenceSet

pytestmark = pytest.mark.component


def test_a_dataconfigs_stored_anchors_are_supplied_evidence():
    reference = reference_from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    for gene in ("V", "J"):
        info = reference.gene(gene).info
        assert {info[name].anchor_evidence for name in reference.gene(gene).names} == {
            AnchorEvidence("supplied", "supplied", None, "HUMAN_IGH_OGRDB")}
    assert reference.gene("D").info is None
    assert anchor_report(reference).complete()


def test_a_zero_anchor_is_resolved_and_its_verdict_recorded():
    reference = reference_from_dataconfigs(gd.HUMAN_TCRB_IMGT)
    evidence = reference.gene("V").info["TRBV17*01"].anchor_evidence
    assert (evidence.method, evidence.status) == ("imgt_position_104", "inferred") and "TAC" in evidence.detail
    v = reference.gene("V")
    assert v.anchors["TRBV17*01"] == len(v.gapped["TRBV17*01"][:309].replace(".", ""))      # codon 104's base
    report = anchor_report(reference)
    assert report.anchored == {"V": 98, "J": 14} and report.total == {"V": 98, "J": 16}
    assert report.missing == {"J": {"unresolved": ("TRBJ2-2P*01", "TRBJ2-7*02")}}
    assert report.noncanonical == {"V": ("TRBV17*01", "TRBV17*02", "TRBV7-3*02", "TRBV7-3*03")}


def test_a_dataconfigs_species_and_source_are_recorded():
    reference = reference_from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    assert reference.loci[0].species == "Human"
    assert reference.provenance == (SourceProvenance(
        "genairr_dataconfig", "HUMAN_IGH_OGRDB", "a66c87e1564177b09371ade7b859783845afb72118b8a6a0cc0bcc80b8ea2da4",
        "OGRDB V8"),)


def test_a_locus_of_two_species_records_each_alleles_species():
    human, mouse = gd.HUMAN_IGK_OGRDB, gd.MOUSE_IGK_IMGT
    reference = reference_from_dataconfigs(human, mouse)
    assert reference.locus_names() == ("IGK",) and reference.loci[0].species is None
    v = reference.gene("V")
    first_human = human.allele_list("v")[0].name
    first_mouse = next(a.name for a in mouse.allele_list("v") if a.name not in {x.name for x in human.allele_list("v")})
    assert v.info[first_human].species == "Human" and v.info[first_mouse].species == "Mouse"


def test_a_union_naming_one_allele_with_two_sequences_is_refused():
    with pytest.raises(ConfigurationError) as caught:
        ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_IMGT, gd.MOUSE_IGH_IMGT)
    assert "IGHV1-18*01" in str(caught.value) and len(caught.value.context["alleles"]) == 20


def test_the_classmethod_is_the_adapters():
    assert ReferenceSet.from_dataconfigs(gd.HUMAN_TCRB_IMGT).to_serializable() == reference_from_dataconfigs(
        gd.HUMAN_TCRB_IMGT).to_serializable()


def test_an_allele_named_twice_inside_one_dataconfig_keeps_its_first_sequence():
    """GenAIRR's MOUSE_IGK_IMGT names IGKV10-94*01 and IGKV10-96*01 twice with two sequences each: its reference keeps
    the first of each, as the trained references always did (the byte pin holds it)."""
    first = {}
    for allele in gd.MOUSE_IGK_IMGT.allele_list("v"):
        first.setdefault(allele.name, allele.ungapped_seq.upper())
    v = reference_from_dataconfigs(gd.MOUSE_IGK_IMGT).gene("V")
    assert v.sequences[v.index["IGKV10-94*01"]] == first["IGKV10-94*01"]


def test_a_dataconfig_unioned_with_a_copy_of_itself_is_itself():
    import copy
    alone = reference_from_dataconfigs(gd.MOUSE_IGK_IMGT)
    doubled = reference_from_dataconfigs(gd.MOUSE_IGK_IMGT, copy.deepcopy(gd.MOUSE_IGK_IMGT))
    assert all(doubled.gene(g).names == alone.gene(g).names for g in ("V", "J"))
