"""What a source tells about its alleles, loci and itself travels with the reference: each allele's functionality,
species, source label and anchor evidence (``AlleleInfo``, ``AnchorEvidence``), each locus's species, the reference's
sources (``SourceProvenance``). The new keys are optional in ``alignair.reference.v1`` and left out when empty, so a
reference without them keeps its bytes and the model card's hashes."""
import hashlib
import pickle

import GenAIRR.data as gd
import pytest

from alignair.errors import DataContractError
from alignair.model_file import serialize
from alignair.reference import AlleleInfo, AnchorEvidence, LocusInfo, SourceProvenance
from alignair.reference.reference_set import GeneReference, ReferenceSet
from tests.support.reference_pin import v1_payload

pytestmark = pytest.mark.component

V0 = "IGHV1-18*01"
EVIDENCE = AnchorEvidence("imgt_gapped_104", "inferred", "declared IMGT alignment; conserved codon validated", None)
INFO = AlleleInfo("F", "Homo sapiens", "M99641", EVIDENCE)
SOURCE = SourceProvenance("imgt", "IGHV.fasta", "ab" * 32, "202643-4", "2026-11-06T00:00:00Z", "CC BY 4.0", "IMGT")


def _reference(*, info=True):
    v = GeneReference([V0, "IGHV1-2*02"], ["ACGTACGTAC", "ACGAACGTAC"], {V0: 0, "IGHV1-2*02": 1},
                      anchors={V0: 4}, gapped={V0: "ACGT..ACGTAC"}, info={V0: INFO} if info else None)
    d = GeneReference(["IGHD1-1*01"], ["GGGCCC"], {"IGHD1-1*01": 0})
    j = GeneReference(["IGHJ1*01"], ["TTTGGA"], {"IGHJ1*01": 0}, anchors={"IGHJ1*01": 0})
    locus = LocusInfo("IGH", "BCR_HEAVY", True, {"V": (0, 2), "D": (0, 1), "J": (0, 1)},
                      species="Homo sapiens" if info else None)
    return ReferenceSet({"V": v, "D": d, "J": j}, True, loci=[locus], provenance=(SOURCE,) if info else ())


def test_allele_info_takes_only_f_orf_or_p():
    for functional in ("F", "ORF", "P", None):
        assert AlleleInfo(functional=functional).functional == functional
    with pytest.raises(ValueError):
        AlleleInfo(functional="pseudogene")


def test_anchor_evidence_takes_only_the_known_statuses():
    with pytest.raises(ValueError):
        AnchorEvidence("x", "ok")


def test_the_new_fields_round_trip():
    reference = _reference()
    back = ReferenceSet.from_serializable(reference.to_serializable())
    assert back.gene("V").info == {V0: INFO}
    assert back.loci == reference.loci and back.loci[0].species == "Homo sapiens"
    assert back.provenance == (SOURCE,)
    assert serialize.reference_to_json(back) == serialize.reference_to_json(reference)


def test_a_v1_reference_round_trips_byte_for_byte():
    v1 = v1_payload(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB).to_serializable())
    data = serialize.reference_to_json(ReferenceSet.from_serializable(v1))
    assert hashlib.sha256(data).hexdigest() == "71c3ff8a1cf3ea8e530cdf64e08ada000abb222cbf535edee41098ed23287bab"


def test_empty_new_fields_are_left_out():
    payload = _reference(info=False).to_serializable()
    assert "provenance" not in payload
    assert not any("info" in gene for gene in payload["genes"].values())
    assert not any("species" in locus for locus in payload["loci"])


def test_unknown_keys_are_ignored():
    payload = _reference().to_serializable()
    payload["future"] = 1
    payload["genes"]["V"]["future"] = 1
    payload["loci"][0]["future"] = 1
    payload["genes"]["V"]["info"][V0]["future"] = 1
    payload["genes"]["V"]["info"][V0]["anchor_evidence"]["future"] = 1
    payload["provenance"][0]["future"] = 1
    back = ReferenceSet.from_serializable(payload)
    assert back.gene("V").info == {V0: INFO} and back.provenance == (SOURCE,)


def test_info_for_an_allele_the_gene_lacks_is_refused():
    payload = _reference().to_serializable()
    payload["genes"]["V"]["info"]["IGHV9-99*01"] = {"functional": "F"}
    with pytest.raises(DataContractError) as caught:
        ReferenceSet.from_serializable(payload)
    assert caught.value.context == {"gene": "V", "allele": "IGHV9-99*01"}


def test_the_card_hashes_ignore_the_new_fields():
    with_info, without = _reference(), _reference(info=False)
    assert serialize.allele_order_sha256(with_info) == serialize.allele_order_sha256(without)
    assert serialize.reference_fasta_sha256(with_info) == serialize.reference_fasta_sha256(without)


def test_a_subset_keeps_info_and_provenance():
    subset = _reference().subset({"V": [V0]})
    assert subset.gene("V").info == {V0: INFO} and subset.provenance == (SOURCE,)
    assert _reference().subset({"V": ["IGHV1-2*02"]}).gene("V").info is None


def test_a_reference_with_info_pickles():
    reference = _reference()
    assert pickle.loads(pickle.dumps(reference)).to_serializable() == reference.to_serializable()
