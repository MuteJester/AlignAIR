"""``ReferenceSet.extend`` adds another reference's alleles (a donor's novel or undocumented ones) at the end of
their locus's block, keeping the locus schema, anchors, gapped forms, info and provenance; an allele it already holds
is skipped, one named twice with two sequences is refused (or kept as it was), one no locus can hold is refused."""
import GenAIRR.data as gd
import pytest

from alignair.errors import InputError
from alignair.reference.reference_set import ReferenceSet

pytestmark = pytest.mark.component


@pytest.fixture(scope="module")
def igk_igl():
    return ReferenceSet.from_dataconfigs(gd.HUMAN_IGK_OGRDB, gd.HUMAN_IGL_OGRDB)


def _snp(sequence, at=100):
    return sequence[:at] + ("A" if sequence[at] != "A" else "C") + sequence[at + 1:]


def test_an_added_allele_goes_to_the_end_of_its_locus(igk_igl):
    v = igk_igl.gene("V")
    extended = igk_igl.extend(ReferenceSet.from_genotype({"V": {"IGKV1-5TCS*99": _snp(v.sequences[0])}, "J": {}}))
    assert extended.gene("V").index["IGKV1-5TCS*99"] == 168
    assert [(l.locus, l.ranges) for l in extended.loci] == [("IGK", {"V": (0, 169), "J": (0, 8)}),
                                                             ("IGL", {"V": (169, 350), "J": (8, 18)})]
    extended.validate()
    lambda_v = v.names[168:]
    assert {n: extended.gene("V").anchors[n] for n in lambda_v if n in v.anchors} == {
        n: v.anchors[n] for n in lambda_v if n in v.anchors}
    assert {n: extended.gene("V").gapped[n] for n in lambda_v} == {n: v.gapped[n] for n in lambda_v}


def test_an_allele_named_twice_with_two_sequences_is_refused(igk_igl):
    other = ReferenceSet.from_genotype({"V": {"IGKV1-5TCS*03": "ACGT"}, "J": {}})
    with pytest.raises(InputError, match=r"IGKV1-5TCS\*03"):
        igk_igl.extend(other)
    assert igk_igl.extend(other, on_conflict="keep").gene("V").names == igk_igl.gene("V").names


def test_an_allele_already_held_is_skipped(igk_igl):
    v = igk_igl.gene("V")
    extended = igk_igl.extend(ReferenceSet.from_genotype({"V": {v.names[0]: v.sequences[0]}, "J": {}}))
    assert extended.gene("V").names == v.names and extended.loci == igk_igl.loci


def test_a_new_locus_is_added_after_the_others():
    igk = ReferenceSet.from_dataconfigs(gd.HUMAN_IGK_OGRDB)
    extended = igk.extend(ReferenceSet.from_genotype({"V": {"IGLV1*01": "ACGT"}, "J": {"IGLJ1*01": "TTTT"}}))
    assert extended.locus_names() == ("IGK", "IGL")
    assert extended.loci[1].ranges == {"V": (168, 169), "J": (8, 9)}
    extended.validate()


def test_an_allele_no_locus_can_hold_is_refused(igk_igl):
    other = ReferenceSet.from_genotype({"V": {"myV*01": "ACGT"}, "J": {}})
    with pytest.raises(InputError, match=r"myV\*01"):
        igk_igl.extend(other)
    igh = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    extended = igh.extend(other)
    assert extended.gene("V").names[-1] == "myV*01" and extended.loci[0].ranges["V"] == (0, len(igh.gene("V")) + 1)


def test_on_conflict_takes_error_or_keep(igk_igl):
    with pytest.raises(ValueError):
        igk_igl.extend(ReferenceSet.from_genotype({"V": {"myV*01": "ACGT"}, "J": {}}), on_conflict="replace")
