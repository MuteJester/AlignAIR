"""A reference keeps its locus schema (the chain head's classes, and each read's locus) through a subset, a genotype
and a FASTA: names that show one locus give it, names that show several are grouped by locus (file order within one),
and a reference whose names show none has no schema unless it is given a locus."""
import GenAIRR.data as gd
import pytest

from alignair.reference import LocusInfo
from alignair.reference.reference_set import GeneReference, ReferenceSet

pytestmark = pytest.mark.component


def _loci(reference):
    return [(l.locus, l.chain_type, l.has_d, l.ranges) for l in reference.loci]


def test_a_single_locus_genotype_gets_its_locus_schema():
    reference = ReferenceSet.from_genotype({"V": {"IGKV1-5*01": "ACGT", "IGKV3-20*01": "ACGG"},
                                            "J": {"IGKJ1*01": "TTTT"}})
    assert _loci(reference) == [("IGK", "BCR_LIGHT_KAPPA", False, {"V": (0, 2), "J": (0, 1)})]


def test_alleles_without_a_locus_name_join_the_one_the_others_name():
    reference = ReferenceSet.from_genotype({"V": {"IGHV1-2*02": "ACGTACGTACGT", "NOVEL-V*01": "TTTTGGGGCCCC"},
                                            "D": {"IGHD3-10*01": "GGGGTTTTAAAA"}, "J": {"IGHJ6*02": "CCCCAAAATTTT"}})
    assert _loci(reference) == [("IGH", "BCR_HEAVY", True, {"V": (0, 2), "D": (0, 1), "J": (0, 1)})]


def test_a_genotype_naming_no_locus_has_none_unless_given_one():
    genes = {"v": {"myV1": "ACGT"}, "j": {"myJ1": "TTTT"}}
    assert ReferenceSet.from_genotype(genes).loci == []
    assert _loci(ReferenceSet.from_genotype(genes, locus="TRB")) == [
        ("TRB", "TCR_BETA", False, {"V": (0, 1), "J": (0, 1)})]


def test_an_interleaved_two_locus_genotype_is_grouped_by_locus():
    reference = ReferenceSet.from_genotype({"V": {"IGKV1*01": "ACGT", "IGLV1*01": "ACGA", "IGKV2*01": "ACGG"},
                                            "J": {"IGLJ1*01": "TTTT", "IGKJ1*01": "TTTA"}})
    assert reference.gene("V").names == ["IGKV1*01", "IGKV2*01", "IGLV1*01"]
    assert reference.gene("J").names == ["IGKJ1*01", "IGLJ1*01"]
    assert reference.gene("V").sequences == ["ACGT", "ACGG", "ACGA"]
    assert _loci(reference) == [("IGK", "BCR_LIGHT_KAPPA", False, {"V": (0, 2), "J": (0, 1)}),
                                ("IGL", "BCR_LIGHT_LAMBDA", False, {"V": (2, 3), "J": (1, 2)})]
    reference.validate()


def test_a_multi_locus_genotype_with_an_unplaceable_allele_keeps_file_order_and_no_schema():
    reference = ReferenceSet.from_genotype({"V": {"IGKV1*01": "ACGT", "IGLV1*01": "ACGA", "myV*01": "ACGG"},
                                            "J": {"IGKJ1*01": "TTTT"}})
    assert reference.loci == [] and reference.gene("V").names == ["IGKV1*01", "IGLV1*01", "myV*01"]


def test_a_fasta_reference_carries_its_locus(tmp_path):
    path = tmp_path / "genotype.fasta"
    path.write_text(">IGHV1-2*02 some desc\nACGTACGT\n>NOVEL_IGHV9*01\nTTTTGGGG\n"
                    ">IGHD3-10*01\nGGGGTTTT\n>IGHJ6*02\nCCCCAAAA\n")
    assert ReferenceSet.from_fasta(str(path)).locus_names() == ("IGH",)


def test_a_subset_keeps_the_locus_schema():
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    subset = reference.subset({"V": reference.gene("V").names[:5]})
    assert _loci(subset) == [("IGH", "BCR_HEAVY", True, {"V": (0, 5), "D": (0, 33), "J": (0, 7)})]
    v = GeneReference(["IGHV1*01", "IGHV2*01"], ["ACGT", "ACGA"], {"IGHV1*01": 0, "IGHV2*01": 1})
    j = GeneReference(["IGHJ1*01"], ["TTTT"], {"IGHJ1*01": 0})
    human = ReferenceSet({"V": v, "J": j}, False,
                         loci=[LocusInfo("IGH", "BCR_HEAVY", False, {"V": (0, 2), "J": (0, 1)}, species="Human")])
    assert human.subset({"V": ["IGHV2*01"]}).loci == [
        LocusInfo("IGH", "BCR_HEAVY", False, {"V": (0, 1), "J": (0, 1)}, species="Human")]


def test_a_given_locus_is_upper_cased():
    reference = ReferenceSet.from_genotype({"v": {"myV1": "ACGT"}, "j": {"myJ1": "TTTT"}}, locus="trb")
    assert reference.locus_names() == ("TRB",) and reference.loci[0].chain_type == "TCR_BETA"


def test_a_fasta_naming_an_allele_twice_with_two_sequences_is_refused(tmp_path):
    from alignair.errors import InputError
    path = tmp_path / "twice.fasta"
    path.write_text(">IGHV1-2*02\nACGTACGT\n>IGHV1-2*02\nACGTACGA\n>IGHJ6*02\nCCCCAAAA\n")
    with pytest.raises(InputError, match=r"IGHV1-2\*02"):
        ReferenceSet.from_fasta(str(path))
    path.write_text(">IGHV1-2*02\nACGTACGT\n>IGHV1-2*02\nACGTACGT\n>IGHJ6*02\nCCCCAAAA\n")
    assert ReferenceSet.from_fasta(str(path)).gene("V").names == ["IGHV1-2*02"]


def _shared_v():
    """The V alleles both the TRA and the TRD dataconfig list (TRAV/DV: they rearrange in either locus)."""
    return sorted({a.name for a in gd.HUMAN_TCRA_IMGT.allele_list("v")}
                  & {a.name for a in gd.HUMAN_TCRD_IMGT.allele_list("v")})


def test_a_v_allele_shared_by_tra_and_trd_belongs_to_both():
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_TCRA_IMGT, gd.HUMAN_TCRD_IMGT)
    v = ref.gene("V")
    tra, trd = ref.locus_mask("TRA", "V"), ref.locus_mask("TRD", "V")
    shared = _shared_v()
    assert len(shared) == 9 and all(tra[v.index[n]] and trd[v.index[n]] for n in shared)
    assert int(trd.sum()) == len({a.name for a in gd.HUMAN_TCRD_IMGT.allele_list("v")})
    assert not trd[v.index["TRAV1-1*01"]]                                     # a TRA-only allele stays TRA's
    assert not ref.locus_mask("TRA", "J")[ref.gene("J").index["TRDJ1*01"]]      # J is not shared


def test_a_reference_without_shared_alleles_keeps_its_range_masks():
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB, gd.HUMAN_IGK_OGRDB)
    again = ReferenceSet.from_serializable(ref.to_serializable())
    assert all("shared" not in locus for locus in ref.to_serializable()["loci"])
    for info in ref.loci:
        for gene, (start, end) in info.ranges.items():
            mask = ref.locus_mask(info.locus, gene)
            assert mask.sum() == end - start and mask[start:end].all()
            assert (again.locus_mask(info.locus, gene) == mask).all()


def test_the_shared_alleles_round_trip_through_the_reference_json():
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_TCRA_IMGT, gd.HUMAN_TCRD_IMGT)
    again = ReferenceSet.from_serializable(ref.to_serializable())
    for locus in ("TRA", "TRD"):
        for gene in ("V", "J"):
            assert (again.locus_mask(locus, gene) == ref.locus_mask(locus, gene)).all()
