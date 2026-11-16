import pytest
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet

pytestmark = pytest.mark.component


def test_build_from_igh():
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    assert rs.has_d is True
    assert set(rs.genes) == {"V", "D", "J"}
    assert len(rs.gene("V").names) == 198 and len(rs.gene("J").names) == 7
    assert len(rs.gene("D").names) == 33  # real D alleles only, no 'Short-D'
    # germline sequences are uppercased nucleotides aligned with names
    v0 = rs.gene("V")
    assert v0.sequences[0] == v0.sequences[0].upper()
    assert set(v0.sequences[0]) <= set("ACGTN")
    assert v0.index[v0.names[0]] == 0


def test_light_chain_has_no_d():
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    assert rs.has_d is False
    assert set(rs.genes) == {"V", "J"}


def test_union_of_two_dataconfigs():
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB, gdata.HUMAN_IGK_OGRDB)
    # union V = IGH V (198) + IGK V (168), names unique
    assert len(rs.gene("V").names) == 198 + 168
    assert rs.has_d is True  # at least one chain has D
    assert len(set(rs.gene("V").names)) == len(rs.gene("V").names)


def test_from_alleles_heavy_and_novel():
    # a genotype with a known-subset plus a NOVEL allele the model never trained on
    genes = {
        "V": {"IGHV1-2*02": "ACGTACGTACGT", "NOVEL-V*01": "TTTTGGGGCCCC"},
        "D": {"IGHD3-10*01": "GGGGTTTTAAAA"},
        "J": {"IGHJ6*02": "CCCCAAAATTTT"},
    }
    rs = ReferenceSet.from_alleles(genes)
    assert rs.has_d is True
    assert rs.gene("V").names == ["IGHV1-2*02", "NOVEL-V*01"]
    assert rs.gene("V").index["NOVEL-V*01"] == 1
    assert rs.gene("V").sequences[1] == "TTTTGGGGCCCC"


def test_from_alleles_light_no_d():
    rs = ReferenceSet.from_alleles({"v": {"IGKV1*01": "ACGT"}, "j": {"IGKJ1*01": "TTTT"}})
    assert rs.has_d is False
    assert set(rs.genes) == {"V", "J"}


def test_from_fasta_infers_gene_and_loads(tmp_path):
    p = tmp_path / "genotype.fasta"
    p.write_text(">IGHV1-2*02 some desc\nACGTACGT\n>NOVEL_IGHV9*01\nTTTTGGGG\n"
                 ">IGHD3-10*01\nGGGGTTTT\n>IGHJ6*02\nCCCCAAAA\n")
    rs = ReferenceSet.from_fasta(str(p))
    assert rs.gene("V").names == ["IGHV1-2*02", "NOVEL_IGHV9*01"]   # V inferred, incl. novel
    assert rs.gene("D").names == ["IGHD3-10*01"]
    assert rs.gene("J").names == ["IGHJ6*02"]
    assert rs.gene("V").sequences[0] == "ACGTACGT"


def test_subset_preserves_anchors():
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    keepV = rs.gene("V").names[:5]
    sub = rs.subset({"V": keepV, "D": rs.gene("D").names[:3], "J": rs.gene("J").names[:2]})
    assert sub.gene("V").names == keepV
    assert len(sub.gene("D")) == 3 and len(sub.gene("J")) == 2
    # anchors carried over for the kept alleles (junction stays computable)
    assert sub.gene("V").anchors is not None
    assert all(sub.gene("V").anchors[n] == rs.gene("V").anchors[n] for n in keepV)


def test_subset_preserves_the_gapped_forms():
    """A genotype subset keeps each kept allele's IMGT-gapped form (the frame a reference-reading model lays V on)."""
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    keepV = rs.gene("V").names[:5]
    sub = rs.subset({"V": keepV})
    assert sub.gene("V").gapped == {n: rs.gene("V").gapped[n] for n in keepV}


def test_from_alleles_with_anchors():
    rs = ReferenceSet.from_alleles(
        {"v": {"IGHV1-2*02": "ACGT"}, "j": {"IGHJ6*02": "TTTT"}},
        anchors={"V": {"IGHV1-2*02": 288}},
    )
    assert rs.gene("V").anchors == {"IGHV1-2*02": 288}
    assert rs.gene("J").anchors is None


def test_a_subset_keeps_the_references_allele_order():
    """The bank's row order is the reference's whatever order (or set) the alleles are named in: the chip's scores
    depend on it."""
    rs = ReferenceSet.from_alleles({"V": {"V*01": "ACGTAC", "V*02": "ACGAAC", "V*03": "ACCTAC", "V*04": "TCGTAC"},
                                     "J": {"J*01": "TTTGGA"}})
    sub = rs.subset({"V": ["V*04", "V*02", "V*02"]})
    assert sub.gene("V").names == ["V*02", "V*04"]
    assert sub.gene("V").index == {"V*02": 0, "V*04": 1} and sub.gene("V").sequences == ["ACGAAC", "TCGTAC"]


@pytest.mark.parametrize("unplaced", ["mystery_allele", "DONOR_V1"])
def test_a_fasta_allele_whose_segment_the_name_does_not_tell_is_refused_by_name(tmp_path, unplaced):
    """An allele is placed by its IG/TR locus code (IGHV..., TRBJ...), else by the one V, D or J letter its name holds;
    a name with none, or with several (``DONOR_V1``), is refused, never dropped or filed under a guessed segment."""
    from alignair.errors import InputError
    p = tmp_path / "genotype.fasta"
    p.write_text(f">IGHV1-2*02\nACGTACGT\n>{unplaced}\nTTTTGGGG\n>IGHJ6*02\nCCCCAAAA\n")
    with pytest.raises(InputError, match=unplaced) as caught:
        ReferenceSet.from_fasta(str(p))
    assert caught.value.context["alleles"] == [unplaced] and "IGHV" in caught.value.hint


def test_a_fasta_name_with_one_segment_letter_is_placed_by_it(tmp_path):
    p = tmp_path / "genotype.fasta"
    p.write_text(">myV1\nACGTACGT\n>myJ1\nCCCCAAAA\n")
    reference = ReferenceSet.from_fasta(str(p))
    assert (reference.gene("V").names, reference.gene("J").names) == (["myV1"], ["myJ1"])
