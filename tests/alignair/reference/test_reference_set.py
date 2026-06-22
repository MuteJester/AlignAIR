import pytest
import torch
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet


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


def test_genotype_mask():
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    names = rs.gene("V").names
    allowed = {names[0], names[5], names[10]}
    mask = rs.genotype_mask("V", allowed)
    assert mask.dtype == torch.bool and mask.shape == (len(names),)
    assert mask.sum().item() == 3
    assert mask[0] and mask[5] and mask[10]


def test_from_genotype_heavy_and_novel():
    # a genotype with a known-subset plus a NOVEL allele the model never trained on
    genes = {
        "V": {"IGHV1-2*02": "ACGTACGTACGT", "NOVEL-V*01": "TTTTGGGGCCCC"},
        "D": {"IGHD3-10*01": "GGGGTTTTAAAA"},
        "J": {"IGHJ6*02": "CCCCAAAATTTT"},
    }
    rs = ReferenceSet.from_genotype(genes)
    assert rs.has_d is True
    assert rs.gene("V").names == ["IGHV1-2*02", "NOVEL-V*01"]
    assert rs.gene("V").index["NOVEL-V*01"] == 1
    assert rs.gene("V").sequences[1] == "TTTTGGGGCCCC"
    # genotype_mask works over the novel-inclusive reference
    m = rs.genotype_mask("V", {"NOVEL-V*01"})
    assert m.tolist() == [False, True]


def test_from_genotype_light_no_d():
    rs = ReferenceSet.from_genotype({"v": {"IGKV1*01": "ACGT"}, "j": {"IGKJ1*01": "TTTT"}})
    assert rs.has_d is False
    assert set(rs.genes) == {"V", "J"}


def test_yaml_roundtrip(tmp_path):
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    p = tmp_path / "genotype.yaml"
    rs.to_yaml(str(p))
    rs2 = ReferenceSet.from_yaml(str(p))
    for g in ("V", "D", "J"):
        assert rs2.gene(g).names == rs.gene(g).names
        assert rs2.gene(g).sequences == rs.gene(g).sequences


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


def test_from_genotype_with_anchors():
    rs = ReferenceSet.from_genotype(
        {"v": {"IGHV1-2*02": "ACGT"}, "j": {"IGHJ6*02": "TTTT"}},
        anchors={"V": {"IGHV1-2*02": 288}},
    )
    assert rs.gene("V").anchors == {"IGHV1-2*02": 288}
    assert rs.gene("J").anchors is None
