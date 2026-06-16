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
