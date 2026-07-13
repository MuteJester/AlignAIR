"""P0-6: the ReferenceSet carries a per-locus schema (ordered loci + per-gene allele-index membership)
so multi-chain prediction can label the locus, mask cross-locus alleles, and assemble per-record."""
import numpy as np
import pytest

import GenAIRR.data as gd
from alignair.reference.reference_set import ReferenceSet


@pytest.fixture(scope="module")
def igk_igl():
    return ReferenceSet.from_dataconfigs(gd.HUMAN_IGK_OGRDB, gd.HUMAN_IGL_OGRDB)


def test_ordered_loci_and_names(igk_igl):
    assert igk_igl.locus_names() == ("IGK", "IGL")
    assert [l.has_d for l in igk_igl.loci] == [False, False]      # light chains: no D


def test_locus_masks_partition_the_v_head(igk_igl):
    k = igk_igl.locus_mask("IGK", "V")
    l = igk_igl.locus_mask("IGL", "V")
    n = len(igk_igl.gene("V"))
    assert k.shape == (n,) and l.shape == (n,)
    assert not np.any(k & l)                                      # disjoint
    assert np.all(k | l)                                         # cover the whole head
    # the mask actually selects that locus's alleles
    names = list(igk_igl.gene("V").names)
    assert all(names[i].upper().startswith("IGKV") for i in np.where(k)[0])
    assert all(names[i].upper().startswith("IGLV") for i in np.where(l)[0])


def test_locus_schema_survives_serialization(igk_igl):
    rebuilt = ReferenceSet.from_serializable(igk_igl.to_serializable())
    assert rebuilt.locus_names() == ("IGK", "IGL")
    assert np.array_equal(rebuilt.locus_mask("IGK", "V"), igk_igl.locus_mask("IGK", "V"))


def test_single_chain_has_one_locus():
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    assert ref.locus_names() == ("IGH",)
    assert ref.loci[0].has_d is True
    assert np.all(ref.locus_mask("IGH", "V"))                    # the whole head is IGH
