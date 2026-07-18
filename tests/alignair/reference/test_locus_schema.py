"""The ReferenceSet carries a per-locus schema (ordered loci + per-gene allele-index membership)
so multi-chain prediction can label the locus, mask cross-locus alleles, and assemble per-record."""
import os

import numpy as np
import pytest

import GenAIRR.data as gd
from alignair.reference.reference_set import ReferenceSet

_IGKL = ".private/models/alignair_igkl_v1.cal.alignair"


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


def test_locus_lacking_a_gene_masks_to_no_call():
    """A known locus that has no alleles for a gene (e.g. D on a light chain) -> all-False, so a
    light-chain read is never handed a heavy-chain D allele."""
    from alignair.reference.reference_set import GeneReference, LocusInfo, ReferenceSet as RS
    genes = {"V": GeneReference(["IGHV1*01", "IGKV1*01"], ["A", "C"], {}),
             "D": GeneReference(["IGHD1*01"], ["G"], {})}
    ref = RS(genes, has_d=True, loci=[
        LocusInfo("IGH", "BCR_HEAVY", True, {"V": (0, 1), "D": (0, 1)}),
        LocusInfo("IGK", "BCR_LIGHT_KAPPA", False, {"V": (1, 2)})])
    assert not ref.locus_mask("IGK", "D").any()                 # IGK has no D -> no-call, NOT all-True
    assert ref.locus_mask("IGH", "D").all()


def test_same_locus_panels_merge_into_one_contiguous_locus():
    """Merging two same-locus references (e.g. IGH panels) is allowed: one IGH locus over the whole
    head (this is the benchmark superset-reference use case)."""
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB, gd.HUMAN_IGH_EXTENDED)
    assert ref.locus_names() == ("IGH",)                        # merged into ONE locus, not two
    assert np.all(ref.locus_mask("IGH", "V"))                   # contiguous over the whole V head


@pytest.mark.skipif(not os.path.exists(_IGKL), reason="shipped IGK+IGL model not present")
def test_real_multichain_model_locus_schema_matches_training():
    """The shipped IGK+IGL model's embedded reference (schema inferred from names, saved before the
    schema existed) yields chain-class order (IGK, IGL) with masks that partition the head — matching
    the dataconfig order the chain_type head was trained on."""
    from alignair.api import load_model
    _, ref = load_model(_IGKL, device="cpu")
    assert ref.locus_names() == ("IGK", "IGL")                  # class 0 = IGK, class 1 = IGL
    for gene in ("V", "J"):
        k, l = ref.locus_mask("IGK", gene), ref.locus_mask("IGL", gene)
        assert not np.any(k & l) and np.all(k | l)              # disjoint + cover
        names = list(ref.gene(gene).names)
        assert all(names[i].upper().startswith("IGK") for i in np.where(k)[0])
        assert all(names[i].upper().startswith("IGL") for i in np.where(l)[0])


def test_interleaved_split_locus_fails_closed():
    """A locus split by another locus's alleles (interleaved) is refused rather than masked by a
    silently-widened min/max range."""
    import pytest
    from alignair.reference.reference_set import GeneReference, ReferenceSet as RS
    genes = {"V": GeneReference(["IGKV1*01", "IGLV1*01", "IGKV2*01"], ["A", "C", "G"], {}),
             "J": GeneReference(["IGKJ1*01"], ["T"], {})}
    with pytest.raises(ValueError, match="contiguous"):
        RS.from_serializable(RS(genes, has_d=False).to_serializable())
