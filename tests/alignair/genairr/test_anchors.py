"""A V or J anchor GenAIRR stores as 0 (its legacy "not found") is no anchor unless GenAIRR's resolver confirms one;
every other stored anchor is kept (the byte pin of the trained references is
``tests/alignair/reference/test_reference_pin.py``)."""
import pytest

import GenAIRR.data as gd
from alignair.reference.reference_set import ReferenceSet

pytestmark = pytest.mark.component


def _stored(dataconfig, gene):
    pool = dataconfig.v_alleles if gene == "V" else dataconfig.j_alleles
    return {allele.name: allele.anchor for alleles in pool.values() for allele in alleles}


@pytest.mark.parametrize("name", ["HUMAN_IGH_OGRDB", "HUMAN_IGK_OGRDB", "HUMAN_IGL_OGRDB"])
def test_every_stored_anchor_of_the_clean_human_sets_is_kept(name):
    dataconfig = getattr(gd, name)
    reference = ReferenceSet.from_dataconfigs(dataconfig)
    for gene in ("V", "J"):
        assert reference.gene(gene).anchors == _stored(dataconfig, gene)


def test_a_zero_anchor_the_resolver_cannot_confirm_is_dropped():
    """TRBJ2-2P*01 (stored 0, no unique motif) gets none; TRBV17*01 (stored 0, codon 104 TAC) is anchored at its IMGT
    position by the owner's rule (``alignair.reference.anchors.imgt_position``), not by the stored 0."""
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_TCRB_IMGT)
    assert "TRBJ2-2P*01" not in reference.gene("J").anchors
    assert reference.gene("V").anchors["TRBV17*01"] > 0
    assert reference.gene("V").anchors["TRBV20-1*01"] > 0          # a real anchor stays


def test_resolving_leaves_the_dataconfig_untouched():
    dataconfig = gd.HUMAN_TCRB_IMGT
    before = dataconfig.compute_checksum()
    ReferenceSet.from_dataconfigs(dataconfig)
    assert _stored(dataconfig, "V")["TRBV17*01"] == 0 and dataconfig.compute_checksum() == before



TRBJ1_1 = "TGAACACTGAAGCTTTCTTTGGACAAGGCACCAGACTCACAGTTGTAG"


def test_resolve_anchor_gives_genairrs_verdict():
    from alignair.genairr.anchors import ResolvedAnchor, resolve_anchor
    igh = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    v, j = igh.gene("V"), igh.gene("J")
    v0 = v.names[0]
    assert resolve_anchor(v.sequences[0], "V", locus="IGH", gapped=v.gapped[v0]) == ResolvedAnchor(
        288, "inferred", "imgt_gapped_104", "declared IMGT alignment; conserved codon validated", (288,))
    two = j.sequences[0][:18] + "TGGGGCAAAGGC" + j.sequences[0][18:]
    assert resolve_anchor(two, "J", locus="IGH") == ResolvedAnchor(
        None, "unknown", "unique_j_fwgxg", "zero or multiple candidate landmarks; source annotation required", (18, 30))
    assert resolve_anchor(TRBJ1_1, "J", locus="TRB").position == 17
    assert resolve_anchor(TRBJ1_1, "J", locus="IGH").position is None


def test_resolve_anchor_refuses_a_d():
    from alignair.genairr.anchors import resolve_anchor
    with pytest.raises(ValueError):
        resolve_anchor("GGGCCC", "D", locus="IGH")


def test_a_genairr_without_the_resolver_is_a_missing_dependency(monkeypatch):
    import types

    from alignair.errors import MissingDependencyError
    from alignair.genairr import anchors
    monkeypatch.setattr(anchors, "_native", lambda: types.SimpleNamespace())
    with pytest.raises(MissingDependencyError) as caught:
        anchors.resolve_anchor(TRBJ1_1, "J", locus="TRB")
    assert "GenAIRR>=2.4.0,<2.5" in caught.value.hint
