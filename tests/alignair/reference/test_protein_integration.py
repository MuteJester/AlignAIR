"""Real-provider protein reference compatibility."""
import pytest
from alignair.reference import ReferenceSet, build_protein_germlines, build_protein_reference

pytestmark = pytest.mark.integration


@pytest.mark.slow
def test_build_protein_germlines_real_igh_all_v_and_j_pass_anchor_qc():
    import GenAIRR as ga
    dc = ga.HUMAN_IGH_OGRDB
    gset = build_protein_germlines(dc, genes=("V", "J"))
    # every V allele translation carries Cys at its anchor residue; every J carries Trp.
    assert len(gset.genes["V"]) >= 190          # 198 V alleles, near-all pass
    assert len(gset.genes["J"]) == 7            # all 7 J pass
    for germ in gset.genes["V"].values():
        assert germ.aa_seq[germ.anchor_aa] == "C"
    for germ in gset.genes["J"].values():
        assert germ.aa_seq[germ.anchor_aa] == "W"
    # quarantine, if any, is small and reported with a reason string.
    assert all(isinstance(r, str) for r in gset.quarantined["V"].values())


def test_named_dataconfig_and_ordered_reference_views_have_identical_labels():
    import GenAIRR as ga
    source = ReferenceSet.from_dataconfigs(ga.HUMAN_IGH_OGRDB)
    direct = build_protein_reference(source)
    named = build_protein_reference('HUMAN_IGH_OGRDB')
    assert direct.source is source
    assert direct.metadata() == named.metadata()
    assert {gene: labels.to_json() for gene, labels in direct.class_maps.items()} == {
        gene: labels.to_json() for gene, labels in named.class_maps.items()}
