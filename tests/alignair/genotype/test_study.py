"""Genotype-constraint study: end-to-end structure + the tight-genotype accuracy effect."""
import pytest


@pytest.mark.slow
def test_genotype_study_returns_constrained_and_unconstrained():
    import GenAIRR.data as gd
    from alignair.core import AlignAIR
    from alignair.core.config import AlignAIRConfig
    from alignair.genotype.study import genotype_study
    from alignair.reference.reference_set import ReferenceSet
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    model, ref = AlignAIR(cfg).eval(), ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    s = genotype_study(model, ref, gd.HUMAN_IGH_OGRDB, n=8, seed=0, method="mask",
                       stratum="moderate", device="cpu", batch_size=8)
    assert s["n"] == 8 and s["method"] == "mask"
    assert "v_acc" in s["unconstrained"] and "v_acc" in s["constrained"]
    assert s["genotype_sizes"]["v"] <= s["reference_sizes"]["v"]         # genotype is a subset


@pytest.mark.slow
def test_tight_genotype_constrains_to_requested_size():
    import GenAIRR.data as gd
    from alignair.core import AlignAIR
    from alignair.core.config import AlignAIRConfig
    from alignair.genotype.study import genotype_study
    from alignair.reference.reference_set import ReferenceSet
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    model, ref = AlignAIR(cfg).eval(), ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    s = genotype_study(model, ref, gd.HUMAN_IGH_OGRDB, n=8, seed=0, method="mask",
                       stratum="moderate", v_genotype_size=25, device="cpu", batch_size=8)
    assert s["genotype_sizes"]["v"] == 25                                 # tight V genotype honored
