"""Genotype task 8: recovery benchmark harness."""
import math

import pytest

from alignair.genotype.benchmark import recovery


def test_recovery_precision_recall_and_zygosity():
    # truth uses IGHV1-2*01/*02 (het), IGHJ4*02 (homo); inferred adds a wrong allele + misses J
    truth = [{"v_call": "IGHV1-2*01", "j_call": "IGHJ4*02"},
             {"v_call": "IGHV1-2*02", "j_call": "IGHJ4*02"}]
    gs = {"genotype_class_list": [{"documented_alleles": [
        {"label": "IGHV1-2*01"}, {"label": "IGHV1-2*02"}, {"label": "IGHV9-9*01"}]}]}
    r = recovery(gs, truth)
    assert r["n_truth_alleles"] == 3 and r["n_inferred"] == 3
    assert abs(r["allele_precision"] - 2 / 3) < 1e-9        # 2 of 3 inferred are true
    assert abs(r["allele_recall"] - 2 / 3) < 1e-9           # J missed -> 2 of 3 truth recovered
    # IGHV1-2 zygosity: truth 2 alleles, inferred 2 -> correct; J absent from inferred
    assert 0.0 <= r["zygosity_acc"] <= 1.0


def test_recovery_handles_empty_inference():
    r = recovery({"genotype_class_list": [{"documented_alleles": []}]}, [{"v_call": "IGHV1-2*01"}])
    assert r["n_inferred"] == 0 and math.isnan(r["allele_precision"]) and r["allele_recall"] == 0.0


@pytest.mark.slow
def test_simulate_repertoire_restricts_to_genotype():
    import GenAIRR.data as gd
    from alignair.genotype.benchmark import simulate_repertoire
    from alignair.reference.reference_set import ReferenceSet
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    genotype = {"v": set(ref.gene("V").names[:40]), "j": set(ref.gene("J").names[:5])}
    seqs, truth = simulate_repertoire(gd.HUMAN_IGH_OGRDB, genotype, 12, seed=0)
    assert len(seqs) >= 6
    for r in truth:                                         # every kept read's truth lies in the genotype
        assert r["v_call"].split(",")[0] in genotype["v"]
        assert r["j_call"].split(",")[0] in genotype["j"]
