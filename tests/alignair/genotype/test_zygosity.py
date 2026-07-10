"""Genotype task 4: evidence-gated zygosity / CNV decision + pruning."""
from alignair.genotype.zygosity import call_gene


def test_prune_leakage_sibling_without_evidence():
    gc = call_gene("v", {"V0": 1.0, "V1": 0.02}, {"V0": True, "V1": False}, min_support=0.1)
    assert [a["name"] for a in gc.alleles] == ["V0"] and gc.zygosity == "homozygous"
    assert any("V1" in r and "excluded" in r for r in gc.reasons)


def test_keep_sibling_with_covered_diagnostic_snp():
    gc = call_gene("v", {"V0": 1.0, "V1": 0.02}, {"V0": True, "V1": True}, min_support=0.1)
    assert {a["name"] for a in gc.alleles} == {"V0", "V1"} and gc.zygosity == "heterozygous"


def test_three_alleles_without_independent_evidence_not_duplication():
    gc = call_gene("v", {"V0": 1.0, "V1": 0.5, "V2": 0.4},
                   {"V0": True, "V1": False, "V2": False}, min_support=0.1)
    assert gc.zygosity == "heterozygous" and len(gc.alleles) == 2
    assert any("demoted" in r for r in gc.reasons)


def test_three_alleles_with_evidence_is_duplication_candidate():
    gc = call_gene("v", {"V0": 1.0, "V1": 0.5, "V2": 0.4},
                   {"V0": True, "V1": True, "V2": True}, min_support=0.1)
    assert "duplication" in gc.zygosity and len(gc.alleles) == 3


def test_deletion_candidate_on_near_zero_support():
    gc = call_gene("v", {"V0": 0.001}, {"V0": False}, min_support=0.1, deletion_floor=0.01)
    assert gc.deletion_candidate and gc.zygosity == "deletion-candidate"
    assert any("deletion" in r.lower() for r in gc.reasons)


def test_every_decision_has_a_reason():
    gc = call_gene("v", {"V0": 1.0, "V1": 0.02}, {"V0": True, "V1": False}, min_support=0.1)
    assert all(isinstance(r, str) and r for r in gc.reasons) and len(gc.reasons) >= 2
