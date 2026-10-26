import pytest
from dataclasses import replace

from alignair.aa.guard.c0_rules import C0_THRESHOLDS, c0_decide
from alignair.aa.guard.features import hmm_features
from alignair.aa.guard.profiles import build_bank_fixture

pytestmark = pytest.mark.component

IGH_V = (b">h1\nQVQLVQSGAEVKKPGASVKVSCKASGYTF\n>h2\nQVQLVQSGAEVKKPGSSVKVSCKASGGTF\n>h3\nEVQLVESGGGLVQPGGSLRLSCAASGFTF\n")
IGH_J = b">hj1\nWGQGTLVTVSSASTKG\n>hj2\nWGQGTTVTVSSASTKG\n>hj3\nWGQGTMVTVSSASTKG\n"
IGK_V = (b">k1\nDIQMTQSPSSLSASVGDRVTITCRASQSI\n>k2\nDIQMTQSPSSLSASVGDRVTITCRASQGI\n>k3\nDIQLTQSPSFLSASVGDRVTITCRASQGI\n")
IGK_J = b">kj1\nFGQGTKVEIKRTVA\n>kj2\nFGQGTKLEIKRTVA\n>kj3\nFGGGTKVEIKRTVA\n"

BANK = build_bank_fixture(regions={"human_igh": {"v": IGH_V, "j": IGH_J},
                                   "human_igk": {"v": IGK_V, "j": IGK_J}})

VH = "QVQLVQSGAEVKKPGASVKVSCKASGYTFAAAAWGQGTLVTVSSASTKG"     # complete human IGH VH
VK = "DIQMTQSPSSLSASVGDRVTITCRASQSIAAAAFGQGTKVEIKRTVA"       # complete human IGK
V_ONLY = "QVQLVQSGAEVKKPGASVKVSCKASGYTF"                       # heavy V, no J


def test_complete_human_vh_is_supported():
    v = c0_decide(hmm_features(BANK, VH))
    assert v.state == "supported" and "human_igh" in v.reason


def test_winning_igk_is_unsupported_locus_when_precision_gated():
    f = hmm_features(BANK, VK)
    assert f.best_v_group == "human_igk"                      # kappa wins the competition
    v = c0_decide(f)
    assert v.state == "unsupported_locus" and "human_igk" in v.reason


def test_winning_igk_is_unknown_ood_when_not_precision_gated():
    f = hmm_features(BANK, VK)
    thr = dict(C0_THRESHOLDS, precision_gated_loci=frozenset())   # gate no alternatives
    v = c0_decide(f, thr)
    assert v.state == "unknown_ood"


def test_v_only_is_insufficient_evidence():
    v = c0_decide(hmm_features(BANK, V_ONLY))
    assert v.state == "insufficient_evidence"


def test_low_coverage_is_insufficient_evidence():
    # a complete VH feature with coverage forced below the floor -> insufficient_evidence
    f = replace(hmm_features(BANK, VH), combined_coverage=0.4, coverage_ok=False)
    v = c0_decide(f)
    assert v.state == "insufficient_evidence"
