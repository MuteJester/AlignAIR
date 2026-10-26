import pytest
from alignair.aa.guard.c0_rules import c0_decide
from alignair.aa.guard.features import hmm_features
from alignair.aa.guard.loco import assert_reason_blind, masked_profiles_for
from alignair.aa.guard.profiles import build_bank_fixture

pytestmark = pytest.mark.component

IGH_V = (b">h1\nQVQLVQSGAEVKKPGASVKVSCKASGYTF\n>h2\nQVQLVQSGAEVKKPGSSVKVSCKASGGTF\n>h3\nEVQLVESGGGLVQPGGSLRLSCAASGFTF\n")
IGH_J = b">hj1\nWGQGTLVTVSSASTKG\n>hj2\nWGQGTTVTVSSASTKG\n>hj3\nWGQGTMVTVSSASTKG\n"
IGK_V = (b">k1\nDIQMTQSPSSLSASVGDRVTITCRASQSI\n>k2\nDIQMTQSPSSLSASVGDRVTITCRASQGI\n>k3\nDIQLTQSPSFLSASVGDRVTITCRASQGI\n")
IGK_J = b">kj1\nFGQGTKVEIKRTVA\n>kj2\nFGQGTKLEIKRTVA\n>kj3\nFGGGTKVEIKRTVA\n"
BANK = build_bank_fixture(regions={"human_igh": {"v": IGH_V, "j": IGH_J},
                                   "human_igk": {"v": IGK_V, "j": IGK_J}})
VK = "DIQMTQSPSSLSASVGDRVTITCRASQSIAAAAFGQGTKVEIKRTVA"       # complete human IGK


def test_reference_covered_sees_the_profile_but_blind_removes_it():
    ref = hmm_features(BANK, VK)
    assert "human_igk" in ref.per_group_v_score and ref.best_v_group == "human_igk"
    blind = hmm_features(BANK, VK, mask_groups=masked_profiles_for("igk_igl"))
    assert "human_igk" not in blind.per_group_v_score and "human_igl" not in blind.per_group_v_score
    assert blind.masked_groups == ("human_igk", "human_igl")
    assert blind.best_v_group not in ("human_igk", "human_igl")       # winner cannot be a masked group


def test_blind_reason_never_names_a_masked_group():
    masked = masked_profiles_for("igk_igl")
    ref = c0_decide(hmm_features(BANK, VK))
    assert "human_igk" in ref.reason                                   # reference-covered names it
    blind = c0_decide(hmm_features(BANK, VK, mask_groups=masked))
    assert_reason_blind(blind.reason, masked)                          # no raise
    assert "human_igk" not in blind.reason and "human_igl" not in blind.reason
