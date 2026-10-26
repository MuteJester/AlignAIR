import pytest
from alignair.aa.guard.features import FEATURE_NAMES, hmm_features
from alignair.aa.guard.profiles import build_bank_fixture

pytestmark = pytest.mark.component

IGH_V = (b">h1\nQVQLVQSGAEVKKPGASVKVSCKASGYTF\n"
         b">h2\nQVQLVQSGAEVKKPGSSVKVSCKASGGTF\n"
         b">h3\nEVQLVESGGGLVQPGGSLRLSCAASGFTF\n")
IGH_J = (b">hj1\nWGQGTLVTVSSASTKG\n>hj2\nWGQGTTVTVSSASTKG\n>hj3\nWGQGTMVTVSSASTKG\n")
IGL_V = (b">l1\nQSVLTQPPSVSGAPGQRVTISCTGSSSNI\n"
         b">l2\nQSVLTQPPSVSGAPGQRVTISCTGSSSNL\n"
         b">l3\nQSALTQPASVSGSPGQSITISCTGTSSDV\n")
IGL_J = (b">lj1\nFGGGTKLTVLGQ\n>lj2\nFGGGTKLTVLRQ\n>lj3\nFGTGTKLTVLGQ\n")

BANK = build_bank_fixture(regions={"human_igh": {"v": IGH_V, "j": IGH_J},
                                   "human_igl": {"v": IGL_V, "j": IGL_J}})

COMPLETE_VH = "QVQLVQSGAEVKKPGASVKVSCKASGYTFAAAAWGQGTLVTVSSASTKG"   # IGH V + CDR3 + IGH J
V_ONLY = "QVQLVQSGAEVKKPGASVKVSCKASGYTF"                              # IGH V, no J


def test_complete_vh_has_positive_human_igh_margin_and_order():
    f = hmm_features(BANK, COMPLETE_VH)
    assert f.best_v_group == "human_igh"
    assert f.margin > 0                    # human IGH beats the IGL alternative
    assert f.has_v and f.has_j and f.v_before_j and f.order_ok
    assert f.coverage_ok and f.combined_coverage > 0.8


def test_features_are_deterministic_and_fixed_length():
    v1 = hmm_features(BANK, COMPLETE_VH).to_vector()
    v2 = hmm_features(BANK, COMPLETE_VH).to_vector()
    assert v1.shape == (len(FEATURE_NAMES),)
    assert (v1 == v2).all()


def test_v_only_never_sets_complete_domain_coverage():
    f = hmm_features(BANK, V_ONLY)
    assert f.has_v and not f.has_j
    assert not f.coverage_ok and not f.order_ok


def test_profile_identity_is_camelid_igh_not_vhh():
    # the camelid germline profile is identified as camelid_igh (evidence), never 'vhh' (an input verdict)
    from alignair.aa.guard.profiles import RECEPTOR_GROUPS
    assert "camelid_igh" in RECEPTOR_GROUPS and "vhh" not in RECEPTOR_GROUPS
    assert all("vhh" not in name for name in FEATURE_NAMES)
    bank = build_bank_fixture(regions={"human_igh": {"v": IGH_V, "j": IGH_J},
                                       "camelid_igh": {"v": IGH_V, "j": IGH_J}})
    f = hmm_features(bank, COMPLETE_VH)
    assert "camelid_igh" in f.per_group_v_score and "vhh" not in f.per_group_v_score
