import pytest
import torch

from alignair.aa.guard.dual_eval import (
    ModeScores, assert_binding_is_blind, binding_scores, dual_eval,
)
from alignair.aa.guard.features import FEATURE_NAMES, hmm_features
from alignair.aa.guard.fusion import fit_fusion
from alignair.aa.guard.model import GuardModel
from alignair.aa.guard.profiles import build_bank_fixture
from alignair.aa.guard.tokenizer import MAX_LEN

IGH_V = b">h1\nQVQLVQSGAEVKKPGASVKVSCKASGYTF\n>h2\nQVQLVQSGAEVKKPGSSVKVSCKASGGTF\n>h3\nEVQLVESGGGLVQPGGSLRLSCAASGFTF\n"
IGH_J = b">hj1\nWGQGTLVTVSSASTKG\n>hj2\nWGQGTTVTVSSASTKG\n>hj3\nWGQGTMVTVSSASTKG\n"
IGK_V = b">k1\nDIQMTQSPSSLSASVGDRVTITCRASQSI\n>k2\nDIQMTQSPSSLSASVGDRVTITCRASQGI\n>k3\nDIQLTQSPSFLSASVGDRVTITCRASQGI\n"
IGK_J = b">kj1\nFGQGTKVEIKRTVA\n>kj2\nFGQGTKLEIKRTVA\n>kj3\nFGGGTKVEIKRTVA\n"
BANK = build_bank_fixture(regions={"human_igh": {"v": IGH_V, "j": IGH_J},
                                   "human_igl": {"v": IGK_V, "j": IGK_J}})
VK = "DIQMTQSPSSLSASVGDRVTITCRASQSIAAAAFGQGTKVEIKRTVA"
VH = "QVQLVQSGAEVKKPGASVKVSCKASGYTFAAAAWGQGTLVTVSSASTKG"


def _setup():
    torch.manual_seed(0)
    ck = BANK.checksum()
    vecs = [hmm_features(BANK, s).to_vector().tolist() for s in (VK, VH)]
    fusion = fit_fusion(FEATURE_NAMES, vecs, ck)
    from alignair.aa.guard.tokenizer import AaTokenizer
    model = GuardModel("C2", len(FEATURE_NAMES), width=16, depth=1, heads=2)
    return model, fusion, AaTokenizer(), ck


def test_dual_eval_identical_ordered_inputs_and_mask_split():
    model, fusion, tok, ck = _setup()
    recs = [{"aa": VH}, {"aa": VK}]
    res = dual_eval(model, BANK, fusion, tok, recs, fold="igk_igl", candidate="C2", bank_checksum=ck)
    assert res.identical_inputs                                    # both modes saw the same ordered inputs
    assert res.reference_covered.masked_groups == () and res.reference_covered.diagnostic_only
    assert res.blind.masked_groups == ("human_igk", "human_igl") and res.blind.binding
    # blind removed the fold profile from the competitive set for every record
    for sc in res.blind.scores:
        assert "human_igl" not in sc.decision_reason and sc.best_v_group != "human_igl"


def test_binding_consumes_only_blind():
    model, fusion, tok, ck = _setup()
    res = dual_eval(model, BANK, fusion, tok, [{"aa": VH}], fold="igk_igl", candidate="C2", bank_checksum=ck)
    assert binding_scores(res) is res.blind
    with pytest.raises(ValueError):
        assert_binding_is_blind(res.reference_covered)            # diagnostic mode can never bind


def test_over_length_input_is_withheld_no_model_score():
    model, fusion, tok, ck = _setup()
    long_seq = "A" * (MAX_LEN + 5)
    res = dual_eval(model, BANK, fusion, tok, [{"aa": long_seq}], fold="igk_igl", candidate="C2", bank_checksum=ck)
    sc = res.blind.scores[0]
    assert not sc.within_length_contract and sc.applicability_logit is None and sc.energy is None
    assert sc.decision_state == "insufficient_evidence"


def test_reference_covered_can_see_the_fold_profile():
    model, fusion, tok, ck = _setup()
    res = dual_eval(model, BANK, fusion, tok, [{"aa": VK}], fold="igk_igl", candidate="C2", bank_checksum=ck)
    # reference-covered keeps human_igl in play; blind removes it -> the two modes are genuinely different
    assert res.reference_covered.scores[0].best_v_group == "human_igl"
    assert res.blind.scores[0].best_v_group != "human_igl"
