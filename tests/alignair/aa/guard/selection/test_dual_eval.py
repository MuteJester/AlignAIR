import pytest
import torch

from alignair.aa.guard.dual_eval import (
    assert_binding_is_blind, binding_scores, dual_eval,
)
from alignair.aa.guard.features import FEATURE_NAMES, hmm_features
from alignair.aa.guard.fusion import fit_fusion
from alignair.aa.guard.model import GuardModel
from alignair.aa.guard.profiles import build_bank_fixture
from alignair.aa.guard.tokenizer import MAX_LEN, AaTokenizer

pytestmark = pytest.mark.component

IGH_V = b">h1\nQVQLVQSGAEVKKPGASVKVSCKASGYTF\n>h2\nQVQLVQSGAEVKKPGSSVKVSCKASGGTF\n>h3\nEVQLVESGGGLVQPGGSLRLSCAASGFTF\n"
IGH_J = b">hj1\nWGQGTLVTVSSASTKG\n>hj2\nWGQGTTVTVSSASTKG\n>hj3\nWGQGTMVTVSSASTKG\n"
IGK_V = b">k1\nDIQMTQSPSSLSASVGDRVTITCRASQSI\n>k2\nDIQMTQSPSSLSASVGDRVTITCRASQGI\n>k3\nDIQLTQSPSFLSASVGDRVTITCRASQGI\n"
IGK_J = b">kj1\nFGQGTKVEIKRTVA\n>kj2\nFGQGTKLEIKRTVA\n>kj3\nFGGGTKVEIKRTVA\n"
BANK = build_bank_fixture(regions={"human_igh": {"v": IGH_V, "j": IGH_J},
                                   "human_igl": {"v": IGK_V, "j": IGK_J}})
VK = "DIQMTQSPSSLSASVGDRVTITCRASQSIAAAAFGQGTKVEIKRTVA"
VH = "QVQLVQSGAEVKKPGASVKVSCKASGYTFAAAAWGQGTLVTVSSASTKG"


def _rec(aa, group="positive_real", expected="supported", lineage="L1", parent="P1"):
    return {"aa": aa, "id": f"id_{aa[:6]}", "group": group, "category": group, "expected_state": expected,
            "lineage_key": lineage, "parent_lineage": parent, "donor": "d1", "study": "s1", "role_split": "selection"}


def _setup():
    torch.manual_seed(0)
    ck = BANK.checksum()
    vecs = [hmm_features(BANK, s).to_vector().tolist() for s in (VK, VH)]
    fusion = fit_fusion(FEATURE_NAMES, vecs, ck)
    model = GuardModel("C2", len(FEATURE_NAMES), width=16, depth=1, heads=2)
    return model, fusion, AaTokenizer(), ck


def test_dual_eval_identical_inputs_and_full_raw_evidence():
    model, fusion, tok, ck = _setup()
    recs = [_rec(VH), _rec(VK, group="igk_igl", expected="unknown_ood")]
    res = dual_eval(model, BANK, fusion, tok, recs, fold="igk_igl", candidate="C2", bank_checksum=ck)
    assert res.identical_inputs                                    # same seq order AND unit/label contract
    assert res.reference_covered.input_contract_digest == res.blind.input_contract_digest
    assert res.reference_covered.masked_groups == () and res.reference_covered.diagnostic_only
    assert res.blind.masked_groups == ("human_igk", "human_igl") and res.blind.binding
    sc = res.blind.scores[0]
    assert len(sc.completeness_logits) == 5 and len(sc.domain_type_logits) == 8   # all decision logits preserved
    assert set(sc.hmm_evidence) >= {"has_v", "has_j", "order_ok", "combined_coverage", "best_v_group", "margin"}
    assert sc.expected_state and sc.group and sc.unit["parent_lineage"] == "P1"
    assert sc.structural_prefilter["length_contract"] == "ok"
    for s in res.blind.scores:                                     # blind removes the fold profile from the winner
        assert "human_igl" not in s.c0_reason and s.hmm_evidence["best_v_group"] != "human_igl"


def test_binding_consumes_only_blind():
    model, fusion, tok, ck = _setup()
    res = dual_eval(model, BANK, fusion, tok, [_rec(VH)], fold="igk_igl", candidate="C2", bank_checksum=ck)
    assert binding_scores(res) is res.blind
    with pytest.raises(ValueError):
        assert_binding_is_blind(res.reference_covered)


def test_over_length_input_is_withheld_no_model_score():
    model, fusion, tok, ck = _setup()
    res = dual_eval(model, BANK, fusion, tok, [_rec("A" * (MAX_LEN + 5))],
                    fold="igk_igl", candidate="C2", bank_checksum=ck)
    sc = res.blind.scores[0]
    assert not sc.within_length_contract and sc.applicability_logit is None and sc.energy is None
    assert sc.completeness_logits is None and sc.domain_type_logits is None
    assert sc.structural_prefilter["length_contract"] == "over_length" and sc.c0_state == "insufficient_evidence"


def test_contract_digest_changes_when_unit_metadata_changes():
    model, fusion, tok, ck = _setup()
    a = dual_eval(model, BANK, fusion, tok, [_rec(VH, lineage="L1")], fold="igk_igl", candidate="C2", bank_checksum=ck)
    b = dual_eval(model, BANK, fusion, tok, [_rec(VH, lineage="L2")], fold="igk_igl", candidate="C2", bank_checksum=ck)
    # same sequence, different independence unit -> the contract digest must differ (not only seq hashes)
    assert a.blind.input_order_sha256 == b.blind.input_order_sha256
    assert a.blind.input_contract_digest != b.blind.input_contract_digest
