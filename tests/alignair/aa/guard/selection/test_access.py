import hashlib
import json
from types import SimpleNamespace

import pytest

from alignair.aa.guard.access import (
    AccessViolation, ChallengeLoader, HashMismatch, NoAdmissibleThreshold, SelectionLoader,
    ThresholdArtifactMissing, freeze_fold_threshold, threshold_grid_hash, verify_challenge_access,
)
from alignair.aa.guard.dual_eval import ModeScores
from alignair.aa.guard.freeze import freeze_pre_unlock_bundle, model_identity
from alignair.aa.guard.loco import masked_profiles_for

pytestmark = pytest.mark.component

FOLD = "igk_igl"
PROV, FILE_SHA = "p" * 64, "f" * 64
MODEL_ID = model_identity(PROV, FILE_SHA)
EXCLUDED = (FOLD, tuple(sorted(masked_profiles_for(FOLD))))
REQ = ["tcr", "positive_real"]


def _rec(group, *, accept, unit_id):
    comp = (5.0, 0, 0, 0, 0) if accept else (0, 5.0, 0, 0, 0)
    typ = (5.0, 0, 0, 0, 0, 0, 0, 0) if accept else (0, 5.0, 0, 0, 0, 0, 0, 0)
    hmm = ({"has_v": True, "has_j": True, "order_ok": True, "combined_coverage": 0.95,
            "best_v_group": "human_igh", "best_alt_score": 0.0, "human_igh_score": 30.0} if accept else
           {"has_v": True, "has_j": True, "order_ok": True, "combined_coverage": 0.95,
            "best_v_group": "human_igl", "best_alt_score": 30.0, "human_igh_score": 0.0})
    return SimpleNamespace(within_length_contract=True, hmm_evidence=hmm, completeness_logits=comp,
                           domain_type_logits=typ, applicability_logit=(5.0 if accept else -5.0), energy=-10.0,
                           group=group, record_id_sha256=unit_id, unit={"lineage": unit_id, "parent_lineage": unit_id})


def _passing_blind():
    scores = ([_rec("tcr", accept=False, unit_id=f"T{i}") for i in range(80)]
              + [_rec("positive_real", accept=True, unit_id=f"R{i}") for i in range(60)])
    return ModeScores(mode="blind", diagnostic_only=False, binding=True,
                      masked_groups=tuple(sorted(masked_profiles_for(FOLD))), input_order_sha256="x",
                      input_contract_digest="c", scores=tuple(scores))


def _ref():
    return ModeScores(mode="reference_covered", diagnostic_only=True, binding=False, masked_groups=(),
                      input_order_sha256="x", input_contract_digest="c", scores=())


def _bundle(model_hash=MODEL_ID):
    return freeze_pre_unlock_bundle(
        model_hash=model_hash, normalization_hash="n" * 64, source_vocab_hash="v" * 64,
        bank_checksum="b" * 64, evaluator_commit="c" * 40, cohort_manifest_hash="h" * 64,
        threshold_grid_hash=threshold_grid_hash())


def _freeze(tmp_path, bundle, chal_sha, *, blind=None, req=REQ):
    return freeze_fold_threshold(FOLD, "C2", blind or _passing_blind(), req, model_identity_hash=MODEL_ID,
                                 checkpoint_provenance_digest=PROV, checkpoint_file_sha256=FILE_SHA,
                                 selection_manifest_sha256="s" * 64, challenge_file_sha256=chal_sha,
                                 frozen_bundle=bundle, out_path=tmp_path / "thr.json", mock=True)


def test_selection_loader_cannot_open_challenge_file(tmp_path):
    sel, chal = tmp_path / "selection", tmp_path / "challenge"
    sel.mkdir(); chal.mkdir()
    (sel / "s.bin").write_bytes(b"ok"); (chal / "c.bin").write_bytes(b"secret")
    loader = SelectionLoader(sel, chal)
    assert loader.load(sel / "s.bin") == b"ok"
    with pytest.raises(AccessViolation):
        loader.load(chal / "c.bin")                               # (1) hard refusal


def test_reference_covered_cannot_freeze_a_threshold(tmp_path):
    with pytest.raises(ValueError):
        freeze_fold_threshold(FOLD, "C2", _ref(), REQ, model_identity_hash=MODEL_ID, checkpoint_provenance_digest=PROV,
                              checkpoint_file_sha256=FILE_SHA, selection_manifest_sha256="s" * 64,
                              challenge_file_sha256="d" * 64, frozen_bundle=_bundle(), out_path=tmp_path / "t.json")


def test_no_admissible_threshold_is_a_stop(tmp_path):
    # a required group that always false-accepts -> no grid point admissible -> STOP
    bad = ModeScores(mode="blind", diagnostic_only=False, binding=True, masked_groups=(), input_order_sha256="x",
                     input_contract_digest="c",
                     scores=tuple(_rec("tcr", accept=True, unit_id=f"T{i}") for i in range(80)))
    with pytest.raises(NoAdmissibleThreshold):
        _freeze(tmp_path, _bundle(), "d" * 64, blind=bad, req=["tcr"])


def test_threshold_freeze_stores_full_per_group_table(tmp_path):
    art = _freeze(tmp_path, _bundle(), "d" * 64)
    assert art.is_complete() and set(art.per_group) >= {"tcr", "positive_real"}
    for g in ("tcr", "positive_real"):
        row = art.per_group[g]
        assert {"k", "n", "estimate", "upper", "ceiling", "passed"} <= set(row) and row["passed"]


def test_challenge_refused_without_artifact(tmp_path):
    chal = tmp_path / "challenge"; chal.mkdir(); (chal / "c.bin").write_bytes(b"x")
    loader = ChallengeLoader(challenge_root=chal, selection_root=tmp_path / "selection",
                             threshold_artifact_path=tmp_path / "missing.json", audit_path=tmp_path / "a.jsonl",
                             frozen_bundle=_bundle(), expected_selection_manifest_sha256="s" * 64,
                             expected_excluded=EXCLUDED, expected_challenge_digest="d" * 64)
    with pytest.raises(ThresholdArtifactMissing):
        loader.open(chal / "c.bin")                               # (2) refuse: no artifact


def test_challenge_unlock_audits_then_returns_matching_bytes(tmp_path):
    bundle = _bundle()
    chal = tmp_path / "challenge"; chal.mkdir()
    payload = b'{"aa":"X"}\n'
    (chal / "c.jsonl").write_bytes(payload)
    chal_sha = hashlib.sha256(payload).hexdigest()
    _freeze(tmp_path, bundle, chal_sha)
    audit = tmp_path / "audit.jsonl"
    loader = ChallengeLoader(challenge_root=chal, selection_root=tmp_path / "selection",
                             threshold_artifact_path=tmp_path / "thr.json", audit_path=audit, frozen_bundle=bundle,
                             expected_selection_manifest_sha256="s" * 64, expected_excluded=EXCLUDED,
                             expected_challenge_digest=chal_sha)
    assert loader.open(chal / "c.jsonl") == payload
    rec = [json.loads(x) for x in audit.read_text().splitlines() if x.strip()][-1]
    assert rec["event"] == "challenge_unlock" and rec["fold"] == FOLD    # (4) audit before returning


def test_challenge_digest_mismatch_writes_failure_audit_and_refuses(tmp_path):
    bundle = _bundle()
    chal = tmp_path / "challenge"; chal.mkdir()
    (chal / "c.jsonl").write_bytes(b"TAMPERED")
    _freeze(tmp_path, bundle, "d" * 64)                           # artifact bound to a DIFFERENT digest
    audit = tmp_path / "audit.jsonl"
    loader = ChallengeLoader(challenge_root=chal, selection_root=tmp_path / "selection",
                             threshold_artifact_path=tmp_path / "thr.json", audit_path=audit, frozen_bundle=bundle,
                             expected_selection_manifest_sha256="s" * 64, expected_excluded=EXCLUDED,
                             expected_challenge_digest="d" * 64)
    with pytest.raises(HashMismatch):
        loader.open(chal / "c.jsonl")                             # (7) opened bytes != pre-frozen digest
    events = [json.loads(x)["event"] for x in audit.read_text().splitlines() if x.strip()]
    assert "challenge_unlock" in events and "challenge_digest_mismatch" in events   # failure recorded


def test_changed_weight_file_hard_fails(tmp_path):
    bundle = _bundle()
    art = _freeze(tmp_path, bundle, "d" * 64)
    # identical metadata, DIFFERENT checkpoint bytes -> model identity drifts -> hard fail
    tampered = _bundle(model_hash=model_identity(PROV, "Z" * 64))
    with pytest.raises(HashMismatch):
        verify_challenge_access(art, tampered, expected_selection_manifest_sha256="s" * 64,
                                expected_excluded=EXCLUDED, expected_challenge_digest="d" * 64)
