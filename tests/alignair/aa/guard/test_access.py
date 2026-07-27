import pytest

from alignair.aa.guard.access import (
    AccessViolation, ChallengeLoader, HashMismatch, ThresholdArtifactMissing,
    freeze_fold_threshold, threshold_grid_hash, verify_challenge_access,
)
from alignair.aa.guard.dual_eval import ModeScores
from alignair.aa.guard.freeze import freeze_pre_unlock_bundle
from alignair.aa.guard.loco import LOCO_MASK_HASH, masked_profiles_for

SEL_MAN = "s" * 64
FOLD = "igk_igl"
EXCLUDED = (FOLD, tuple(sorted(masked_profiles_for(FOLD))))
# zero-error synthetic selection counts: (0,74) passes 5%-gate FAR, (0,59) passes FRR
GRID = [({"applicability_logit_cutoff": 1.0}, 0, 74, 0, 59)]


def _bundle(model_hash="m" * 64):
    return freeze_pre_unlock_bundle(
        model_hash=model_hash, normalization_hash="n" * 64, source_vocab_hash="v" * 64,
        bank_checksum="b" * 64, evaluator_commit="c" * 40, cohort_manifest_hash="h" * 64,
        threshold_grid_hash=threshold_grid_hash())


def _blind():
    return ModeScores(mode="blind", diagnostic_only=False, binding=True,
                      masked_groups=tuple(sorted(masked_profiles_for(FOLD))), input_order_sha256="x", scores=())


def _ref():
    return ModeScores(mode="reference_covered", diagnostic_only=True, binding=False,
                      masked_groups=(), input_order_sha256="x", scores=())


def _freeze(tmp_path, bundle):
    return freeze_fold_threshold(FOLD, _blind(), fold_checkpoint_sha256="m" * 64,
                                 selection_manifest_sha256=SEL_MAN, frozen_bundle=bundle,
                                 guardband_grid=GRID, out_path=tmp_path / "thr.json", mock=True)


def test_selection_loader_cannot_open_challenge_file(tmp_path):
    from alignair.aa.guard.access import SelectionLoader
    sel, chal = tmp_path / "selection", tmp_path / "challenge"
    sel.mkdir(); chal.mkdir()
    (sel / "s.bin").write_bytes(b"ok"); (chal / "c.bin").write_bytes(b"secret")
    loader = SelectionLoader(sel, chal)
    assert loader.load(sel / "s.bin") == b"ok"
    with pytest.raises(AccessViolation):
        loader.load(chal / "c.bin")                               # (1) hard refusal


def test_reference_covered_cannot_freeze_a_threshold(tmp_path):
    with pytest.raises(ValueError):
        freeze_fold_threshold(FOLD, _ref(), fold_checkpoint_sha256="m" * 64,
                              selection_manifest_sha256=SEL_MAN, frozen_bundle=_bundle(),
                              guardband_grid=GRID, out_path=tmp_path / "thr.json")


def test_challenge_refused_without_artifact(tmp_path):
    chal = tmp_path / "challenge"; chal.mkdir(); (chal / "c.bin").write_bytes(b"x")
    loader = ChallengeLoader(challenge_root=chal, selection_root=tmp_path / "selection",
                             threshold_artifact_path=tmp_path / "missing.json", audit_path=tmp_path / "audit.jsonl",
                             frozen_bundle=_bundle(), expected_selection_manifest_sha256=SEL_MAN,
                             expected_excluded=EXCLUDED)
    with pytest.raises(ThresholdArtifactMissing):
        loader.open(chal / "c.bin")                               # (2) refuse: no artifact


def test_challenge_unlock_writes_audit_before_returning(tmp_path):
    bundle = _bundle()
    _freeze(tmp_path, bundle)
    chal = tmp_path / "challenge"; chal.mkdir(); (chal / "c.bin").write_bytes(b"payload")
    audit = tmp_path / "audit.jsonl"
    loader = ChallengeLoader(challenge_root=chal, selection_root=tmp_path / "selection",
                             threshold_artifact_path=tmp_path / "thr.json", audit_path=audit,
                             frozen_bundle=bundle, expected_selection_manifest_sha256=SEL_MAN,
                             expected_excluded=EXCLUDED)
    data = loader.open(chal / "c.bin")
    assert data == b"payload"
    assert audit.exists()                                         # (4) audit written
    import json
    rec = [json.loads(x) for x in audit.read_text().splitlines() if x.strip()][-1]
    assert rec["event"] == "challenge_unlock" and rec["fold"] == FOLD


def test_any_changed_hash_hard_fails(tmp_path):
    bundle = _bundle()
    art = _freeze(tmp_path, bundle)
    tampered = _bundle(model_hash="Z" * 64)                       # model hash drifted
    with pytest.raises(HashMismatch):
        verify_challenge_access(art, tampered, expected_selection_manifest_sha256=SEL_MAN,
                                expected_excluded=EXCLUDED)         # (5) no silent recompute


def test_wrong_selection_manifest_hard_fails(tmp_path):
    bundle = _bundle()
    art = _freeze(tmp_path, bundle)
    with pytest.raises(HashMismatch):
        verify_challenge_access(art, bundle, expected_selection_manifest_sha256="different" * 8,
                                expected_excluded=EXCLUDED)
