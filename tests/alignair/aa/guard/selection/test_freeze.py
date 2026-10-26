import pytest

from alignair.aa.guard.classes import CLASS_MAP_HASH
from alignair.aa.guard.freeze import (
    REQUIRED_SLOTS, FrozenBundle, feature_order_hash, freeze_pre_unlock_bundle, model_identity, preprocessing_hash,
)
from alignair.aa.guard.loco import LOCO_MASK_HASH

pytestmark = pytest.mark.component


def test_model_identity_binds_both_metadata_and_weight_bytes():
    base = model_identity("prov" * 16, "file" * 16)
    assert base == model_identity("prov" * 16, "file" * 16)                 # deterministic
    assert base != model_identity("prov" * 16, "OTHR" * 16)                 # weight-file change drifts it
    assert base != model_identity("OTHR" * 16, "file" * 16)                 # metadata change drifts it


def _bundle():
    return freeze_pre_unlock_bundle(
        model_hash="m" * 64, normalization_hash="n" * 64, source_vocab_hash="s" * 64,
        bank_checksum="b" * 64, evaluator_commit="c" * 40, cohort_manifest_hash="h" * 64,
        threshold_grid_hash="g" * 64)


def test_bundle_carries_every_required_slot_and_self_derived_hashes():
    b = _bundle()
    assert set(b.hashes) == set(REQUIRED_SLOTS)
    # self-derivable slots are pinned from source, not passed in
    assert b.hashes["class_map"] == CLASS_MAP_HASH
    assert b.hashes["mask_graph"] == LOCO_MASK_HASH
    assert b.hashes["feature_order"] == feature_order_hash()
    assert b.hashes["preprocessing"] == preprocessing_hash()


def test_bundle_sha_is_deterministic_and_sensitive():
    assert _bundle().bundle_sha256() == _bundle().bundle_sha256()
    other = freeze_pre_unlock_bundle(
        model_hash="X" * 64, normalization_hash="n" * 64, source_vocab_hash="s" * 64,
        bank_checksum="b" * 64, evaluator_commit="c" * 40, cohort_manifest_hash="h" * 64,
        threshold_grid_hash="g" * 64)
    assert other.bundle_sha256() != _bundle().bundle_sha256()      # a changed slot changes the bundle sha


def test_missing_slot_is_rejected():
    with pytest.raises(ValueError):
        FrozenBundle(hashes={"model": "m" * 64})                   # incomplete bundle is a defect
