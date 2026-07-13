"""Phase 1 / Task 9: publish into a local registry + validator gate."""
import hashlib
import json
import shutil

import GenAIRR.data as gd

from alignair import model_file as mf
from alignair.core import AlignAIR
from alignair.core.config import AlignAIRConfig
from alignair.registry.publish import publish_local
from alignair.registry.validate import validate_registry


def _save(path, **kw):
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    mf.save_model(str(path), AlignAIR(cfg).eval(), dataconfigs=["HUMAN_IGH_OGRDB"],
                  training={"steps": 1, "batch_size": 1}, **kw)


def test_publish_local_valid_and_writes_registry(tmp_path):
    art = tmp_path / "m.alignair"
    _save(art, include_trusted_pickle=False, model_id="human-igh", model_version="2.1.0")
    regdir = tmp_path / "registry"
    assert publish_local(str(art), "human-igh", "2.1.0", str(regdir)) == []       # valid
    reg = json.loads((regdir / "registry.json").read_text())
    e = reg["models"]["human-igh"]["versions"]["2.1.0"]
    assert reg["models"]["human-igh"]["latest"] == "2.1.0"
    assert e["size"] and len(e["artifact_sha256"]) == 64 and e["allele_order_sha256"]
    assert (regdir / "human-igh" / "2.1.0.alignair").exists()


def test_publish_sets_latest_to_max_semver(tmp_path):
    regdir = tmp_path / "registry"
    for v in ("2.0.0", "2.1.0", "2.0.5"):
        art = tmp_path / f"{v}.alignair"
        _save(art, include_trusted_pickle=False, model_id="human-igh", model_version=v)
        assert publish_local(str(art), "human-igh", v, str(regdir)) == []
    assert json.loads((regdir / "registry.json").read_text())["models"]["human-igh"]["latest"] == "2.1.0"


def test_publish_is_transactional_on_validation_failure(tmp_path):
    """A publish that fails validation leaves NEITHER an updated catalog NOR a copied artifact (P0-11)."""
    regdir = tmp_path / "registry"
    good = tmp_path / "good.alignair"
    _save(good, include_trusted_pickle=False, model_id="human-igh", model_version="2.1.0")
    assert publish_local(str(good), "human-igh", "2.1.0", str(regdir)) == []
    before = (regdir / "registry.json").read_text()

    bad = tmp_path / "bad.alignair"                    # a resumable checkpoint (has pickle) -> invalid
    _save(bad, include_trusted_pickle=True, model_id="human-igh", model_version="9.9.9")
    problems = publish_local(str(bad), "human-igh", "9.9.9", str(regdir))
    assert any("pickle" in p for p in problems)         # rejected
    assert (regdir / "registry.json").read_text() == before   # catalog unchanged
    assert not (regdir / "human-igh" / "9.9.9.alignair").exists()   # invalid artifact not copied
    assert not (regdir / "human-igh" / "9.9.9.alignair.staging").exists()   # stage cleaned up


def test_validator_rejects_pickle_artifact(tmp_path):
    art = tmp_path / "ckpt.alignair"
    _save(art, include_trusted_pickle=True, model_id="human-igh", model_version="1.0.0")   # has pickle
    regdir = tmp_path / "registry"
    (regdir / "human-igh").mkdir(parents=True)
    shutil.copy(art, regdir / "human-igh" / "1.0.0.alignair")
    data = (regdir / "human-igh" / "1.0.0.alignair").read_bytes()
    reg = {"models": {"human-igh": {"latest": "1.0.0", "versions": {"1.0.0": {
        "file": "human-igh/1.0.0.alignair", "artifact_sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data)}}}}}
    problems = validate_registry(reg, lambda r: str(regdir / r))
    assert any("pickle" in p for p in problems)


def test_validator_flags_sha_size_and_id_mismatch(tmp_path):
    art = tmp_path / "m.alignair"
    _save(art, include_trusted_pickle=False, model_id="human-igh", model_version="2.1.0")
    regdir = tmp_path / "registry"
    (regdir / "other").mkdir(parents=True)
    shutil.copy(art, regdir / "other" / "2.1.0.alignair")
    reg = {"models": {"other": {"latest": "2.1.0", "versions": {"2.1.0": {
        "file": "other/2.1.0.alignair", "artifact_sha256": "0" * 64, "size": 1}}}}}
    problems = validate_registry(reg, lambda r: str(regdir / r))
    assert any("artifact_sha256 mismatch" in p for p in problems)
    assert any("size" in p for p in problems)
    assert any("model_id" in p for p in problems)          # card=human-igh, registry id=other
