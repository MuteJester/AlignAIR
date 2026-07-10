"""Phase 1 / Task 5: verified, atomic cache + resolve_model."""
import hashlib
import json
import shutil

import pytest

from alignair.registry import cache


def _registry(tmp_path, payload=b"ALIGNAIR-ARTIFACT" * 2000):
    d = tmp_path / "registry"
    (d / "human-igh").mkdir(parents=True)
    (d / "human-igh" / "2.1.0.alignair").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    reg = {"schema": "alignair.registry.v1", "models": {"human-igh": {"latest": "2.1.0", "versions": {
        "2.1.0": {"file": "human-igh/2.1.0.alignair", "artifact_sha256": sha}}}}}
    (d / "registry.json").write_text(json.dumps(reg))
    return f"file://{d}", sha


def test_cache_path_layout(monkeypatch, tmp_path):
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path / "c"))
    assert cache.cache_path("human-igh", "2.1.0") == tmp_path / "c" / "models" / "human-igh" / "2.1.0.alignair"


def test_download_verified_atomic_and_hashed(monkeypatch, tmp_path):
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path / "c"))
    src, sha = _registry(tmp_path)
    dest = cache.cache_path("human-igh", "2.1.0")
    out = cache.download_verified(src, "human-igh/2.1.0.alignair", dest, sha)
    assert out.exists() and hashlib.sha256(out.read_bytes()).hexdigest() == sha
    assert not out.with_name(out.name + ".part").exists()          # no leftover temp
    assert cache.download_verified(src, "human-igh/2.1.0.alignair", dest, sha) == out  # idempotent


def test_download_rejects_bad_hash_and_installs_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path / "c"))
    src, _ = _registry(tmp_path)
    dest = cache.cache_path("human-igh", "2.1.0")
    with pytest.raises(cache.IntegrityError):
        cache.download_verified(src, "human-igh/2.1.0.alignair", dest, "0" * 64)
    assert not dest.exists() and not dest.with_name(dest.name + ".part").exists()


def test_resolve_model_path_passthrough_and_id_download(monkeypatch, tmp_path):
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.delenv("ALIGNAIR_REGISTRY", raising=False)
    src, sha = _registry(tmp_path)
    f = tmp_path / "local.alignair"
    f.write_bytes(b"x")
    assert cache.resolve_model(str(f)) == f                         # a filesystem path passes through
    p = cache.resolve_model("human-igh@2.1.0", sources=[src])       # id@version -> verified download
    assert p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == sha
    with pytest.raises(ValueError, match="unknown"):
        cache.resolve_model("nope", sources=[src])


def test_pinned_cached_needs_no_network(monkeypatch, tmp_path):
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path / "c"))
    src, _ = _registry(tmp_path)
    dest = cache.cache_path("human-igh", "2.1.0")
    dest.parent.mkdir(parents=True)
    shutil.copy(src[len("file://"):] + "/human-igh/2.1.0.alignair", dest)
    # pinned + already cached -> returns without contacting any (here unreachable) registry
    assert cache.resolve_model("human-igh@2.1.0", sources=["hf://unreachable/x"], offline=True) == dest
