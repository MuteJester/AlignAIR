"""Verified, atomic cache + resolve_model."""
import hashlib
import json
import shutil

import pytest

from alignair.registry import cache, OfflineError
from alignair.errors import ModelError, RegistryError

pytestmark = pytest.mark.integration


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


@pytest.mark.parametrize("spec", [None, b"model.alignair", "", " ", 5])
def test_invalid_model_spec_is_a_contextual_error(spec):
    with pytest.raises(ModelError, match="model specification") as caught:
        cache.resolve_model(spec)
    assert caught.value.context["stage"] == "model_resolution"


def test_local_paths_ignore_revision_and_registry_configuration(tmp_path, monkeypatch):
    from alignair.registry import sources

    path = tmp_path / "model.alignair"
    path.write_bytes(b"local")
    monkeypatch.setenv("HOME", str(tmp_path))
    def refuse(**kwargs):
        raise AssertionError("local paths must not read registry configuration")
    monkeypatch.setattr(sources, "resolve_sources", refuse)
    for spec in (path, str(path), "~/model.alignair"):
        assert cache.resolve_model(spec, revision="unused") == path


@pytest.mark.parametrize("kind", ["absolute", "relative", "pathlike", "directory"])
def test_invalid_local_paths_fail_before_hub_or_catalog_lookup(kind, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = {"absolute": str(tmp_path / "missing"), "relative": "./missing",
            "pathlike": tmp_path / "missing", "directory": str(tmp_path)}[kind]
    with pytest.raises(ModelError, match="model (file does not exist|path must be a file)") as caught:
        cache.resolve_model(spec, sources=[])
    assert caught.value.context["stage"] == "model_resolution"
    assert "path" in caught.value.context


@pytest.mark.parametrize("spec,revision", [("human-igh", "2.1.0"), ("human-igh@2.1.0", "0.0.0")])
def test_catalog_revision_and_inline_version_precedence(spec, revision, tmp_path):
    src, sha = _registry(tmp_path)
    path = cache.resolve_model(spec, sources=[src], revision=revision)
    assert path.name == "2.1.0.alignair"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha


@pytest.mark.parametrize("spec,revision", [("human-igh", None), ("human-igh@2.1.0", None),
                                         ("human-igh", "2.1.0")])
def test_existing_historical_alias_without_registry_configuration(spec, revision, tmp_path):
    path = tmp_path / "legacy.alignair"
    path.write_bytes(b"historical")
    aliases = {"human-igh": path, "human-igh@2.1.0": path}
    assert cache.resolve_model(spec, revision=revision, local_aliases=aliases, offline=True) == path


@pytest.mark.parametrize("configuration", ["argument", "environment", "config"])
def test_configured_registry_takes_precedence_over_alias(configuration, tmp_path, monkeypatch):
    src, sha = _registry(tmp_path)
    legacy = tmp_path / "legacy.alignair"
    legacy.write_bytes(b"must not be loaded")
    kwargs = {}
    if configuration == "argument":
        kwargs["sources"] = [src]
    elif configuration == "environment":
        monkeypatch.setenv("ALIGNAIR_REGISTRY", src)
    else:
        cfg = tmp_path / "config"
        cfg.mkdir()
        (cfg / "config.toml").write_text(f'registries = ["{src}"]\n')
        monkeypatch.setenv("ALIGNAIR_CONFIG_DIR", str(cfg))
    path = cache.resolve_model("human-igh@2.1.0", local_aliases={"human-igh@2.1.0": legacy}, **kwargs)
    assert path != legacy
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha


@pytest.mark.parametrize("target", ["missing", "directory", "different_version", "unmapped"])
def test_unusable_alias_falls_through_to_default_catalog(target, tmp_path, monkeypatch):
    from alignair.registry import sources

    src, sha = _registry(tmp_path)
    monkeypatch.setattr(sources, "DEFAULT_REGISTRY", src)
    legacy = tmp_path / "legacy.alignair"
    legacy.write_bytes(b"old version")
    aliases = {"unrelated": legacy} if target == "unmapped" else {
        "human-igh@2.0.0" if target == "different_version" else "human-igh@2.1.0":
            {"missing": tmp_path / "missing", "directory": tmp_path}.get(target, legacy)}
    path = cache.resolve_model("human-igh", revision="2.1.0", local_aliases=aliases)
    assert path != legacy
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha


def test_explicit_empty_sources_disables_alias_and_default_catalog(tmp_path):
    legacy = tmp_path / "legacy.alignair"
    legacy.write_bytes(b"legacy")
    with pytest.raises(ModelError, match="unknown model"):
        cache.resolve_model("human-igh", sources=[], local_aliases={"human-igh": legacy})


def test_configured_registry_failure_never_silently_loads_alias(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy.alignair"
    legacy.write_bytes(b"legacy")
    monkeypatch.setenv("ALIGNAIR_REGISTRY", f"file://{tmp_path / 'missing-registry'}")
    with pytest.raises(RegistryError):
        cache.resolve_model("human-igh", local_aliases={"human-igh": legacy})


def test_pinned_cached_needs_no_network(monkeypatch, tmp_path):
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path / "c"))
    src, _ = _registry(tmp_path)
    dest = cache.cache_path("human-igh", "2.1.0")
    dest.parent.mkdir(parents=True)
    shutil.copy(src[len("file://"):] + "/human-igh/2.1.0.alignair", dest)
    # pinned + already cached (no sidecar -> legacy, can't verify) -> returns without the registry
    assert cache.resolve_model("human-igh@2.1.0", sources=["hf://unreachable/x"], offline=True) == dest


def test_pinned_cache_with_valid_sidecar_returns(monkeypatch, tmp_path):
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path / "c"))
    payload = b"ARTIFACT" * 100
    dest = cache.cache_path("human-igh", "2.1.0")
    dest.parent.mkdir(parents=True)
    dest.write_bytes(payload)
    cache._write_sidecar(dest, hashlib.sha256(payload).hexdigest())
    assert cache.resolve_model("human-igh@2.1.0", sources=["hf://unreachable/x"], offline=True) == dest


def test_pinned_cache_tampered_file_is_not_returned(monkeypatch, tmp_path):
    """A cached pinned artifact whose sidecar no longer matches is NOT returned — it falls through to a
    registry re-resolve instead of silently loading the wrong/corrupt model."""
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path / "c"))
    dest = cache.cache_path("human-igh", "2.1.0")
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"TAMPERED-OR-CORRUPTED-BYTES")
    cache._write_sidecar(dest, hashlib.sha256(b"the-original-artifact").hexdigest())   # != file
    with pytest.raises(OfflineError, match="offline"):
        cache.resolve_model("human-igh@2.1.0", sources=["hf://unreachable/x"], offline=True)


def test_unpinned_offline_cached_resolution_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path / "c"))
    payload = b"MODEL-BYTES-VERSION-1.0.0"
    dest = cache.cache_path("human-igh", "1.0.0")
    dest.parent.mkdir(parents=True)
    dest.write_bytes(payload)
    cache._write_sidecar(dest, hashlib.sha256(payload).hexdigest())

    # Try resolving 'human-igh' (unpinned) in offline mode
    # Since it is in the cache, it should fall back to using the latest installed version
    p = cache.resolve_model("human-igh", sources=["hf://unreachable/x"], offline=True)
    assert p == dest
    assert p.read_bytes() == payload
