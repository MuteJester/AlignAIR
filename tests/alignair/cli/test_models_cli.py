"""Phase 1 / Task 6: `alignair models …` CLI."""
import hashlib
import json

import pytest

from alignair.cli.main import build_parser


def _run(argv):
    args = build_parser().parse_args(argv)
    return args.func(args)


def _registry(tmp_path, versions=("2.0.0", "2.1.0")):
    d = tmp_path / "registry"
    entries = {}
    for v in versions:
        (d / "human-igh").mkdir(parents=True, exist_ok=True)
        payload = f"ARTIFACT-{v}".encode() * 500
        (d / "human-igh" / f"{v}.alignair").write_bytes(payload)
        entries[v] = {"file": f"human-igh/{v}.alignair", "artifact_sha256": hashlib.sha256(payload).hexdigest()}
    reg = {"schema": "alignair.registry.v1", "models": {
        "human-igh": {"description": "Human IGH", "latest": versions[-1], "versions": entries}}}
    (d / "registry.json").write_text(json.dumps(reg))
    return f"file://{d}"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("ALIGNAIR_REGISTRY", raising=False)
    return _registry(tmp_path)


def test_get_then_list_shows_installed(env, capsys):
    assert _run(["models", "get", "human-igh@2.1.0", "--registry", env]) == 0
    assert "cached ->" in capsys.readouterr().out
    assert _run(["models", "list", "--registry", env]) == 0
    out = capsys.readouterr().out
    assert "human-igh" in out and "installed" in out


def test_path_prints_cached_location(env, capsys):
    _run(["models", "get", "human-igh@2.1.0", "--registry", env]); capsys.readouterr()
    assert _run(["models", "path", "human-igh@2.1.0", "--registry", env]) == 0
    assert capsys.readouterr().out.strip().endswith("2.1.0.alignair")


def test_verify_ok_then_mismatch(env, capsys, tmp_path):
    _run(["models", "get", "human-igh@2.1.0", "--registry", env]); capsys.readouterr()
    assert _run(["models", "verify", "--registry", env]) == 0
    assert "OK" in capsys.readouterr().out
    # corrupt the cached file -> verify must fail
    import os
    p = tmp_path / "cache" / "models" / "human-igh" / "2.1.0.alignair"
    p.write_bytes(b"tampered")
    assert _run(["models", "verify", "--registry", env]) == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_update_and_prune(env, capsys, tmp_path):
    _run(["models", "get", "human-igh@2.0.0", "--registry", env])          # install an older version
    _run(["models", "update", "--registry", env])                          # -> pulls latest 2.1.0
    capsys.readouterr()
    from alignair.registry import cache
    assert cache.installed_models()["human-igh"] == ["2.0.0", "2.1.0"]
    assert _run(["models", "prune", "--keep", "1"]) == 0
    assert cache.installed_models()["human-igh"] == ["2.1.0"]              # only newest kept


def test_get_unknown_id_errors(env):
    with pytest.raises(ValueError, match="unknown"):
        _run(["models", "get", "does-not-exist", "--registry", env])
