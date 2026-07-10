"""Phase 1 / Task 7: passive, careful update notices."""
import hashlib
import io
import json

import pytest

from alignair.registry import cache, updates


def _registry(tmp_path, latest="2.1.0"):
    d = tmp_path / "registry"
    (d / "human-igh").mkdir(parents=True)
    payload = b"art"
    (d / "human-igh" / f"{latest}.alignair").write_bytes(payload)
    reg = {"models": {"human-igh": {"latest": latest, "versions": {
        latest: {"file": f"human-igh/{latest}.alignair", "artifact_sha256": hashlib.sha256(payload).hexdigest(),
                 "trained": "2026-01-01"}}}}}
    (d / "registry.json").write_text(json.dumps(reg))
    return f"file://{d}"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("ALIGNAIR_NO_NETWORK", raising=False)
    # install an OLD version locally
    old = cache.cache_path("human-igh", "2.0.0")
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old")
    return _registry(tmp_path, latest="2.1.0")


def test_notice_when_newer_available(env):
    s = io.StringIO()
    notices = updates.maybe_notify_updates(sources_list=[env], stream=s)
    assert notices == [("human-igh", "2.1.0")]
    assert "human-igh 2.1.0" in s.getvalue() and "models update human-igh" in s.getvalue()


def test_suppressed_when_pinned_offline_quiet_or_env(env, monkeypatch):
    for kw in ({"pinned": True}, {"offline": True}, {"quiet": True}):
        s = io.StringIO()
        assert updates.maybe_notify_updates(sources_list=[env], stream=s, **kw) == []
        assert s.getvalue() == ""
    monkeypatch.setenv("ALIGNAIR_NO_NETWORK", "1")
    s = io.StringIO()
    assert updates.maybe_notify_updates(sources_list=[env], stream=s) == []
    assert s.getvalue() == ""


def test_no_notice_when_up_to_date(tmp_path, monkeypatch):
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("ALIGNAIR_NO_NETWORK", raising=False)
    latest = cache.cache_path("human-igh", "2.1.0")
    latest.parent.mkdir(parents=True)
    latest.write_bytes(b"cur")
    src = _registry(tmp_path, latest="2.1.0")
    s = io.StringIO()
    assert updates.maybe_notify_updates(sources_list=[src], stream=s) == []
    assert s.getvalue() == ""


def test_unreachable_registry_is_silent(env):
    s = io.StringIO()
    assert updates.maybe_notify_updates(sources_list=["hf://unreachable/x"], stream=s) == []
    assert s.getvalue() == ""
