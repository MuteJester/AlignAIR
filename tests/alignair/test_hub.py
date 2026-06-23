import sys
import types

import pytest

from alignair import hub


def test_resolve_local_path_returned_as_is(tmp_path):
    p = tmp_path / "bundle"
    p.mkdir()
    assert hub.resolve_model(str(p)) == str(p)


def test_resolve_unknown_spec_errors():
    with pytest.raises(SystemExit, match="unknown model"):
        hub.resolve_model("not-a-path-or-id")


def _mock_hf(monkeypatch, calls):
    mod = types.ModuleType("huggingface_hub")
    def snapshot_download(repo_id, revision=None, local_dir=None):
        calls.append((repo_id, revision, local_dir))
        return f"/cache/{repo_id}"
    mod.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)


def test_unavailable_catalog_id_errors():
    # catalog entries not yet published give a clear "train your own / demo" message, not a 404
    assert hub.MODEL_CATALOG["human-igh-ogrdb"]["available"] is False
    with pytest.raises(SystemExit, match="not published yet"):
        hub.resolve_model("human-igh-ogrdb")


def test_available_catalog_id_downloads(monkeypatch):
    calls = []
    _mock_hf(monkeypatch, calls)
    monkeypatch.setitem(hub.MODEL_CATALOG["human-igh-ogrdb"], "available", True)
    path = hub.resolve_model("human-igh-ogrdb")
    assert calls and calls[0][0] == hub.MODEL_CATALOG["human-igh-ogrdb"]["repo"]
    assert path == f"/cache/{hub.MODEL_CATALOG['human-igh-ogrdb']['repo']}"


def test_resolve_hf_repo_id_downloads(monkeypatch):
    calls = []
    _mock_hf(monkeypatch, calls)
    hub.resolve_model("someorg/somemodel")
    assert calls[0][0] == "someorg/somemodel"


def test_list_models_nonempty():
    assert "human-igh-ogrdb" in hub.list_models()
