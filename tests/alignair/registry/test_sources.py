"""Registry source resolution + stdlib fetch."""
import json

import pytest

from alignair.registry import sources

pytestmark = pytest.mark.integration


def test_resolve_precedence_cli_env_config_default(tmp_path, monkeypatch):
    monkeypatch.delenv("ALIGNAIR_REGISTRY", raising=False)
    # default
    assert sources.resolve_sources(config_path=str(tmp_path / "none.toml")) == [sources.DEFAULT_REGISTRY]
    # config file
    cfg = tmp_path / "config.toml"
    cfg.write_text('registries = ["file:///srv/reg", "hf://lab/models"]\n')
    assert sources.resolve_sources(config_path=str(cfg)) == ["file:///srv/reg", "hf://lab/models"]
    # env (comma-separated; URLs contain colons so it must NOT colon-split)
    monkeypatch.setenv("ALIGNAIR_REGISTRY", "https://x.org/registry.json, hf://a/b")
    assert sources.resolve_sources(config_path=str(cfg)) == ["https://x.org/registry.json", "hf://a/b"]
    # explicit CLI wins over everything
    assert sources.resolve_sources(registry=["file:///override"], config_path=str(cfg)) == ["file:///override"]


def test_artifact_url_per_scheme():
    assert sources.artifact_url("hf://AlignAIR/AlignAIR-pretrained", "alignair-igh-human/1.0.0.alignair") == \
        "https://huggingface.co/AlignAIR/AlignAIR-pretrained/resolve/main/alignair-igh-human/1.0.0.alignair"
    assert sources.artifact_url("https://host/x/registry.json", "human-igh/1.0.0.alignair") == \
        "https://host/x/human-igh/1.0.0.alignair"                       # registry.json stripped -> base
    assert sources.artifact_url("file:///srv/reg", "registry.json") == "file:///srv/reg/registry.json"


def test_a_local_source_is_joined_as_a_path_a_windows_drive_too():
    """A directory source is a path, not a URL: a Windows drive (``C:``) is not read as a URL scheme, which would drop
    the directory and leave the artifact's path relative to the working directory."""
    assert sources.artifact_url("/srv/reg", "models/m.alignair") == "/srv/reg/models/m.alignair"
    assert sources.artifact_url("C:\\Users\\me\\reg", "models/m.alignair") == "C:\\Users\\me\\reg/models/m.alignair"
    assert sources.artifact_url("C:/reg/registry.json", "registry.json") == "C:/reg/registry.json"


def test_a_file_url_of_a_windows_drive_names_the_drive_path():
    """``file:///C:/reg`` (RFC 8089's form of a Windows path) names ``C:/reg``, as ``url2pathname`` reads it, not
    ``/C:/reg``, which Windows cannot open; ``file:///srv/reg`` stays absolute, the docs' ``file://catalog`` relative
    and ``file://C:\\reg`` as written."""
    assert sources.local_path("file:///C:/reg/models/m.alignair") == "C:/reg/models/m.alignair"
    assert sources.local_path("file:///d:") == "d:"
    assert sources.local_path(sources.artifact_url("file:///C:/reg/registry.json", "models/m.alignair")) == \
        "C:/reg/models/m.alignair"
    assert sources.local_path("file:///srv/reg/registry.json") == "/srv/reg/registry.json"
    assert sources.local_path("file://catalog/registry.json") == "catalog/registry.json"
    assert sources.local_path("file://C:\\reg\\registry.json") == "C:\\reg\\registry.json"
    assert sources.local_path("file:///CD:/x") == "/CD:/x"                      # not a drive letter


def test_load_registry_from_file_scheme(tmp_path):
    reg = {"schema": "alignair.registry.v1", "models": {
        "human-igh": {"latest": "2.1.0", "versions": {
            "2.1.0": {"file": "human-igh/2.1.0.alignair", "artifact_sha256": "abc"},
            "2.0.0": {"file": "human-igh/2.0.0.alignair", "artifact_sha256": "def"}}}}}
    (tmp_path / "registry.json").write_text(json.dumps(reg))
    src = f"file://{tmp_path}"
    assert sources.load_registry(src)["models"]["human-igh"]["latest"] == "2.1.0"
    # find latest and pinned
    s, ver, entry, _ = sources.find_model("human-igh", None, [src])
    assert ver == "2.1.0" and entry["artifact_sha256"] == "abc"
    _, ver2, entry2, _ = sources.find_model("human-igh", "2.0.0", [src])
    assert ver2 == "2.0.0" and entry2["artifact_sha256"] == "def"
    assert sources.find_model("nope", None, [src]) is None


def test_offline_fetch_raises(tmp_path):
    with pytest.raises(sources.OfflineError):
        sources.fetch_bytes("hf://a/b", "registry.json", offline=True)


def test_an_artifact_opens_as_a_stream_from_a_local_source_and_never_offline_from_a_remote_one(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "m.alignair").write_bytes(b"model bytes")
    with sources.open_artifact(str(tmp_path), "models/m.alignair") as stream:
        assert stream.read() == b"model bytes"
    assert sources.fetch_bytes(str(tmp_path), "models/m.alignair") == b"model bytes"
    with pytest.raises(sources.OfflineError):
        sources.open_artifact("hf://a/b", "models/m.alignair", offline=True)


def test_a_config_file_naming_one_registry_as_a_string_is_that_registry(tmp_path, monkeypatch):
    monkeypatch.delenv("ALIGNAIR_REGISTRY", raising=False)
    cfg = tmp_path / "config.toml"
    cfg.write_text('registries = "file:///srv/reg"\n')
    assert sources.resolve_sources(config_path=str(cfg)) == ["file:///srv/reg"]
