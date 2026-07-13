"""P0-11: direct Hugging Face repo loading — spec parsing, routing, offline behavior, provenance
commit, and save_pretrained. Network is never touched (offline / mocked)."""
import pytest

from alignair.registry import hf
from alignair.registry.cache import resolve_model


def test_is_hf_repo_spec():
    assert hf.is_hf_repo_spec("hf://alignair/models")
    assert hf.is_hf_repo_spec("alignair/human-igh")            # bare org/repo
    assert not hf.is_hf_repo_spec("alignair-igh-v1")           # a plain catalog id (no slash)
    assert not hf.is_hf_repo_spec("https://example.com/x")     # a different scheme
    assert not hf.is_hf_repo_spec("./local/model.alignair")    # a relative path


def test_parse_hf_spec():
    assert hf.parse_hf_spec("hf://org/repo") == ("org/repo", None, None)
    assert hf.parse_hf_spec("hf://org/repo@v1.0.0") == ("org/repo", "v1.0.0", None)
    assert hf.parse_hf_spec("hf://org/repo/sub/model.alignair@abc123") == (
        "org/repo", "abc123", "sub/model.alignair")
    assert hf.parse_hf_spec("org/repo") == ("org/repo", None, None)
    with pytest.raises(ValueError, match="expected org/repo"):
        hf.parse_hf_spec("hf://justoneword")


def test_resolved_commit_from_snapshot_path():
    p = "/home/u/.cache/huggingface/hub/models--org--repo/snapshots/deadbeef1234/model.alignair"
    assert hf.resolved_commit(p) == "deadbeef1234"
    assert hf.resolved_commit("/tmp/local/model.alignair") is None


def test_resolve_model_routes_hf_spec(monkeypatch, tmp_path):
    """A hf:// spec goes to download_from_hub (not the catalog path); the catalog id path is untouched."""
    calls = {}

    def fake_dl(spec, *, revision=None, token=None, offline=False):
        calls.update(spec=spec, revision=revision, token=token, offline=offline)
        return tmp_path / "model.alignair"
    monkeypatch.setattr(hf, "download_from_hub", fake_dl)
    out = resolve_model("hf://org/repo", revision="v2", token="tok", offline=True)
    assert out == tmp_path / "model.alignair"
    assert calls == {"spec": "hf://org/repo", "revision": "v2", "token": "tok", "offline": True}


def test_download_from_hub_offline_miss_is_clean_error():
    # offline + a repo definitely not in the local HF cache -> a clear ValueError, no network, no crash
    with pytest.raises(ValueError, match="Hugging Face|offline"):
        hf.download_from_hub("hf://alignair/definitely-not-a-real-repo-xyz", offline=True)


def test_aligner_from_pretrained_captures_hf_commit(monkeypatch, tmp_path):
    """Real routing (is_hf_repo_spec + resolve_model), only the actual download is faked -> no network."""
    import alignair.aligner as aln

    snap = tmp_path / "models--org--repo" / "snapshots" / "cafebabe99" / "model.alignair"
    snap.parent.mkdir(parents=True)
    snap.write_text("x")
    monkeypatch.setattr("alignair.registry.hf.download_from_hub", lambda spec, **k: snap)
    monkeypatch.setattr(aln, "resolve_device", lambda d: "cpu")
    monkeypatch.setattr("alignair.api.load_model", lambda *a, **k: ("MODEL", "REF"))
    a = aln.Aligner.from_pretrained("hf://org/repo", device="cpu")
    assert a.model_path == str(snap) and a.source_commit == "cafebabe99"


def test_save_pretrained_copies_artifact(tmp_path):
    from alignair.aligner import Aligner
    src = tmp_path / "loaded.alignair"
    src.write_bytes(b"ALIGNAIR-artifact-bytes")
    a = Aligner(model=None, reference=None, model_path=str(src))
    dest = a.save_pretrained(str(tmp_path / "out"))
    assert open(dest, "rb").read() == b"ALIGNAIR-artifact-bytes"

    bare = Aligner(model=None, reference=None)                 # not loaded from a file
    with pytest.raises(ValueError, match="loaded from a file"):
        bare.save_pretrained(str(tmp_path / "out2"))
