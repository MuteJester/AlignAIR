"""Direct Hugging Face repo loading — spec parsing, routing, offline behavior, provenance
commit, and save_pretrained. Network is never touched (offline / mocked)."""
from types import SimpleNamespace

import pytest

from alignair.registry import hf
from alignair.registry.cache import resolve_model
from alignair.errors import RegistryError

pytestmark = pytest.mark.integration


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
    with pytest.raises(RegistryError, match="offline") as caught:
        hf.download_from_hub("hf://alignair/definitely-not-a-real-repo-xyz", offline=True)
    assert caught.value.context["offline"] is True
    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_offline_hub_cache_hit_never_calls_the_download_api(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub import constants

    root = tmp_path / "hub"
    repo = root / "models--org--model"
    snapshot = repo / "snapshots" / "fixture-commit" / "model.alignair"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"cached artifact")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("fixture-commit")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(root))
    def refuse(**kwargs):
        raise AssertionError("offline lookup invoked the download API")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", refuse)
    assert hf.download_from_hub("hf://org/model", offline=True) == snapshot


def test_aligner_from_pretrained_captures_hf_commit(monkeypatch, tmp_path):
    """Real routing (is_hf_repo_spec + resolve_model), only the actual download is faked -> no network."""
    import alignair.aligner as aln

    snap = tmp_path / "models--org--repo" / "snapshots" / "cafebabe99" / "model.alignair"
    snap.parent.mkdir(parents=True)
    snap.write_text("x")
    monkeypatch.setattr("alignair.registry.hf.download_from_hub", lambda spec, **k: snap)
    monkeypatch.setattr(aln, "resolve_device", lambda d: "cpu")
    monkeypatch.setattr("alignair.model_file.load_model",
                        lambda *a, **k: SimpleNamespace(model="MODEL", reference="REF"))
    a = aln.Aligner.from_pretrained("hf://org/repo", device="cpu")
    assert a.model_path == str(snap) and a.source_commit == "cafebabe99"


def test_save_pretrained_copies_artifact(monkeypatch, tmp_path):
    from alignair.aligner import Aligner
    src = tmp_path / "loaded.alignair"
    src.write_bytes(b"ALIGNAIR-artifact-bytes")
    monkeypatch.setattr("alignair.model_file.read_metadata", lambda p: {"sections": {"config": {}}})
    a = Aligner(model=None, reference=None, model_path=str(src))
    dest = a.save_pretrained(str(tmp_path / "out"))
    assert open(dest, "rb").read() == b"ALIGNAIR-artifact-bytes"

    bare = Aligner(model=None, reference=None)                 # not loaded from a file
    with pytest.raises(ValueError, match="loaded from a file"):
        bare.save_pretrained(str(tmp_path / "out2"))


def test_save_pretrained_refuses_pickle_checkpoint(monkeypatch, tmp_path):
    """A resumable checkpoint (pickle sections) must not be distributed via save_pretrained."""
    from alignair.aligner import Aligner
    src = tmp_path / "ckpt.alignair"
    src.write_bytes(b"x")
    monkeypatch.setattr("alignair.model_file.read_metadata",
                        lambda p: {"sections": {"config": {}, "dataconfig/0": {}, "train_state": {}}})
    a = Aligner(model=None, reference=None, model_path=str(src))
    with pytest.raises(ValueError, match="pickle sections|not distributable"):
        a.save_pretrained(str(tmp_path / "out"))


@pytest.mark.parametrize("spec", ["runs/igh/model.alignair", "runs/model.alignair", "model.alignair"])
def test_a_missing_model_file_is_reported_missing_not_fetched_from_the_hub(spec, tmp_path, monkeypatch):
    """A spec that names a model file (it ends in .alignair, or has a path below org/repo) and does not exist is a
    missing file; only hf://... or a bare org/repo goes to the Hub."""
    from alignair.errors import ModelError
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hf, "download_from_hub", lambda *a, **k: pytest.fail(f"{spec} was fetched from the Hub"))
    assert not hf.is_hf_repo_spec(spec)
    with pytest.raises(ModelError, match="model file does not exist"):
        resolve_model(spec)


def test_a_hub_token_is_the_given_one_then_hf_token_then_the_hub_variable(monkeypatch):
    from alignair.registry.hf import hf_token
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert hf_token(None) is None
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hub")
    assert hf_token(None) == "hub"
    monkeypatch.setenv("HF_TOKEN", "hf")
    assert hf_token(None) == "hf" and hf_token("given") == "given"
