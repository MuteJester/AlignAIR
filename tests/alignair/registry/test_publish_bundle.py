"""A bundle goes into the shared catalog with its model card and reference manifest: ``publish_bundle`` lays the three
files out as the catalog reads them (``<id>/<version>.alignair`` and, beside it, ``.model_card.md`` and
``.reference_manifest.json``, each recorded in ``registry.json`` with its SHA-256), the model then loads by id from that
catalog, and ``upload_version`` sends exactly those files and the catalog in one commit. ``push_to_hub`` uploads the
card and manifest that sit beside a model file with it. Nothing here touches the network: the Hugging Face client is
replaced by a recorder of the files it would be given."""
import hashlib
import json

import GenAIRR.data as gd
import pytest

from alignair import Aligner
from alignair import model_file as mf
from alignair.cli.main import main
from alignair.core import prepare_model
from alignair.reference import ReferenceSet
from alignair.registry.publish import publish_bundle, upload_version
from alignair.registry.validate import validate_registry

pytestmark = pytest.mark.integration

ID, VERSION = "alignair4-igh-human", "0.1.0"
COMPANIONS = ("model_card.md", "reference_manifest.json")


class RecordingApi:
    """The one HfApi call the publishing code makes, recorded: the repo and the files of the commit."""

    def __init__(self, *args, **kwargs):
        self.commits = []

    def create_repo(self, repo_id, **kwargs):
        self.created = repo_id

    def create_commit(self, repo_id, operations, *, commit_message, repo_type="model", revision=None, **kwargs):
        self.commits.append({"repo_id": repo_id, "repo_type": repo_type, "message": commit_message,
                             "files": {op.path_in_repo: op.path_or_fileobj for op in operations}})


@pytest.fixture
def bundle(tmp_path):
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), max_seq_length=128, embed_dim=4,
                          trunk_channels=8)
    ckpt = tmp_path / "run" / "model.alignair"
    ckpt.parent.mkdir()
    mf.save_model(str(ckpt), setup.build_model().eval(), setup.reference, dataconfigs=["HUMAN_IGH_OGRDB"],
                  training={"steps": 3, "batch_size": 2})
    out = tmp_path / "bundle"
    assert main(["export", str(ckpt), str(out), "--model-id", ID, "--model-version", VERSION, "--no-validate"]) == 0
    return out


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_bundle_publishes_with_its_card_and_manifest_and_loads_by_id(bundle, tmp_path, monkeypatch):
    registry = tmp_path / "registry"
    assert publish_bundle(str(bundle), ID, VERSION, str(registry), description="Human IGH (test)") == []
    files = {p.relative_to(registry).as_posix() for p in registry.rglob("*") if p.is_file() and p.suffix != ".lock"}
    assert files == {"registry.json", f"{ID}/{VERSION}.alignair", f"{ID}/{VERSION}.model_card.md",
                     f"{ID}/{VERSION}.reference_manifest.json"}
    entry = json.loads((registry / "registry.json").read_text())["models"][ID]["versions"][VERSION]
    assert entry["companions"] == {
        "model_card": {"file": f"{ID}/{VERSION}.model_card.md", "sha256": _sha(bundle / "model_card.md")},
        "reference_manifest": {"file": f"{ID}/{VERSION}.reference_manifest.json",
                               "sha256": _sha(bundle / "reference_manifest.json")}}
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path / "cache"))
    aligner = Aligner.from_pretrained(ID, registry=[f"file://{registry}"], device="cpu")
    reference = aligner.reference
    read = reference.gene("V").sequences[5][-80:] + "GGGAAA" + reference.gene("J").sequences[3]
    assert len(aligner.predict([read]).records) == 1


def test_a_bundle_without_its_card_is_refused_and_nothing_is_written(bundle, tmp_path):
    (bundle / "model_card.md").unlink()
    registry = tmp_path / "registry"
    with pytest.raises(FileNotFoundError, match="model_card.md"):
        publish_bundle(str(bundle), ID, VERSION, str(registry))
    assert not (registry / "registry.json").exists()


def test_the_validator_flags_a_companion_that_changed(bundle, tmp_path):
    registry = tmp_path / "registry"
    assert publish_bundle(str(bundle), ID, VERSION, str(registry)) == []
    (registry / ID / f"{VERSION}.model_card.md").write_text("edited\n")
    reg = json.loads((registry / "registry.json").read_text())
    problems = validate_registry(reg, lambda rel: str(registry / rel))
    assert problems == [f"{ID}@{VERSION}: model_card sha256 mismatch"]


def test_upload_version_sends_the_model_its_card_manifest_and_the_catalog_in_one_commit(bundle, tmp_path):
    registry = tmp_path / "registry"
    assert publish_bundle(str(bundle), ID, VERSION, str(registry)) == []
    api = RecordingApi()
    uploaded = upload_version(str(registry), ID, VERSION, api=api)
    (commit,) = api.commits
    assert commit["repo_id"] == "AlignAIR/AlignAIR-pretrained" and commit["repo_type"] == "model"
    assert sorted(commit["files"]) == sorted(uploaded) == sorted(
        ["registry.json", f"{ID}/{VERSION}.alignair", f"{ID}/{VERSION}.model_card.md",
         f"{ID}/{VERSION}.reference_manifest.json"])
    assert commit["files"][f"{ID}/{VERSION}.model_card.md"] == str(registry / ID / f"{VERSION}.model_card.md")


def test_upload_version_refuses_a_version_the_catalog_lacks(bundle, tmp_path):
    from alignair.errors import RegistryError
    registry = tmp_path / "registry"
    assert publish_bundle(str(bundle), ID, VERSION, str(registry)) == []
    with pytest.raises(RegistryError, match="9.9.9"):
        upload_version(str(registry), ID, "9.9.9", api=RecordingApi())


def test_push_to_hub_uploads_the_card_and_manifest_beside_the_model(bundle, monkeypatch):
    import huggingface_hub
    recorder = RecordingApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: recorder)
    Aligner.from_pretrained(str(bundle / "model.alignair"), device="cpu").push_to_hub("org/repo", token="x")
    (commit,) = recorder.commits
    assert commit["repo_id"] == "org/repo"
    assert sorted(commit["files"]) == ["model.alignair", "model_card.md", "reference_manifest.json"]
