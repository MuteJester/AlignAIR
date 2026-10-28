"""Both product loaders consume the shared resolution policy before format-specific validation."""
import hashlib
import json

import pytest
import torch

from alignair import Aligner, load_aa_model
from alignair.predict.protein import loader as aa_loader
from alignair.cli.main import build_parser
from alignair.errors import ModelError
from alignair.registry import cache, hf
from tests.support.protein_bundle import build_protein_bundle_setup, write_protein_bundle

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    path = tmp_path_factory.mktemp("protein-registry") / "model.alignair"
    write_protein_bundle(build_protein_bundle_setup(), path)
    return path


@pytest.fixture
def registry(bundle, tmp_path):
    root = tmp_path / "registry"
    root.mkdir()
    artifact = root / "model.alignair"
    artifact.write_bytes(bundle.read_bytes())
    entry = {"file": artifact.name, "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
    (root / "registry.json").write_text(json.dumps({"models": {
        "protein-test": {"latest": "1.0.0", "versions": {"1.0.0": entry}}}}))
    return root.as_uri()


def test_aa_catalog_load_then_offline_cache_reuse_preserves_weights_and_calibration(bundle, registry):
    direct = load_aa_model(bundle, device="cpu", revision="ignored-for-local-path")
    downloaded = load_aa_model("protein-test", revision="1.0.0", registry=[registry], device="cpu")
    offline = load_aa_model("protein-test@1.0.0", registry=["hf://unreachable/catalog"],
                            offline=True, device="cpu")
    for actual in (downloaded, offline):
        assert actual.card == direct.card
        assert actual.conformal["V"]["pooled"].qhat == direct.conformal["V"]["pooled"].qhat
        for name, weights in direct.model.state_dict().items():
            assert torch.equal(actual.model.state_dict()[name], weights), name


def test_aa_loader_validates_vj_only_bundle_without_inventing_a_d_head(tmp_path):
    path = tmp_path / "vj.alignair"
    write_protein_bundle(build_protein_bundle_setup(has_d=False), path)
    predictor = load_aa_model(path, device="cpu", offline=True)
    assert not predictor.cfg.has_d
    assert set(predictor.model.branches) == {"v", "j"}
    assert set(predictor.conformal) == {"V", "J"}
    assert predictor.d_allele_order == []


def test_aa_catalog_resolution_still_checks_calibration_binding(registry, monkeypatch):
    from alignair.model_file import container

    cache.resolve_model("protein-test@1.0.0", sources=[registry])
    original = container.read_header
    def swapped_binding(path):
        header = original(path)
        header["reference_provenance"]["source_model_sha256"] = "0" * 64
        return header
    monkeypatch.setattr(container, "read_header", swapped_binding)
    with pytest.raises(ModelError, match="model_sha256"):
        load_aa_model("protein-test@1.0.0", registry=[registry], device="cpu")


@pytest.mark.parametrize("spec", ["hf://org/protein@v1", "org/protein/sub/model.alignair@v1"])
def test_aa_hub_resolution_honors_options_before_safe_decoding(spec, bundle, monkeypatch):
    calls = []
    def downloaded(model, **kwargs):
        calls.append((model, kwargs))
        return bundle
    monkeypatch.setattr(hf, "download_from_hub", downloaded)
    predictor = load_aa_model(spec, revision="v2", token="test-token", offline=True, device="cpu")
    assert predictor.card["artifact_type"] == "alignair-aa-model"
    assert calls == [(spec, {"revision": "v2", "token": "test-token", "offline": True})]


def test_aa_cli_forwards_resolution_options_without_token_in_run_metadata(registry, tmp_path):
    reads = tmp_path / "reads.txt"
    reads.write_text("EVQLVESGGG\n")
    out = tmp_path / "predictions.jsonl"
    args = build_parser().parse_args([
        "predict-aa", str(reads), "--out", str(out), "--model", "protein-test",
        "--registry", registry, "--revision", "1.0.0", "--offline", "--device", "cpu",
        "--token", "test-secret-token",
    ])
    assert args.func(args) == 0
    assert json.loads(out.read_text())["input_length"] == 10
    metadata = out.with_suffix(out.suffix + ".run.json").read_text()
    assert "test-secret-token" not in metadata
    assert json.loads(metadata)["command"]["revision"] == "1.0.0"
    assert json.loads(metadata)["command"]["offline"] is True


def test_aa_registry_does_not_get_shadowed_by_checkout_alias(tmp_path, registry, monkeypatch):
    legacy = tmp_path / "non-bundle.pt"
    legacy.write_bytes(b"must not load this artifact")
    monkeypatch.setitem(aa_loader._LOCAL_MODEL_ALIASES, "protein-test@1.0.0", legacy)
    predictor = load_aa_model("protein-test@1.0.0", registry=[registry], device="cpu")
    assert predictor.card["model_id"] == "alignair-aa-igh-human-test"


def test_aligner_local_revision_does_not_change_artifact_path(bundle, monkeypatch):
    # The decoder boundary is injected here; real nucleotide decode is covered by CLI registry tests.
    calls = []
    def decoded(path, **kwargs):
        calls.append(path)
        return "model", "reference"
    monkeypatch.setattr("alignair.api.load_model", decoded)
    aligner = Aligner.from_pretrained(bundle, revision="v1", device="cpu")
    assert calls == [str(bundle)]
    assert aligner.model_path == str(bundle)
