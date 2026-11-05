"""Reference and label preflight precedes model allocation in safe artifact restoration."""
import pytest
import torch

from alignair import model_file
from alignair.core import AlignAIR, prepare_model
from alignair.errors import ModelIntegrityError
from alignair.model_file import container, serialize
from alignair.reference import ReferenceSet

pytestmark = pytest.mark.integration


@pytest.fixture(params=["nucleotide"])
def artifact(request, tmp_path):
    reference = ReferenceSet.from_genotype({"V": {"V2": "ATGTGTAAA", "V1": "ATGTGCAAA"},
                                            "J": {"J1": "TGGAAA"}})
    setup = prepare_model(reference, input_space=request.param, max_seq_length=128,
                           embed_dim=4, filters=4, block_out=8)
    model = setup.build_model().eval()
    header = {"model": {"label_schema": setup.label_schema()}, "reference": {
        "allele_order_sha256": serialize.allele_order_sha256(reference),
        "reference_fasta_sha256": serialize.reference_fasta_sha256(reference)}}
    sections = {"config": (serialize.config_to_bytes(setup.config), "zlib"),
                "weights": (serialize.state_dict_to_bytes(model.state_dict()), "none"),
                "reference_json": (serialize.reference_to_json(reference), "zlib")}
    path = tmp_path / "model.alignair"
    container.write_container(path, header, sections)
    return path, header, sections, model


def test_safe_restore_preserves_predictions_and_label_identity(artifact):
    path, header, _, original = artifact
    loaded = model_file.load_model(path)
    assert loaded.metadata["model"]["label_schema"] == header["model"]["label_schema"]
    assert loaded.config is loaded.model.cfg
    assert not loaded.model.training
    tokens = torch.ones(1, 128, dtype=torch.long)
    with torch.no_grad():
        expected = original({"tokenized_sequence": tokens})
        actual = loaded.model({"tokenized_sequence": tokens})
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("binding", ["allele_order_sha256", "reference_fasta_sha256", "label_schema"])
def test_invalid_binding_fails_before_model_allocation(artifact, binding, monkeypatch):
    path, header, sections, _ = artifact
    if binding == "label_schema":
        header["model"]["label_schema"]["genes"]["v"].reverse()
    else:
        header["reference"][binding] = "0" * 64
    container.write_container(path, header, sections)
    def forbidden(*args, **kwargs):
        raise AssertionError("model allocated before reference/label preflight")
    monkeypatch.setattr(AlignAIR, "__init__", forbidden)
    with pytest.raises(ModelIntegrityError, match="label schema" if binding == "label_schema" else binding):
        model_file.load_model(path)


def test_safe_artifact_weight_mismatch_uses_shared_diagnostics(artifact):
    path, header, sections, model = artifact
    state = model.state_dict()
    del state[next(iter(state))]
    sections["weights"] = (serialize.state_dict_to_bytes(state), "none")
    container.write_container(path, header, sections)
    with pytest.raises(ModelIntegrityError, match="Missing key") as caught:
        model_file.load_model(path)
    assert caught.value.context["stage"] == "model_restoration"
    assert caught.value.context["path"] == str(path)
