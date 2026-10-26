"""Shared model contracts, architecture reconstruction, and component inspection."""
from dataclasses import replace
import json

import GenAIRR.data as gd
import pytest
import torch

from alignair import model_file
from alignair.core import AlignAIR, AlignAIRConfig, ModelComponents, register_architecture
from alignair.core import components
from alignair.core.layers import TokenAndPositionEmbedding
from alignair.core.gene_branch import GeneBranch
from alignair.errors import ConfigurationError
from alignair.model_file.serialize import config_from_bytes, config_to_bytes

pytestmark = pytest.mark.component


def _config():
    return AlignAIRConfig(v_allele_count=3, j_allele_count=2, d_allele_count=2, has_d=True,
                          max_seq_length=128, embed_dim=4, filters=4, block_out=8)


def _batch(vocab_size):
    tokens = torch.zeros(2, 128, dtype=torch.long)
    tokens[0, :31] = torch.randint(1, vocab_size, (31,))
    tokens[1, :45] = torch.randint(1, vocab_size, (45,))
    return {"tokenized_sequence": tokens}


@pytest.mark.parametrize("categorical", [False, True], ids=["nucleotide", "protein"])
def test_shared_model_score_semantics_and_boundary_slots(categorical):
    cfg = _config()
    if categorical:
        cfg = replace(cfg, vocab_size=23).with_amino_acid_heads()
    model = AlignAIR(cfg).eval()
    with torch.no_grad():
        model.branches["v"].cls_head.weight.zero_()
        model.branches["v"].cls_head.bias.copy_(torch.tensor([-2.0, 0.0, 2.0]))
        out = model(_batch(cfg.vocab_size))
    expected = torch.tensor([-2.0, 0.0, 2.0])
    if categorical:
        torch.testing.assert_close(out["v_class_logits"], expected.expand(2, -1))
        assert out["end_mask"][0, 31] and not out["start_mask"][0, 31]
        assert "orientation_logits" not in out and "mutation_rate" not in out
    else:
        torch.testing.assert_close(out["v_allele"], expected.sigmoid().expand(2, -1))
        assert not out["position_mask"][0, 31]
        assert "orientation_logits" in out and "mutation_rate" in out


def test_categorical_scores_can_be_composed_with_auxiliary_and_state_heads():
    cfg = replace(_config(), classification_mode="categorical", use_orientation=False, state_head=True)
    model = AlignAIR(cfg).eval()
    out = model(_batch(cfg.vocab_size))
    assert {"v_class_logits", "mutation_rate", "productive", "state_logits"} <= out.keys()
    assert "v_allele" not in out and "orientation_logits" not in out
    sum(t.square().mean() for t in out.values() if t.is_floating_point()).backward()
    assert model.branches["v"].cls_head.weight.grad is not None
    assert model.meta_heads["productive"].head.weight.grad is not None
    assert model.state_branch.head.fc.weight.grad is not None


@pytest.mark.parametrize("protein", [False, True])
def test_config_roundtrip_preserves_construction_choices(protein):
    cfg = _config().with_amino_acid_heads() if protein else _config()
    restored = config_from_bytes(config_to_bytes(cfg))
    assert restored == cfg
    model = AlignAIR(cfg).eval()
    rebuilt = AlignAIR(restored).eval()
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    batch = _batch(cfg.vocab_size)
    with torch.no_grad():
        original, actual = model(batch), rebuilt(batch)
    assert original.keys() == actual.keys()
    for key in original:
        torch.testing.assert_close(actual[key], original[key], rtol=0, atol=0)


def test_old_config_without_construction_choices_retains_nucleotide_defaults():
    cfg = config_from_bytes(json.dumps({"v_allele_count": 3, "j_allele_count": 2,
                                       "has_d": False, "max_seq_length": 128}).encode())
    model = AlignAIR(cfg)
    assert model.orientation_head is not None and model.meta_tower is not None
    assert cfg.classification_key("v") == "v_allele"


@pytest.mark.parametrize("protein", [False, True])
def test_component_inspection_uses_registered_modules_without_duplicating_weights(protein):
    cfg = _config().with_amino_acid_heads() if protein else _config()
    model = AlignAIR(cfg)
    before = tuple(model.state_dict())
    points = model.component_modules()
    assert points["v.cls_head"] is model.branches["v"].cls_head
    assert points["embedding"] is model.embedding
    assert ("orientation_head" in points) is not protein
    assert ("meta_tower" in points) is not protein
    assert tuple(model.state_dict()) == before
    assert all(any(module is child for child in model.modules()) for module in points.values())


class _ScaledEmbedding(TokenAndPositionEmbedding):
    def forward(self, tokens):
        return 2 * super().forward(tokens)


def _scaled_embedding(cfg):
    return _ScaledEmbedding(cfg.vocab_size, cfg.embed_dim, cfg.max_seq_length)


def test_custom_architecture_rebuilds_from_safe_checkpoint_and_requires_its_factories(tmp_path, monkeypatch):
    name = "test-scaled-conv-v1"
    # Let pytest restore the registry even if an assertion fails after registration.
    monkeypatch.setattr(components, "_ARCHITECTURES", dict(components._ARCHITECTURES))
    register_architecture(name, replace(ModelComponents(), embedding=_scaled_embedding))
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=128,
                                          embed_dim=4, filters=4, block_out=8, architecture=name)
    model = AlignAIR(cfg).eval()
    path = tmp_path / "custom.alignair"
    model_file.save_model(path, model, dataconfigs=[gd.HUMAN_IGH_OGRDB], training={},
                          include_trusted_pickle=False)
    loaded = model_file.load_model(path)
    assert isinstance(loaded.model.embedding, _ScaledEmbedding)
    batch = _batch(cfg.vocab_size)
    with torch.no_grad():
        expected, actual = model(batch), loaded.model(batch)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    monkeypatch.delitem(components._ARCHITECTURES, name)
    with pytest.raises(ConfigurationError, match="unknown model architecture"):
        model_file.load_model(path)


def test_architecture_names_cannot_silently_replace_existing_definitions():
    with pytest.raises(ConfigurationError, match="already registered"):
        register_architecture("conv-v1", ModelComponents())


@pytest.mark.parametrize("field,value", [("classification_mode", "softmax"), ("boundary_mode", "unknown")])
def test_invalid_construction_choices_fail_with_configuration_context(field, value):
    with pytest.raises(ConfigurationError) as caught:
        replace(_config(), **{field: value})
    assert caught.value.context["field"] == field


def test_invalid_component_factory_identifies_the_component():
    with pytest.raises(ConfigurationError) as caught:
        ModelComponents(embedding=None)
    assert caught.value.context["component"] == "embedding"


class _MeanTower(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.projection = torch.nn.Linear(in_dim, out_dim)

    def forward(self, embeddings):
        return self.projection(embeddings.mean(dim=1))


def _mean_branch(spec, cfg):
    return GeneBranch(spec, cfg,
                      segmentation_tower=_MeanTower(cfg.embed_dim, cfg.block_out),
                      classification_tower=_MeanTower(cfg.embed_dim, cfg.cls_spec(spec)[0]))


@pytest.mark.parametrize("protein", [False, True])
def test_same_replacement_towers_work_with_both_head_presets(protein, monkeypatch):
    monkeypatch.setattr(components, "_ARCHITECTURES", dict(components._ARCHITECTURES))
    register_architecture("test-mean-towers-v1", replace(ModelComponents(), gene_branch=_mean_branch))
    cfg = replace(_config(), architecture="test-mean-towers-v1")
    if protein:
        cfg = cfg.with_amino_acid_heads()
    model = AlignAIR(cfg).eval()
    out = model(_batch(cfg.vocab_size))
    scores = out[cfg.classification_key("v")]
    assert scores.shape == (2, cfg.v_allele_count)
    scores.sum().backward()
    assert model.branches["v"].seg_tower.projection.weight.grad is not None
    assert model.branches["v"].cls_tower.projection.weight.grad is not None
    assert model.component_modules()["v.seg_tower"] is model.branches["v"].seg_tower
