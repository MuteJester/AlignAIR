import pytest
from dataclasses import replace
import GenAIRR.data as gd
import torch
from alignair.core.config import AlignAIRConfig
from alignair.core import AlignAIR
from alignair import model_file as mf
from alignair.api import load_model as api_load
from alignair.core import prepare_model
from alignair.errors import ConfigurationError, ModelError
from alignair.reference import ReferenceSet

pytestmark = pytest.mark.integration


def test_api_load_model_reads_alignair_without_dataconfig(tmp_path):
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    m = AlignAIR(cfg)
    p = tmp_path / "m.alignair"
    mf.save_model(str(p), m, dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1})
    model, reference = api_load(str(p))                 # NO dataconfigs= needed
    assert reference.gene("V").names[0].startswith("IGH")
    assert model.cfg.v_allele_count == cfg.v_allele_count


@pytest.fixture
def legacy_checkpoint(tmp_path):
    reference = ReferenceSet.from_genotype({"V": {"V1": "ATGTGTAAA"}, "J": {"J1": "TGGAAA"}})
    setup = prepare_model(reference, max_seq_length=128, embed_dim=4, filters=4, block_out=8)
    original = setup.build_model().eval()
    # A historical key layout still belongs to the legacy decoder, before strict restoration.
    state = {name.replace("branches.v.cls_head.", "cls_head.v."): value
             for name, value in original.state_dict().items()}
    path = tmp_path / "legacy.pt"
    torch.save({"config": setup.config.__dict__, "model": state}, path)
    return path, reference, original


def test_legacy_restoration_keeps_trust_gate_key_migration_and_reference(legacy_checkpoint, monkeypatch):
    path, reference, original = legacy_checkpoint
    original_load = torch.load
    calls = []
    def tracked_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", tracked_load)
    with pytest.raises(ModelError, match="trust_pickle"):
        api_load(path, reference=reference)
    assert calls == []
    model, bound_reference = api_load(path, reference=reference, trust_pickle=True)
    assert bound_reference is reference
    assert all(not module.training for module in model.modules())
    assert calls == [{"map_location": "cpu", "weights_only": False}]
    for name, weights in original.state_dict().items():
        assert torch.equal(model.state_dict()[name], weights), name


def test_legacy_missing_reference_fails_before_model_allocation(legacy_checkpoint, monkeypatch):
    path, _, _ = legacy_checkpoint
    def forbidden(*args, **kwargs):
        raise AssertionError("model allocated before reference preflight")
    monkeypatch.setattr(AlignAIR, "__init__", forbidden)
    with pytest.raises(ConfigurationError, match="reference"):
        api_load(path, trust_pickle=True)


def test_legacy_reference_head_mismatch_fails_before_allocation(legacy_checkpoint, monkeypatch):
    path, _, _ = legacy_checkpoint
    wrong = ReferenceSet.from_genotype({'V': {'V1': 'ATGTGTAAA', 'V2': 'ATGTGTTTT'},
                                        'J': {'J1': 'TGGAAA'}})
    def forbidden(*args, **kwargs):
        raise AssertionError('legacy model allocated before binding preflight')
    monkeypatch.setattr(AlignAIR, '__init__', forbidden)
    with pytest.raises(ConfigurationError, match='head size') as caught:
        api_load(path, reference=wrong, trust_pickle=True)
    assert caught.value.context['gene'] == 'v'
    assert caught.value.context['stage'] == 'model_binding'
    assert caught.value.context['path'] == str(path)


@pytest.mark.parametrize('has_schema', [False, True])
def test_legacy_protein_class_loading_uses_saved_labels_instead_of_allele_counts(legacy_checkpoint, has_schema):
    path, reference, original = legacy_checkpoint
    cfg = replace(original.cfg, vocab_size=23).with_amino_acid_heads()
    model = AlignAIR(cfg)
    payload = {'config': cfg.__dict__, 'model': model.state_dict()}
    if has_schema:
        payload['label_schema'] = {'schema_version': 1, 'input_space': 'protein',
            'label_space': 'protein_classes', 'genes': {'v': [['V1']], 'j': [['J1']]}}
    torch.save(payload, path)
    if not has_schema:
        with pytest.raises(ConfigurationError, match='ordered class labels'):
            api_load(path, reference=reference, trust_pickle=True)
    else:
        loaded, bound = api_load(path, reference=reference, trust_pickle=True)
        assert bound is reference
        for name, weights in model.state_dict().items():
            assert torch.equal(loaded.state_dict()[name], weights), name
