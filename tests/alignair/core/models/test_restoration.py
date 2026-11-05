"""Strict, format-independent construction from decoded model state."""
from dataclasses import replace

import pytest
import torch

from alignair.core import AlignAIR, build_model_config, restore_model_state
from alignair.errors import ConfigurationError, ModelIntegrityError

pytestmark = pytest.mark.component


def _config(input_space="nucleotide", label_space="alleles", has_d=True):
    return build_model_config(3, 2, 2 if has_d else 0, has_d=has_d,
                              input_space=input_space, label_space=label_space,
                              max_seq_length=128, embed_dim=4, filters=4, block_out=8,
                              state_head=True)


@pytest.mark.parametrize("input_space,label_space", [
    ("nucleotide", "alleles"),
])
@pytest.mark.parametrize("has_d", [False, True])
def test_restoration_preserves_weights_and_outputs_across_model_policies(input_space, label_space, has_d):
    cfg = _config(input_space, label_space, has_d)
    original = AlignAIR(cfg).eval()
    restored = AlignAIR.from_state_dict(cfg, original.state_dict())
    assert restored.cfg == cfg and restored.cfg is not cfg
    assert all(not module.training for module in restored.modules())
    assert all(parameter.requires_grad for parameter in restored.parameters())
    for name, weights in original.state_dict().items():
        assert torch.equal(restored.state_dict()[name], weights), name
        assert restored.state_dict()[name].data_ptr() != weights.data_ptr(), name
    tokens = torch.zeros(2, 128, dtype=torch.long)
    tokens[:, :12] = 1
    with torch.no_grad():
        expected = original({"tokenized_sequence": tokens})
        actual = restored({"tokenized_sequence": tokens})
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("fault,detail", [
    ("missing", "Missing key"), ("unexpected", "Unexpected key"),
    ("shape", "size mismatch"), ("not_tensor", "Tensor"), ("not_mapping", "dict-like"),
])
def test_incompatible_state_is_contextual_and_preserves_the_pytorch_cause(fault, detail, tmp_path):
    cfg = _config()
    state = AlignAIR(cfg).state_dict()
    key = next(iter(state))
    if fault == "missing":
        del state[key]
    elif fault == "unexpected":
        state["retired.weight"] = torch.ones(1)
    elif fault == "shape":
        state[key] = torch.ones(1)
    elif fault == "not_tensor":
        state[key] = "invalid weights"
    else:
        state = "invalid state"
    path = tmp_path / "broken.alignair"
    with pytest.raises(ModelIntegrityError, match=detail) as caught:
        AlignAIR.from_state_dict(cfg, state, source=path)
    assert isinstance(caught.value.__cause__, (RuntimeError, TypeError))
    assert caught.value.context == {"stage": "model_restoration", "path": str(path),
                                    "architecture": cfg.architecture}


def test_cpu_restoration_ignores_ambient_default_device():
    cfg = _config()
    original = AlignAIR(cfg)
    with torch.device("meta"):
        restored = AlignAIR.from_state_dict(cfg, original.state_dict(), device="cpu")
    assert all(parameter.device.type == "cpu" for parameter in restored.parameters())
    for name, weights in original.state_dict().items():
        assert torch.equal(restored.state_dict()[name], weights), name


def test_unknown_architecture_keeps_its_configuration_error():
    with pytest.raises(ConfigurationError, match="unknown model architecture"):
        AlignAIR.from_state_dict(replace(_config(), architecture="unregistered"), {})


def test_device_failure_is_not_reported_as_weight_corruption():
    cfg = _config()
    with pytest.raises(RuntimeError) as caught:
        AlignAIR.from_state_dict(cfg, AlignAIR(cfg).state_dict(), device="invalid-device")
    assert not isinstance(caught.value, ModelIntegrityError)


def test_existing_instance_restoration_retains_modes_and_optimizer_parameter_bindings():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Dropout())
    model.train()
    model[0].eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    identities = [id(parameter) for parameter in model.parameters()]
    modes = [module.training for module in model.modules()]
    state = {name: torch.full_like(value, .25) for name, value in model.state_dict().items()}
    restore_model_state(model, state)
    assert [id(parameter) for parameter in model.parameters()] == identities
    assert [id(parameter) for group in optimizer.param_groups for parameter in group['params']] == identities
    assert [module.training for module in model.modules()] == modes
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name])
    model[0](torch.ones(1, 2)).sum().backward()
    optimizer.step()
    assert not torch.equal(model[0].weight, state['0.weight'])


def test_existing_instance_failure_uses_shared_diagnostics_without_requiring_alignair_config(tmp_path):
    source = tmp_path / 'checkpoint.alignair'
    with pytest.raises(ModelIntegrityError, match='Missing key') as caught:
        restore_model_state(torch.nn.Linear(2, 1), {}, source=source)
    assert caught.value.context == {'stage': 'model_restoration', 'path': str(source), 'architecture': None}
    assert isinstance(caught.value.__cause__, RuntimeError)
