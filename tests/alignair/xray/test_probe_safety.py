"""Diagnostics preserve mixed module modes, buffers, gradients and temporary hooks."""
import pytest
import torch

from alignair.xray import ModelXRay, network

pytestmark = pytest.mark.component


def model():
    result = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3), torch.nn.Dropout())
    result.train()
    result[2].eval()
    return result


@pytest.mark.parametrize('probe', [network.activation_stats, network.capture_activations])
@pytest.mark.parametrize('failure', [None, 'forward', 'registration'])
def test_activation_probes_restore_mixed_modes_and_clean_hooks(probe, failure, monkeypatch):
    module = model()
    modes = [child.training for child in module.modules()]
    state = {name: value.clone() for name, value in module.state_dict().items()}
    def fail(*args, **kwargs):
        raise RuntimeError('injected failure')
    if failure == 'forward':
        monkeypatch.setattr(module[1], 'forward', fail)
    elif failure == 'registration':
        monkeypatch.setattr(module[1], 'register_forward_hook', fail)
    if failure:
        with pytest.raises(RuntimeError, match='injected failure'):
            probe(module, torch.ones(2, 3), module_names=['0', '1'])
    else:
        assert set(probe(module, torch.ones(2, 3), module_names=['0', '1'])) == {'0', '1'}
    assert [child.training for child in module.modules()] == modes
    assert all(torch.equal(value, state[name]) for name, value in module.state_dict().items())
    assert all(not child._forward_hooks for child in module.modules())
    assert torch.is_grad_enabled()


def test_captured_activations_snapshot_cpu_storage():
    module = torch.nn.Identity()
    source = torch.ones(2, 3)
    captured = network.capture_activations(module, source, module_names=[''])['']
    source.fill_(7)
    assert torch.equal(captured, torch.ones(2, 3))


@pytest.mark.parametrize('failure', [False, True])
def test_conflict_probe_enables_gradients_but_preserves_caller_state(failure):
    module = model()
    modes = [child.training for child in module.modules()]
    state = {name: value.clone() for name, value in module.state_dict().items()}
    def losses(current, inputs):
        assert not any(child.training for child in current.modules())
        assert torch.is_grad_enabled()
        if failure:
            raise RuntimeError('injected failure')
        output = current(inputs)
        return {'a': output.sum(), 'b': output.square().sum()}
    probe = ModelXRay(module, lr=.001, task_losses=losses)
    with torch.no_grad():
        if failure:
            with pytest.raises(RuntimeError, match='injected failure'):
                probe._conflict(torch.ones(2, 3))
        else:
            assert 'a|b' in probe._conflict(torch.ones(2, 3))['cosine']
        assert not torch.is_grad_enabled()
    assert [child.training for child in module.modules()] == modes
    assert all(torch.equal(value, state[name]) for name, value in module.state_dict().items())
    assert all(parameter.grad is None for parameter in module.parameters())


def test_task_evaluation_freezes_batchnorm_and_restores_modes_on_failure():
    module = model()
    modes = [child.training for child in module.modules()]
    state = {name: value.clone() for name, value in module.state_dict().items()}
    def evaluate(current, inputs):
        assert not any(child.training for child in current.modules())
        assert not torch.is_grad_enabled()
        current(inputs)
        raise RuntimeError('evaluation failed')
    probe = ModelXRay(module, lr=.001, deep_every=1, task_eval=evaluate, hook_targets=['0'])
    with pytest.raises(RuntimeError, match='evaluation failed'):
        probe.observe(1, 1., probe_input=torch.ones(2, 3))
    assert [child.training for child in module.modules()] == modes
    assert all(torch.equal(value, state[name]) for name, value in module.state_dict().items())
