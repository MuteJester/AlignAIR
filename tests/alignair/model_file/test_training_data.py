"""Checkpoint decoding and model restoration are separate allocation and validation phases."""
import GenAIRR.data as gd
import numpy as np
import pytest
import torch

from alignair import model_file
from alignair.core import AlignAIR, prepare_model
from alignair.errors import ModelIntegrityError, ModelError
from alignair.model_file import container, serialize
from alignair.reference import ReferenceSet
from alignair.train.checkpoints import load_training_checkpoint
from alignair.train.random_state import capture_rng_state

pytestmark = pytest.mark.integration


@pytest.fixture
def checkpoint(tmp_path):
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), max_seq_length=128,
                          embed_dim=4, trunk_channels=8)
    model = setup.build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=9, eta_min=.00001)
    for group in optimizer.param_groups:
        for parameter in group['params']:
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()
    rng = capture_rng_state(include_cuda=False)
    arguments = {'lr': .001, 'schedule': {'total_steps': 9, 'min_lr': .00001}}
    path = tmp_path / 'resume.alignair'
    model_file.save_model(path, model, setup.reference, dataconfigs=[gd.HUMAN_IGH_OGRDB],
                          training={'steps': 1, 'batch_size': 2, 'train_args': arguments},
                          optimizer=optimizer, scheduler=scheduler, rng=rng)
    return path, setup, model, optimizer, scheduler, rng, arguments


def _equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _equal(left, right)
    else:
        assert actual == expected


def _rewrite(path, mutate):
    header = container.read_header(path)
    header.pop('_sections_base', None)
    sections = {name: (container.read_section(path, name), info['codec'])
                for name, info in header['sections'].items()}
    mutate(header, sections)
    container.write_container(path, header, sections)


def test_decode_preserves_all_state_without_model_construction_or_rng_consumption(checkpoint, monkeypatch):
    path, setup, model, optimizer, scheduler, rng, arguments = checkpoint
    before = capture_rng_state(include_cuda=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('checkpoint decoding constructed a temporary model')
    monkeypatch.setattr(AlignAIR, '__init__', forbidden)
    decoded = model_file.read_model_state(path)
    assert decoded.config == setup.config
    _equal(decoded.model_state, model.state_dict())
    for data in (model_file.read_training_state(path), load_training_checkpoint(path)):
        assert data.config == setup.config and data.step == 1
        _equal(data.model_state, model.state_dict())
        _equal(data.optimizer_state, optimizer.state_dict())
        _equal(data.scheduler_state, scheduler.state_dict())
        _equal(data.rng, rng)
    decoded = model_file.read_training_state(path)
    assert isinstance(decoded, model_file.TrainingStateData)
    assert decoded.train_args == arguments
    assert decoded.metadata['reference']['allele_order_sha256'] == serialize.allele_order_sha256(setup.reference)
    assert decoded.reference.to_serializable() == setup.reference.to_serializable()
    _equal(capture_rng_state(include_cuda=False), before)


def test_existing_model_bearing_api_still_restores_an_evaluation_model(checkpoint):
    path, setup, model, optimizer, scheduler, rng, arguments = checkpoint
    restored = model_file.load_training_state(path)
    assert isinstance(restored, model_file.TrainingState)
    assert isinstance(restored.model, AlignAIR) and not restored.model.training
    assert restored.model is not model and restored.config == setup.config
    assert restored.config is restored.model.cfg and restored.config is not setup.config
    _equal(restored.model.state_dict(), model.state_dict())
    _equal(restored.optimizer_state, optimizer.state_dict())
    _equal(restored.scheduler_state, scheduler.state_dict())
    _equal(restored.rng, rng)
    assert restored.train_args == arguments


@pytest.mark.parametrize('digest', ['allele_order_sha256', 'reference_fasta_sha256'])
def test_decode_keeps_the_reference_integrity_checks(checkpoint, digest, monkeypatch):
    path = checkpoint[0]
    def alter(header, sections):
        header['reference'][digest] = '0' * 64
    _rewrite(path, alter)
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid checkpoint caused model construction')
    monkeypatch.setattr(AlignAIR, '__init__', forbidden)
    with pytest.raises(ModelIntegrityError) as caught:
        model_file.read_training_state(path)
    assert caught.value.context['stage'] == 'model_binding'
    assert caught.value.context['path'] == str(path)


def test_missing_training_section_fails_without_model_construction(checkpoint, monkeypatch):
    path = checkpoint[0]
    _rewrite(path, lambda header, sections: sections.pop('train_state'))
    def forbidden(*args, **kwargs):
        raise AssertionError('inference artifact caused temporary model construction')
    monkeypatch.setattr(AlignAIR, '__init__', forbidden)
    with pytest.raises(ModelError, match='no train_state') as caught:
        model_file.read_training_state(path)
    assert caught.value.context['path'] == str(path)


def test_decoded_weights_are_checked_against_an_architecture_at_restoration(checkpoint):
    path, setup, *_ = checkpoint
    def alter(header, sections):
        state = serialize.state_dict_from_bytes(sections['weights'][0])
        state.pop(next(iter(state)))
        sections['weights'] = (serialize.state_dict_to_bytes(state), 'none')
    _rewrite(path, alter)
    decoded = model_file.read_training_state(path)
    for restore in (lambda: setup.restore_model(decoded.model_state, source=path),
                    lambda: model_file.load_training_state(path)):
        with pytest.raises(ModelIntegrityError, match='Missing key') as caught:
            restore()
        assert caught.value.context['stage'] == 'model_restoration'
        assert caught.value.context['path'] == str(path)


def test_read_maps_weights_to_requested_device_without_constructing_a_model(checkpoint, monkeypatch):
    path = checkpoint[0]
    def forbidden(*args, **kwargs):
        raise AssertionError('mapping decoded tensors constructed a model')
    monkeypatch.setattr(AlignAIR, '__init__', forbidden)
    for data in (model_file.read_training_state(path, device='meta'),
                 load_training_checkpoint(path, device='meta')):
        assert all(tensor.device.type == 'meta' for tensor in data.model_state.values())


def test_a_checkpoint_without_reference_json_is_refused_before_any_model_is_built(checkpoint, monkeypatch):
    path, *_ = checkpoint
    _rewrite(path, lambda header, sections: sections.pop('reference_json'))
    def forbidden(*args, **kwargs):
        raise AssertionError('reading a training checkpoint constructed an AlignAIR model')
    monkeypatch.setattr(AlignAIR, '__init__', forbidden)
    with pytest.raises(ModelError, match='no reference_json: it was written before AlignAIR 4'):
        model_file.read_training_state(path)
