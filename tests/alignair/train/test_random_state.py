"""RNG continuation and rollback use independently advanced generator sequences."""
import random

import numpy as np
import pytest
import torch

from alignair.errors import TrainingError
from alignair.train.random_state import capture_rng_state, restore_rng_state

pytestmark = pytest.mark.component


def _draw():
    return random.random(), np.random.random(4), torch.rand(4)


def _assert_draw(actual, expected):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_process_generators_continue_exactly_after_restore_including_python():
    snapshot = capture_rng_state()
    expected = _draw()
    _draw()
    restore_rng_state(snapshot)
    _assert_draw(_draw(), expected)


@pytest.mark.parametrize('bad_field', ['numpy', 'torch', 'python'])
def test_invalid_state_raises_and_rolls_back_every_generator(bad_field):
    snapshot = capture_rng_state()
    expected = _draw()
    restore_rng_state(snapshot)
    alternate = random.Random(999).getstate()
    invalid = {'python': alternate, bad_field: 'malformed'}
    with pytest.raises(TrainingError, match='cannot be restored') as caught:
        restore_rng_state(invalid)
    assert caught.value.context['stage'] == 'training_resume_rng'
    assert caught.value.__cause__ is not None
    _assert_draw(_draw(), expected)


def test_absent_fields_are_untouched_and_serialized_torch_bytes_are_supported():
    snapshot = capture_rng_state()
    expected = _draw()
    restore_rng_state(snapshot)
    restore_rng_state({})
    restore_rng_state({'torch': snapshot['torch'].tolist(), 'numpy': None, 'python': None})
    _assert_draw(_draw(), expected)


def test_cuda_state_uses_cpu_byte_tensors_without_requiring_a_gpu(monkeypatch):
    device_state = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
    restored = []
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, 'is_available', lambda: True)
        patch.setattr(torch.cuda, 'get_rng_state_all', lambda: device_state)
        patch.setattr(torch.cuda, 'set_rng_state_all', restored.append)
        assert capture_rng_state()['cuda'] is device_state
        restore_rng_state({'cuda': [[4, 5, 6]]})
    assert restored[0][0].device.type == 'cpu'
    assert restored[0][0].dtype == torch.uint8
    assert restored[0][0].tolist() == [4, 5, 6]


@pytest.mark.parametrize('state', [torch.tensor([1., 2.]), [256, 0], [-1, 0], [[1, 2]], [True]])
def test_rng_buffers_cannot_silently_coerce_invalid_byte_values(state):
    with pytest.raises(TrainingError, match='cannot be restored'):
        restore_rng_state({'torch': state})


@pytest.mark.parametrize('invalid', [False, True])
def test_cuda_exclusion_skips_cuda_capture_apply_and_rollback_on_any_host(monkeypatch, invalid):
    def forbidden(*args): pytest.fail('CUDA must not be queried when excluded')
    with monkeypatch.context() as patch:
        for operation in ('is_available', 'get_rng_state_all', 'set_rng_state_all'):
            patch.setattr(torch.cuda, operation, forbidden)
        snapshot = capture_rng_state(include_cuda=False)
        assert set(snapshot) == {'python', 'numpy', 'torch'}
        expected = _draw()
        restore_rng_state(snapshot, include_cuda=False)
        requested = {**snapshot, 'cuda': 'ignored malformed CUDA state'}
        if invalid:
            requested.update(python=random.Random(99).getstate(), numpy=())
            with pytest.raises(TrainingError): restore_rng_state(requested, include_cuda=False)
        else:
            restore_rng_state(requested, include_cuda=False, require_process_state=True)
        _assert_draw(_draw(), expected)


@pytest.mark.parametrize('field', ['python', 'numpy', 'torch'])
@pytest.mark.parametrize('missing', [True, False])
def test_complete_process_requirement_checks_all_fields_before_applying_any(field, missing):
    snapshot = capture_rng_state(include_cuda=False)
    expected = _draw(); restore_rng_state(snapshot, include_cuda=False)
    requested = {**snapshot, 'python': random.Random(99).getstate()}
    if missing: del requested[field]
    else: requested[field] = None
    with pytest.raises(TrainingError, match='incomplete') as caught:
        restore_rng_state(requested, include_cuda=False, require_process_state=True)
    assert caught.value.context['fields'] == [field]
    _assert_draw(_draw(), expected)


@pytest.mark.parametrize('value', [[], 'state', False, 1, None])
def test_nonmapping_state_is_classified_instead_of_raising_attribute_errors(value):
    with pytest.raises(TrainingError, match='mapping'):
        restore_rng_state(value, require_process_state=True)


def test_none_remains_a_noop_for_sparse_restore():
    snapshot = capture_rng_state(include_cuda=False)
    expected = _draw(); restore_rng_state(snapshot, include_cuda=False)
    restore_rng_state(None)
    _assert_draw(_draw(), expected)


@pytest.mark.parametrize('field', ['python', 'numpy'])
def test_truncated_generator_tuples_raise_and_roll_back_other_generators(field):
    snapshot = capture_rng_state(include_cuda=False)
    expected = _draw(); restore_rng_state(snapshot, include_cuda=False)
    requested = {**snapshot, 'python': random.Random(99).getstate(), field: ()}
    with pytest.raises(TrainingError) as caught: restore_rng_state(requested, include_cuda=False)
    assert isinstance(caught.value.__cause__, (IndexError, ValueError))
    _assert_draw(_draw(), expected)


def test_partial_cuda_apply_failure_rolls_back_process_and_cuda_state(monkeypatch):
    snapshot = capture_rng_state(include_cuda=False)
    expected = _draw(); restore_rng_state(snapshot, include_cuda=False)
    original_cuda = [torch.tensor([1, 2], dtype=torch.uint8), torch.tensor([3, 4], dtype=torch.uint8)]
    current_cuda = [value.clone() for value in original_cuda]
    calls = []
    def apply(values):
        calls.append(values)
        current_cuda[0] = values[0].clone()
        if values[1].tolist() == [9, 9]: raise RuntimeError('second CUDA generator failed')
        current_cuda[1] = values[1].clone()
    requested = {**snapshot, 'python': random.Random(99).getstate(), 'cuda': [[8, 8], [9, 9]]}
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, 'is_available', lambda: True)
        patch.setattr(torch.cuda, 'get_rng_state_all', lambda: [value.clone() for value in current_cuda])
        patch.setattr(torch.cuda, 'set_rng_state_all', apply)
        with pytest.raises(TrainingError): restore_rng_state(requested)
    assert len(calls) == 2
    assert all(torch.equal(actual, expected) for actual, expected in zip(current_cuda, original_cuda))
    _assert_draw(_draw(), expected)
