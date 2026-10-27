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
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: device_state)
    restored = []
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', restored.append)
    assert capture_rng_state()['cuda'] is device_state
    restore_rng_state({'cuda': [[4, 5, 6]]})
    assert restored[0][0].device.type == 'cpu'
    assert restored[0][0].dtype == torch.uint8
    assert restored[0][0].tolist() == [4, 5, 6]


@pytest.mark.parametrize('state', [torch.tensor([1., 2.]), [256, 0], [-1, 0], [[1, 2]], [True]])
def test_rng_buffers_cannot_silently_coerce_invalid_byte_values(state):
    with pytest.raises(TrainingError, match='cannot be restored'):
        restore_rng_state({'torch': state})
