"""Serial reference runs for the parallel training-data tests: the trainer's own loop over each stream."""
import random

import numpy as np
import torch

from alignair.train import StepResult, training_iterations
from alignair.train.gym.streams import training_records
from alignair.train.random_state import capture_rng_state


class RecordingEngine:
    """Keeps each step's batch and the CPU random state the step starts with; may draw from that state."""

    def __init__(self, *, draw=False):
        self.steps, self._draw = [], draw

    def step(self, batch, *, step):
        self.steps.append((step, batch, capture_rng_state(include_cuda=False)))
        if self._draw:
            torch.rand(1)
        return StepResult(0., {}, 0.)


def serial_records(stream):
    """The trainer's serial construction (``train``'s ``_build_stream``) of a ``TrainingStream``."""
    records = training_records(stream.dataconfigs, policy=stream.policy, seed=stream.seed, start_step=stream.start_step)
    return stream.data.prepare_records(records, augment=stream.augment, seed=stream.prepare_seed,
                                       start_step=stream.start_step)


def run(records, batch_adapter, *, start_step, steps, batch_size, engine=None, seed=1234):
    """Drive ``training_iterations`` from seeded process generators; return its steps, batches and error."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    engine = engine or RecordingEngine()
    iterations, error = [], None
    try:
        for iteration in training_iterations(engine, records, batch_adapter, start_step=start_step,
                                             stop_step=start_step + steps, batch_size=batch_size):
            iterations.append((iteration.step, iteration.records))
    except Exception as exc:
        error = exc
    return iterations, engine.steps, error


def same_rng(left, right):
    return (left['python'] == right['python'] and torch.equal(left['torch'], right['torch'])
            and left['numpy'][0] == right['numpy'][0] and np.array_equal(left['numpy'][1], right['numpy'][1])
            and left['numpy'][2:] == right['numpy'][2:])


def assert_same_batch(left, right):
    for part in ('inputs', 'targets'):
        expected, actual = getattr(left, part), getattr(right, part)
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            other = actual[key]
            assert other.dtype == value.dtype and other.device.type == value.device.type, key
            assert other.shape == value.shape and torch.equal(other.cpu(), value.cpu()), key


def assert_same_runs(serial, parallel, *, steps):
    """Equal step numbers, records, batches and CPU random states; a serial error must also match."""
    serial_iterations, serial_steps, serial_error = serial
    parallel_iterations, parallel_steps, parallel_error = parallel
    assert len(serial_iterations) == steps or serial_error is not None
    assert [step for step, _ in parallel_iterations] == [step for step, _ in serial_iterations]
    assert [list(records) for _, records in parallel_iterations] == [list(records) for _, records in serial_iterations]
    assert len(parallel_steps) == len(serial_steps)
    for (step, expected, expected_rng), (other_step, actual, actual_rng) in zip(serial_steps, parallel_steps):
        assert other_step == step
        assert_same_batch(expected, actual)
        assert same_rng(expected_rng, actual_rng), step
    assert type(parallel_error) is type(serial_error)
    if serial_error is not None:
        assert str(parallel_error) == str(serial_error)
        assert getattr(parallel_error, 'context', None) == getattr(serial_error, 'context', None)
