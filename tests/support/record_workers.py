"""Spawn-safe record providers for worker-process tests: a spawned worker imports them by module name.

A component is known by its seed: a test names the component a provider singles out by its planned seed
(``gym.workers.component_plan``)."""
from dataclasses import dataclass
import os
import threading

from alignair.errors import TrainingDataError, error_context


def counted(dataconfig, parameters, seed):
    """An endless, deterministic component stream identified by panel, seed and position."""
    ordinal = 0
    while True:
        yield {'panel': dataconfig.name, 'seed': seed, 'ordinal': ordinal, 'rate': parameters['mutation_rate']}
        ordinal += 1


def bulky(dataconfig, parameters, seed):
    """``counted`` records padded to 20 kB, so a chunk of them outgrows a pipe's 64 kB buffer."""
    for record in counted(dataconfig, parameters, seed):
        yield {**record, 'padding': 'N' * 20_000}


def finite(dataconfig, parameters, seed):
    """Component seed s yields s % 5 + 3 records, so components end at different positions."""
    for ordinal in range(seed % 5 + 3):
        yield {'seed': seed, 'ordinal': ordinal}


@dataclass(frozen=True)
class Failing:
    """The component of seed ``seed`` fails after 70 records, inside its component context like the GenAIRR provider."""
    seed: int

    def __call__(self, dataconfig, parameters, seed):
        for ordinal in range(70 if seed == self.seed else 10**9):
            yield {'seed': seed, 'ordinal': ordinal}
        with error_context(stage='curriculum_component', seed=seed):
            raise TrainingDataError('simulated provider failure', context={'ordinal': 70})


@dataclass(frozen=True)
class FailingPlainly:
    """A third-party failure of the component of seed ``seed``: a built-in exception that carries no AlignAIR context."""
    seed: int

    def __call__(self, dataconfig, parameters, seed):
        if seed == self.seed:
            raise ValueError(f'unsupported parameters for seed {seed}')
        yield from counted(dataconfig, parameters, seed)


@dataclass(frozen=True)
class Crashing:
    """The process serving the component of seed ``seed`` dies without raising, as after a native crash or an
    out-of-memory kill."""
    seed: int

    def __call__(self, dataconfig, parameters, seed):
        yield {'seed': seed, 'ordinal': 0}
        if seed == self.seed:
            os._exit(7)
        yield from counted(dataconfig, parameters, seed)


def genairr_origin(dataconfig, parameters, seed):
    """Report which GenAIRR package the worker process imported."""
    import GenAIRR
    while True:
        yield {'seed': seed, 'genairr': GenAIRR.__file__, 'pid': os.getpid()}


def idle(payload, inputs, link):
    """A worker task that only receives its payload, then waits to be stopped."""


def ordinal_batch(records):
    """A batch adapter for synthetic records that, like the nucleotide adapter, draws from torch's CPU generator."""
    import torch
    from alignair.data.batching import TrainingBatch
    tokens = torch.tensor([[record['ordinal'] + 1, record['seed'] + 1] for record in records])
    return TrainingBatch({'tokenized_sequence': tokens}, {'orientation': torch.randint(0, 4, (len(records),))})


def strict_batch(records):
    """``ordinal_batch`` that rejects the 50th record of any component."""
    for index, record in enumerate(records):
        with error_context(stage='batch_adaptation', record_index=index):
            if record['ordinal'] == 50:
                raise TrainingDataError('record 50 is malformed')
    return ordinal_batch(records)


def consume_within(records, seconds, count=100_000):
    """Read up to ``count`` records in a daemon thread; return whether it is still blocked, and its outcome."""
    outcome = {}

    def consume():
        try:
            outcome['records'] = sum(1 for _ in zip(range(count), records))
        except BaseException as exc:            # the outcome under test may be any exception
            outcome['error'] = exc
    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    thread.join(seconds)
    return thread.is_alive(), outcome


def shifted(dataconfig, parameters, seed):
    """Real GenAIRR component records, of seed + 1000: other records than the default provider's."""
    from alignair.train.gym.streams import stream_records
    yield from stream_records(dataconfig, parameters, seed + 1000)
