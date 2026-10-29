"""Spawn-safe record providers for worker-process tests: a spawned worker imports them by module name."""
import os

from alignair.errors import TrainingDataError, error_context


def counted(dataconfig, parameters, seed):
    """An endless, deterministic component stream identified by panel, seed and position."""
    ordinal = 0
    while True:
        yield {'panel': dataconfig.name, 'seed': seed, 'ordinal': ordinal, 'rate': parameters['mutation_rate']}
        ordinal += 1


def finite(dataconfig, parameters, seed):
    """Component seed s yields s % 5 + 3 records, so components end at different positions."""
    for ordinal in range(seed % 5 + 3):
        yield {'seed': seed, 'ordinal': ordinal}


def failing(dataconfig, parameters, seed):
    """Component seed 3 fails after 70 records, inside its component context like the GenAIRR provider."""
    for ordinal in range(70 if seed == 3 else 10**9):
        yield {'seed': seed, 'ordinal': ordinal}
    with error_context(stage='curriculum_component', seed=seed):
        raise TrainingDataError('simulated provider failure', context={'ordinal': 70})


def failing_plainly(dataconfig, parameters, seed):
    """A third-party failure: a built-in exception that carries no AlignAIR context."""
    if seed == 1:
        raise ValueError(f'unsupported parameters for seed {seed}')
    yield from counted(dataconfig, parameters, seed)


def crashing(dataconfig, parameters, seed):
    """The process dies without raising, as after a native crash or an out-of-memory kill."""
    yield {'seed': seed, 'ordinal': 0}
    if seed == 2:
        os._exit(7)
    yield from counted(dataconfig, parameters, seed)


def genairr_origin(dataconfig, parameters, seed):
    """Report which GenAIRR package the worker process imported."""
    import GenAIRR
    while True:
        yield {'seed': seed, 'genairr': GenAIRR.__file__, 'pid': os.getpid()}


def idle(payload, inputs, link):
    """A worker task that only receives its payload, then waits to be stopped."""
