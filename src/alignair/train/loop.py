"""Shared finite step windows over records and a model batch adapter."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice
from operator import index
from typing import NamedTuple

from ..data.batching import BatchAdapter
from ..errors import ConfigurationError, error_context
from .engine import StepResult, TrainingEngine


class TrainingIteration(NamedTuple):
    """One successful update, with its detached result and source batch for runner diagnostics."""
    step: int
    result: StepResult
    records: tuple


def _count(value, field, minimum):
    try:
        if isinstance(value, bool):
            raise ValueError
        value = index(value)
        if value < minimum:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field} must be an integer >= {minimum}",
                                 context={"stage": "training_iteration_preflight", "field": field}) from exc
    return value


def training_iterations(engine: TrainingEngine, records: Iterable, batch_adapter: BatchAdapter, *,
                        stop_step: int, batch_size: int, start_step: int = 0) -> Iterator[TrainingIteration]:
    """Yield successful updates numbered start_step+1 through stop_step, inclusive.

    Only complete batches are trained. A finite source's incomplete tail is consumed and dropped.
    The source is never read beyond the requested step window; the next batch is pulled only when
    the caller advances the iterator. Validation is eager, while source iteration is lazy.

    This function does not resume source/RNG state or save checkpoints. Runners own those policies,
    and can use each result to log, evaluate, checkpoint, or stop before requesting another update.
    """
    start_step = _count(start_step, "start_step", 0)
    stop_step = _count(stop_step, "stop_step", 0)
    batch_size = _count(batch_size, "batch_size", 1)
    if stop_step < start_step:
        raise ConfigurationError("stop_step must be at least start_step",
                                 context={"stage": "training_iteration_preflight",
                                          "start_step": start_step, "stop_step": stop_step})
    if not callable(batch_adapter):
        raise ConfigurationError("batch_adapter must be callable", context={"field": "batch_adapter"})

    def iterate():
        # Delay iter(records), too: an empty window must not initialize a simulator or loader.
        stream = None
        for step in range(start_step + 1, stop_step + 1):
            with error_context(stage="training_iteration", step=step):
                if stream is None:
                    stream = iter(records)
                batch_records = tuple(islice(stream, batch_size))
                if len(batch_records) < batch_size:
                    return
                # Keep adapted tensors out of the suspended iterator during runner callbacks.
                result = engine.step(batch_adapter(batch_records), step=step)
            # Runner callbacks must not inherit the previous iteration's error context.
            yield TrainingIteration(step, result, batch_records)

    return iterate()
