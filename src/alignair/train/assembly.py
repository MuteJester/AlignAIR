"""Training batches from worker processes, identical to the batches of serial training.

``parallel_training_data`` replaces the serial record stream and batch adapter that ``train`` hands to
``training_iterations``. Curriculum components always run in record workers (``gym.workers``). Where
the batches are built depends on the device:

* on an accelerator, an assembler process zips the component queues, prepares the records and
  adapts them into CPU batches, working a few batches ahead; this process only places each batch on
  the device and steps;
* on the CPU, the zip, preparation and adaptation stay in this process (``LocalAssembly``), because a
  CPU training step draws from the same CPU random generators as the batch adapter.

The batch adapter draws from the process random generators (orientation augmentation). The assembler
starts from this process's generator state and sends the state after each batch; the batch adapter
adopts it, so this process's generators, and its checkpoints, advance exactly as in serial training.
A CPU draw between steps cannot be replayed by an assembler that has already worked ahead, so it is
refused instead of silently changing the batches.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import islice
from operator import index

import numpy as np
import torch

from ..data.training import TrainingDataSetup
from ..errors import ConfigurationError, TrainingError, error_context
from .gym.streams import CurriculumStreamPolicy, RecordSource, stream_records, training_records
from .gym.workers import (CHUNK_SIZE, QUEUE_CHUNKS, ComponentChannels, RecordWorkers, WorkerGroup, component_plan,
                          plan_signature, raise_failure, receive, start_record_workers, worker_options)
from .random_state import capture_rng_state, restore_rng_state

QUEUE_BATCHES = 4        # assembled batches in flight (about 1 MB each at batch 64): the assembler's lead


@dataclass(frozen=True)
class TrainingStream:
    """The trainer's serial training stream as data, so that a worker process can rebuild it exactly: the run
    ``seed``'s training records from ``start_step`` on (a resumed run draws its own seed blocks)."""
    dataconfigs: tuple
    policy: CurriculumStreamPolicy
    seed: int
    data: TrainingDataSetup
    start_step: int = 0

    def records(self, record_source: RecordSource | None = None):
        """The curriculum's records, built as ``train`` builds its training stream."""
        return training_records(self.dataconfigs, policy=self.policy, seed=self.seed, start_step=self.start_step,
                                record_source=record_source)


def _count(value, field):
    try:
        if isinstance(value, bool) or index(value) < 1:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f'{field} must be a positive integer', context={'field': field}) from exc
    return index(value)


def _device(value) -> torch.device:
    try:
        return torch.device(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ConfigurationError('invalid training device', context={'field': 'device'}) from exc


def _same_rng(left, right) -> bool:
    return (left['python'] == right['python'] and torch.equal(left['torch'], right['torch'])
            and left['numpy'][0] == right['numpy'][0] and np.array_equal(left['numpy'][1], right['numpy'][1])
            and left['numpy'][2:] == right['numpy'][2:])


def assemble_batches(payload, inputs, link):
    """The assembler process: rebuild the stream over the component queues; send each batch and the
    CPU random state after it, numbering steps and scoping errors as ``training_iterations`` does."""
    stream, signature, batch_size, rng = payload
    restore_rng_state(rng, include_cuda=False, require_process_state=True)
    records = stream.records(record_source=ComponentChannels(signature, inputs, link.check))
    step = stream.start_step
    while link.running():
        step += 1
        with error_context(stage='training_iteration', step=step):
            batch_records = tuple(islice(records, batch_size))
            if len(batch_records) < batch_size:
                link.send(0, ('end', batch_records))
                return
            batch = stream.data.batch_adapter(batch_records)
        # Channels carry plain pickles, so tensors travel as bytes rather than as torch shared memory,
        # whose per-tensor descriptor handoff would wait on this process's GIL and stall the trainer.
        if not link.send(0, ('batch', (batch_records, batch, capture_rng_state(include_cuda=False)))):
            return


class AssembledBatches:
    """Batches built by an assembler process; this process places them on the device.

    ``records`` and ``batch_adapter`` stand in for the serial stream and adapter: the records arrive a
    batch at a time, and the adapter returns the batch assembled for exactly those records.
    """

    def __init__(self, stream: TrainingStream, *, batch_size: int, workers: int, device,
                 provider: RecordSource = stream_records, chunk_size: int = CHUNK_SIZE,
                 queue_chunks: int = QUEUE_CHUNKS, queue_batches: int = QUEUE_BATCHES):
        if not isinstance(stream, TrainingStream):
            raise ConfigurationError('batch assembly requires a TrainingStream', context={'field': 'stream'})
        self._options = worker_options(workers=workers, provider=provider, chunk_size=chunk_size,
                                       queue_chunks=queue_chunks)
        self._batch_size = _count(batch_size, 'batch_size')
        self._queue_batches = _count(queue_batches, 'queue_batches')
        self._stream, self._device = stream, _device(device)
        stream.records()                               # rejects a bad request now; nothing runs until iterated
        self._group = self._batches = self._pending = self._expected = None
        self._records, self._ended = deque(), False

    @property
    def records(self):
        return self

    @property
    def batch_adapter(self):
        return self._adapt

    def __iter__(self):
        return self

    def __next__(self):
        while not self._records:
            if self._ended:
                raise StopIteration
            self._receive()
        return self._records.popleft()

    def _start(self):
        stream = self._stream
        plan = component_plan(stream.dataconfigs, policy=stream.policy, seed=stream.seed, start_step=stream.start_step)
        self._group = WorkerGroup()
        channels = start_record_workers(self._group, plan, **self._options)
        self._batches = self._group.channel(self._queue_batches)
        self._expected = capture_rng_state(include_cuda=False)
        self._group.start(assemble_batches, (stream, plan_signature(plan), self._batch_size, self._expected),
                          outputs=[self._batches], inputs=channels, name='alignair-batches')

    def _receive(self):
        if self._group is None:
            self._start()
        kind, payload = receive(self._batches.receiver, self._group.check)
        if kind == 'batch':
            self._pending = payload
            self._records.extend(payload[0])
        else:
            self._ended = True
            if kind == 'error':
                raise_failure(payload)
            self._records.extend(payload)

    def _adapt(self, records):
        pending, self._pending = self._pending, None
        if pending is None or len(records) != len(pending[0]) or any(a is not b for a, b in zip(records, pending[0])):
            raise TrainingError('the batch adapter received records other than those it assembled',
                                context={'stage': 'training_batch_assembly'})
        _, batch, rng = pending
        if not _same_rng(capture_rng_state(include_cuda=False), self._expected):
            raise TrainingError('a CPU random draw between training steps cannot be replayed by batches '
                                'assembled ahead', context={'stage': 'training_batch_assembly'},
                                hint='Keep CPU random draws out of the training step and its callbacks, '
                                     'or train with data_workers=0.')
        restore_rng_state(rng, include_cuda=False)
        self._expected = rng
        if self._device.type == 'cuda':
            return batch.pin_memory().to(self._device, non_blocking=True)
        return batch.to(self._device)

    def close(self):
        """Stop the assembler and every record worker; the stream then ends. Idempotent."""
        self._ended, self._pending = True, None
        self._records.clear()
        if self._group is not None:
            self._group.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class LocalAssembly:
    """Records from worker processes, prepared and adapted in this process by the stream's own adapter."""

    def __init__(self, stream: TrainingStream, *, workers: int, device, provider: RecordSource = stream_records,
                 chunk_size: int = CHUNK_SIZE, queue_chunks: int = QUEUE_CHUNKS):
        if not isinstance(stream, TrainingStream):
            raise ConfigurationError('batch assembly requires a TrainingStream', context={'field': 'stream'})
        self._workers = RecordWorkers(stream.dataconfigs, policy=stream.policy, seed=stream.seed, workers=workers,
                                      provider=provider, chunk_size=chunk_size, queue_chunks=queue_chunks,
                                      start_step=stream.start_step)
        self._adapter, self._device = stream.data.batch_adapter, _device(device)
        self.records = stream.records(record_source=self._workers)

    @property
    def batch_adapter(self):
        return self._adapt

    def _adapt(self, records):
        return self._adapter(records).to(self._device)

    def close(self):
        """Stop every record worker. Idempotent."""
        self._workers.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def parallel_training_data(stream: TrainingStream, *, batch_size: int, workers: int, device,
                           assemble_in_worker: bool | None = None, provider: RecordSource = stream_records,
                           chunk_size: int = CHUNK_SIZE, queue_chunks: int = QUEUE_CHUNKS,
                           queue_batches: int = QUEUE_BATCHES) -> AssembledBatches | LocalAssembly:
    """Records and a batch adapter for ``training_iterations``, from ``workers`` record processes.

    ``stream.data`` must adapt on the CPU. Batches are assembled in a worker process unless training
    runs on the CPU; ``assemble_in_worker`` overrides that choice. Nothing starts before the first
    record is requested; close the result (or use it as a context manager) to stop every process.
    """
    device = _device(device)
    if assemble_in_worker is None:
        assemble_in_worker = device.type != 'cpu'
    elif type(assemble_in_worker) is not bool:
        raise ConfigurationError('assemble_in_worker must be a boolean or None', context={'field': 'assemble_in_worker'})
    options = {'workers': workers, 'device': device, 'provider': provider, 'chunk_size': chunk_size,
               'queue_chunks': queue_chunks}
    if assemble_in_worker:
        return AssembledBatches(stream, batch_size=batch_size, queue_batches=queue_batches, **options)
    return LocalAssembly(stream, **options)
