"""Worker processes for curriculum components: the serial record stream, generated in parallel.

``training_records`` zips independent component streams, and component ``i`` of a reference panel
depends only on that panel's reference, its parameters and its seed. Each provider can therefore
run in its own process while the consuming process zips the component streams exactly as the
serial stream does: the same records in the same order, and the same errors at the same records.

Topology: ``min(workers, components)`` spawned daemon processes. Worker ``w`` serves components
``w, w + W, ...`` round-robin, one chunk at a time, into one bounded queue per component; the
consumer reads each queue in the order the zip asks for records. Processes start at the first
record, ignore SIGINT, and exit only when their group stops them (close, garbage collection or
interpreter exit) or when the process that started them has died.

A failure inside a worker travels as data: the exception keeps its type, message and AlignAIR
context, and its cause is a ``WorkerTraceback`` holding the worker's traceback and scoped context.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import islice, repeat
import multiprocessing
from multiprocessing import util as process_util
import os
import pickle
import queue as queues
import signal
import sys
import time
import traceback

from ...errors import AlignAIRError, ConfigurationError, TrainingError, current_error_context, error_context
from ...errors.context import snapshot
from .streams import CurriculumStreamPolicy, RecordSource, _integer, stream_records, training_records

# Measured on the 1.0.0 recipe (7 components, batch 64): the data side delivers 27-28.5 batches/s for
# any chunk size from 8 to 256 records and any depth from 1 to 8 messages, about twice what the GPU
# update consumes. 64 records keeps the message count low; 4 messages bound the records in flight
# at 1,792 (about 6 MB) for the 7 components.
CHUNK_SIZE = 64          # records per message
QUEUE_CHUNKS = 4         # messages in flight per component: the memory bound
_POLL_SECONDS = 0.25     # how often a waiting process checks that its peers are alive
_STOP_SECONDS = 5.0      # how long a stopping process may take before it is terminated


@dataclass(frozen=True)
class Component:
    """One provider call of ``training_records``: a reference, its parameters and the component seed."""
    reference: object
    parameters: Mapping
    seed: int


def component_plan(dataconfigs, *, policy: CurriculumStreamPolicy, seed: int) -> tuple[Component, ...]:
    """The provider calls ``training_records`` makes, in order, found by asking it for one record."""
    calls = []

    def record(reference, parameters, component_seed):
        calls.append(Component(reference, dict(parameters), component_seed))
        return repeat({})

    next(training_records(dataconfigs, policy=policy, seed=seed, record_source=record))
    return tuple(calls)


class WorkerTraceback(Exception):
    """A data worker's traceback and the error context in which its failure was raised."""

    def __init__(self, text: str, context: dict):
        super().__init__(text)
        self.context = context


def pack_failure(error: BaseException) -> bytes:
    """Serialize a worker failure; an exception that cannot cross processes becomes a TrainingError."""
    text = ''.join(traceback.format_exception(error))
    try:
        payload = pickle.dumps(error)
        pickle.loads(payload)
    except Exception:
        payload = pickle.dumps(TrainingError(f'{type(error).__name__}: {error}',
                                             context={'exception_type': type(error).__name__}))
    return pickle.dumps((payload, text, current_error_context()))


def raise_failure(failure: bytes):
    """Raise a worker failure here, caused by the worker's traceback.

    An AlignAIR error also gains this process's scoped context beneath its own, as if it had been
    raised through these scopes: the serial stream reports the same context for the same failure.
    """
    payload, text, context = pickle.loads(failure)
    error = pickle.loads(payload)
    if isinstance(error, AlignAIRError):
        error._context = snapshot({**current_error_context(), **error._context})
    raise error from WorkerTraceback(text, context)


class WorkerStopped(Exception):
    """The worker group is stopping; the task unwinds without reporting a failure."""


class WorkerLink:
    """A worker's stop-aware view of its output queues."""

    def __init__(self, outputs, stop):
        self._outputs, self._stop = outputs, stop
        self._parent = multiprocessing.parent_process()

    def running(self) -> bool:
        return not self._stop.is_set() and (self._parent is None or self._parent.is_alive())

    def check(self):
        if not self.running():
            raise WorkerStopped

    def send(self, index: int, message) -> bool:
        """Put a message, waiting for space while the group runs; False once it stops."""
        while self.running():
            try:
                self._outputs[index].put(message, timeout=_POLL_SECONDS)
                return True
            except queues.Full:
                pass
        return False

    def broadcast(self, message):
        for index in range(len(self._outputs)):
            if not self.send(index, message):
                return

    def wait(self):
        while self.running():
            self._stop.wait(_POLL_SECONDS)

    def release(self):
        # The consumer has stopped reading: exiting must not wait to flush undelivered messages.
        for output in self._outputs:
            output.cancel_join_thread()


def _genairr_origin():
    return getattr(sys.modules.get('GenAIRR'), '__file__', None)


def _check_genairr(parent):
    if parent is None:
        return
    import GenAIRR
    if os.path.realpath(GenAIRR.__file__) != os.path.realpath(parent):
        raise ConfigurationError('a data worker imported a different GenAIRR than its parent process',
                                 context={'stage': 'training_record_worker', 'parent': parent,
                                          'worker': GenAIRR.__file__},
                                 hint='Put the intended GenAIRR first on PYTHONPATH before starting training.')


def _run(task, inbox, outputs, inputs, stop, genairr):
    """The body of every worker process: run one task, report its failure, then wait to be stopped."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # the parent handles interrupts and stops its workers
    import torch
    torch.set_num_threads(1)
    link = WorkerLink(outputs, stop)
    try:
        _check_genairr(genairr)
        task(pickle.loads(receive(inbox, link.check)), inputs, link)
    except WorkerStopped:
        pass
    except Exception as exc:
        link.broadcast(('error', pack_failure(exc)))
    finally:
        link.wait()
        link.release()


def _stop_group(stop, processes, owned_queues):
    stop.set()
    deadline = time.monotonic() + _STOP_SECONDS
    for process in processes:
        if process.pid is not None:
            process.join(max(0., deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(_STOP_SECONDS)
        if process.is_alive():
            process.kill()
            process.join()
    for owned in owned_queues:
        owned.cancel_join_thread()               # an undelivered payload must not hold up exit
        owned.close()


class WorkerGroup:
    """Spawned daemon processes sharing one stop event, stopped together exactly once.

    Spawning keeps CUDA state out of the workers. Closing, collecting the group and interpreter
    exit all stop it: each process is asked to stop, then terminated if it has not exited in time.
    """

    def __init__(self):
        self._context = multiprocessing.get_context('spawn')
        self._stop = self._context.Event()
        self._processes, self._queues = [], []
        self._finalizer = process_util.Finalize(self, _stop_group, args=(self._stop, self._processes, self._queues),
                                                exitpriority=10)

    @property
    def processes(self) -> tuple:
        return tuple(self._processes)

    def queue(self, maxsize: int):
        owned = self._context.Queue(maxsize)
        self._queues.append(owned)
        return owned

    def start(self, task, payload, *, outputs, inputs=(), name):
        """Run ``task(payload, inputs, link)`` in a new process; the payload is unpickled there.

        The payload travels through the process's inbox queue, not its start arguments: a large
        start argument would hold ``start()`` until the child had imported the task's modules.
        """
        inbox = self.queue(1)
        process = self._context.Process(target=_run, name=name, daemon=True,
                                        args=(task, inbox, tuple(outputs), tuple(inputs), self._stop,
                                              _genairr_origin()))
        self._processes.append(process)
        process.start()
        inbox.put(pickle.dumps(payload))

    def check(self):
        """Raise if a process has exited: workers leave only when stopped, so an exit is a crash."""
        for process in self._processes:
            if process.exitcode is not None:
                raise TrainingError('a training data worker exited unexpectedly',
                                    context={'stage': 'training_record_worker', 'process': process.name,
                                             'exitcode': process.exitcode},
                                    hint='Check the worker for memory exhaustion or a native crash; '
                                         'data_workers=0 generates records in this process.')

    def close(self):
        self._finalizer()


def serve_components(payload, inputs, link: WorkerLink):
    """Generate this worker's components round-robin, a chunk at a time, into one queue each."""
    provider, components, chunk_size = payload
    streams = [None] * len(components)
    active = list(range(len(components)))
    while active and link.running():
        for slot in tuple(active):
            component, chunk, ending = components[slot], [], None
            with error_context(stage='curriculum_component', seed=component.seed):
                try:
                    if streams[slot] is None:
                        streams[slot] = iter(provider(component.reference, component.parameters, component.seed))
                    for record in islice(streams[slot], chunk_size):
                        chunk.append(record)
                    if len(chunk) < chunk_size:
                        ending = ('end', None)
                except Exception as exc:
                    ending = ('error', pack_failure(exc))
            if chunk and not link.send(slot, ('records', chunk)):
                return
            if ending is not None:
                active.remove(slot)
                if not link.send(slot, ending):
                    return


def worker_options(*, workers, provider, chunk_size, queue_chunks) -> dict:
    """Validate a record-worker request before any process starts."""
    if not callable(provider):
        raise ConfigurationError('provider must be a callable record source', context={'field': 'provider'})
    return {'workers': _integer(workers, 'workers', 1), 'provider': provider,
            'chunk_size': _integer(chunk_size, 'chunk_size', 1), 'queue_chunks': _integer(queue_chunks, 'queue_chunks', 1)}


def start_record_workers(group: WorkerGroup, plan, *, provider: RecordSource, workers: int,
                         chunk_size: int = CHUNK_SIZE, queue_chunks: int = QUEUE_CHUNKS) -> tuple:
    """Start ``min(workers, components)`` processes for a plan; return one queue per component."""
    channels = tuple(group.queue(queue_chunks) for _ in plan)
    count = min(workers, len(plan))
    for worker in range(count):
        slots = range(worker, len(plan), count)
        group.start(serve_components, (provider, tuple(plan[slot] for slot in slots), chunk_size),
                    outputs=[channels[slot] for slot in slots], name=f'alignair-records-{worker}')
    return channels


def plan_signature(plan) -> tuple:
    """The seeds and parameters that identify each planned component, without its reference."""
    return tuple((component.seed, component.parameters) for component in plan)


def receive(channel, check):
    """Wait for the next message, calling ``check`` while none arrives."""
    while True:
        try:
            return channel.get(timeout=_POLL_SECONDS)
        except queues.Empty:
            check()


class _ComponentRecords(Iterator):
    """One component's records, read from its queue as the zip asks for them."""

    def __init__(self, channel, check):
        self._channel, self._check = channel, check
        self._records, self._done = deque(), False

    def __next__(self):
        while not self._records:
            if self._done:
                raise StopIteration
            kind, payload = receive(self._channel, self._check)
            if kind == 'records':
                self._records.extend(payload)
            else:
                self._done = True
                if kind == 'error':
                    raise_failure(payload)
        return self._records.popleft()


class ComponentChannels:
    """A record source answering ``training_records``' provider calls from planned component queues."""

    def __init__(self, signature, channels, check):
        self._signature, self._channels, self._check = signature, channels, check
        self._position = 0

    def __call__(self, reference, parameters, seed):
        position = self._position
        if position >= len(self._signature) or self._signature[position] != (seed, dict(parameters)):
            raise TrainingError('the record stream requested a component outside the worker plan',
                                context={'stage': 'training_record_worker', 'component': position, 'seed': seed})
        self._position += 1
        return _ComponentRecords(self._channels[position], self._check)


class RecordWorkers:
    """A record source for ``training_records`` that plans its components and starts their workers at its
    first call. Pass the same references, policy and seed to both; ``close`` stops the workers."""

    def __init__(self, dataconfigs, *, policy: CurriculumStreamPolicy, seed: int, workers: int,
                 provider: RecordSource = stream_records, chunk_size: int = CHUNK_SIZE,
                 queue_chunks: int = QUEUE_CHUNKS):
        self._request = dataconfigs, policy, seed
        self._options = worker_options(workers=workers, provider=provider, chunk_size=chunk_size,
                                       queue_chunks=queue_chunks)
        self.group = self._channels = None

    def __call__(self, reference, parameters, seed):
        if self._channels is None:
            dataconfigs, policy, base_seed = self._request
            plan = component_plan(dataconfigs, policy=policy, seed=base_seed)
            self.group = WorkerGroup()
            channels = start_record_workers(self.group, plan, **self._options)
            self._channels = ComponentChannels(plan_signature(plan), channels, self.group.check)
        return self._channels(reference, parameters, seed)

    def close(self):
        if self.group is not None:
            self.group.close()


class ParallelRecords(Iterator):
    """``training_records`` with each component generated in a worker process; close it to stop them."""

    def __init__(self, dataconfigs, *, policy: CurriculumStreamPolicy = CurriculumStreamPolicy(), seed: int = 0,
                 translated_only: bool = False, workers: int, provider: RecordSource = stream_records,
                 chunk_size: int = CHUNK_SIZE, queue_chunks: int = QUEUE_CHUNKS):
        if not isinstance(dataconfigs, (str, bytes)):
            try:
                dataconfigs = tuple(dataconfigs)
            except TypeError:
                pass                                     # training_records reports the request
        self._source = RecordWorkers(dataconfigs, policy=policy, seed=seed, workers=workers, provider=provider,
                                     chunk_size=chunk_size, queue_chunks=queue_chunks)
        self._records = training_records(dataconfigs, policy=policy, seed=seed, translated_only=translated_only,
                                         record_source=self._source)

    def __next__(self):
        if self._records is None:
            raise StopIteration
        return next(self._records)

    def close(self):
        """Stop every worker; the stream then ends. Idempotent."""
        self._records = None
        self._source.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def parallel_records(dataconfigs, *, policy: CurriculumStreamPolicy = CurriculumStreamPolicy(), seed: int = 0,
                     translated_only: bool = False, workers: int, provider: RecordSource = stream_records,
                     chunk_size: int = CHUNK_SIZE, queue_chunks: int = QUEUE_CHUNKS) -> ParallelRecords:
    """The ``training_records`` stream, with its component providers running in worker processes."""
    return ParallelRecords(dataconfigs, policy=policy, seed=seed, translated_only=translated_only, workers=workers,
                           provider=provider, chunk_size=chunk_size, queue_chunks=queue_chunks)
