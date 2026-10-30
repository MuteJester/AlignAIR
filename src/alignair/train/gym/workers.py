"""Worker processes for curriculum components: the serial record stream, generated in parallel.

``training_records`` zips independent streams (a curriculum component interleaves several), and each
stream, one provider call, depends only on its reference, its parameters and its seed. Here a
"component" is one such provider call. Each provider can therefore run in its own process while the
consuming process zips the component streams exactly as the serial stream does: the same records in
the same order, and the same errors at the same records.

Topology: ``min(workers, components)`` spawned daemon processes. Worker ``w`` serves components
``w, w + W, ...`` round-robin, one chunk at a time, into one channel per component; the consumer
reads each channel in the order the zip asks for records. This relies on ``training_records``
drawing every component at the same rate: a worker that wrote ahead on a component drawn more
slowly than another it serves would wait on that full channel while the consumer waits on the
other. Processes start at the first record,
ignore SIGINT, and exit only when their group stops them (close, garbage collection or interpreter
exit) or when the process that started them has died.

A channel is a one-way pipe whose two ends each have a single holder: the parent closes its copies
once the processes that use them have started. A reader therefore sees end-of-file as soon as its
writer dies, even halfway through a message, and a writer sees a broken pipe once its reader is
gone; no wait is unbounded. Writers hold back up to ``depth`` messages while a thread writes them.

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
from multiprocessing.connection import wait as wait_for
from operator import index
import os
import pickle
import queue as queues
import signal
import sys
import threading
import time
import traceback

from ...errors import AlignAIRError, ConfigurationError, TrainingError, current_error_context, error_context
from ...errors.base import rescoped
from .streams import CurriculumStreamPolicy, RecordSource, stream_records, training_records

# Measured on the 1.0.0 recipe's mix before selection (7 streams, batch 64): the data side delivers
# 27-28.5 batches/s for any chunk size from 8 to 256 records and any depth from 1 to 8 messages, about
# twice what the GPU update consumes. 64 records keeps the message count low; 4 held-back messages,
# plus the one being written, bound the records in flight at 320 per stream (8,960, about 30 MB, for
# the 28 streams of the selected mix). With selection (recipe v3) the selected streams, each simulated
# serially, bound the data side at about 4 batches/s on 8 cores, whatever the worker count above 14.
CHUNK_SIZE = 64          # records per message
QUEUE_CHUNKS = 4         # messages a worker holds back per component: the memory bound
_POLL_SECONDS = 0.25     # how often a waiting process checks that its peers are alive
_STOP_SECONDS = 5.0      # how long a stopping process may take before it is terminated
_EXIT_HINT = ('A worker exits early when it runs out of memory or crashes natively, or when the training '
              'script has no `if __name__ == "__main__":` guard: workers are spawned and import the main '
              'module. data_workers=0 generates training data in the training process.')


def _count(value, field, minimum=1):
    try:
        if isinstance(value, bool):
            raise ValueError
        value = index(value)
        if value < minimum:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f'{field} must be an integer >= {minimum}', context={'field': field}) from exc
    return value


@dataclass(frozen=True)
class Component:
    """One provider call of ``training_records``: a reference, its parameters and the component seed."""
    reference: object
    parameters: Mapping
    seed: int


def component_plan(dataconfigs, *, policy: CurriculumStreamPolicy, seed: int, start_step: int = 0,
                   purpose: str = 'training') -> tuple[Component, ...]:
    """The provider calls ``training_records`` makes, in order, found by asking it for one record."""
    calls = []

    def record(reference, parameters, component_seed):
        calls.append(Component(reference, dict(parameters), component_seed))
        return repeat({})

    next(training_records(dataconfigs, policy=policy, seed=seed, start_step=start_step, purpose=purpose,
                          record_source=record))
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
    raise (rescoped(error) if isinstance(error, AlignAIRError) else error) from WorkerTraceback(text, context)


class WorkerStopped(Exception):
    """The worker group is stopping; the task unwinds without reporting a failure."""


@dataclass(frozen=True)
class Channel:
    """A one-way pipe between two processes, and how many messages its writer may hold back."""
    receiver: object
    sender: object
    depth: int


class _Outbox:
    """A writer's end of a channel: up to ``depth`` pickled messages wait while a thread writes them."""

    def __init__(self, sender, depth):
        self._sender, self._buffer, self.broken = sender, queues.Queue(depth), False
        threading.Thread(target=self._write, daemon=True).start()

    def _write(self):
        while True:
            data = self._buffer.get()
            try:
                self._sender.send_bytes(data)
            except OSError:                              # the reader is gone
                self.broken = True
                return

    def put(self, data, timeout):
        self._buffer.put(data, timeout=timeout)


class WorkerLink:
    """A worker's stop-aware view of its output channels."""

    def __init__(self, outputs, stop):
        self._outputs = [_Outbox(sender, depth) for sender, depth in outputs]
        self._stop = stop
        self._parent = multiprocessing.parent_process()

    def running(self) -> bool:
        return not self._stop.is_set() and (self._parent is None or self._parent.is_alive())

    def check(self, *, ended=False):
        """Unwind once the group stops; a channel that ended otherwise lost its writer."""
        if not self.running():
            raise WorkerStopped
        if ended:
            raise TrainingError('a training data worker exited unexpectedly',
                                context={'stage': 'training_record_worker'}, hint=_EXIT_HINT)

    def send(self, position: int, message) -> bool:
        """Queue a message, waiting for space while the group runs; False once it stops or the reader is gone."""
        data = pickle.dumps(message, pickle.HIGHEST_PROTOCOL)
        outbox = self._outputs[position]
        while self.running() and not outbox.broken:
            try:
                outbox.put(data, _POLL_SECONDS)
                return True
            except queues.Full:
                pass
        return False

    def broadcast(self, message):
        for position in range(len(self._outputs)):
            if not self.send(position, message):
                return

    def wait(self):
        while self.running():
            self._stop.wait(_POLL_SECONDS)


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
        task(receive(inbox, link.check), inputs, link)
    except WorkerStopped:
        pass
    except Exception as exc:
        link.broadcast(('error', pack_failure(exc)))
    finally:
        link.wait()


def _deliver(sender, data):
    try:
        sender.send_bytes(data)
    except OSError:                                      # the worker died before reading its payload
        pass
    finally:
        sender.close()


def _stop_group(stop, processes, channels):
    stop.set()
    for channel in channels:                             # a writer blocked on a pipe this process reads
        channel.receiver.close()                         # now finds it broken
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


class WorkerGroup:
    """Spawned daemon processes sharing one stop event, stopped together exactly once.

    Spawning keeps CUDA state out of the workers. Closing, collecting the group and interpreter
    exit all stop it: each process is asked to stop, then terminated if it has not exited in time.
    """

    def __init__(self):
        self._context = multiprocessing.get_context('spawn')
        self._stop = self._context.Event()
        self._processes, self._channels = [], []
        self._finalizer = process_util.Finalize(self, _stop_group, args=(self._stop, self._processes, self._channels),
                                                exitpriority=10)

    @property
    def processes(self) -> tuple:
        return tuple(self._processes)

    def channel(self, depth: int) -> Channel:
        """A one-way pipe whose writer may hold back ``depth`` messages."""
        receiver, sender = self._context.Pipe(duplex=False)
        channel = Channel(receiver, sender, _count(depth, 'depth'))
        self._channels.append(channel)
        return channel

    def start(self, task, payload, *, outputs, inputs=(), name):
        """Run ``task(payload, inputs, link)`` in a new process that writes ``outputs`` and reads ``inputs``.

        The payload is pickled here, before any process starts, and a thread delivers it: a large
        start argument would hold ``start()`` until the child had imported the task's modules. The
        new process becomes the only holder of its channel ends.
        """
        try:
            data = pickle.dumps(payload, pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            raise ConfigurationError('a training data task cannot be sent to a worker process',
                                     context={'field': 'payload', 'process': name},
                                     hint='Worker processes are spawned: use module-level providers and '
                                          'picklable references.') from exc
        inbox = self.channel(1)
        process = self._context.Process(target=_run, name=name, daemon=True,
                                        args=(task, inbox.receiver, tuple((c.sender, c.depth) for c in outputs),
                                              tuple(c.receiver for c in inputs), self._stop, _genairr_origin()))
        self._processes.append(process)
        process.start()
        for end in (inbox.receiver, *(c.sender for c in outputs), *(c.receiver for c in inputs)):
            end.close()
        threading.Thread(target=_deliver, args=(inbox.sender, data), daemon=True).start()

    def check(self, *, ended=False):
        """Raise once the group is stopped, or if a process has exited: workers leave only when stopped,
        so any other exit is a crash. After a channel ended, wait briefly for its writer's exit code."""
        if self._stop.is_set():
            raise TrainingError('the training data workers were stopped', context={'stage': 'training_record_worker'})
        if ended:
            wait_for([process.sentinel for process in self._processes if process.pid is not None], _STOP_SECONDS)
        for process in self._processes:
            if process.exitcode is not None:
                raise TrainingError('a training data worker exited unexpectedly',
                                    context={'stage': 'training_record_worker', 'process': process.name,
                                             'exitcode': process.exitcode}, hint=_EXIT_HINT)
        if ended:
            raise TrainingError('a training data channel ended unexpectedly',
                                context={'stage': 'training_record_worker'}, hint=_EXIT_HINT)

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
    try:
        pickle.dumps(provider)
    except Exception as exc:
        raise ConfigurationError('a worker process cannot receive this provider', context={'field': 'provider'},
                                 hint='Worker processes are spawned: pass a module-level function.') from exc
    return {'workers': _count(workers, 'workers'), 'provider': provider,
            'chunk_size': _count(chunk_size, 'chunk_size'), 'queue_chunks': _count(queue_chunks, 'queue_chunks')}


def start_record_workers(group: WorkerGroup, plan, *, provider: RecordSource, workers: int,
                         chunk_size: int = CHUNK_SIZE, queue_chunks: int = QUEUE_CHUNKS) -> tuple:
    """Start ``min(workers, components)`` processes for a plan; return one channel per component."""
    channels = tuple(group.channel(queue_chunks) for _ in plan)
    count = min(workers, len(plan))
    for worker in range(count):
        slots = range(worker, len(plan), count)
        group.start(serve_components, (provider, tuple(plan[slot] for slot in slots), chunk_size),
                    outputs=[channels[slot] for slot in slots], name=f'alignair-records-{worker}')
    return channels


def plan_signature(plan) -> tuple:
    """The seeds and parameters that identify each planned component, without its reference."""
    return tuple((component.seed, component.parameters) for component in plan)


def receive(receiver, check):
    """Wait for the next message on a channel, calling ``check`` while none arrives.

    Every wait is bounded: a writer that dies, even mid-message, ends its pipe, and so does this
    process closing the pipe on stopping; ``check(ended=True)`` then raises.
    """
    while True:
        try:
            if receiver.poll(_POLL_SECONDS):
                return pickle.loads(receiver.recv_bytes())
        except (EOFError, OSError):
            check(ended=True)
            raise
        check()


class _ComponentRecords(Iterator):
    """One component's records, read from its queue as the zip asks for them."""

    def __init__(self, receiver, check):
        self._receiver, self._check = receiver, check
        self._records, self._done = deque(), False

    def __next__(self):
        while not self._records:
            if self._done:
                raise StopIteration
            kind, payload = receive(self._receiver, self._check)
            if kind == 'records':
                self._records.extend(payload)
            else:
                self._done = True
                if kind == 'error':
                    raise_failure(payload)
        return self._records.popleft()


class ComponentChannels:
    """A record source answering ``training_records``' provider calls from planned component channels."""

    def __init__(self, signature, receivers, check):
        self._signature, self._receivers, self._check = signature, receivers, check
        self._position = 0

    def __call__(self, reference, parameters, seed):
        position = self._position
        if position >= len(self._signature) or self._signature[position] != (seed, dict(parameters)):
            raise TrainingError('the record stream requested a component outside the worker plan',
                                context={'stage': 'training_record_worker', 'component': position, 'seed': seed})
        self._position += 1
        return _ComponentRecords(self._receivers[position], self._check)


class RecordWorkers:
    """A record source for ``training_records`` that plans its components and starts their workers at its
    first call. Pass the same references, policy, seed and start step to both; ``close`` stops the workers."""

    def __init__(self, dataconfigs, *, policy: CurriculumStreamPolicy, seed: int, workers: int,
                 provider: RecordSource = stream_records, chunk_size: int = CHUNK_SIZE,
                 queue_chunks: int = QUEUE_CHUNKS, start_step: int = 0):
        self._request = dataconfigs, policy, seed, start_step
        self._options = worker_options(workers=workers, provider=provider, chunk_size=chunk_size,
                                       queue_chunks=queue_chunks)
        self.group = self._channels = None
        self._closed = False

    def __call__(self, reference, parameters, seed):
        if self._closed:
            raise TrainingError('the record workers are closed', context={'stage': 'training_record_worker'})
        if self._channels is None:
            dataconfigs, policy, base_seed, start_step = self._request
            plan = component_plan(dataconfigs, policy=policy, seed=base_seed, start_step=start_step)
            self.group = WorkerGroup()
            channels = start_record_workers(self.group, plan, **self._options)
            self._channels = ComponentChannels(plan_signature(plan), [channel.receiver for channel in channels],
                                               self.group.check)
        return self._channels(reference, parameters, seed)

    def close(self):
        """Stop the workers; a closed source never starts them again. Idempotent."""
        self._closed = True
        if self.group is not None:
            self.group.close()


class ParallelRecords(Iterator):
    """``training_records`` with each component generated in a worker process; close it to stop them."""

    def __init__(self, dataconfigs, *, policy: CurriculumStreamPolicy = CurriculumStreamPolicy(), seed: int = 0,
                 translated_only: bool = False, workers: int, provider: RecordSource = stream_records,
                 chunk_size: int = CHUNK_SIZE, queue_chunks: int = QUEUE_CHUNKS, start_step: int = 0):
        if not isinstance(dataconfigs, (str, bytes)):
            try:
                dataconfigs = tuple(dataconfigs)
            except TypeError:
                pass                                     # training_records reports the request
        self._source = RecordWorkers(dataconfigs, policy=policy, seed=seed, workers=workers, provider=provider,
                                     chunk_size=chunk_size, queue_chunks=queue_chunks, start_step=start_step)
        self._records = training_records(dataconfigs, policy=policy, seed=seed, translated_only=translated_only,
                                         record_source=self._source, start_step=start_step)

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
                     chunk_size: int = CHUNK_SIZE, queue_chunks: int = QUEUE_CHUNKS,
                     start_step: int = 0) -> ParallelRecords:
    """The ``training_records`` stream, with its component providers running in worker processes."""
    return ParallelRecords(dataconfigs, policy=policy, seed=seed, translated_only=translated_only, workers=workers,
                           provider=provider, chunk_size=chunk_size, queue_chunks=queue_chunks, start_step=start_step)
