"""Worker processes and the channels between them, for training's data side.

A channel is a one-way pipe whose two ends each have a single holder: the parent closes its copies
once the processes that use them have started. A reader therefore sees end-of-file as soon as its
writer dies, even halfway through a message, and a writer sees a broken pipe once its reader is
gone; no wait is unbounded. Writers hold back up to ``depth`` messages while a thread writes them.

A failure inside a worker travels as data: the exception keeps its type, message and AlignAIR
context, and its cause is a ``WorkerTraceback`` holding the worker's traceback and scoped context.
"""
from __future__ import annotations

import multiprocessing
import os
import pickle
import queue as queues
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from multiprocessing import util as process_util
from multiprocessing.connection import wait as wait_for

from ..errors import AlignAIRError, ConfigurationError, TrainingError, current_error_context
from ..errors.base import rescoped
from .checks import integer

_POLL_SECONDS = 0.25     # how often a waiting process checks that its peers are alive
_STOP_SECONDS = 5.0      # how long a stopping process may take before it is terminated
_REAP_SECONDS = 1.0      # join timeout for a process whose sentinel is ready: it bounds no wait (see check)
_EXIT_HINT = ('A worker exits early when it runs out of memory or crashes natively, or when the training '
              'script has no `if __name__ == "__main__":` guard: workers are spawned and import the main '
              'module. data_workers=0 generates training data in the training process.')


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
        channel = Channel(receiver, sender, integer(depth, 'depth'))
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
        so any other exit is a crash. After a channel ended, wait briefly for its writer's exit code.

        A process's sentinel is ready once the process has closed its files, which can be before it can be
        reaped; its exit code is then still unknown. Each process whose sentinel is ready is therefore joined
        before the exit codes are read. The join's timeout (``_REAP_SECONDS``) bounds only its wait for the
        sentinel, which is already ready, so the join then blocks in ``waitpid`` until the process is reaped:
        not long for a process that has closed its files, but not bounded here either. The check waits at most
        ``_STOP_SECONDS`` for any sentinel to be ready; a process whose sentinel is not ready then is not
        joined."""
        if self._stop.is_set():
            raise TrainingError('the training data workers were stopped', context={'stage': 'training_record_worker'})
        if ended:
            started = [process for process in self._processes if process.pid is not None]
            ready = set(wait_for([process.sentinel for process in started], _STOP_SECONDS))
            for process in started:
                if process.sentinel in ready:
                    process.join(_REAP_SECONDS)
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
