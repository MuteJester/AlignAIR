"""Worker processes for curriculum components: the serial record stream, generated in parallel.

``training_records`` zips independent streams (a curriculum component interleaves several), and each
stream, one provider call, depends only on its reference, its parameters and its seed. Here a
"component" is one such provider call. Each provider can therefore run in its own process while the
consuming process zips the component streams exactly as the serial stream does: the same records in
the same order, and the same errors at the same records.

Topology: ``min(workers, components)`` spawned daemon processes. Each worker serves the components
``worker_slots`` deals it (the slow selected ones spread evenly) round-robin, one chunk at a time,
into one channel per component; the consumer
reads each channel in the order the zip asks for records. This relies on ``training_records``
drawing every component at the same rate: a worker that wrote ahead on a component drawn more
slowly than another it serves would wait on that full channel while the consumer waits on the
other. Processes start at the first record,
ignore SIGINT, and exit only when their group stops them (close, garbage collection or interpreter
exit) or when the process that started them has died.

The processes and their channels are ``train.processes``'s.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from itertools import islice, repeat
import pickle

from ...errors import ConfigurationError, TrainingError, error_context
from ..checks import integer
from .streams import CurriculumStreamPolicy, RecordSource, stream_records, training_records
from ..processes import WorkerGroup, WorkerLink, pack_failure, raise_failure, receive


# Measured on a 7-stream mix without selection (batch 64): the data side delivers
# 27-28.5 batches/s for any chunk size from 8 to 256 records and any depth from 1 to 8 messages, about
# twice what the GPU update consumes. 64 records keeps the message count low; 4 held-back messages,
# plus the one being written, bound the records in flight at 320 per stream (8,960, about 30 MB, for
# the 28 streams of the selected mix). With selection, the 12 selected streams, each simulated
# serially, bound the data side: about 5.8 batches/s with 7 workers on 8 cores, and 14 workers are no faster.
CHUNK_SIZE = 64          # records per message
QUEUE_CHUNKS = 4         # messages a worker holds back per component: the memory bound


@dataclass(frozen=True)
class Component:
    """One provider call of ``training_records``: a reference, its parameters and the component seed."""
    reference: object
    parameters: Mapping
    seed: int


def component_plan(dataconfigs, *, policy: CurriculumStreamPolicy, seed: int, start_step: int = 0,
                   purpose: str = 'training') -> tuple[Component, ...]:
    """The provider calls ``training_records`` makes, in order, found by asking it for one record. Flanks are drawn
    after the provider and change none of its calls, so the plan is found without them."""
    calls = []

    def record(reference, parameters, component_seed):
        calls.append(Component(reference, dict(parameters), component_seed))
        return repeat({})

    next(training_records(dataconfigs, policy=replace(policy, flanks=None), seed=seed, start_step=start_step,
                          purpose=purpose, record_source=record))
    return tuple(calls)


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
    return {'workers': integer(workers, 'workers'), 'provider': provider,
            'chunk_size': integer(chunk_size, 'chunk_size'), 'queue_chunks': integer(queue_chunks, 'queue_chunks')}


def worker_slots(plan, workers: int) -> tuple[tuple[int, ...], ...]:
    """The plan positions each of ``min(workers, components)`` workers serves, each in plan order.

    Every component is drawn at the same rate, so a worker keeps pace only if the records of all its components
    together are not much slower to simulate than another worker's. A selected component (``productive_only``)
    is 10 to 25 times slower per record than an unselected one, and slower the more it mutates. The components are
    therefore dealt slowest first (selected, then by mutation rate, then in plan order) in a snake over the workers:
    0 to W-1, then W-1 to 0, and so on. Each worker gets as many selected components as any other, within one, and
    the W most mutated selected components (the heavy-SHM ones first) go to different workers.
    """
    count = min(workers, len(plan))

    def slowest_first(position):
        parameters = plan[position].parameters
        return (not parameters.get('productive_only', False), -float(parameters.get('mutation_rate') or 0.),
                position)

    served = [[] for _ in range(count)]
    for rank, position in enumerate(sorted(range(len(plan)), key=slowest_first)):
        lap, offset = divmod(rank, count)
        served[offset if lap % 2 == 0 else count - 1 - offset].append(position)
    return tuple(tuple(sorted(slots)) for slots in served)


def start_record_workers(group: WorkerGroup, plan, *, provider: RecordSource, workers: int,
                         chunk_size: int = CHUNK_SIZE, queue_chunks: int = QUEUE_CHUNKS) -> tuple:
    """Start ``min(workers, components)`` processes for a plan, serving the components ``worker_slots`` deals
    them; return one channel per component."""
    channels = tuple(group.channel(queue_chunks) for _ in plan)
    for worker, slots in enumerate(worker_slots(plan, workers)):
        group.start(serve_components, (provider, tuple(plan[slot] for slot in slots), chunk_size),
                    outputs=[channels[slot] for slot in slots], name=f'alignair-records-{worker}')
    return channels


def plan_signature(plan) -> tuple:
    """The seeds and parameters that identify each planned component, without its reference."""
    return tuple((component.seed, component.parameters) for component in plan)


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
