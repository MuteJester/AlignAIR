"""Which data worker serves which stream: the slow selected streams spread evenly over the workers, the heavy-SHM
selected streams on different workers."""
from types import SimpleNamespace

import pytest

from alignair.train.records.streams import CurriculumStreamPolicy
from alignair.train.records.workers import component_plan, worker_slots

pytestmark = pytest.mark.component

RECIPE = CurriculumStreamPolicy((0.3, 0.6, 0.9), 0.25, 1)      # the 1.0.0 mix: 7 components of 4 streams


def plan():
    return component_plan((SimpleNamespace(name='IGH', metadata=SimpleNamespace(chain_type='BCR_HEAVY')),),
                          policy=RECIPE, seed=0)


def selected(component):
    return component.parameters.get('productive_only', False)


@pytest.mark.parametrize('workers', [3, 7, 14])
def test_selected_streams_spread_evenly_and_heavy_shm_selected_streams_never_share_a_worker(workers):
    streams = plan()
    slots = worker_slots(streams, workers)
    assert len(slots) == workers and sorted(p for served in slots for p in served) == list(range(len(streams)))
    assert all(list(served) == sorted(served) for served in slots)          # each worker in plan order
    counts = [sum(selected(streams[p]) for p in served) for served in slots]
    assert sum(counts) == 12 and max(counts) - min(counts) <= 1
    sizes = [len(served) for served in slots]
    assert max(sizes) - min(sizes) <= 1
    heavy = [worker for worker, served in enumerate(slots) for p in served
             if selected(streams[p]) and streams[p].parameters['mutation_rate'] == RECIPE.heavy_shm]
    assert len(heavy) == len(set(heavy)) == 3
    assert worker_slots(streams, workers) == slots                         # deterministic


def test_one_worker_serves_every_stream_and_extra_workers_are_not_started():
    streams = plan()
    assert worker_slots(streams, 1) == (tuple(range(28)),)
    assert sorted(worker_slots(streams, 40)) == [(p,) for p in range(28)]
