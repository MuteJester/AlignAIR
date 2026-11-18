"""Worker processes reproduce the serial curriculum record stream and never outlive it."""
import gc
from itertools import islice
import multiprocessing
import os
import signal
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

from alignair.errors import ConfigurationError, TrainingDataError, TrainingError
from alignair.train.records.seeds import seed_block
from alignair.train.records.streams import CurriculumStreamPolicy, training_records
from alignair.train.processes import WorkerGroup
from alignair.train.records.workers import RecordWorkers, component_plan
from tests.support.record_workers import parallel_records
from tests.support import record_workers

pytestmark = pytest.mark.integration

RECIPE = CurriculumStreamPolicy((0.3, 0.6, 0.9), 0.25, 1)      # the 1.0.0 recipe: 7 components of 4 streams


def reference(name='IGH', chain_type='BCR_HEAVY'):
    return SimpleNamespace(name=name, metadata=SimpleNamespace(chain_type=chain_type))


def children():
    return set(multiprocessing.active_children())


def stopped(processes):
    for process in processes:
        assert not process.is_alive() and process.exitcode is not None
        with pytest.raises(ProcessLookupError):
            os.kill(process.pid, 0)
    return True


def test_component_plan_lists_every_provider_call_in_order():
    panels = (reference(), reference('TRA', 'TCR_ALPHA'))
    policy = CurriculumStreamPolicy((.3,), 0., selected_fraction=0)
    plan = component_plan(panels, policy=policy, seed=17)
    assert [component.seed for component in plan] == [seed_block(17, 'training', 0, panel, component, 'unselected-0')
                                                      for panel in range(2) for component in range(4)]
    resumed = component_plan(panels, policy=policy, seed=17, start_step=40)
    assert [component.seed for component in resumed] == [
        seed_block(17, 'training', 40, panel, component, 'unselected-0') for panel in range(2) for component in range(4)]
    assert [component.reference.name for component in plan] == ['IGH'] * 4 + ['TRA'] * 4
    assert plan[4].parameters['mutation_rate'] == 0.
    recipe = component_plan((reference(),), policy=RECIPE, seed=0)
    assert len(recipe) == 28 and [component.parameters.get('productive_only', False) for component in recipe[:4]] == [
        True, True, True, False]


def test_fewer_workers_than_components_keep_the_serial_order_across_panels():
    panels = (reference(), reference('TRA', 'TCR_ALPHA'))
    serial = list(islice(training_records(panels, policy=RECIPE, seed=5, record_source=record_workers.counted), 600))
    before = children()
    with parallel_records(panels, policy=RECIPE, seed=5, workers=3, provider=record_workers.counted) as records:
        assert children() == before                    # nothing starts before the first record
        assert list(islice(records, 600)) == serial
        started = children() - before
        assert len(started) == 3
    assert stopped(started) and children() == before


def test_finite_components_end_where_the_serial_stream_ends():
    panels = (reference(),)
    serial = list(training_records(panels, policy=RECIPE, seed=0, record_source=record_workers.finite))
    with parallel_records(panels, policy=RECIPE, seed=0, workers=2, provider=record_workers.finite,
                          chunk_size=2) as records:
        assert list(records) == serial


def test_worker_failure_reaches_the_consumer_at_the_serial_position_with_type_and_context():
    panels = (reference(),)
    def consume(records):
        seen = []
        with pytest.raises(TrainingDataError) as caught:
            for record in records:
                seen.append(record)
        return seen, caught.value
    plan = component_plan(panels, policy=RECIPE, seed=0)
    failing = record_workers.Failing(plan[3].seed)            # the unselected stream of the first component
    serial, expected = consume(training_records(panels, policy=RECIPE, seed=0, record_source=failing))
    with parallel_records(panels, policy=RECIPE, seed=0, workers=2, provider=failing, chunk_size=16) as records:
        seen, error = consume(records)
    assert seen == serial and len(seen) == len(plan) * 70     # 70 complete periods of every component
    assert type(error) is type(expected) and str(error) == str(expected) == 'simulated provider failure'
    assert error.context['stage'] == 'curriculum_component' and error.context['seed'] == failing.seed
    assert error.context['ordinal'] == 70
    assert 'simulated provider failure' in str(error.__cause__)          # the worker's traceback
    assert error.__cause__.context == {'stage': 'curriculum_component', 'seed': failing.seed}


def test_a_third_party_exception_keeps_its_type_and_message():
    failing = record_workers.FailingPlainly(component_plan((reference(),), policy=RECIPE, seed=0)[1].seed)
    with parallel_records((reference(),), policy=RECIPE, seed=0, workers=2, provider=failing) as records:
        with pytest.raises(ValueError, match=f'unsupported parameters for seed {failing.seed}') as caught:
            list(islice(records, 50))
    assert caught.type is ValueError
    assert caught.value.__cause__.context == {'stage': 'curriculum_component', 'seed': failing.seed}


def test_a_crashed_worker_is_reported_instead_of_waited_for():
    before = children()
    crashing = record_workers.Crashing(component_plan((reference(),), policy=RECIPE, seed=0)[2].seed)
    records = parallel_records((reference(),), policy=RECIPE, seed=0, workers=7, provider=crashing)
    with pytest.raises(TrainingError, match='exited unexpectedly') as caught:
        list(islice(records, 500))
    assert caught.value.context['exitcode'] == 7 and caught.value.context['stage'] == 'training_record_worker'
    assert '__main__' in caught.value.hint and 'data_workers=0' in caught.value.hint
    records.close()
    assert children() == before


def test_a_worker_killed_while_sending_a_large_message_is_reported_not_waited_for():
    before = children()
    records = parallel_records((reference(),), policy=RECIPE, seed=0, workers=1, provider=record_workers.bulky)
    next(records)
    (worker,) = children() - before
    time.sleep(1.5)                     # the worker runs ahead until a 1.3 MB chunk fills its pipe mid-message
    os.kill(worker.pid, signal.SIGKILL)
    blocked, outcome = record_workers.consume_within(records, 15)
    records.close()
    assert not blocked and isinstance(outcome.get('error'), TrainingError)
    assert outcome['error'].context['stage'] == 'training_record_worker'
    assert children() == before


def test_closing_or_collecting_the_stream_stops_every_worker():
    panels = (reference(),)
    before = children()
    records = parallel_records(panels, policy=RECIPE, seed=0, workers=3, provider=record_workers.counted)
    next(records)
    started = children() - before
    records.close()
    assert len(started) == 3 and stopped(started)
    records.close()                                    # idempotent
    with pytest.raises(StopIteration):
        next(records)

    records = parallel_records(panels, policy=RECIPE, seed=0, workers=3, provider=record_workers.counted)
    next(records)
    started = children() - before
    del records
    gc.collect()
    assert len(started) == 3 and stopped(started)


def test_closed_workers_never_restart_and_say_they_were_stopped():
    before = children()
    source = RecordWorkers((reference(),), policy=RECIPE, seed=0, workers=2, provider=record_workers.counted)
    source.close()
    with pytest.raises(TrainingError, match='closed'):
        next(training_records((reference(),), policy=RECIPE, seed=0, record_source=source))
    assert children() == before

    source = RecordWorkers((reference(),), policy=RECIPE, seed=0, workers=2, provider=record_workers.counted)
    records = training_records((reference(),), policy=RECIPE, seed=0, record_source=source)
    next(records)
    source.close()
    with pytest.raises(TrainingError, match='stopped'):
        list(islice(records, 5000))                    # past the first chunk of each of the 28 streams
    assert children() == before


def test_workers_import_the_parents_genairr():
    import GenAIRR
    with parallel_records((reference(),), policy=RECIPE, seed=0, workers=2,
                          provider=record_workers.genairr_origin) as records:
        rows = list(islice(records, 14))
    assert {row['genairr'] for row in rows} == {GenAIRR.__file__}
    assert os.getpid() not in {row['pid'] for row in rows} and len({row['pid'] for row in rows}) == 2


def test_a_worker_that_imports_a_different_genairr_refuses_to_generate(monkeypatch):
    import GenAIRR
    elsewhere = ModuleType('GenAIRR')
    elsewhere.__file__ = os.path.join(os.sep, 'elsewhere', 'GenAIRR', '__init__.py')
    monkeypatch.setitem(sys.modules, 'GenAIRR', elsewhere)
    records = parallel_records((reference(),), policy=RECIPE, seed=0, workers=1, provider=record_workers.counted)
    with pytest.raises(ConfigurationError, match='GenAIRR') as caught:
        next(records)
    records.close()
    assert caught.value.context['worker'] == GenAIRR.__file__
    assert caught.value.context['parent'] == elsewhere.__file__


@pytest.mark.performance
def test_starting_workers_does_not_wait_for_each_one_to_import_its_payload():
    group = WorkerGroup()
    started = time.perf_counter()
    for worker in range(4):
        group.start(record_workers.idle, b'reference' * 1_000_000, outputs=(), name=f'idle-{worker}')
    elapsed = time.perf_counter() - started
    group.close()
    assert elapsed < 1.0 and stopped(group.processes)


@pytest.mark.parametrize('options', [{'workers': 0}, {'workers': True}, {'workers': 1.5}, {'workers': '2'},
                                     {'chunk_size': 0}, {'queue_chunks': False}, {'provider': 1}])
def test_invalid_worker_requests_are_rejected_before_any_process_starts(options):
    before = children()
    with pytest.raises(ConfigurationError):
        parallel_records((reference(),), policy=RECIPE, **{'workers': 2, **options})
    assert children() == before


def test_a_task_a_worker_cannot_receive_is_refused_before_its_process_starts():
    before = children()
    with pytest.raises(ConfigurationError) as caught:
        parallel_records((reference(),), policy=RECIPE, workers=2, provider=lambda *args: iter(()))
    assert caught.value.context['field'] == 'provider'
    group = WorkerGroup()
    with pytest.raises(ConfigurationError) as caught:
        group.start(record_workers.idle, {'unpicklable': lambda: None}, outputs=(), name='refused')
    group.close()
    assert caught.value.context == {'field': 'payload', 'process': 'refused'}
    assert group.processes == () and children() == before


def test_genairr_records_are_identical_to_the_serial_stream():
    import GenAIRR.data as gd
    sources = (gd.HUMAN_IGH_OGRDB,)
    serial = list(islice(training_records(sources, policy=RECIPE, seed=0), 300))
    with parallel_records(sources, policy=RECIPE, seed=0, workers=7) as records:
        assert list(islice(records, 300)) == serial

