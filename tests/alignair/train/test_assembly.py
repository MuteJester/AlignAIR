"""Batches assembled in worker processes are the serial trainer's batches, and the workers never outlive them."""
import gc
from itertools import islice
import multiprocessing
import os
import signal
import time

import pytest
import torch

from alignair.data.training import TrainingDataSetup
from alignair.errors import ConfigurationError, TrainingDataError, TrainingError
from alignair.train.assembly import AssembledBatches, LocalAssembly, TrainingStream, parallel_training_data
from alignair.train.gym.streams import CurriculumStreamPolicy
from tests.support import record_workers
from tests.support.training_streams import RecordingEngine, assert_same_runs, run, serial_records

pytestmark = pytest.mark.integration

RECIPE = CurriculumStreamPolicy((0.3, 0.6, 0.9), 0.25, 1)


@pytest.fixture(scope='module')
def igh():
    import GenAIRR.data as gd
    from alignair.core import prepare_model
    from alignair.data.training import prepare_training_data
    from alignair.reference.reference_set import ReferenceSet
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), input_space='nucleotide',
                          label_space='alleles')
    return gd.HUMAN_IGH_OGRDB, prepare_training_data(setup, device='cpu')


def synthetic(adapter=record_workers.ordinal_batch, start=0):
    from types import SimpleNamespace
    data = TrainingDataSetup(adapter, adapter, False)
    panel = SimpleNamespace(name='IGH', metadata=SimpleNamespace(chain_type='BCR_HEAVY'))
    return TrainingStream((panel,), RECIPE, start, data, augment=True, prepare_seed=0, start_step=start)


def children():
    return set(multiprocessing.active_children())


@pytest.mark.parametrize('start', [0, 3])
@pytest.mark.parametrize('in_worker', [True, False])
def test_parallel_batches_equal_the_serial_trainer_batches(igh, start, in_worker):
    dataconfig, data = igh
    stream = TrainingStream((dataconfig,), RECIPE, start, data, augment=True, prepare_seed=0, start_step=start)
    serial = run(serial_records(stream), data.batch_adapter, start_step=start, steps=4, batch_size=16)
    with parallel_training_data(stream, batch_size=16, workers=3, device='cpu', assemble_in_worker=in_worker) as batches:
        parallel = run(batches.records, batches.batch_adapter, start_step=start, steps=4, batch_size=16)
    assert_same_runs(serial, parallel, steps=4)


@pytest.mark.parametrize('in_worker', [True, False])
def test_a_record_failure_stops_training_at_the_serial_step_with_its_type_and_context(in_worker):
    stream = synthetic()
    serial = run(serial_records_with(stream, record_workers.failing), stream.data.batch_adapter,
                 start_step=0, steps=200, batch_size=16)
    with parallel_training_data(stream, batch_size=16, workers=2, device='cpu', assemble_in_worker=in_worker,
                                provider=record_workers.failing) as batches:
        parallel = run(batches.records, batches.batch_adapter, start_step=0, steps=200, batch_size=16)
    assert_same_runs(serial, parallel, steps=200)
    error = parallel[2]
    assert isinstance(error, TrainingDataError) and len(parallel[0]) == 7 * 70 // 16
    assert error.context['stage'] == 'curriculum_component' and error.context['seed'] == 3
    assert error.context['step'] == len(parallel[0]) + 1


@pytest.mark.parametrize('in_worker', [True, False])
def test_a_batch_failure_carries_the_serial_stage_and_step(in_worker):
    stream = synthetic(record_workers.strict_batch)
    serial = run(serial_records_with(stream, record_workers.counted), stream.data.batch_adapter,
                 start_step=0, steps=40, batch_size=16)
    with parallel_training_data(stream, batch_size=16, workers=2, device='cpu', assemble_in_worker=in_worker,
                                provider=record_workers.counted) as batches:
        parallel = run(batches.records, batches.batch_adapter, start_step=0, steps=40, batch_size=16)
    assert_same_runs(serial, parallel, steps=40)
    assert parallel[2].context == {'stage': 'batch_adaptation', 'step': 22, 'record_index': 14}


def serial_records_with(stream, provider):
    from alignair.train.gym.streams import training_records
    records = training_records(stream.dataconfigs, policy=stream.policy, seed=stream.seed, record_source=provider)
    return stream.data.prepare_records(records, augment=stream.augment, seed=stream.prepare_seed,
                                       start_step=stream.start_step)


def test_a_cpu_random_draw_inside_the_step_is_refused_rather_than_silently_diverging():
    with AssembledBatches(synthetic(), batch_size=8, workers=2, device='cpu',
                          provider=record_workers.counted) as batches:
        iterations, _, error = run(batches.records, batches.batch_adapter, start_step=0, steps=3, batch_size=8,
                                   engine=RecordingEngine(draw=True))
    assert len(iterations) == 1 and isinstance(error, TrainingError)
    assert 'random' in str(error) and error.context['step'] == 2


def test_an_assembler_killed_while_sending_a_batch_is_reported_not_waited_for():
    before = children()
    batches = AssembledBatches(synthetic(), batch_size=64, workers=1, device='cpu', provider=record_workers.bulky)
    next(batches.records)
    (assembler,) = [process for process in children() - before if process.name == 'alignair-batches']
    time.sleep(2)                       # the assembler runs ahead until a 1.3 MB batch fills its pipe mid-message
    os.kill(assembler.pid, signal.SIGKILL)
    blocked, outcome = record_workers.consume_within(batches.records, 15)
    batches.close()
    assert not blocked and isinstance(outcome.get('error'), TrainingError)
    assert children() == before


def test_the_adapter_only_accepts_the_records_it_assembled():
    with AssembledBatches(synthetic(), batch_size=8, workers=1, device='cpu',
                          provider=record_workers.counted) as batches:
        records = tuple(islice(batches.records, 8))
        with pytest.raises(TrainingError, match='assembled'):
            batches.batch_adapter(tuple(dict(record) for record in records))


@pytest.mark.parametrize('in_worker', [True, False])
def test_closing_or_collecting_the_batches_stops_every_worker(in_worker):
    before = children()
    batches = parallel_training_data(synthetic(), batch_size=8, workers=2, device='cpu', assemble_in_worker=in_worker,
                                     provider=record_workers.counted)
    assert children() == before
    next(batches.records)
    started = children() - before
    batches.close()
    batches.close()
    assert len(started) == (3 if in_worker else 2)
    assert all(not process.is_alive() for process in started) and children() == before

    batches = parallel_training_data(synthetic(), batch_size=8, workers=2, device='cpu', assemble_in_worker=in_worker,
                                     provider=record_workers.counted)
    next(batches.records)
    started = children() - before
    del batches
    gc.collect()
    assert started and all(not process.is_alive() for process in started) and children() == before


def test_batches_are_assembled_in_a_worker_unless_training_runs_on_the_cpu():
    stream = synthetic()
    with parallel_training_data(stream, batch_size=8, workers=2, device='cpu') as local:
        assert isinstance(local, LocalAssembly)
    with parallel_training_data(stream, batch_size=8, workers=2, device=torch.device('cuda')) as assembled:
        assert isinstance(assembled, AssembledBatches)


@pytest.mark.parametrize('options', [{'workers': 0}, {'workers': False}, {'batch_size': 0}, {'queue_batches': 0},
                                     {'device': 'nowhere'}, {'assemble_in_worker': 1}, {'provider': None}])
def test_invalid_requests_are_rejected_before_any_process_starts(options):
    before = children()
    with pytest.raises(ConfigurationError):
        parallel_training_data(synthetic(), **{'batch_size': 8, 'workers': 2, 'device': 'cpu',
                                               'assemble_in_worker': True, **options})
    with pytest.raises(ConfigurationError):
        AssembledBatches(object(), batch_size=8, workers=2, device='cpu')
    assert children() == before
