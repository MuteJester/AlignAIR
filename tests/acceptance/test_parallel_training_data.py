"""Acceptance: under the 1.0.0 recipe and seed 0, data workers produce the serial records and batches."""
from itertools import islice

import pytest

from alignair.train.assembly import TrainingStream, parallel_training_data
from alignair.train.gym.streams import CurriculumStreamPolicy, training_records
from alignair.train.gym.workers import parallel_records
from tests.support.training_streams import assert_same_runs, run, serial_records

pytestmark = [pytest.mark.acceptance, pytest.mark.slow]

RECIPE = CurriculumStreamPolicy((0.3, 0.6, 0.9), 0.25, 1)      # progresses, heavy_shm, short_boost


@pytest.fixture(scope='module')
def igh():
    import GenAIRR.data as gd
    from alignair.core import prepare_model
    from alignair.data.training import prepare_training_data
    from alignair.reference.reference_set import ReferenceSet
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), input_space='nucleotide',
                          label_space='alleles')
    return gd.HUMAN_IGH_OGRDB, prepare_training_data(setup, device='cpu')


@pytest.mark.parametrize('workers', [7, 3])
def test_two_thousand_records_are_the_serial_records(igh, workers):
    dataconfig, _ = igh
    serial = list(islice(training_records((dataconfig,), policy=RECIPE, seed=0), 2048))
    with parallel_records((dataconfig,), policy=RECIPE, seed=0, workers=workers) as records:
        assert list(islice(records, 2048)) == serial


@pytest.mark.parametrize('start', [0, 1000])
@pytest.mark.parametrize('in_worker', [True, False])
def test_thirty_two_batches_of_64_are_the_serial_batches_from_the_start_and_from_a_resume(igh, start, in_worker):
    dataconfig, data = igh
    stream = TrainingStream((dataconfig,), RECIPE, 0, data, augment=True, prepare_seed=0, start_step=start)
    serial = run(serial_records(stream), data.batch_adapter, start_step=start, steps=32, batch_size=64)
    with parallel_training_data(stream, batch_size=64, workers=7, device='cpu', assemble_in_worker=in_worker) as batches:
        parallel = run(batches.records, batches.batch_adapter, start_step=start, steps=32, batch_size=64)
    assert_same_runs(serial, parallel, steps=32)
