"""The record cache replays the live curriculum stream record for record, and refuses any other stream.

One small cache of the 1.0.0 recipe (40 steps of 64: 2,560 records, 366 per component) is written per module; the
tests that alter a cache work on a copy of it.
"""
import json
import pickle
import shutil
from itertools import islice

import pytest

from alignair.errors import ConfigurationError, TrainingDataError
from alignair.train.assembly import TrainingStream, parallel_training_data
from alignair.train.gym.streams import CurriculumStreamPolicy, training_records
from alignair.train.gym.workers import component_plan, parallel_records
from alignair_lab.recipe import BATCH_SIZE, DATACONFIGS, POLICY, SEED
from alignair_lab.records import CachedRecordSource, component_counts, genairr_digest, write_record_cache
from tests.support.training_streams import assert_same_runs, run, serial_records

pytestmark = pytest.mark.integration

STEPS = 40
RECORDS = STEPS * BATCH_SIZE                       # 2,560 records of the zipped stream
COMPONENT_RECORDS = 366                            # the 366th round of 7 components holds records 2,556 to 2,562


@pytest.fixture(scope='module')
def cache(tmp_path_factory):
    path = tmp_path_factory.mktemp('records') / 'igh'
    write_record_cache(path, dataconfigs=DATACONFIGS, policy=POLICY, seed=SEED, steps=STEPS, batch_size=BATCH_SIZE,
                       workers=7)
    return path


@pytest.fixture(scope='module')
def data():
    import GenAIRR.data as gd
    from alignair.core import prepare_model
    from alignair.data.training import prepare_training_data
    from alignair.reference.reference_set import ReferenceSet
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), input_space='nucleotide',
                          label_space='alleles')
    return prepare_training_data(setup, device='cpu')


def live(records=RECORDS, *, policy=POLICY, seed=SEED, record_source=None):
    return list(islice(training_records(DATACONFIGS, policy=policy, seed=seed, record_source=record_source), records))


def altered(cache, tmp_path, **changes):
    """A copy of the cache whose manifest has ``changes`` applied to it."""
    copy = tmp_path / 'altered'
    shutil.copytree(cache, copy)
    manifest = json.loads((copy / 'manifest.json').read_text())
    for field, value in changes.items():
        target = manifest
        *parents, leaf = field.split('.')
        for parent in parents:
            target = target[parent]
        target[leaf] = value
    (copy / 'manifest.json').write_text(json.dumps(manifest))
    return CachedRecordSource(copy)


def test_the_recipe_is_the_curriculum_run_training_trains_on():
    import inspect
    from alignair.train.trainer import train
    defaults = {name: parameter.default for name, parameter in inspect.signature(train).parameters.items()}
    assert POLICY == CurriculumStreamPolicy(defaults['progresses'], defaults['heavy_shm'], 1, defaults['mutation_cap'])
    assert DATACONFIGS == ('HUMAN_IGH_OGRDB',) and (SEED, BATCH_SIZE) == (0, 64)


def test_component_counts_are_what_the_zipped_stream_draws():
    assert component_counts(DATACONFIGS, policy=POLICY, seed=SEED, records=RECORDS) == (COMPONENT_RECORDS,) * 7
    assert component_counts(DATACONFIGS, policy=POLICY, seed=SEED, records=7) == (1,) * 7
    assert component_counts(DATACONFIGS, policy=POLICY, seed=SEED, records=8) == (2,) * 7


def test_the_manifest_records_genairr_the_recipe_and_each_components_records(cache):
    import GenAIRR
    from pathlib import Path
    manifest = json.loads((cache / 'manifest.json').read_text())
    assert manifest['genairr']['genairr_source_sha256'] == genairr_digest(str(Path(GenAIRR.__file__).parent))
    assert manifest['dataconfigs'] == ['HUMAN_IGH_OGRDB']
    assert manifest['policy'] == {'progresses': [0.3, 0.6, 0.9], 'heavy_shm': 0.25, 'short_boost': 1,
                                  'mutation_cap': None, 'mode': 'auto'}
    assert (manifest['seed'], manifest['steps'], manifest['batch_size'], manifest['records']) == (0, 40, 64, RECORDS)
    plan = component_plan(DATACONFIGS, policy=POLICY, seed=SEED)
    assert [component['seed'] for component in manifest['components']] == [component.seed for component in plan]
    assert all(component['records'] == COMPONENT_RECORDS and component['reference'] == 'HUMAN_IGH_OGRDB'
               for component in manifest['components'])
    assert all(sum(chunk['records'] for chunk in component['chunks']) == COMPONENT_RECORDS
               for component in manifest['components'])


def test_the_cache_replays_the_live_stream_record_for_record(cache):
    assert live(record_source=CachedRecordSource(cache)) == live()


def test_a_source_crosses_processes_as_its_path(cache):
    source = CachedRecordSource(cache)
    source.manifest                                # loaded here; a copy loads its own
    copy = pickle.loads(pickle.dumps(source))
    assert copy.path == source.path and len(pickle.dumps(source)) < 500
    with parallel_records(DATACONFIGS, policy=POLICY, seed=SEED, workers=3, provider=copy) as records:
        assert list(islice(records, 2048)) == live(2048)


@pytest.mark.parametrize('in_worker', [None, False, True])
def test_batches_from_the_cache_are_the_live_batches(cache, data, in_worker):
    stream = TrainingStream(DATACONFIGS, POLICY, SEED, data, augment=True, prepare_seed=SEED)
    expected = run(serial_records(stream), data.batch_adapter, start_step=0, steps=20, batch_size=BATCH_SIZE)
    source = CachedRecordSource(cache)
    if in_worker is None:                          # the trainer's serial path, with the cache as its provider
        actual = run(stream.records(record_source=source), data.batch_adapter, start_step=0, steps=20,
                     batch_size=BATCH_SIZE)
    else:
        with parallel_training_data(stream, batch_size=BATCH_SIZE, workers=2, device='cpu',
                                    assemble_in_worker=in_worker, provider=source) as batches:
            actual = run(batches.records, batches.batch_adapter, start_step=0, steps=20, batch_size=BATCH_SIZE)
    assert_same_runs(expected, actual, steps=20)


def test_a_cache_read_to_its_end_raises_and_never_wraps_around(cache):
    records = training_records(DATACONFIGS, policy=POLICY, seed=SEED, record_source=CachedRecordSource(cache))
    served = list(islice(records, 7 * COMPONENT_RECORDS))
    assert served == live(7 * COMPONENT_RECORDS)
    with pytest.raises(TrainingDataError, match='exhausted') as caught:
        next(records)
    assert caught.value.context['records'] == COMPONENT_RECORDS


def test_exhaustion_in_a_data_worker_reaches_the_training_process(cache):
    with parallel_records(DATACONFIGS, policy=POLICY, seed=SEED, workers=2, provider=CachedRecordSource(cache)) \
            as records:
        assert len(list(islice(records, 7 * COMPONENT_RECORDS))) == 7 * COMPONENT_RECORDS
        with pytest.raises(TrainingDataError, match='exhausted') as caught:
            next(records)
    assert caught.value.context['stage'] == 'record_cache'


@pytest.mark.parametrize('seed,problem', [(100, 'no component of seed 100'), (1, 'other parameters')])
def test_another_seed_is_refused(cache, seed, problem):
    with pytest.raises(ConfigurationError, match=problem):
        live(1, seed=seed, record_source=CachedRecordSource(cache))


def test_other_parameters_are_refused(cache):
    with pytest.raises(ConfigurationError, match='other parameters') as caught:
        live(1, policy=CurriculumStreamPolicy(heavy_shm=0.3), record_source=CachedRecordSource(cache))
    assert caught.value.context['fields'] == ['mutation_rate']


@pytest.mark.parametrize('field,problem', [('genairr.genairr_source_sha256', 'another GenAIRR'),
                                           ('experiment_sha256', 'another experiment builder')])
def test_a_cache_of_another_generator_is_refused(cache, tmp_path, field, problem):
    source = altered(cache, tmp_path, **{field: '0' * 64})
    with pytest.raises(ConfigurationError, match=problem):
        live(1, record_source=source)


def test_require_checks_a_whole_stream_before_any_record(cache):
    source = CachedRecordSource(cache)
    source.require(DATACONFIGS, policy=POLICY, seed=SEED, records=RECORDS)
    with pytest.raises(ConfigurationError, match='serves 2560 records') as caught:
        source.require(DATACONFIGS, policy=POLICY, seed=SEED, records=RECORDS + 1)
    assert caught.value.context['needed'] == RECORDS + 1
    with pytest.raises(ConfigurationError, match='other parameters'):
        source.require(DATACONFIGS, policy=CurriculumStreamPolicy(heavy_shm=0.3), seed=SEED, records=7)
    with pytest.raises(ConfigurationError, match='another stream'):         # 6 of the 7 cached components
        source.require(DATACONFIGS, policy=CurriculumStreamPolicy(heavy_shm=0.), seed=SEED, records=7)


def test_the_cache_command_writes_the_recipes_cache(tmp_path, capsys):
    from alignair_lab.cli import main
    assert main(['cache', '--out', str(tmp_path / 'one'), '--steps', '1', '--workers', '2']) == 0
    manifest = CachedRecordSource(tmp_path / 'one').require(DATACONFIGS, policy=POLICY, seed=SEED, records=BATCH_SIZE)
    assert (manifest['steps'], manifest['batch_size'], manifest['workers']) == (1, BATCH_SIZE, 2)
    assert f"wrote {tmp_path / 'one'}: 64 records" in capsys.readouterr().out
    assert main(['cache', '--out', str(tmp_path / 'one'), '--steps', '1']) == 2
    assert 'exists' in capsys.readouterr().err


def test_an_existing_cache_or_an_unfinished_one_is_not_a_cache(cache, tmp_path):
    with pytest.raises(FileExistsError):
        write_record_cache(cache, dataconfigs=DATACONFIGS, policy=POLICY, seed=SEED, steps=1, batch_size=1)
    (tmp_path / 'partial').mkdir()
    with pytest.raises(ConfigurationError, match='not a complete record cache'):
        CachedRecordSource(tmp_path / 'partial').manifest
