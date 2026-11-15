"""The record cache replays the live curriculum stream record for record, and refuses any other stream.

Two caches are written per module, and the live streams they are compared with are drawn once per module:

- ``unselected``: the recipe's 7 components without selection (one fast GenAIRR stream each), 40 steps of 64 records
  (366 per stream) in chunks of 100 records, so every long replay crosses chunk boundaries. It carries the tests
  that need thousands of records.
- ``recipe``: the recipe itself, whose 7 components interleave 4 streams each (3 under GenAIRR's productive_only(),
  about 20 times slower to simulate), for one step of 64 records: 3 records per stream, 84 records in all.

The tests that alter a cache work on a copy of it.
"""
import dataclasses
import json
import pickle
import re
import shutil
from itertools import islice

import pytest

from alignair.errors import ConfigurationError, TrainingDataError, TrainingError
from alignair.train.assembly import TrainingStream, parallel_training_data
from alignair.train.gym.streams import CurriculumStreamPolicy, training_records
from alignair.train.gym.workers import component_plan
from tests.support.record_workers import parallel_records
from alignair_lab.recipe import BATCH_SIZE, DATACONFIGS, POLICY, SEED
from alignair_lab.records import CachedRecordSource, component_counts, genairr_digest, write_record_cache
from tests.support.training_streams import assert_same_runs, run, serial_records

pytestmark = pytest.mark.integration

UNSELECTED = dataclasses.replace(POLICY, selected_fraction=0)
STEPS = 40
RECORDS = STEPS * BATCH_SIZE                       # 2,560 records of the zipped stream
STREAM_RECORDS = 366                               # its 366th round of 7 streams holds records 2,556 to 2,562
SERVED = 7 * STREAM_RECORDS                        # what the unselected cache serves before a stream is exhausted
CHUNK = 100
RECIPE_STREAMS = 28
RECIPE_STREAM_RECORDS = 3                          # 64 records need 10 of each component's: 3 periods of 4 streams
RECIPE_SERVED = RECIPE_STREAMS * RECIPE_STREAM_RECORDS


@pytest.fixture(scope='module')
def cache(tmp_path_factory):
    path = tmp_path_factory.mktemp('records') / 'unselected'
    write_record_cache(path, dataconfigs=DATACONFIGS, policy=UNSELECTED, seed=SEED, steps=STEPS,
                       batch_size=BATCH_SIZE, workers=7, chunk_records=CHUNK)
    return path


@pytest.fixture(scope='module')
def recipe_cache(tmp_path_factory):
    path = tmp_path_factory.mktemp('records') / 'recipe'
    write_record_cache(path, dataconfigs=DATACONFIGS, policy=POLICY, seed=SEED, steps=1, batch_size=BATCH_SIZE)
    return path


@pytest.fixture(scope='module')
def expected():
    """The live unselected stream, as far as the unselected cache serves it."""
    return live(SERVED)


@pytest.fixture(scope='module')
def data():
    import GenAIRR.data as gd
    from alignair.core import prepare_model
    from alignair.data.training import prepare_training_data
    from alignair.reference.reference_set import ReferenceSet
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB))
    return prepare_training_data(setup, device='cpu')


@pytest.fixture(scope='module')
def serial_run(data):
    """The trainer's serial run of 20 steps over the live unselected stream."""
    stream = TrainingStream(DATACONFIGS, UNSELECTED, SEED, data)
    return stream, run(serial_records(stream), data.batch_adapter, start_step=0, steps=20, batch_size=BATCH_SIZE)


def live(records, *, policy=UNSELECTED, seed=SEED, record_source=None):
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
    assert POLICY == CurriculumStreamPolicy(defaults['progresses'], defaults['heavy_shm'], 1, defaults['mutation_cap'],
                                            indel_fraction=defaults['indel_fraction'],
                                            indel_form=defaults['indel_form'],
                                            selected_fraction=defaults['selected_fraction'])
    assert DATACONFIGS == ('HUMAN_IGH_OGRDB',) and (SEED, BATCH_SIZE) == (0, 64)


def test_a_flank_policy_is_no_part_of_a_caches_identity():
    """Flanks are drawn after the provider (``training_records``), so a cache of the provider's records serves a
    flanked run as it serves a bare one."""
    from alignair.data.flanks import FlankPolicy
    from alignair_lab.records import policy_fields
    assert policy_fields(dataclasses.replace(POLICY, flanks=FlankPolicy())) == policy_fields(POLICY)
    assert 'flanks' not in policy_fields(POLICY)


def test_component_counts_are_what_the_zipped_stream_draws():
    assert component_counts(DATACONFIGS, policy=POLICY, seed=SEED, records=BATCH_SIZE) == (3,) * RECIPE_STREAMS
    assert component_counts(DATACONFIGS, policy=POLICY, seed=SEED, records=28) == (1,) * RECIPE_STREAMS
    assert component_counts(DATACONFIGS, policy=POLICY, seed=SEED, records=29) == (2,) * RECIPE_STREAMS
    assert component_counts(DATACONFIGS, policy=UNSELECTED, seed=SEED, records=RECORDS) == (STREAM_RECORDS,) * 7


def test_the_manifest_records_genairr_the_recipe_and_each_streams_records(recipe_cache):
    import hashlib
    import GenAIRR
    from pathlib import Path
    import alignair
    from alignair.train.trainer import RECIPE
    manifest = json.loads((recipe_cache / 'manifest.json').read_text())
    assert manifest['genairr']['genairr_source_sha256'] == genairr_digest(str(Path(GenAIRR.__file__).parent))
    assert manifest['dataconfigs'] == ['HUMAN_IGH_OGRDB']
    assert (manifest['recipe'], manifest['purpose'], manifest['start_step']) == (RECIPE, 'training', 0)
    assert manifest['policy'] == {'progresses': {'__tuple__': [0.3, 0.6, 0.9]}, 'heavy_shm': 0.25, 'short_boost': 1,
                                  'mutation_cap': None, 'mode': 'auto', 'indel_fraction': {'__tuple__': [0.05, 0.25]},
                                  'indel_form': 'weighted', 'selected_fraction': 0.75}
    files = manifest['code']['files']           # provenance: the code that produced the records
    assert {'train/gym/streams.py', 'train/gym/experiment.py', 'train/gym/curriculum/base.py'} <= set(files)
    root = Path(alignair.__file__).parent
    assert all(digest == hashlib.sha256((root / name).read_bytes()).hexdigest() for name, digest in files.items())
    assert (manifest['seed'], manifest['steps'], manifest['batch_size'], manifest['records']) == (0, 1, 64, 64)
    plan = component_plan(DATACONFIGS, policy=POLICY, seed=SEED)
    assert [stream['seed'] for stream in manifest['components']] == [stream.seed for stream in plan]
    assert all(stream['records'] == RECIPE_STREAM_RECORDS and stream['reference'] == 'HUMAN_IGH_OGRDB'
               for stream in manifest['components'])
    assert all(sum(chunk['records'] for chunk in stream['chunks']) == RECIPE_STREAM_RECORDS
               for stream in manifest['components'])


def test_the_writer_runs_a_process_per_physical_core_by_default(recipe_cache):
    from alignair_lab.records import physical_cores
    manifest = json.loads((recipe_cache / 'manifest.json').read_text())
    assert manifest['workers'] == min(physical_cores(), manifest['segments'])


def test_the_writer_reports_records_per_second_and_its_eta(tmp_path):
    lines = []
    manifest = write_record_cache(tmp_path / 'watched', dataconfigs=DATACONFIGS, policy=UNSELECTED, seed=SEED,
                                  steps=8, batch_size=BATCH_SIZE, workers=2, progress=lines.append,
                                  progress_seconds=0.05)
    total = sum(stream['records'] for stream in manifest['components'])
    assert len(lines) >= 2
    assert all(re.search(r'\d+/\d+ records \(\d+%\), \d+ of 7 streams done, [\d.]+ records/s, ETA ', line)
               for line in lines), lines
    assert lines[-1].startswith(f'{total}/{total} records (100%), 7 of 7 streams done')


def test_a_cache_written_in_segments_is_the_serial_stream(cache, tmp_path, expected):
    split = tmp_path / 'split'
    write_record_cache(split, dataconfigs=DATACONFIGS, policy=UNSELECTED, seed=SEED, steps=STEPS,
                       batch_size=BATCH_SIZE, workers=7, chunk_records=CHUNK, segment_records=50)
    streams = json.loads((split / 'manifest.json').read_text())['components']
    assert all(stream['segments'] == 8 and [chunk['records'] for chunk in stream['chunks']] == [50] * 7 + [16]
               for stream in streams)
    assert live(SERVED, record_source=CachedRecordSource(split)) == expected


def test_selected_streams_written_in_segments_are_the_serial_streams(tmp_path):
    split = tmp_path / 'split'
    write_record_cache(split, dataconfigs=DATACONFIGS, policy=POLICY, seed=SEED, steps=1, batch_size=BATCH_SIZE,
                       segment_records=1)                                 # each of a stream's 3 records alone
    streams = json.loads((split / 'manifest.json').read_text())['components']
    assert all(stream['segments'] == RECIPE_STREAM_RECORDS for stream in streams)
    assert live(RECIPE_SERVED, policy=POLICY, record_source=CachedRecordSource(split)) == \
        live(RECIPE_SERVED, policy=POLICY)


def test_segments_of_a_shiftable_provider_are_its_serial_stream(tmp_path):
    from tests.alignair_lab.providers import shiftable
    split = tmp_path / 'split'
    manifest = write_record_cache(split, dataconfigs=DATACONFIGS, policy=UNSELECTED, seed=SEED, steps=1,
                                  batch_size=BATCH_SIZE, segment_records=4, provider=shiftable)
    assert manifest['boundaries_checked'] == sum(stream['segments'] - 1 for stream in manifest['components']) > 0
    assert manifest['interiors_checked'] >= len(manifest['components'])     # a record inside each split stream
    source = CachedRecordSource(split)
    for planned in component_plan(DATACONFIGS, policy=UNSELECTED, seed=SEED):
        entry = source.component(planned.reference, planned.parameters, planned.seed)
        assert list(islice(source(planned.reference, planned.parameters, planned.seed), entry['records'])) == \
            list(islice(shiftable(None, None, planned.seed), entry['records']))


@pytest.mark.parametrize('provider,problem', [('unshiftable', 'not shiftable'), ('renamed', 'seq<k>'),
                                              ('joins_only', 'not shiftable: record 2 of')])
def test_a_stream_that_cannot_be_drawn_in_segments_fails_the_write(tmp_path, provider, problem):
    from tests.alignair_lab import providers
    with pytest.raises(ConfigurationError, match=problem):
        write_record_cache(tmp_path / 'split', dataconfigs=DATACONFIGS, policy=UNSELECTED, seed=SEED, steps=1,
                           batch_size=BATCH_SIZE, segment_records=4, provider=getattr(providers, provider))
    assert list(tmp_path.iterdir()) == []                                 # no cache, no partial directory


@pytest.mark.timeout(120)
def test_a_writer_killed_outright_fails_the_write_instead_of_hanging(tmp_path):
    """A pool replaces a worker killed outright (the out-of-memory killer, kill -9) but never finishes its task."""
    from tests.alignair_lab.providers import killed
    with pytest.raises(TrainingError, match='died before it finished'):
        write_record_cache(tmp_path / 'cache', dataconfigs=DATACONFIGS, policy=UNSELECTED, seed=SEED, steps=1,
                           batch_size=BATCH_SIZE, workers=2, provider=killed)
    assert list(tmp_path.iterdir()) == []                                 # no cache, no partial directory


@pytest.mark.timeout(120)
def test_a_write_whose_writers_cannot_be_signalled_is_not_failed(tmp_path, monkeypatch):
    """A sandbox may refuse kill(pid, 0) for a live process: the writers are judged by the pool's own children."""
    import os
    from tests.alignair_lab.providers import slow
    signal = os.kill

    def refused(pid, number):
        if number == 0:
            raise PermissionError(1, 'Operation not permitted')
        return signal(pid, number)

    monkeypatch.setattr(os, 'kill', refused)
    manifest = write_record_cache(tmp_path / 'cache', dataconfigs=DATACONFIGS, policy=UNSELECTED, seed=SEED,
                                  steps=1, batch_size=BATCH_SIZE, workers=7, provider=slow)
    assert manifest['segments'] == 7 and (tmp_path / 'cache' / 'manifest.json').is_file()


def test_a_writer_names_its_process_before_it_checks_its_genairr(tmp_path, monkeypatch):
    """A writer killed before its first record must still be found gone: it names its process first of all."""
    import os
    from alignair_lab import records
    from alignair_lab.records import Segment, _write_segment
    from tests.alignair_lab.providers import shiftable
    writers = [0]
    monkeypatch.setattr(records, '_PROGRESS', ([0], [0.], writers))
    with pytest.raises(ConfigurationError, match='another GenAIRR'):
        _write_segment((0, (str(tmp_path), 'stream', None, {}, 0, Segment(0, 0, 4, 0, True, False), 4, 'other',
                            shiftable)))
    assert writers == [os.getpid()]


def test_a_cache_directory_has_a_new_directorys_permissions(cache, tmp_path):
    (tmp_path / 'plain').mkdir()
    assert cache.stat().st_mode & 0o777 == (tmp_path / 'plain').stat().st_mode & 0o777


def test_the_cache_replays_the_live_stream_record_for_record_across_its_chunks(cache, expected):
    streams = json.loads((cache / 'manifest.json').read_text())['components']
    assert all([chunk['records'] for chunk in stream['chunks']] == [CHUNK] * 3 + [66] for stream in streams)
    assert live(SERVED, record_source=CachedRecordSource(cache)) == expected


def test_the_recipes_cache_replays_its_selected_and_unselected_streams(recipe_cache):
    assert live(RECIPE_SERVED, policy=POLICY, record_source=CachedRecordSource(recipe_cache)) == \
        live(RECIPE_SERVED, policy=POLICY)


def test_a_source_crosses_processes_as_its_path(cache, expected):
    source = CachedRecordSource(cache)
    source.manifest                                # loaded here; a copy loads its own
    copy = pickle.loads(pickle.dumps(source))
    assert copy.path == source.path and len(pickle.dumps(source)) < 500
    with parallel_records(DATACONFIGS, policy=UNSELECTED, seed=SEED, workers=3, provider=copy) as records:
        assert list(islice(records, 2048)) == expected[:2048]


@pytest.mark.parametrize('in_worker', [None, False, True])
def test_batches_from_the_cache_are_the_live_batches(cache, data, serial_run, in_worker):
    stream, expected_run = serial_run
    source = CachedRecordSource(cache)
    if in_worker is None:                          # the trainer's serial path, with the cache as its provider
        actual = run(stream.records(record_source=source), data.batch_adapter, start_step=0, steps=20,
                     batch_size=BATCH_SIZE)
    else:
        with parallel_training_data(stream, batch_size=BATCH_SIZE, workers=2, device='cpu',
                                    assemble_in_worker=in_worker, provider=source) as batches:
            actual = run(batches.records, batches.batch_adapter, start_step=0, steps=20, batch_size=BATCH_SIZE)
    assert_same_runs(expected_run, actual, steps=20)


def test_a_cache_read_to_its_end_raises_and_never_wraps_around(cache, expected):
    records = training_records(DATACONFIGS, policy=UNSELECTED, seed=SEED, record_source=CachedRecordSource(cache))
    assert list(islice(records, SERVED)) == expected
    with pytest.raises(TrainingDataError, match='exhausted') as caught:
        next(records)
    assert caught.value.context['records'] == STREAM_RECORDS


def test_exhaustion_in_a_data_worker_reaches_the_training_process(cache):
    with parallel_records(DATACONFIGS, policy=UNSELECTED, seed=SEED, workers=2,
                          provider=CachedRecordSource(cache)) as records:
        assert len(list(islice(records, SERVED))) == SERVED
        with pytest.raises(TrainingDataError, match='exhausted') as caught:
            next(records)
    assert caught.value.context['stage'] == 'record_cache'


def test_a_chunk_whose_bytes_changed_is_refused_when_read(cache, tmp_path, expected):
    copy = tmp_path / 'corrupt'
    shutil.copytree(cache, copy)
    stream = json.loads((copy / 'manifest.json').read_text())['components'][3]
    chunk = copy / stream['chunks'][1]['file']
    data = bytearray(chunk.read_bytes())
    data[-1] ^= 1
    chunk.write_bytes(bytes(data))
    records = training_records(DATACONFIGS, policy=UNSELECTED, seed=SEED, record_source=CachedRecordSource(copy))
    assert list(islice(records, 7 * CHUNK)) == expected[:7 * CHUNK]        # every stream's first chunk
    with pytest.raises(ConfigurationError, match='SHA-256') as caught:
        list(islice(records, 7 * CHUNK))
    assert caught.value.context['chunk'] == stream['chunks'][1]['file']


@pytest.mark.parametrize('seed', [100, 1])
def test_another_seed_is_refused(cache, seed):
    first = component_plan(DATACONFIGS, policy=UNSELECTED, seed=seed)[0].seed
    with pytest.raises(ConfigurationError, match=f'no component of seed {first}'):
        live(1, seed=seed, record_source=CachedRecordSource(cache))


def test_other_parameters_are_refused(cache):
    with pytest.raises(ConfigurationError, match='other parameters') as caught:
        live(1, policy=dataclasses.replace(UNSELECTED, heavy_shm=0.3), record_source=CachedRecordSource(cache))
    assert caught.value.context['fields'] == ['mutation_rate']


def test_a_cache_of_another_genairr_is_refused(cache, tmp_path):
    source = altered(cache, tmp_path, **{'genairr.genairr_source_sha256': '0' * 64})
    with pytest.raises(ConfigurationError, match='another GenAIRR'):
        live(1, record_source=source)


def test_a_cache_serves_a_flanked_request_and_its_counts_are_the_bare_streams(cache):
    from alignair.data.flanks import FlankPolicy
    flanked = dataclasses.replace(UNSELECTED, flanks=FlankPolicy())
    CachedRecordSource(cache).require(DATACONFIGS, policy=flanked, seed=SEED, records=RECORDS)
    assert (component_counts(DATACONFIGS, policy=flanked, seed=SEED, records=29)
            == component_counts(DATACONFIGS, policy=UNSELECTED, seed=SEED, records=29))


def test_require_checks_a_whole_stream_before_any_record(cache):
    source = CachedRecordSource(cache)
    source.require(DATACONFIGS, policy=UNSELECTED, seed=SEED, records=RECORDS)
    with pytest.raises(ConfigurationError, match='serves 2560 records') as caught:
        source.require(DATACONFIGS, policy=UNSELECTED, seed=SEED, records=RECORDS + 1)
    assert caught.value.context['needed'] == RECORDS + 1


@pytest.mark.parametrize('field,request_', [
    ('selected_fraction', {'policy': dataclasses.replace(UNSELECTED, selected_fraction=0.5)}),
    ('indel_form', {'policy': dataclasses.replace(UNSELECTED, indel_form='uniform')}),
    ('indel_fraction', {'policy': dataclasses.replace(UNSELECTED, indel_fraction=(0.1, 0.3))}),
    ('heavy_shm', {'policy': dataclasses.replace(UNSELECTED, heavy_shm=0.3)}),
    ('seed', {'seed': 1}),
    ('dataconfigs', {'dataconfigs': ('HUMAN_IGH_OGRDB', 'HUMAN_IGK_OGRDB')})])
def test_require_refuses_another_stream_naming_the_field(cache, field, request_):
    options = {'dataconfigs': DATACONFIGS, 'policy': UNSELECTED, 'seed': SEED, **request_}
    with pytest.raises(ConfigurationError, match=field) as caught:
        CachedRecordSource(cache).require(options.pop('dataconfigs'), records=7, **options)
    assert caught.value.context['field'] == field


@pytest.mark.parametrize('change,problem', [({'recipe': 'v2'}, 'recipe'), ({'version': 1}, 'not a version 2')])
def test_require_refuses_a_cache_of_another_recipe_or_format(cache, tmp_path, change, problem):
    with pytest.raises(ConfigurationError, match=problem):
        altered(cache, tmp_path, **change).require(DATACONFIGS, policy=UNSELECTED, seed=SEED, records=7)


def rare_change(monkeypatch, **parameters):
    """The live provider with ``parameters`` set in every stream: a change to the producing code that leaves the
    manifest as it was."""
    from alignair.train.gym import streams
    original = streams.stream_records

    def changed(dataconfig, stream_parameters, seed):
        return original(dataconfig, {**stream_parameters, **parameters}, seed)

    monkeypatch.setattr(streams, 'stream_records', changed)


def test_a_rare_change_passes_two_records_a_stream_and_fails_the_deep_check(cache, tmp_path, monkeypatch):
    rare_change(monkeypatch, invert_d_prob=0.03)                                     # first alters record 2
    source = CachedRecordSource(cache)
    source.require(DATACONFIGS, policy=UNSELECTED, seed=SEED, records=RECORDS)      # the first 2 records unchanged
    assert source.verification == {'code_changed': False, 'deep': False, 'records': 14}
    changed = altered(cache, tmp_path, **{'code.sha256': '0' * 64})                 # written by other producing code
    with pytest.raises(ConfigurationError, match='does not reproduce the live stream') as caught:
        changed.require(DATACONFIGS, policy=UNSELECTED, seed=SEED, records=RECORDS)
    assert caught.value.context['deep'] and caught.value.context['record'] >= 2
    assert caught.value.context['seed'] in [stream.seed for stream in component_plan(DATACONFIGS, policy=UNSELECTED,
                                                                                     seed=SEED)]


def test_the_deep_check_accepts_a_cache_whose_records_are_the_live_ones(cache, tmp_path):
    source = altered(cache, tmp_path, **{'code.sha256': '0' * 64})
    source.require(DATACONFIGS, policy=UNSELECTED, seed=SEED, records=RECORDS)
    assert source.verification == {'code_changed': True, 'deep': True, 'records': 14 + 7 * 256}


def test_require_draws_the_first_records_of_every_stream_live_and_compares_them(recipe_cache, monkeypatch):
    """A change to the code that produces the records (here the provider; equally the seed blocks, the interleave or
    the curriculum) leaves the manifest as it was: only drawing the stream again shows it."""
    from alignair.train.gym import streams
    source = CachedRecordSource(recipe_cache)
    source.require(DATACONFIGS, policy=POLICY, seed=SEED, records=BATCH_SIZE)
    original = streams.stream_records

    def changed(dataconfig, parameters, seed):
        return original(dataconfig, {**parameters, 'seq_error_rate': 0.5}, seed)

    monkeypatch.setattr(streams, 'stream_records', changed)
    with pytest.raises(ConfigurationError, match='does not reproduce the live stream') as caught:
        source.require(DATACONFIGS, policy=POLICY, seed=SEED, records=BATCH_SIZE)
    assert caught.value.context['record'] == 0
    assert caught.value.context['seed'] == component_plan(DATACONFIGS, policy=POLICY, seed=SEED)[0].seed


def test_the_cache_command_writes_the_recipes_cache(tmp_path, capsys):
    from alignair_lab.cli import main
    assert main(['cache', '--out', str(tmp_path / 'one'), '--steps', '1', '--workers', '2',
                 '--progress-seconds', '0.05', '--segment-records', '2']) == 0
    manifest = CachedRecordSource(tmp_path / 'one').require(DATACONFIGS, policy=POLICY, seed=SEED, records=BATCH_SIZE)
    assert (manifest['steps'], manifest['batch_size'], manifest['workers']) == (1, BATCH_SIZE, 2)
    assert manifest['segment_records'] == 2 and manifest['segments'] == 2 * RECIPE_STREAMS    # 3 records: 2 + 1
    printed = capsys.readouterr().out
    assert f"wrote {tmp_path / 'one'}: 64 records" in printed and 'records/s, ETA' in printed
    assert main(['cache', '--out', str(tmp_path / 'one'), '--steps', '1']) == 2
    assert 'exists' in capsys.readouterr().err


def test_an_existing_cache_or_an_unfinished_one_is_not_a_cache(cache, tmp_path):
    with pytest.raises(FileExistsError):
        write_record_cache(cache, dataconfigs=DATACONFIGS, policy=UNSELECTED, seed=SEED, steps=1, batch_size=1)
    (tmp_path / 'partial').mkdir()
    with pytest.raises(ConfigurationError, match='not a complete record cache'):
        CachedRecordSource(tmp_path / 'partial').manifest


def test_a_policy_without_novel_alleles_keeps_the_manifest_fields_of_before():
    from alignair_lab.records import policy_fields
    assert set(policy_fields(POLICY)) == {"progresses", "heavy_shm", "short_boost", "mutation_cap", "mode",
                                          "indel_fraction", "indel_form", "selected_fraction"}


def test_a_policy_with_inverted_ds_records_their_share_in_the_manifest():
    from alignair_lab.records import policy_fields
    fields = policy_fields(dataclasses.replace(POLICY, inverted_d=0.05))
    assert fields["inverted_d"] == 0.05
    assert set(fields) - {"inverted_d"} == set(policy_fields(POLICY))


def test_a_novel_cache_written_in_segments_is_the_serial_stream(tmp_path):
    from alignair.reference.variants import NOVEL_FIELD
    from alignair.train.gym.novel import NovelAlleles
    spec = NovelAlleles(epoch_records=2)
    # the unselected streams (selected ones are about 20 times slower to simulate, and their shift across an epoch is
    # tests/alignair/train/gym/test_novel_streams.py's): each holds 10 records, its segments of 3 cross epochs of 2
    policy = dataclasses.replace(UNSELECTED, novel_alleles=spec)
    path = tmp_path / "novel"
    manifest = write_record_cache(path, dataconfigs=DATACONFIGS, policy=policy, seed=SEED, steps=1,
                                  batch_size=BATCH_SIZE, workers=2, segment_records=3)
    assert manifest["policy"]["novel_alleles"] == spec.to_dict()
    source = CachedRecordSource(path)
    assert all(stream["segments"] > 1 for stream in manifest["components"])
    source.require(DATACONFIGS, policy=policy, seed=SEED, records=BATCH_SIZE)
    served = live(BATCH_SIZE, policy=policy, record_source=source)
    assert served == live(BATCH_SIZE, policy=policy) and any(NOVEL_FIELD in r for r in served)


def test_a_cache_of_another_novel_allele_spec_is_refused(tmp_path):
    from alignair.train.gym.novel import NovelAlleles
    policy = dataclasses.replace(UNSELECTED, novel_alleles=NovelAlleles(epoch_records=2))
    write_record_cache(tmp_path / "novel", dataconfigs=DATACONFIGS, policy=policy, seed=SEED, steps=1,
                       batch_size=BATCH_SIZE, workers=2)
    other = dataclasses.replace(UNSELECTED, novel_alleles=NovelAlleles(gene_fraction=0.3, epoch_records=2))
    with pytest.raises(ConfigurationError) as caught:
        CachedRecordSource(tmp_path / "novel").require(DATACONFIGS, policy=other, seed=SEED, records=BATCH_SIZE)
    assert caught.value.context["field"] == "novel_alleles"
