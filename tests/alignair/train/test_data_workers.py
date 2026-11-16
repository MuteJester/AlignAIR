"""Training with data workers is serial training: same batches and weights, any worker count on resume,
and no worker outlives the run however it ends."""
import multiprocessing

import pytest
import torch

from alignair import TrainingConfig, run_training
from alignair.errors import ConfigurationError
from alignair.model_file import load_model, read_metadata
from alignair.train.engine import TrainingEngine
from alignair.train.random_state import capture_rng_state
from tests.support.training_streams import assert_same_batch, same_rng

pytestmark = pytest.mark.integration


def request(**options):
    return TrainingConfig.from_genairr('HUMAN_IGH_OGRDB', **{
        'steps': 3, 'batch_size': 4, 'device': 'cpu', 'seed': 5,
        'model_options': {'embed_dim': 4, 'trunk_channels': 8}, **options})


def weights(path):
    return load_model(path).model.state_dict()


def assert_same_weights(left, right):
    assert left.keys() == right.keys()
    for key, value in left.items():
        assert torch.equal(value, right[key]), key


def trained(config, path):
    records = []
    run = run_training(config, out_path=str(path), on_iteration=lambda it: records.append((it.step, list(it.records))))
    return run, records


def test_parallel_cpu_training_reproduces_serial_weights_and_records_the_worker_count(tmp_path):
    serial, serial_records = trained(request(), tmp_path / 'serial.alignair')
    parallel, parallel_records = trained(request(data_workers=3), tmp_path / 'parallel.alignair')
    assert parallel_records == serial_records and parallel.steps == serial.steps == 3
    assert_same_weights(weights(serial.model_path), weights(parallel.model_path))
    assert read_metadata(parallel.model_path)['training']['train_args']['data_workers'] == 3
    assert read_metadata(serial.model_path)['training']['train_args']['data_workers'] == 0


def test_parallel_training_with_flanks_reproduces_the_serial_flanked_records_and_weights(tmp_path):
    """Flanks are drawn after the provider, so worker-built streams carry the serial run's flanks."""
    from alignair.data.flanks import FlankPolicy
    flanks = FlankPolicy(p_prefix=0.5, prefix=(5, 60), p_mask=0.5, p_suffix=0.5, suffix=(5, 40))
    serial, serial_records = trained(request(flanks=flanks), tmp_path / 'serial.alignair')
    parallel, parallel_records = trained(request(flanks=flanks, data_workers=3), tmp_path / 'parallel.alignair')
    assert parallel_records == serial_records
    assert any(record['flank_5_length'] for _, records in serial_records for record in records)
    assert_same_weights(weights(serial.model_path), weights(parallel.model_path))


def test_resume_accepts_another_worker_count_and_continues_the_serial_stream(tmp_path):
    first = run_training(request(steps=2), out_path=str(tmp_path / 'first.alignair'))
    serial, serial_records = trained(request(resume_path=first.model_path), tmp_path / 'serial.alignair')
    parallel, parallel_records = trained(request(resume_path=first.model_path, data_workers=2),
                                         tmp_path / 'parallel.alignair')
    assert [step for step, _ in parallel_records] == [3] and parallel_records == serial_records
    assert_same_weights(weights(serial.model_path), weights(parallel.model_path))
    resumed_again = run_training(request(steps=4, resume_path=parallel.model_path),
                                 out_path=str(tmp_path / 'again.alignair'))
    assert resumed_again.steps == 4


def test_workers_stop_when_training_finishes_fails_or_is_interrupted(tmp_path, monkeypatch):
    before = set(multiprocessing.active_children())
    seen = []

    def watch(iteration):
        seen.append(set(multiprocessing.active_children()) - before)

    def gone():
        started = set().union(*seen)
        seen.clear()
        return len(started) == 2 and all(not process.is_alive() for process in started) \
            and set(multiprocessing.active_children()) == before

    run_training(request(data_workers=2), out_path=str(tmp_path / 'finished.alignair'), on_iteration=watch)
    assert gone()

    def interrupt(iteration):
        watch(iteration)
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_training(request(data_workers=2), out_path=str(tmp_path / 'interrupted.alignair'), on_iteration=interrupt)
    assert gone()

    original = TrainingEngine.step

    def failing_step(self, batch, *, step=0):
        if step == 2:
            raise RuntimeError('step failed')
        return original(self, batch, step=step)
    monkeypatch.setattr(TrainingEngine, 'step', failing_step)
    with pytest.raises(RuntimeError, match='step failed'):
        run_training(request(data_workers=2), out_path=str(tmp_path / 'failed.alignair'), on_iteration=watch)
    assert gone()


@pytest.mark.parametrize('value', [-1, True, 1.5, '2', None])
def test_the_worker_count_is_a_non_negative_integer(value, tmp_path):
    with pytest.raises(ConfigurationError) as caught:
        request(data_workers=value)
    assert caught.value.context['field'] == 'data_workers'


def test_the_cli_passes_the_worker_count_to_training(tmp_path, monkeypatch):
    from alignair.cli.main import main
    from alignair.train import workflow
    from alignair.train.guards import TrainingConfigError
    requested = []

    def capture(config, **_):
        requested.append(config.data_workers)
        raise TrainingConfigError('captured')
    monkeypatch.setattr(workflow, 'run_training', capture)
    for extra, expected in (([], 0), (['--data-workers', '7'], 7)):
        assert main(['train', '--dataconfig', 'HUMAN_IGH_OGRDB', '--out', str(tmp_path / str(expected)),
                     '--preset', 'quick', '--device', 'cpu', *extra]) == 1
        assert requested.pop() == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason='batch assembly in a worker serves accelerator training')
def test_cuda_training_with_assembled_batches_sees_the_serial_batches(tmp_path, monkeypatch):
    original = TrainingEngine.step

    def run(**options):
        steps = []

        def recording(self, batch, *, step=0):
            steps.append((step, batch, capture_rng_state(include_cuda=False)))
            return original(self, batch, step=step)
        monkeypatch.setattr(TrainingEngine, 'step', recording)
        trained(request(device='cuda', steps=6, batch_size=8, **options), tmp_path / f'{len(options)}.alignair')
        return steps

    serial, parallel = run(), run(data_workers=3)
    assert [step for step, _, _ in parallel] == [step for step, _, _ in serial] == list(range(1, 7))
    for (_, expected, expected_rng), (_, actual, actual_rng) in zip(serial, parallel):
        assert_same_batch(expected, actual)
        assert same_rng(expected_rng, actual_rng)
