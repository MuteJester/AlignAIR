"""``train`` and ``run_training`` draw every curriculum record from a ``record_source`` provider when given one,
in the training process and in data workers alike."""
from itertools import islice

import pytest

from alignair import TrainingConfig, run_training
from alignair.data.flanks import FlankPolicy
from alignair.errors import ConfigurationError
from alignair.train.records.streams import CurriculumStreamPolicy, stream_records, training_records
from alignair.train.records.workers import component_plan
from tests.support.record_workers import shifted

pytestmark = pytest.mark.integration

STEPS, BATCH, SEED = 3, 4, 5


def request(**options):
    return TrainingConfig.from_genairr('HUMAN_IGH_OGRDB', **{
        'steps': STEPS, 'batch_size': BATCH, 'device': 'cpu', 'seed': SEED,
        'model_options': {'embed_dim': 4, 'trunk_channels': 8}, **options})


def stream(record_source=None):
    """The training stream ``run_training`` draws: the default curriculum with the default library flanks."""
    import GenAIRR.data as gd
    records = training_records((gd.HUMAN_IGH_OGRDB,), policy=CurriculumStreamPolicy(flanks=FlankPolicy()), seed=SEED,
                               record_source=record_source)
    return list(islice(records, STEPS * BATCH))


@pytest.mark.parametrize('workers', [0, 2])
def test_training_records_come_from_the_record_source(tmp_path, workers):
    seen = []
    run_training(request(data_workers=workers), out_path=str(tmp_path / 'model.alignair'),
                 on_iteration=lambda iteration: seen.extend(iteration.records), record_source=shifted)
    assert seen == stream(shifted) and seen != stream()


def test_validation_records_come_from_the_record_source_too(tmp_path):
    seeds = []

    def watched(dataconfig, parameters, seed):
        seeds.append(seed)
        return stream_records(dataconfig, parameters, seed)

    import GenAIRR.data as gd
    run_training(request(val_every=2), out_path=str(tmp_path / 'model.alignair'), record_source=watched)
    training, validation = (component_plan((gd.HUMAN_IGH_OGRDB,), policy=CurriculumStreamPolicy(), seed=SEED,
                                           purpose=purpose) for purpose in ('training', 'validation'))
    assert sorted(seeds) == sorted(component.seed for component in (*training, *validation))
    assert not {component.seed for component in training} & {component.seed for component in validation}


@pytest.mark.parametrize('workers,provider,problem', [
    (0, 42, 'record_source must be callable'),
    (2, lambda dataconfig, parameters, seed: iter(()), 'cannot receive this provider'),
])
def test_an_unusable_record_source_is_refused_before_training(tmp_path, workers, provider, problem):
    steps = []
    with pytest.raises(ConfigurationError, match=problem):
        run_training(request(data_workers=workers), out_path=str(tmp_path / 'model.alignair'),
                     on_iteration=steps.append, record_source=provider)
    assert steps == [] and not (tmp_path / 'model.alignair').exists()
