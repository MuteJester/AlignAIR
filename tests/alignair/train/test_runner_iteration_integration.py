"""Every runner saves the actual completed step when its record source ends early."""
from itertools import islice

import pytest

from alignair.core import AlignAIR
from alignair.reference import ReferenceSet
from alignair.train import trainer
from alignair.train.config import TrainingConfig
from alignair.train.records.streams import training_records, CurriculumStreamPolicy
from alignair.train.checkpoints import load_training_checkpoint
from alignair.core import prepare_model

pytestmark = pytest.mark.integration


@pytest.mark.parametrize('source_size', [0, 1, 3])
def test_exhausted_source_saves_completed_steps(source_size, tmp_path, monkeypatch):
    import GenAIRR as ga
    source = training_records(('HUMAN_IGH_OGRDB',), policy=CurriculumStreamPolicy(), seed=21)
    record = next(islice(source, 1))
    setup = prepare_model(ReferenceSet.from_dataconfigs(ga.HUMAN_IGH_OGRDB), trunk_channels=8, embed_dim=4)
    model = AlignAIR(setup.config)
    model.bind_reference(setup.reference)
    config = TrainingConfig(dataconfigs=(ga.HUMAN_IGH_OGRDB,), steps=5, batch_size=2)
    expected_step = source_size // 2

    monkeypatch.setattr(trainer, 'training_records', lambda *args, **kwargs: iter([record] * source_size))
    path = tmp_path / 'model.alignair'
    trainer._train(setup, model, ga.HUMAN_IGH_OGRDB, config, CurriculumStreamPolicy(progresses=(.3,), heavy_shm=0.),
                   save_path=path, log_every=1000, save_every=100)
    saved = load_training_checkpoint(path)
    assert saved.step == expected_step
