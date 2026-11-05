"""Every runner saves the actual completed step when its record source ends early."""
from dataclasses import replace
from itertools import islice

import pytest

from alignair.core import AlignAIR
from alignair.reference import ReferenceSet
from alignair.tasks import build_tasks
from alignair.train import trainer
from alignair.train.gym.streams import training_records, CurriculumStreamPolicy
from alignair.train.checkpoints import load_training_checkpoint
from alignair.core import prepare_model

pytestmark = pytest.mark.integration


@pytest.mark.parametrize('runner', ['nucleotide', 'nucleotide_auto_weights'])
@pytest.mark.parametrize('source_size', [0, 1, 3])
def test_exhausted_source_saves_completed_steps(runner, source_size, tmp_path, monkeypatch):
    import GenAIRR as ga
    source = training_records(('HUMAN_IGH_OGRDB',), policy=CurriculumStreamPolicy(mode='amplicon'), seed=21)
    record = next(islice(source, 1))
    setup = prepare_model(ReferenceSet.from_dataconfigs(ga.HUMAN_IGH_OGRDB), trunk_channels=8)
    ref, cfg = setup.reference, replace(setup.config, embed_dim=4)
    model = AlignAIR(cfg)
    model.bind_reference(ref)
    options = dict(steps=5, batch_size=2, log_every=1000, save_every=100)
    expected_step = source_size // 2

    monkeypatch.setattr(trainer, 'training_records', lambda *args, **kwargs: iter([record] * source_size))
    path = tmp_path / 'model.alignair'
    weights = None if runner == 'nucleotide_auto_weights' else build_tasks(cfg).make_weights()
    trainer.train(model, ref, ga.HUMAN_IGH_OGRDB, cfg, weights,
                  progresses=(.3,), heavy_shm=0., save_path=path, **options)
    saved = load_training_checkpoint(path)
    assert saved.step == expected_step and saved.weight_state is not None
