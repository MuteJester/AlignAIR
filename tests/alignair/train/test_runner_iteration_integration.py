"""Every runner saves the actual completed step when its record source ends early."""
from dataclasses import replace
from itertools import islice

import pytest

from alignair.core import AlignAIR
from alignair.data.sampling import ProteinSampleBuilder, ProteinSamplePolicy
from alignair.data.tokenizer import AAProteinTokenizer
from alignair.reference import build_protein_reference
from alignair.tasks import build_tasks
from alignair.train import trainer
from alignair.train.gym.streams import training_records, CurriculumStreamPolicy
from alignair.train.checkpoints import load_training_checkpoint
from alignair.data.augmentation import ProteinCropPolicy
from alignair.core import prepare_model

pytestmark = pytest.mark.integration


@pytest.mark.parametrize('runner', ['nucleotide', 'nucleotide_auto_weights', 'protein_alleles', 'protein_classes', 'protein_crops'])
@pytest.mark.parametrize('source_size', [0, 1, 3])
def test_exhausted_source_saves_completed_steps(runner, source_size, tmp_path, monkeypatch):
    reference = build_protein_reference('HUMAN_IGH_OGRDB')
    builder = ProteinSampleBuilder(reference.class_maps, AAProteinTokenizer(), 192,
        reference.source.gene('D').index, policy=ProteinSamplePolicy(False, True))
    source = training_records(('HUMAN_IGH_OGRDB',), policy=CurriculumStreamPolicy(mode='amplicon'), seed=21, translated_only=True)
    record = next(record for record in islice(source, 50) if builder(record).sample is not None)

    nucleotide = runner.startswith('nucleotide')
    setup = prepare_model(reference.source, input_space='nucleotide' if nucleotide else 'protein',
                          label_space='protein_classes' if runner in ('protein_classes', 'protein_crops') else 'alleles')
    ref, cfg = setup.reference, setup.config
    cfg = replace(cfg, embed_dim=4, filters=4, block_out=8)
    model = AlignAIR(cfg)
    options = dict(steps=5, batch_size=2, log_every=1000, save_every=100)
    expected_step = source_size // 2

    import GenAIRR as ga
    monkeypatch.setattr(trainer, 'training_records', lambda *args, **kwargs: iter([record] * source_size))
    path = tmp_path / 'model.alignair'
    weights = None if runner == 'nucleotide_auto_weights' else build_tasks(cfg).make_weights()
    trainer.train(model, ref, ga.HUMAN_IGH_OGRDB, cfg, weights,
                  progresses=(.3,), heavy_shm=0., save_path=path,
                  augmentation=ProteinCropPolicy(crop_probability=1.) if runner == 'protein_crops' else None, **options)
    saved = load_training_checkpoint(path)
    assert saved.step == expected_step and saved.weight_state is not None
