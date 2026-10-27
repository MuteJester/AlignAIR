"""Every runner saves the actual completed step when its record source ends early."""
from dataclasses import replace
from itertools import islice

import pytest
import torch

from alignair.core import AlignAIR
from alignair.data.sampling import ProteinSampleBuilder, ProteinSamplePolicy
from alignair.data.tokenizer import AAProteinTokenizer
from alignair.reference import build_protein_reference
from alignair.tasks import build_tasks
from alignair.train import trainer
from alignair.train.gym.streams import training_records, CurriculumStreamPolicy
from alignair.train.checkpoints import load_training_checkpoint
from alignair.aa import train_gym, strategic, strategic_trunc

pytestmark = pytest.mark.integration


@pytest.mark.parametrize('runner', ['nucleotide', 'protein_alleles', 'protein_classes', 'protein_crops'])
@pytest.mark.parametrize('source_size', [0, 1, 3])
def test_exhausted_source_saves_completed_steps(runner, source_size, tmp_path, monkeypatch):
    reference = build_protein_reference('HUMAN_IGH_OGRDB')
    builder = ProteinSampleBuilder(reference.class_maps, AAProteinTokenizer(), 192,
        reference.source.gene('D').index, policy=ProteinSamplePolicy(False, True))
    source = training_records(('HUMAN_IGH_OGRDB',), policy=CurriculumStreamPolicy(mode='amplicon'), seed=21, translated_only=True)
    record = next(record for record in islice(source, 50) if builder(record).sample is not None)

    if runner in ('nucleotide', 'protein_alleles'):
        _, ref, cfg, _ = train_gym.setup_aa(max_seq_length=576 if runner == 'nucleotide' else 192)
        if runner == 'nucleotide':
            cfg = replace(cfg, vocab_size=6, use_orientation=True)
    else:
        _, cfg, maps, meta = strategic.strategic_setup(max_seq_length=192)
    cfg = replace(cfg, embed_dim=4, filters=4, block_out=8)
    model = AlignAIR(cfg)
    options = dict(steps=5, batch_size=2, log_every=1000, save_every=100)
    expected_step = source_size // 2

    if runner == 'nucleotide':
        import GenAIRR as ga
        monkeypatch.setattr(trainer, 'training_records', lambda *args, **kwargs: iter([record] * source_size))
        path = tmp_path / 'model.alignair'
        trainer.train(model, ref, ga.HUMAN_IGH_OGRDB, cfg, build_tasks(cfg).make_weights(),
                      progresses=(.3,), heavy_shm=0., save_path=path, **options)
        saved = load_training_checkpoint(path)
        assert saved.step == expected_step
    else:
        module = {'protein_alleles': train_gym, 'protein_classes': strategic,
                  'protein_crops': strategic_trunc}[runner]
        monkeypatch.setattr(module, 'training_records', lambda *args, **kwargs: iter([record] * source_size))
        path = tmp_path / 'model.pt'
        if runner == 'protein_alleles':
            history = train_gym.train_aa(model, ref, 'HUMAN_IGH_OGRDB', cfg,
                                        build_tasks(cfg).make_weights(), save_path=path, **options)
        elif runner == 'protein_classes':
            history = strategic.strategic_train(model, cfg, maps, d_index=meta['d_index'],
                                               save_path=path, **options)
        else:
            history = strategic_trunc.stage1_train(model, cfg, maps, meta, p_crop=1.,
                                                  save_path=path, **options)
        assert len(history) == expected_step
        assert torch.load(path, weights_only=True)['step'] == expected_step
