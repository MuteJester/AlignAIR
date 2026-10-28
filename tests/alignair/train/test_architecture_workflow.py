"""A replacement architecture must train, reload, inspect and resume through the public workflow."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from alignair import TrainingConfig, run_training
from alignair.core import ModelComponents, register_architecture
from alignair.core import components
from alignair.core.gene_branch import GeneBranch
from alignair.data import SequenceInputAdapter
from alignair.data.tokenizer import AAProteinTokenizer, CenterPaddedTokenizer
from alignair.model_file import read_training_state
from alignair.predict import class_probabilities, component_features
from alignair.errors import ConfigurationError

pytestmark = pytest.mark.integration


class MeanTower(torch.nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.projection = torch.nn.Linear(channels, width)

    def forward(self, embeddings):
        return self.projection(embeddings.mean(dim=1))


def mean_branch(spec, cfg):
    return GeneBranch(spec, cfg,
        segmentation_tower=MeanTower(cfg.embed_dim, cfg.block_out),
        classification_tower=MeanTower(cfg.embed_dim, cfg.cls_spec(spec)[0]))


def test_training_cannot_save_a_different_construction_or_loss_contract(tmp_path, monkeypatch):
    import GenAIRR.data as gd
    from alignair.core import prepare_model
    from alignair.reference import ReferenceSet
    from alignair.train import trainer
    source = gd.HUMAN_IGH_OGRDB
    setup = prepare_model(ReferenceSet.from_dataconfigs(source), embed_dim=4, filters=4, block_out=8)
    model = setup.build_model()
    changed = replace(setup.config, short_d_penalty=True)
    def forbidden(*args, **kwargs):
        pytest.fail('mismatched configuration reached data or optimizer setup')
    monkeypatch.setattr(trainer, 'prepare_training_data', forbidden)
    monkeypatch.setattr(trainer, 'build_training_engine', forbidden)
    path = tmp_path / 'model.alignair'
    with pytest.raises(ConfigurationError, match='constructed model') as caught:
        trainer.train(model, setup.reference, [source], changed, steps=1, save_path=path)
    assert caught.value.context['field'] == 'config' and not path.exists()


@pytest.mark.parametrize('space,labels', [('nucleotide', 'alleles'), ('protein', 'protein_classes')])
def test_registered_replacement_architecture_uses_the_complete_shared_workflow(space, labels, tmp_path, monkeypatch):
    monkeypatch.setattr(components, '_ARCHITECTURES', dict(components._ARCHITECTURES))
    name = 'test-architecture-workflow-v1'
    register_architecture(name, replace(ModelComponents(), gene_branch=mean_branch))
    request = TrainingConfig.from_genairr('HUMAN_IGH_OGRDB', input_space=space, label_space=labels,
        steps=2, batch_size=2, device='cpu', seed=31, val_every=1,
        model_options={'architecture': name, 'embed_dim': 4, 'filters': 4, 'block_out': 8})
    observed = []
    run = run_training(request, output_dir=tmp_path / 'first',
                       on_iteration=lambda item: observed.append(item.result.total))
    assert len(observed) == 2 and np.isfinite(observed).all()
    predictor = run.best_aligner(device='cpu')
    model = predictor.model
    assert model.cfg.architecture == name
    assert all(isinstance(branch.seg_tower, MeanTower) and isinstance(branch.cls_tower, MeanTower)
               for branch in model.branches.values())
    sequences = ['ACGTN', 'AC'] if space == 'nucleotide' else ['ACDX*', 'EVQL']
    raw = predictor.predict_raw(sequences, batch_size=1)
    for spec in model.cfg.gene_specs:
        assert raw[model.cfg.classification_key(spec.name)].shape == (2, spec.allele_count)
    tokenizer = CenterPaddedTokenizer() if space == 'nucleotide' else AAProteinTokenizer()
    adapter = SequenceInputAdapter(tokenizer, model.cfg.max_seq_length, vocab_size=model.cfg.vocab_size)
    if space == 'protein':
        probabilities = class_probabilities(model, sequences, adapter, keys=('v_class_logits',), batch_size=1)
        np.testing.assert_allclose(probabilities['v_class_logits'].sum(1), 1., atol=1e-6)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    modes = [module.training for module in model.modules()]
    features = component_features(model, sequences, adapter, components=('v.cls_head', 'j.cls_head'))
    assert features.shape[0] == len(sequences) and np.isfinite(features).all()
    assert [module.training for module in model.modules()] == modes
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    completed = read_training_state(run.model_path)
    assert completed.step == 2 and completed.optimizer_state['state']
    steps = []
    resumed = run_training(request.with_(steps=3, resume_path=run.model_path), output_dir=tmp_path / 'resumed',
                           on_iteration=lambda item: steps.append(item.step))
    assert steps == [3] and resumed.steps == 3
    restored = resumed.best_aligner(device='cpu')
    assert restored.label_schema == predictor.label_schema
    assert isinstance(restored.model.branches['v'].cls_tower, MeanTower)
    actual = read_training_state(resumed.model_path)
    assert actual.step == 3
    assert all(torch.isfinite(value).all() for value in actual.model_state.values() if value.is_floating_point())
