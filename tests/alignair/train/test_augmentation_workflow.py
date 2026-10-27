"""Protein substitution experiments use the public training/checkpoint/validation lifecycle."""
from itertools import islice

import numpy as np
import pytest
import torch

from alignair import TrainingConfig, run_training
from alignair.data.augmentation import ProteinSubstitutionPolicy
from alignair.data.training import prepare_training_data
from alignair.errors import ConfigurationError
from alignair.model_file import read_metadata
from alignair.train import trainer
from alignair.train.checkpoints import load_training_checkpoint
from alignair.train.engine import TrainingEngine
from alignair.train.gym.streams import CurriculumStreamPolicy, training_records
from tests.support.model_setup import protein_class_setup

pytestmark = pytest.mark.integration


def request(**fields):
    return TrainingConfig.from_genairr('HUMAN_IGH_OGRDB', **{
        'input_space': 'protein', 'label_space': 'protein_classes', 'steps': 2, 'batch_size': 2,
        'device': 'cpu', 'seed': 1, 'grad_clip': 1., 'augmentation': ProteinSubstitutionPolicy(),
        'model_options': {'embed_dim': 8, 'filters': 8, 'block_out': 32}, **fields})


def test_reference_bound_class_model_retains_established_head_sizes():
    setup = protein_class_setup()
    assert setup.label_counts == {'v': 149, 'd': 33, 'j': 7}
    model = setup.build_model()
    assert set(model.branches) == {'v', 'd', 'j'}
    outputs = model({'tokenized_sequence': torch.randint(1, 23, (2, 192))})
    assert all(outputs[f'{gene}_class_logits'].shape == (2, count) for gene, count in setup.label_counts.items())


@pytest.mark.slow
def test_substitution_experiment_learns_through_public_training(tmp_path):
    history = []
    run = run_training(request(steps=30, batch_size=16), output_dir=tmp_path,
                       on_iteration=lambda item: history.append(item.result))
    assert run.steps == len(history) == 30
    assert all(np.isfinite(item.total) for item in history)
    assert set(history[-1].parts) >= {'segmentation', 'classification'}
    assert history[-1].total < history[0].total
    assert run.best_aligner(device='cpu').label_schema['label_space'] == 'protein_classes'


def test_validation_is_unaugmented_and_training_provenance_is_saved(tmp_path, monkeypatch):
    setup = protein_class_setup()
    data = prepare_training_data(setup)
    source = training_records(('HUMAN_IGH_OGRDB',), policy=CurriculumStreamPolicy(mode='amplicon'), seed=21)
    record = next(record for record in islice(source, 100) if data.sample_builder(record).sample is not None)
    sample = data.sample_builder(record).sample
    expected = data.validation_adapter([sample, sample]).inputs['tokenized_sequence']
    original = TrainingEngine.validate
    validation_calls, training_tokens = [], []

    def validate(engine, batches):
        batches = list(batches)
        assert all(torch.equal(batch.inputs['tokenized_sequence'], expected) for batch in batches)
        validation_calls.append(len(batches))
        return original(engine, batches)

    monkeypatch.setattr(TrainingEngine, 'validate', validate)
    monkeypatch.setattr(trainer, 'training_records', lambda *args, **kw: iter([record] * 8))
    policy = ProteinSubstitutionPolicy(1, 1, 1)
    run = run_training(request(val_every=1, augmentation=policy), output_dir=tmp_path,
        on_iteration=lambda item: training_tokens.extend(row['tokens'] for row in item.records))
    assert validation_calls == [4, 4] and run.best_model_path
    assert training_tokens and all(not np.array_equal(tokens, sample['tokens']) for tokens in training_tokens)
    state = load_training_checkpoint(run.model_path)
    assert state.augmentation == policy.to_dict() and state.optimizer_state['state']
    assert read_metadata(run.model_path)['training']['train_args']['augmentation'] == policy.to_dict()


def test_augmentation_resume_reseeds_source_and_local_generator_deterministically(tmp_path):
    config = request()
    first = run_training(config, output_dir=tmp_path / 'first')
    one = run_training(config.with_(steps=3, resume_path=first.model_path), output_dir=tmp_path / 'one')
    two = run_training(config.with_(steps=3, resume_path=first.model_path), output_dir=tmp_path / 'two')
    assert one.steps == two.steps == 3
    first_state, second_state = load_training_checkpoint(one.model_path), load_training_checkpoint(two.model_path)
    assert first_state.augmentation == config.augmentation.to_dict() == second_state.augmentation
    assert all(torch.equal(value, second_state.model_state[key]) for key, value in first_state.model_state.items())


@pytest.mark.parametrize('initial,requested', [(None, ProteinSubstitutionPolicy()),
    (ProteinSubstitutionPolicy(), None), (ProteinSubstitutionPolicy(), ProteinSubstitutionPolicy(.7))])
def test_resume_refuses_changed_augmentation_before_loading_weights(tmp_path, monkeypatch, initial, requested):
    first = run_training(request(steps=1, augmentation=initial), output_dir=tmp_path / 'first')
    from alignair.core import ModelSetup
    original = ModelSetup.build_model
    allocated = []
    def build(setup):
        model = original(setup)
        allocated.append((model, {key: value.clone() for key, value in model.state_dict().items()}))
        return model
    monkeypatch.setattr(ModelSetup, 'build_model', build)
    with pytest.raises(ConfigurationError, match='augmentation policy differs'):
        run_training(request(augmentation=requested, resume_path=first.model_path), output_dir=tmp_path / 'rejected')
    assert allocated
    for model, before in allocated:
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_nucleotide_augmentation_is_rejected_before_allocation_or_output(tmp_path, monkeypatch):
    from alignair.core import ModelSetup
    monkeypatch.setattr(ModelSetup, 'build_model', lambda self: pytest.fail('allocated invalid model'))
    config = request(input_space='nucleotide', label_space='alleles')
    with pytest.raises(ConfigurationError, match='protein-class'):
        run_training(config, output_dir=tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
