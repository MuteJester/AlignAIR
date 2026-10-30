"""Partial protein supervision, study callbacks and cosine continuation use one public lifecycle."""
from dataclasses import asdict

import numpy as np
import pytest
import torch

from alignair import TrainingConfig, run_training
from alignair.data.augmentation import ProteinCropPolicy, SAMPLE_POLICY_ID
from alignair.errors import ConfigurationError
from alignair.model_file import container, read_metadata
from alignair.train import CosineSchedule, TrainingStudy, trainer
from alignair.train.checkpoints import load_training_checkpoint, save_research_checkpoint, save_snapshot
from tests.support.model_setup import protein_class_setup

pytestmark = pytest.mark.integration


def request(**fields):
    return TrainingConfig.from_genairr('HUMAN_IGH_OGRDB', **{
        'input_space': 'protein', 'label_space': 'protein_classes', 'steps': 2, 'batch_size': 2,
        'lr': 1e-4, 'grad_clip': 1., 'seed': 1, 'device': 'cpu', 'augmentation': ProteinCropPolicy(),
        'schedule': CosineSchedule(12), 'model_options': {'embed_dim': 8, 'filters': 8, 'block_out': 32}, **fields})


@pytest.mark.slow
def test_crop_training_learns_with_partial_class_targets_and_checkpoint_reload(tmp_path):
    history, reasons = [], []
    def observe(item):
        history.append(item.result.total)
        reasons.extend(row['v_class_reason'] for row in item.records)
    run = run_training(request(steps=40, batch_size=16, schedule=CosineSchedule(40)),
                       output_dir=tmp_path, on_iteration=observe)
    # learning, not one batch's luck: the last ten steps' mean loss is below the first ten steps'
    assert run.steps == len(history) == 40 and np.isfinite(history).all()
    assert np.mean(history[-10:]) < np.mean(history[:10])
    assert 'source_label' in reasons and 'reference_fragments' in reasons
    predictor = run.best_aligner(device='cpu')
    assert predictor.model.cfg.label_space == 'protein_classes'
    assert predictor.predict_raw(['EVQL'])['v_class_logits'].shape == (1,149)


def test_cosine_optimizer_and_policy_state_continue_without_resetting_decayed_lr(tmp_path):
    config = request(steps=6)
    first = run_training(config, output_dir=tmp_path/'first')
    before = load_training_checkpoint(first.model_path)
    assert before.scheduler_state['last_epoch'] == 6
    assert before.optimizer_state['param_groups'][0]['lr'] == pytest.approx(.000055)
    run = run_training(config.with_(steps=12, resume_path=first.model_path), output_dir=tmp_path/'continued')
    after = load_training_checkpoint(run.model_path)
    assert run.steps == after.step == after.scheduler_state['last_epoch'] == 12
    assert after.optimizer_state['param_groups'][0]['lr'] == pytest.approx(.00001)
    assert after.schedule == config.schedule.to_dict() and after.augmentation == config.augmentation.to_dict()
    assert all(value['step'].item() == 12 for value in after.optimizer_state['state'].values())
    arguments = read_metadata(run.model_path)['training']['train_args']
    assert len(arguments['source_checkpoint_sha256']) == 64


def test_study_callbacks_restore_mode_and_selected_checkpoints_survive_rotation(tmp_path, monkeypatch):
    from alignair.core import ModelSetup
    original = ModelSetup.build_model
    modes, validation_modes = [], []
    def build(setup):
        model = original(setup)
        model.register_forward_pre_hook(lambda module, _: modes.append(module.training))
        return model
    def evaluate(model, step):
        validation_modes.append(model.training)
        model.eval()
        return {'objective': float(step)}
    monkeypatch.setattr(ModelSetup, 'build_model', build)
    config = request(steps=4, study=TrainingStudy(tmp_path, (1,2,3,4), 1, evaluate))
    run = run_training(config, output_dir=tmp_path)
    assert modes == [True]*4 and validation_modes == [False]*4
    assert set(run.checkpoints) == {1,2,3,4} and len(run.validations) == 4
    assert all(load_training_checkpoint(path).step == step for step,path in run.checkpoints.items())
    # Routine rotating checkpoints have a different naming contract from retained study artifacts.
    from alignair.train.checkpoints import save_rotating
    predictor = run.best_aligner(device='cpu')
    import GenAIRR.data as gd
    for step in (5,6,7,8):
        save_rotating(tmp_path/'model.alignair', predictor.model.cfg, predictor.model, None, step, None,
                      dataconfigs=[gd.HUMAN_IGH_OGRDB], label_schema=predictor.label_schema, keep=1)
    assert all(container.is_alignair_file(path) for path in run.checkpoints.values())


@pytest.mark.parametrize('changes', [{'schedule': None}, {'schedule': CosineSchedule(20)},
    {'schedule': CosineSchedule(12, .00002)}, {'lr': .0002}, {'augmentation': ProteinCropPolicy(.5)}])
def test_changed_schedule_or_crop_policy_cannot_resume(tmp_path, changes):
    first = run_training(request(), output_dir=tmp_path/'first')
    with pytest.raises(ConfigurationError, match='schedule|augmentation policy'):
        run_training(request(resume_path=first.model_path, steps=3, **changes), output_dir=tmp_path/'bad')


def test_legacy_crop_state_requires_trust_and_preserves_optimizer_scheduler_and_policy(tmp_path):
    setup = protein_class_setup(embed_dim=8, filters=8, block_out=32)
    model = setup.build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
    scheduler = CosineSchedule(12).build(optimizer)
    path = tmp_path/'historical.pt'
    save_research_checkpoint(path, model=model, cfg=asdict(setup.config), step=0, optimizer=optimizer,
        scheduler=scheduler, lr=.0001, history=[], provenance={'sample_policy': SAMPLE_POLICY_ID,
        'hyperparams': {'p_crop': .4, 'p_sub': .4, 'p_il': .15, 'base_lr': .0001}})
    from alignair.errors import ModelError
    with pytest.raises(ModelError, match='resume_trust_pickle'):
        load_training_checkpoint(path)
    run = run_training(request(steps=1, resume_path=path, trust_pickle=True), output_dir=tmp_path/'new')
    state = load_training_checkpoint(run.model_path)
    assert state.step == state.scheduler_state['last_epoch'] == 1
    assert state.augmentation['policy'] == SAMPLE_POLICY_ID and state.label_schema is not None


def test_old_snapshot_requires_explicit_warm_start_and_incompatible_policy_cannot_continue(tmp_path):
    setup = protein_class_setup(embed_dim=8, filters=8, block_out=32)
    model = setup.build_model()
    snapshot = tmp_path/'snapshot.pt'
    save_snapshot(snapshot, model, setup.config, 12)
    with pytest.raises(ConfigurationError, match='augmentation policy|optimizer continuation'):
        run_training(request(resume_path=snapshot, trust_pickle=True), output_dir=tmp_path/'bad')
    run = run_training(request(init_from=snapshot, trust_pickle=True), output_dir=tmp_path/'warm')
    assert run.steps == load_training_checkpoint(run.model_path).scheduler_state['last_epoch'] == 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
    scheduler = CosineSchedule(12).build(optimizer)
    invalid = tmp_path/'old_policy.pt'
    save_research_checkpoint(invalid, model=model, cfg=asdict(setup.config), step=1, optimizer=optimizer,
                            scheduler=scheduler, lr=.0001, history=[], provenance={'sample_policy': 'old-policy'})
    with pytest.raises(ConfigurationError, match='crop-label policy differs') as caught:
        run_training(request(resume_path=invalid, trust_pickle=True), output_dir=tmp_path/'old')
    assert 'init_from' in caught.value.hint


def test_resume_reseeds_corpus_and_crop_rng_from_seed_and_step(tmp_path, monkeypatch):
    first = run_training(request(), output_dir=tmp_path/'first')
    original = trainer.training_records
    seeds, rows = [], []
    def source(*args, **kw):
        seeds.append(kw['seed'])
        return original(*args, **kw)
    monkeypatch.setattr(trainer, 'training_records', source)
    config = request(steps=3, resume_path=first.model_path)
    for name in ('one','two'):
        run_training(config, output_dir=tmp_path/name,
                     on_iteration=lambda item: rows.append([row['tokens'].copy() for row in item.records]))
    assert seeds == [3,3]
    assert all(np.array_equal(first,second) for first,second in zip(rows[0],rows[1]))


@pytest.mark.parametrize('fields', [{'schedule': {}}, {'study': {}}, {'trust_pickle': 1},
                                  {'init_from': 'a', 'resume_path': 'b'}])
def test_invalid_public_study_request_is_rejected(fields):
    with pytest.raises(ConfigurationError):
        request(**fields)


@pytest.mark.parametrize('space', ['nucleotide', 'protein'])
def test_cosine_and_retained_study_checkpoints_are_shared_with_allele_models(tmp_path, space):
    config = request(input_space=space, label_space='alleles', augmentation=None,
                      study=TrainingStudy(tmp_path/'selected', (1,)))
    first = run_training(config, output_dir=tmp_path/'first')
    assert load_training_checkpoint(first.checkpoints[1]).scheduler_state['last_epoch'] == 1
    continued = run_training(config.with_(steps=3, resume_path=first.model_path), output_dir=tmp_path/'continued')
    state = load_training_checkpoint(continued.model_path)
    assert state.step == state.scheduler_state['last_epoch'] == 3 and state.weight_state
    assert state.label_schema['input_space'] == space and state.label_schema['label_space'] == 'alleles'


def test_missing_resume_checkpoint_never_starts_fresh_training(tmp_path, monkeypatch):
    def source(*args, **kwargs):
        pytest.fail('training source was constructed despite a missing resume checkpoint')
    monkeypatch.setattr(trainer, 'training_records', source)
    with pytest.raises(ConfigurationError, match='does not exist'):
        run_training(request(resume_path=tmp_path/'missing.alignair'), output_dir=tmp_path/'new')
