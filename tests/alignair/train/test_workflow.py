"""One public training/checkpoint/reload/raw-prediction workflow across supported input/label spaces."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from alignair import Aligner, TrainingConfig, TrainingRun, run_training
from alignair.errors import ConfigurationError, InputError, ModelIntegrityError
from alignair.model_file import container, load_model, read_metadata

pytestmark = pytest.mark.integration


def request(input_space='nucleotide', label_space='alleles', **options):
    return TrainingConfig.from_genairr('HUMAN_IGH_OGRDB', input_space=input_space, label_space=label_space,
        **{'steps': 2, 'batch_size': 2, 'device': 'cpu', 'grad_clip': 1.,
           'model_options': {'embed_dim': 4, 'filters': 4, 'block_out': 8}, **options})


@pytest.mark.parametrize('space,labels,has_d', [('nucleotide', 'alleles', True), ('protein', 'alleles', True),
                                             ('protein', 'protein_classes', True), ('protein', 'protein_classes', False)])
def test_public_train_validate_reload_predict_path_is_shared(space, labels, has_d, tmp_path):
    observations = []
    run = run_training(request(space, labels, has_d=has_d, val_every=1), output_dir=tmp_path,
                       on_iteration=lambda iteration: observations.append(iteration.result.total))
    assert isinstance(run, TrainingRun) and run.steps == 2 and run.best_model_path
    assert len(observations) == 2 and np.isfinite(observations).all()
    predictor = run.best_aligner(device='cpu')
    assert type(predictor) is Aligner
    assert predictor.model.cfg.input_space == space and predictor.model.cfg.label_space == labels
    assert predictor.model.cfg.has_d is has_d
    schema = predictor.label_schema
    assert schema['input_space'] == space and schema['label_space'] == labels
    assert set(schema['genes']) == ({'v', 'd', 'j'} if has_d else {'v', 'j'})
    outputs = predictor.predict_raw(iter(['ACGTN', 'AC'] if space == 'nucleotide' else ['ACDX*', 'EVQL']))
    for spec in predictor.model.cfg.gene_specs:
        assert len(schema['genes'][spec.name]) == spec.allele_count
        assert outputs[predictor.model.cfg.classification_key(spec.name)].shape == (2, spec.allele_count)
    assert ('orientation' in outputs) is (space == 'nucleotide')
    if space == 'protein':
        with pytest.raises(ConfigurationError, match='predict_raw'):
            predictor.predict(['ACDX*'])
    if labels == 'protein_classes':
        assert all(isinstance(group, list) and group for group in schema['genes']['v'])
    final = read_metadata(run.model_path)
    assert final['training']['steps'] == 2
    assert final['training']['train_args']['input_space'] == space
    assert load_model(run.model_path).config.input_space == space


@pytest.mark.parametrize('space,labels', [('nucleotide', 'alleles'), ('protein', 'alleles'), ('protein', 'protein_classes')])
def test_public_resume_retains_profile_label_identity_and_optimizer_state(space, labels, tmp_path):
    original = request(space, labels)
    first = run_training(original, output_dir=tmp_path / 'first')
    before = first.best_aligner(device='cpu').label_schema
    steps = []
    resumed = run_training(original.with_(steps=3, resume_path=first.model_path), output_dir=tmp_path / 'resumed',
                           on_iteration=lambda iteration: steps.append(iteration.step))
    assert resumed.steps == 3 and steps == [3]
    assert resumed.best_aligner(device='cpu').label_schema == before


def test_training_seed_reproduces_weights_and_raw_predictions(tmp_path):
    config = request('protein', 'protein_classes', seed=17)
    first = run_training(config, output_dir=tmp_path / 'one').best_aligner(device='cpu')
    second = run_training(config, output_dir=tmp_path / 'two').best_aligner(device='cpu')
    assert first.label_schema == second.label_schema
    for key, value in first.model.state_dict().items():
        assert torch.equal(value, second.model.state_dict()[key]), key
    one, two = first.predict_raw(['EVQL']), second.predict_raw(['EVQL'])
    assert all(np.array_equal(value, two[key]) for key, value in one.items())


def test_protein_allele_training_exercises_existing_losses_through_the_shared_runner(tmp_path):
    history = []
    run_training(request('protein', 'alleles', steps=30, batch_size=16, seed=0,
                          model_options={'embed_dim': 8, 'filters': 8, 'block_out': 32}),
                 output_dir=tmp_path, on_iteration=lambda iteration: history.append(iteration.result))
    assert len(history) == 30 and all(np.isfinite(item.total) for item in history)
    assert {'segmentation', 'classification', 'mutation', 'indel', 'productive'} <= set(history[-1].parts)
    assert history[-1].total < history[0].total


def test_training_config_has_one_owner_and_snapshots_model_options():
    from alignair.aligner import TrainingConfig as old_config, TrainingRun as old_run, run_training as old_train
    from alignair.train import TrainingConfig as owned_config, TrainingRun as owned_run, run_training as owned_train
    assert TrainingConfig is old_config is owned_config
    assert TrainingRun is old_run is owned_run and run_training is old_train is owned_train
    options = {'filters': 4}
    config = request(model_options=options)
    options['filters'] = 999
    assert config.model_options['filters'] == 4
    with pytest.raises(FrozenInstanceError):
        config.steps = 99
    assert config.with_(steps=3).steps == 3 and config.steps == 2
    assert TrainingConfig.from_genairr('HUMAN_IGH_OGRDB', preset='full').steps == 300_000


@pytest.mark.parametrize('options', [{'input_space': 'aa'}, {'label_space': 'classes'},
                                    {'input_space': 'nucleotide', 'label_space': 'protein_classes'},
                                    {'model_options': []}, {'model_options': {'v_allele_count': 2}},
                                    {'steps': True}, {'batch_size': 0}, {'short_boost': False}, {'augmentation': {}}])
def test_bad_public_requests_fail_at_construction(options):
    with pytest.raises(ConfigurationError):
        request(**options)


@pytest.mark.parametrize('sources', [None, 'HUMAN_IGH_OGRDB', (), iter(())])
def test_source_collection_is_explicit(sources):
    with pytest.raises(ConfigurationError):
        TrainingConfig(dataconfigs=sources)


@pytest.mark.parametrize('options', [{'model_options': {'classification_mode': 'categorical'}},
                                    {'seed': -1}, {'seed': True}, {'val_every': -1}, {'lr': float('nan')}])
def test_invalid_workflow_fails_before_model_or_directory_creation(options, tmp_path, monkeypatch):
    import alignair.core.setup as module
    monkeypatch.setattr(module.ModelSetup, 'build_model', lambda self: pytest.fail('allocated invalid model'))
    out = tmp_path / 'should-not-exist'
    with pytest.raises(ConfigurationError):
        run_training(request(**options), output_dir=out)
    assert not out.exists()


def test_unknown_source_and_invalid_callback_fail_before_output_creation(tmp_path):
    with pytest.raises(ConfigurationError):
        run_training(TrainingConfig.from_genairr('MISSING_SOURCE'), output_dir=tmp_path / 'unknown')
    with pytest.raises(ConfigurationError):
        run_training(request(), output_dir=tmp_path / 'callback', on_iteration=1)
    assert not list(tmp_path.iterdir())


def test_public_workflow_requires_a_request_and_destination():
    with pytest.raises(ConfigurationError, match='TrainingConfig'):
        run_training(object())
    with pytest.raises(ConfigurationError, match='output_dir'):
        run_training(request())


def test_explicit_checkpoint_path_reloads_schema_even_with_legacy_filename_suffix(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    run = run_training(request(steps=1), out_path=path)
    assert run.model_path == str(path) and run.steps == 1
    assert run.best_aligner(device='cpu').label_schema['label_space'] == 'alleles'


@pytest.mark.parametrize('labels', ['alleles', 'protein_classes'])
def test_protein_safe_export_validation_and_conversion_preserve_schema_and_predictions(labels, tmp_path):
    import GenAIRR.data as gd
    from alignair.cli.convert import _convert_legacy_alignair
    from alignair.train.build import export_bundle, _validation_report
    run = run_training(request('protein', labels), output_dir=tmp_path / 'run')
    source = run.best_aligner(device='cpu')
    expected = source.predict_raw(['EVQL'])
    metrics = _validation_report(run.model_path, [gd.HUMAN_IGH_OGRDB], n_batches=1, batch_size=2)
    assert metrics and all(np.isfinite(value) for value in metrics.values())
    path = export_bundle(run.model_path, [gd.HUMAN_IGH_OGRDB], str(tmp_path / 'bundle'))
    exported = Aligner.from_pretrained(path, device='cpu')
    assert exported.label_schema == source.label_schema
    assert not any(name.startswith('dataconfig/') or name == 'train_state' for name in read_metadata(path)['sections'])
    actual = exported.predict_raw(['EVQL'])
    assert all(np.array_equal(value, actual[key]) for key, value in expected.items())
    converted_path = str(tmp_path / 'converted.alignair')
    assert _convert_legacy_alignair(run.model_path, converted_path) == 0
    assert Aligner.from_pretrained(converted_path, device='cpu').label_schema == source.label_schema
    from alignair.api import predict_sequences
    with pytest.raises(ConfigurationError, match='predict_raw'):
        predict_sequences(source.model, source.reference, ['EVQL'])


@pytest.mark.parametrize('empty_source', [False, True])
def test_training_run_never_adopts_a_stale_best_checkpoint(tmp_path, monkeypatch, empty_source):
    from alignair.train import trainer
    stale = tmp_path / 'model.best.alignair'
    stale.write_text('an earlier run with different semantics')
    if empty_source:
        monkeypatch.setattr(trainer, 'training_records', lambda *args, **kwargs: iter(()))
    run = run_training(request('protein', 'protein_classes', val_every=1 if empty_source else 0), output_dir=tmp_path)
    assert run.best_model_path is None and run.steps == (0 if empty_source else 2)
    assert stale.read_text() == 'an earlier run with different semantics'
    assert run.best_aligner(device='cpu').model.cfg.input_space == 'protein'


def test_shared_raw_prediction_reserves_half_open_endpoint_for_nucleotide_models(tmp_path):
    config = request(model_options={'embed_dim': 4, 'filters': 4, 'block_out': 8, 'boundary_mode': 'half_open'})
    predictor = run_training(config, output_dir=tmp_path).best_aligner(device='cpu')
    outputs = predictor.predict_raw(['A' * 600])
    assert int(outputs['start_mask'].sum()) == 575
    with pytest.raises(ConfigurationError, match='predict_raw'):
        predictor.predict(['ACGT'])


def test_label_schema_is_checked_during_load_instead_of_silently_reinterpreted(tmp_path):
    run = run_training(request('protein', 'protein_classes'), output_dir=tmp_path / 'run')
    header = container.read_header(run.model_path)
    payloads = {name: (container.read_section(run.model_path, name), info['codec'])
                for name, info in header['sections'].items()}
    header['model']['label_schema']['genes']['v'][0] = ['UNRECOGNIZED_ALLELE']
    bad = tmp_path / 'bad.alignair'
    container.write_container(bad, header, payloads)
    with pytest.raises(ModelIntegrityError, match='label schema'):
        load_model(bad)


def test_resume_rejects_incompatible_weights_before_optimizer_or_rng_continuation(tmp_path, monkeypatch):
    from alignair.model_file import serialize
    from alignair.train import trainer

    config = request('protein', 'protein_classes', steps=1)
    run = run_training(config, output_dir=tmp_path / 'source')
    header = container.read_header(run.model_path)
    header.pop('_sections_base', None)
    sections = {name: (container.read_section(run.model_path, name), info['codec'])
                for name, info in header['sections'].items()}
    state = serialize.state_dict_from_bytes(sections['weights'][0])
    state.pop(next(iter(state)))
    sections['weights'] = (serialize.state_dict_to_bytes(state), 'none')
    bad = tmp_path / 'incompatible.alignair'
    container.write_container(bad, header, sections)
    def forbidden(*args, **kwargs):
        raise AssertionError('continuation state applied before strict model restoration')
    monkeypatch.setattr(torch.optim.AdamW, 'load_state_dict', forbidden)
    monkeypatch.setattr(trainer, '_restore_rng', forbidden)
    with pytest.raises(ModelIntegrityError, match='Missing key') as caught:
        run_training(config.with_(steps=2, resume_path=bad), output_dir=tmp_path / 'resumed')
    assert caught.value.context['stage'] == 'model_restoration'
    assert caught.value.context['path'] == str(bad)


def test_raw_prediction_restores_mode_and_rejects_invalid_protein_inputs(tmp_path):
    predictor = run_training(request('protein', 'protein_classes'), output_dir=tmp_path).best_aligner(device='cpu')
    predictor.model.train()
    predictor.predict_raw(['EVQL'])
    assert predictor.model.training
    for values in ([], 'EVQL', [None], ['?'], ['A' * 192]):
        with pytest.raises(InputError):
            predictor.predict_raw(values)
    predictor.model.cfg.input_space = 'nucleotide'
    with pytest.raises(ConfigurationError, match='vocabulary'):
        predictor.predict_raw(['EVQL'])
