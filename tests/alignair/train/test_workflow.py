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
           'model_options': {'embed_dim': 4, 'trunk_channels': 8}, **options})


@pytest.mark.parametrize('has_d', [True, False])
def test_public_train_validate_reload_predict_path_is_shared(has_d, tmp_path):
    space, labels = 'nucleotide', 'alleles'
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
    outputs = predictor.predict_raw(iter(['ACGTN', 'AC']))
    cfg = predictor.model.cfg
    for gene in cfg.genes:
        assert len(schema['genes'][gene]) == cfg.allele_count(gene)
        assert outputs[f"{gene}_allele"].shape == (2, cfg.allele_count(gene))
    assert 'orientation' in outputs
    final = read_metadata(run.model_path)
    assert final['training']['steps'] == 2
    assert final['training']['train_args']['input_space'] == space
    assert load_model(run.model_path).config.input_space == space


def test_public_resume_retains_profile_label_identity_and_optimizer_state(tmp_path):
    original = request()
    first = run_training(original, output_dir=tmp_path / 'first')
    before = first.best_aligner(device='cpu').label_schema
    steps = []
    resumed = run_training(original.with_(steps=3, resume_path=first.model_path), output_dir=tmp_path / 'resumed',
                           on_iteration=lambda iteration: steps.append(iteration.step))
    assert resumed.steps == 3 and steps == [3]
    assert resumed.best_aligner(device='cpu').label_schema == before


def test_training_seed_reproduces_weights_and_raw_predictions(tmp_path):
    config = request(seed=17)
    first = run_training(config, output_dir=tmp_path / 'one').best_aligner(device='cpu')
    second = run_training(config, output_dir=tmp_path / 'two').best_aligner(device='cpu')
    assert first.label_schema == second.label_schema
    for key, value in first.model.state_dict().items():
        assert torch.equal(value, second.model.state_dict()[key]), key
    one, two = first.predict_raw(['ACGTACGT']), second.predict_raw(['ACGTACGT'])
    assert all(np.array_equal(value, two[key]) for key, value in one.items())


def test_training_config_has_one_owner_and_snapshots_model_options():
    from alignair.aligner import TrainingConfig as old_config, TrainingRun as old_run, run_training as old_train
    from alignair.train import TrainingConfig as owned_config, TrainingRun as owned_run, run_training as owned_train
    assert TrainingConfig is old_config is owned_config
    assert TrainingRun is old_run is owned_run and run_training is old_train is owned_train
    options = {'trunk_channels': 4}
    config = request(model_options=options)
    options['trunk_channels'] = 999
    assert config.model_options['trunk_channels'] == 4
    with pytest.raises(FrozenInstanceError):
        config.steps = 99
    assert config.with_(steps=3).steps == 3 and config.steps == 2
    assert TrainingConfig.from_genairr('HUMAN_IGH_OGRDB', preset='full').steps == 300_000


@pytest.mark.parametrize('options', [{'input_space': 'aa'}, {'label_space': 'classes'},
                                    {'input_space': 'nucleotide', 'label_space': 'protein_classes'},
                                    {'model_options': []}, {'model_options': {'v_allele_count': 2}},
                                    {'steps': True}, {'batch_size': 0}, {'short_boost': False}])
def test_bad_public_requests_fail_at_construction(options):
    with pytest.raises(ConfigurationError):
        request(**options)


@pytest.mark.parametrize('sources', [None, 'HUMAN_IGH_OGRDB', (), iter(())])
def test_source_collection_is_explicit(sources):
    with pytest.raises(ConfigurationError):
        TrainingConfig(dataconfigs=sources)


def test_an_init_seed_changes_the_initial_weights_and_leaves_the_training_batches_alone(tmp_path, monkeypatch):
    """Replicate runs of one data seed: the init seed draws the initial weights, and the process generators the batch
    adapter draws from (orientation augmentation) start where the data seed leaves them, whatever the model's build
    draws. On the CPU the training step draws from them too, so equal states at every step mean equal batches."""
    from alignair.core import ModelSetup
    original = ModelSetup.build_model
    built = []

    def build(setup):
        model = original(setup)
        built.append({key: value.clone() for key, value in model.state_dict().items()})
        return model

    monkeypatch.setattr(ModelSetup, 'build_model', build)

    def watched(init_seed):
        seen = []
        run = run_training(request(seed=5, init_seed=init_seed), output_dir=tmp_path / str(init_seed),
                           on_iteration=lambda iteration: seen.append((iteration.records, torch.get_rng_state())))
        return seen, read_metadata(run.model_path)['training']['train_args']

    (first, first_args), (second, second_args) = watched(0), watched(1)
    assert any(not torch.equal(value, built[1][key]) for key, value in built[0].items())
    assert len(first) == len(second) == 2
    for (records, state), (other_records, other_state) in zip(first, second):
        assert records == other_records and torch.equal(state, other_state)
    assert (first_args['init_seed'], second_args['init_seed'], first_args['seed']) == (0, 1, 5)


def test_an_init_seeded_build_leaves_every_process_generator_as_the_data_seed_left_it(tmp_path, monkeypatch):
    """Python's and NumPy's generators too, should a model's build draw from them."""
    import random
    from alignair.core import ModelSetup
    original = ModelSetup.build_model

    def drawing(setup):
        random.random()
        np.random.random()
        return original(setup)

    def states(build, name):
        monkeypatch.setattr(ModelSetup, 'build_model', build)
        seen = []
        run_training(request(seed=5, init_seed=3, steps=1), output_dir=tmp_path / name,
                     on_iteration=lambda iteration: seen.append((random.getstate(), np.random.get_state()[1].copy())))
        return seen[0]

    (python, numpy), (python_drawn, numpy_drawn) = states(original, 'plain'), states(drawing, 'drawing')
    assert python == python_drawn and np.array_equal(numpy, numpy_drawn)


def test_a_genairr_source_that_cannot_be_read_is_recorded_as_unknown(tmp_path, monkeypatch, capsys):
    """The digest is provenance: an unreadable file of the GenAIRR package never fails a training run."""
    from alignair.train import trainer

    def unreadable():
        raise PermissionError(13, 'Permission denied', 'GenAIRR/data/x.pkl')

    monkeypatch.setattr(trainer, 'genairr_source_sha256', unreadable)
    run = run_training(request(steps=1), output_dir=tmp_path)
    assert read_metadata(run.model_path)['training']['train_args']['genairr_source_sha256'] is None
    assert capsys.readouterr().out.count('GenAIRR source digest not recorded') == 1


def test_without_an_init_seed_the_data_seed_draws_the_initial_weights(tmp_path, monkeypatch):
    import GenAIRR.data as gd
    from alignair.core import ModelSetup, prepare_model
    from alignair.reference.reference_set import ReferenceSet
    original = ModelSetup.build_model
    built = []

    def build(setup):
        model = original(setup)
        built.append({key: value.clone() for key, value in model.state_dict().items()})
        return model

    monkeypatch.setattr(ModelSetup, 'build_model', build)
    run = run_training(request(seed=5, steps=1), output_dir=tmp_path)
    torch.manual_seed(5)
    expected = original(prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), embed_dim=4, trunk_channels=8)).state_dict()
    assert all(torch.equal(value, expected[key]) for key, value in built[0].items())
    assert read_metadata(run.model_path)['training']['train_args']['init_seed'] is None


def test_a_run_records_its_gradient_clip_how_it_started_and_its_genairr_source(tmp_path):
    from alignair.train.trainer import genairr_source_sha256
    scratch = run_training(request(steps=1, grad_clip=.5), output_dir=tmp_path / 'scratch')
    resumed = run_training(request(steps=2, grad_clip=.5, resume_path=scratch.model_path),
                           output_dir=tmp_path / 'resumed')
    warm = run_training(request(steps=1, grad_clip=None, init_from=scratch.model_path), output_dir=tmp_path / 'warm')
    found = {name: read_metadata(run.model_path)['training']['train_args']
             for name, run in (('scratch', scratch), ('resumed', resumed), ('warm', warm))}
    assert [found[name]['grad_clip'] for name in found] == [.5, .5, None]
    assert [found[name]['started_from'] for name in found] == [None, 'resume', 'init_from']
    assert 'source_checkpoint_sha256' not in found['scratch']
    assert len(found['resumed']['source_checkpoint_sha256']) == len(found['warm']['source_checkpoint_sha256']) == 64
    digest = genairr_source_sha256()
    assert len(digest) == 64 and all(args['genairr_source_sha256'] == digest for args in found.values())


@pytest.mark.parametrize('options', [{'model_options': {'filters': 4}},
                                    {'seed': -1}, {'seed': True}, {'val_every': -1}, {'lr': float('nan')},
                                    {'init_seed': -1}, {'init_seed': True}, {'init_seed': 2**32},
                                    {'init_seed': 1.0}, {'init_seed': '1'}])
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


@pytest.mark.parametrize('empty_source', [False, True])
def test_training_run_never_adopts_a_stale_best_checkpoint(tmp_path, monkeypatch, empty_source):
    from alignair.train import trainer
    stale = tmp_path / 'model.best.alignair'
    stale.write_text('an earlier run with different semantics')
    if empty_source:
        monkeypatch.setattr(trainer, 'training_records', lambda *args, **kwargs: iter(()))
    run = run_training(request(val_every=1 if empty_source else 0), output_dir=tmp_path)
    assert run.best_model_path is None and run.steps == (0 if empty_source else 2)
    assert stale.read_text() == 'an earlier run with different semantics'
    assert run.best_aligner(device='cpu').model.cfg.input_space == 'nucleotide'


def test_shared_raw_prediction_reserves_half_open_endpoint_for_nucleotide_models(tmp_path):
    config = request(model_options={'embed_dim': 4, 'trunk_channels': 8})
    predictor = run_training(config, output_dir=tmp_path).best_aligner(device='cpu')
    outputs = predictor.predict_raw(['A' * 600])
    assert int(outputs['start_mask'].sum()) == 575
    assert len(predictor.predict(['A' * 600, 'ACGT'] )) == 2   # AIRR prediction crops to the same window


def test_label_schema_is_checked_during_load_instead_of_silently_reinterpreted(tmp_path):
    run = run_training(request(), output_dir=tmp_path / 'run')
    header = container.read_header(run.model_path)
    payloads = {name: (container.read_section(run.model_path, name), info['codec'])
                for name, info in header['sections'].items()}
    header['model']['label_schema']['genes']['v'][0] = 'UNRECOGNIZED_ALLELE'
    bad = tmp_path / 'bad.alignair'
    container.write_container(bad, header, payloads)
    with pytest.raises(ModelIntegrityError, match='label schema'):
        load_model(bad)


def test_resume_rejects_incompatible_weights_before_optimizer_or_rng_continuation(tmp_path, monkeypatch):
    from alignair.model_file import serialize
    from alignair.train import trainer

    config = request(steps=1)
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


def test_raw_prediction_restores_mode_and_rejects_invalid_inputs(tmp_path):
    predictor = run_training(request(), output_dir=tmp_path).best_aligner(device='cpu')
    predictor.model.train()
    predictor.predict_raw(['ACGT'])
    assert predictor.model.training
    for values in ([], [None]):
        with pytest.raises(InputError):
            predictor.predict_raw(values)


def test_a_flank_policy_trains_on_flanked_reads_and_is_recorded(tmp_path, monkeypatch):
    """``TrainingConfig.flanks``: the run's curriculum puts library flanks around its reads and records the policy."""
    from alignair.data.flanks import FlankPolicy
    from alignair.train import trainer

    flanks = FlankPolicy(p_prefix=1.0, prefix=(30, 30), p_mask=0.0, p_suffix=0.0)
    seen = []
    original = trainer.training_records

    def observed(*args, **kwargs):
        seen.append(kwargs['policy'].flanks)
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer, 'training_records', observed)
    run = run_training(request(flanks=flanks), output_dir=tmp_path)
    assert seen and all(policy == flanks for policy in seen)
    assert read_metadata(run.model_path)['training']['train_args']['flanks'] == flanks.to_dict()
    with pytest.raises(ConfigurationError, match='flanks'):
        TrainingConfig.from_genairr('HUMAN_IGH_OGRDB', flanks={'p_prefix': 1.0})
