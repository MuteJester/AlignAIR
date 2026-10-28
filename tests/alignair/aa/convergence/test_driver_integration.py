"""Reference-bound model construction for convergence checkpoint evaluation."""
import pytest

pytestmark = pytest.mark.integration


def test_candidate_reload_uses_shared_reference_bound_setup(tmp_path, monkeypatch):
    import torch
    from alignair.aa.convergence import driver
    from alignair.train.checkpoints import save_snapshot
    setup = driver._study_setup('HUMAN_IGH_OGRDB', 192)
    model = setup.build_model()
    path = tmp_path / 'candidate.pt'
    save_snapshot(path, model, setup.config, 7)
    calls = []
    original = driver._study_setup
    def prepare(name, length):
        calls.append((name, length))
        return original(name, length)
    monkeypatch.setattr(driver, '_study_setup', prepare)
    loaded, rebound = driver._load_fresh_candidate(str(path), 'HUMAN_IGH_OGRDB', 192, 'cpu')
    assert calls == [('HUMAN_IGH_OGRDB', 192)]
    assert rebound.label_schema() == setup.label_schema() and not loaded.training
    assert loaded is not model
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items())


@pytest.mark.parametrize('full_state', [False, True])
def test_study_trains_real_candidates_and_evaluates_the_same_frozen_cohort(tmp_path, monkeypatch, full_state):
    import json
    import torch
    from alignair import TrainingConfig, run_training
    from alignair.data.augmentation import ProteinCropPolicy
    from alignair.train import CosineSchedule
    from alignair.train.checkpoints import load_training_checkpoint, save_snapshot
    from alignair.aa.convergence import driver
    from tests.support.model_setup import protein_class_setup
    if full_state:
        config = TrainingConfig.from_genairr('HUMAN_IGH_OGRDB', input_space='protein', label_space='protein_classes',
            steps=1, batch_size=2, lr=.0001, device='cpu', augmentation=ProteinCropPolicy(),
            schedule=CosineSchedule(4), model_options={'embed_dim':4, 'filters':4, 'block_out':8})
        source = run_training(config, output_dir=tmp_path/'source').model_path
        start = 1
    else:
        setup = protein_class_setup(embed_dim=4, filters=4, block_out=8)
        source = tmp_path/'historical.pt'
        save_snapshot(source, setup.build_model(), setup.config, 7)
        start = 7
    cohort = {'frozen': object()}
    evaluations = []
    monkeypatch.setattr(driver, 'build_cohort', lambda *a, **kw: (cohort,'fixed-fingerprint'))
    def evaluate(model, cfg, maps, meta, received, **kw):
        assert received is cohort and not model.training
        evaluations.append({key:value.clone() for key,value in model.state_dict().items()})
        return {'clean_v_acc':.99, 'clean_j_acc':.99, 'd_allele_macro':.3, 'by_stratum':{}}
    monkeypatch.setattr(driver, 'evaluate_checkpoint', evaluate)
    result = driver.run_study(str(source), budget=2, ckpt_steps=(1,2), device='cpu',
        art_dir=str(tmp_path/'artifacts'), ckpt_dir=str(tmp_path/'selected'), batch_size=2, val_every=1)
    assert [row['step'] for row in result['rows']] == [start,start+1,start+2]
    assert len(evaluations) == 5  # incumbent, two in-training evaluations, two fresh candidates
    assert any(not torch.equal(value, evaluations[-1][key]) for key,value in evaluations[0].items())
    final = load_training_checkpoint(tmp_path/'selected'/f'checkpoint-{start+2}.alignair')
    assert final.step == start+2 and final.scheduler_state['last_epoch'] == (3 if full_state else 2)
    assert final.label_schema['label_space'] == 'protein_classes'
    stored = json.loads((tmp_path/'artifacts'/'checkpoints_metrics.json').read_text())
    assert stored['rows'] == result['rows'] and stored['cohort_fingerprint']['fingerprint'] == 'fixed-fingerprint'
    assert stored['cohort_fingerprint']['schema'] == 'alignair.convergence.cohort.v2'
    assert (tmp_path/'artifacts'/'predeclaration.md').read_text() == driver.PREDECLARATION_TEXT


def test_existing_checkpoint_directory_is_refused_before_loading_the_incumbent(tmp_path, monkeypatch):
    from alignair.aa.convergence import driver
    from alignair.errors import ConfigurationError
    directory = tmp_path/'selected'
    directory.mkdir()
    stale = directory/'checkpoint-1.alignair'
    stale.write_text('previous run')
    def forbidden(*args, **kwargs): pytest.fail('existing output must fail before checkpoint loading')
    monkeypatch.setattr(driver, 'load_checkpoint', forbidden)
    with pytest.raises(ConfigurationError):
        driver.run_study('source.pt', budget=1, ckpt_steps=(1,), device='cpu',
            art_dir=tmp_path/'artifacts', ckpt_dir=directory)
    assert stale.read_text() == 'previous run' and not (tmp_path/'artifacts').exists()


@pytest.fixture
def incumbent(tmp_path, monkeypatch):
    from alignair.aa.convergence import driver
    from alignair.train.checkpoints import save_snapshot
    from tests.support.model_setup import protein_class_setup
    setup = protein_class_setup(embed_dim=4, filters=4, block_out=8)
    source = tmp_path/'incumbent.pt'
    save_snapshot(source, setup.build_model(), setup.config, 0)
    monkeypatch.setattr(driver, 'build_cohort', lambda *a,**kw: ({},'fixed'))
    monkeypatch.setattr(driver, 'evaluate_checkpoint', lambda *a,**kw:
                        {'clean_v_acc':.99,'clean_j_acc':.99,'d_allele_macro':.3})
    return source


def test_missing_current_candidate_is_not_replaced_by_an_unreported_file(tmp_path, monkeypatch, incumbent):
    from types import SimpleNamespace
    from alignair.aa.convergence import driver
    directory = tmp_path/'selected'
    def train(*args, **kwargs):
        (directory/'checkpoint-1.alignair').write_text('unreported candidate')
        return SimpleNamespace(checkpoints={})
    def forbidden(*args, **kwargs): pytest.fail('candidate was not reported by this training run')
    monkeypatch.setattr(driver, 'train', train)
    monkeypatch.setattr(driver, '_load_fresh_candidate', forbidden)
    result = driver.run_study(incumbent, budget=1, ckpt_steps=(1,), device='cpu',
        art_dir=tmp_path/'artifacts', ckpt_dir=directory, batch_size=2, val_every=0)
    assert [row['name'] for row in result['rows']] == ['incumbent']
    assert (directory/'checkpoint-1.alignair').read_text() == 'unreported candidate'


@pytest.mark.parametrize('failure_stage', ['baseline', 'training', 'markdown'])
def test_failed_runs_retain_recovery_material_without_publishing_final_metrics(tmp_path, monkeypatch, incumbent, failure_stage):
    from types import SimpleNamespace
    from alignair.aa.convergence import driver
    art, ckpt = tmp_path/'artifacts', tmp_path/'selected'
    def fail(*args, **kwargs): raise RuntimeError('injected failure')
    def train(*args, **kwargs):
        (ckpt/'recovery.alignair').write_text('saved recovery material')
        if failure_stage == 'training': fail()
        return SimpleNamespace(checkpoints={})
    monkeypatch.setattr(driver, 'train', train)
    if failure_stage == 'baseline': monkeypatch.setattr(driver, 'evaluate_checkpoint', fail)
    if failure_stage == 'markdown': monkeypatch.setattr(driver, '_render_convergence_md', fail)
    with pytest.raises(RuntimeError, match='injected failure'):
        driver.run_study(incumbent, budget=1, ckpt_steps=(1,), device='cpu', art_dir=art, ckpt_dir=ckpt)
    if failure_stage == 'baseline':
        assert not art.exists() and not ckpt.exists()
    else:
        assert (art/'cohort_fingerprint.json').is_file()
        assert (art/'predeclaration.md').read_text() == driver.PREDECLARATION_TEXT
        assert (ckpt/'recovery.alignair').read_text() == 'saved recovery material'
        assert not (art/'checkpoints_metrics.json').exists() and not (art/'convergence.md').exists()
