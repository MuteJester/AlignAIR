"""``python -m alignair_lab screen``: a candidate trained from the record cache and compared with the baseline's
milestone at the same step, end to end on a tiny budget.

The rig is built once per module in a temporary directory: a dev build of S0 (2 reads), a small conv-v1 baseline
trained 2 steps and kept as its step-2 milestone, and a record cache of the recipe for 2 steps.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from alignair import run_training
from alignair.model_file import read_metadata
from alignair.train.trainer import RECIPE
from alignair_lab.cli import main
from alignair_lab.recipe import BATCH_SIZE, DATACONFIGS, POLICY, SEED, recipe_train_args, training_config
from alignair_lab.records import write_record_cache

pytestmark = pytest.mark.integration

SMALL = {'embed_dim': 4, 'filters': 4, 'block_out': 8}


@pytest.fixture(scope='module')
def rig(tmp_path_factory):
    from alignair_assay.core.layout import RunLayout
    from alignair_assay.workflow.build import build_suite
    root = tmp_path_factory.mktemp('rig')
    build_suite('S0', RunLayout(root / 'dev'), reference_name='HUMAN_IGH_OGRDB', split='dev', seed=5, scale=0.02,
                workers=1)
    baseline = run_training(training_config(steps=2, model_options=SMALL, device='cpu'), output_dir=root / 'baseline')
    (root / 'milestones').mkdir()
    shutil.copyfile(baseline.model_path, root / 'milestones' / 'model.step2.alignair')
    write_record_cache(root / 'cache', dataconfigs=DATACONFIGS, policy=POLICY, seed=SEED, steps=2,
                       batch_size=BATCH_SIZE, workers=2)
    return SimpleNamespace(root=root, dev=root / 'dev', milestones=root / 'milestones', cache=root / 'cache',
                           screens=root / 'screens')


def arguments(rig, name, *extra, steps=2):
    options = [option for key, value in SMALL.items() for option in ('--model-option', f'{key}={value}')]
    return ['screen', '--name', name, '--steps', str(steps), '--architecture', 'lab-null-v1', *options,
            '--device', 'cpu', '--data-workers', '1', '--eval-batch-size', '16', '--cache', str(rig.cache),
            '--dev-build', str(rig.dev), '--milestones', str(rig.milestones), '--out-root', str(rig.screens), *extra]


def test_a_screen_trains_from_the_cache_and_compares_with_the_milestone_at_its_step(rig, capsys):
    assert main(arguments(rig, 'smoke')) == 0
    printed = capsys.readouterr().out
    screen = rig.screens / 'smoke'
    record = json.loads((screen / 'screen.json').read_text())
    assert record['status'] == 'done' and record['steps'] == 2 and record['architecture'] == 'lab-null-v1'
    assert record['model_options'] == SMALL and record['suites'] == ['S0'] and record['variants'] == ['baseline']
    milestone = read_metadata(rig.milestones / 'model.step2.alignair')
    baseline = record['baseline']
    assert baseline['milestone'] == str(rig.milestones / 'model.step2.alignair')
    assert (baseline['step'], baseline['recipe']) == (2, RECIPE)
    assert baseline['git_commit'] == milestone['provenance']['git_commit'] and baseline['git_commit']
    assert baseline['genairr_version'] == milestone['genairr_version']
    assert baseline['reference_schema_sha256'] == [dataconfig['schema_sha256']
                                                   for dataconfig in milestone['reference']['dataconfigs']]
    assert record['cache']['path'] == str(rig.cache.resolve()) and record['cache']['steps'] == 2
    assert sorted(path.name for path in (screen / 'train').iterdir()) == ['model.alignair']   # no rotation copies
    metadata = read_metadata(screen / 'train' / 'model.alignair')
    assert metadata['training']['train_args']['data_workers'] == 1
    assert metadata['training']['train_args']['steps'] == 2
    assert Path(record['report']).is_file() and Path(record['report']).is_relative_to(screen)
    assert set(record['comparison']['suites']) == {'S0'}
    headline = (screen / 'headline.txt').read_text()
    assert headline.strip() in printed and 'clone_key.exact' in headline and record['report'] in headline
    assert f"git {baseline['git_commit'][:12]}" in headline and f"GenAIRR {baseline['genairr_version']}" in headline
    assert 'Wall time' in headline and (screen / 'train.log').is_file()     # the trainer logs every 50 steps
    assert '[screen smoke] step 2/2' in printed                             # progress, past the trainer's log
    assert not (rig.dev / 'diagnostics').exists()                  # the dev build is only read


def test_the_null_candidate_reproduces_the_baselines_weights_on_the_cpu(rig):
    """lab-null-v1 is conv-v1, trained from the cache with a data worker: on the CPU, the baseline trained serially
    on live GenAIRR records, weight for weight."""
    import torch
    from alignair.model_file import load_model
    if not (rig.screens / 'smoke').exists():
        assert main(arguments(rig, 'smoke')) == 0
    candidate = load_model(rig.screens / 'smoke' / 'train' / 'model.alignair').model.state_dict()
    baseline = load_model(rig.milestones / 'model.step2.alignair').model.state_dict()
    assert candidate.keys() == baseline.keys()
    assert all(torch.equal(value, baseline[key]) for key, value in candidate.items())


def test_the_baseline_records_the_recipes_train_args(rig):
    train_args = read_metadata(rig.milestones / 'model.step2.alignair')['training']['train_args']
    assert {key: train_args.get(key) for key in recipe_train_args()} == recipe_train_args()


REMOVED = object()


def milestone_copy(rig, tmp_path, *, step=2, saved_step=2, **changes):
    """A milestones directory holding a copy of the rig's baseline as ``model.step<step>.alignair``, its header saying
    it was saved at ``saved_step`` with ``changes`` to its train_args (``REMOVED`` removes one)."""
    from alignair.model_file import update_card
    source = rig.milestones / 'model.step2.alignair'
    training = read_metadata(source)['training']
    train_args = {**training['train_args'], **changes}
    train_args = {key: value for key, value in train_args.items() if value is not REMOVED}
    milestones = tmp_path / 'milestones'
    milestones.mkdir()
    update_card(source, milestones / f'model.step{step}.alignair',
                {'training': {**training, 'steps': saved_step, 'train_args': train_args}})
    return milestones


@pytest.mark.parametrize('copy,problem', [
    ({'step': 3}, 'saved at step 2'),
    ({'recipe': REMOVED}, 'before recipe v3'),
    ({'indel_form': 'uniform'}, 'indel_form'),
    ({'selected_fraction': 0.5}, 'selected_fraction'),
    ({'seed': 1}, 'seed'),
    ({'lr': 1e-3}, 'lr')])
def test_a_milestone_of_another_run_is_refused_before_anything_runs(rig, tmp_path, capsys, copy, problem):
    milestones = milestone_copy(rig, tmp_path, **copy)
    assert main([*arguments(rig, 'mismatched', steps=copy.get('step', 2)), '--milestones', str(milestones)]) == 2
    assert problem in capsys.readouterr().err
    assert not (rig.screens / 'mismatched').exists()


def test_a_milestone_of_another_model_family_is_refused_before_anything_runs(rig, tmp_path, capsys):
    import GenAIRR.data as gd
    import alignair_lab.architectures  # noqa: F401  (registers lab-null-v1)
    from alignair.core import prepare_model
    from alignair.reference.reference_set import ReferenceSet
    from alignair.train.checkpoints import save_checkpoint
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), architecture='lab-null-v1', **SMALL)
    milestones = tmp_path / 'milestones'
    milestones.mkdir()
    train_args = read_metadata(rig.milestones / 'model.step2.alignair')['training']['train_args']
    save_checkpoint(str(milestones / 'model.step2.alignair'), setup.config, setup.build_model(), None, 2,
                    dataconfigs=list(DATACONFIGS), train_args=train_args, label_schema=setup.label_schema())
    assert main([*arguments(rig, 'other-family'), '--milestones', str(milestones)]) == 2
    assert "architecture 'lab-null-v1', not 'conv-v1'" in capsys.readouterr().err
    assert not (rig.screens / 'other-family').exists()


def test_a_step_without_a_baseline_milestone_is_refused_before_anything_runs(rig, capsys):
    assert main(arguments(rig, 'unmatched', steps=3)) == 2
    error = capsys.readouterr().err
    assert 'no milestone at step 3' in error and 'steps 2' in error
    assert not (rig.screens / 'unmatched').exists()


def test_a_screen_the_cache_cannot_feed_is_refused_before_anything_runs(rig, tmp_path, capsys):
    milestones = milestone_copy(rig, tmp_path, step=3, saved_step=3)
    assert main([*arguments(rig, 'hungry', steps=3), '--milestones', str(milestones)]) == 2
    assert 'serves 128 records' in capsys.readouterr().err
    assert not (rig.screens / 'hungry').exists()


@pytest.mark.parametrize('extra,problem', [(('--architecture', 'lab-absent-v1'), 'unknown model architecture'),
                                           (('--model-option', 'no_such_option=1'), 'invalid model configuration'),
                                           (('--model-option', 'filters'), 'KEY=VALUE'),
                                           (('--suites', 'S1'), 'suite S1 is not built')])
def test_a_request_that_cannot_run_is_refused_before_anything_runs(rig, capsys, extra, problem):
    assert main([*arguments(rig, 'refused'), *extra]) == 2
    assert problem in capsys.readouterr().err
    assert not (rig.screens / 'refused').exists()


def test_a_screen_name_in_use_is_refused(rig, capsys):
    (rig.screens / 'taken').mkdir(parents=True)
    assert main(arguments(rig, 'taken')) == 2
    assert 'exists' in capsys.readouterr().err
    assert list((rig.screens / 'taken').iterdir()) == []


def test_two_screens_run_at_once_from_the_command_line(rig):
    import GenAIRR
    path = os.pathsep.join([str(Path(GenAIRR.__file__).resolve().parents[1]), str(Path(__file__).parents[2] / 'src')])
    command = [sys.executable, '-m', 'alignair_lab']
    processes = [subprocess.Popen([*command, *arguments(rig, name)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, env={**os.environ, 'PYTHONPATH': path})
                 for name in ('first', 'second')]
    outcomes = [process.communicate(timeout=600) for process in processes]
    for (out, err), process, name in zip(outcomes, processes, ('first', 'second')):
        assert process.returncode == 0, err
        record = json.loads((rig.screens / name / 'screen.json').read_text())
        assert record['status'] == 'done' and record['report'] in out
    assert not (rig.dev / 'diagnostics').exists()
