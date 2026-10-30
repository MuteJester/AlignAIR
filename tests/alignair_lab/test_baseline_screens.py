"""Baseline screens: screens of lab-null-v1, one per init seed, which the screens of candidates compare with.

A baseline screen trains with a cosine cooldown and compares with nothing. A candidate compares with the registered
baseline screen of its protocol (the recipe, the record cache, the steps, the schedule) and its baseline's options,
init seed 0, or with the one it names (``--against-screen``); the baseline screen's checkpoint is trusted by its
contents. The candidate's headline weighs its Δs against the protocol's null band, or says it is uncalibrated. On the
rig (``rig``), whose registered baseline screen is ``base0``.
"""
import json
import shutil

import pytest

from alignair.model_file import read_metadata
from alignair_lab.cli import main
from alignair_lab.recipe import MIN_LR
from tests.alignair_lab.rig import REMOVED, SMALL, candidate_arguments, copy_card

pytestmark = pytest.mark.integration


def test_a_baseline_screen_cools_its_rate_down_and_compares_with_nothing(baseline):
    from alignair.train.checkpoints import load_training_checkpoint
    record = json.loads((baseline / 'screen.json').read_text())
    assert (record['status'], record['role'], record['architecture']) == ('done', 'baseline', 'lab-null-v1')
    assert (record['init_seed'], record['schedule'], record['baseline']) == (0, 'cosine', None)
    assert record['protocol'] == {'recipe': 'v3', 'cache': record['cache']['identity'], 'steps': 2,
                                  'schedule': {'policy': 'cosine-v1', 'total_steps': 2, 'min_lr': MIN_LR}}
    assert record['model_options'] == SMALL and not (baseline / 'assay').exists()
    checkpoint = baseline / 'train' / 'model.alignair'
    train_args = read_metadata(checkpoint)['training']['train_args']
    assert train_args['schedule'] == record['protocol']['schedule'] and train_args['init_seed'] == 0
    state = load_training_checkpoint(checkpoint)
    assert state.scheduler_state['last_epoch'] == 2
    assert state.optimizer_state['param_groups'][0]['lr'] == pytest.approx(MIN_LR)      # cooled down to its end
    assert 'Baseline screen base0' in (baseline / 'headline.txt').read_text()


def test_a_candidate_compares_with_the_registered_baseline_screen_of_its_protocol(rig, baseline):
    assert main(candidate_arguments(rig, 'candidate')) == 0
    screen = rig.screens / 'candidate'
    record = json.loads((screen / 'screen.json').read_text())
    reference = record['baseline']
    assert (reference['kind'], reference['name'], reference['init_seed']) == ('screen', 'base0', 0)
    assert reference['checkpoint'] == str(baseline / 'train' / 'model.alignair')
    assert record['protocol'] == json.loads((baseline / 'screen.json').read_text())['protocol']
    assert (record['role'], record['init_seed'], record['schedule']) == ('candidate', 0, 'cosine')
    assert record['comparison']['code']['same'] is True                     # one process, one AlignAIR code
    assert record['null_band'] is None and set(record['decisions']) >= {'clone_key.exact', 'junction.nt_exact'}
    headline = (screen / 'headline.txt').read_text()
    assert 'against the baseline screen base0 (lab-null-v1, init seed 0) at step 2' in headline
    noise = next(line for line in headline.splitlines() if line.startswith('noise '))
    assert 'Null band: none' in headline and noise.split()[1:] == ['uncalibrated'] * 5


def screens_with(tmp_path, baseline, *names, changes=None):
    """A screens directory holding copies of the baseline screen under ``names``, each screen.json changed by
    ``changes`` (``"protocol.steps"`` changes a field of the protocol; ``REMOVED`` removes one)."""
    root = tmp_path / 'screens'
    for name in names:
        shutil.copytree(baseline, root / name)
        path = root / name / 'screen.json'
        record = {**json.loads(path.read_text()), 'name': name}
        for key, value in (changes or {}).items():
            owner, _, field = key.rpartition('.')
            target = record[owner] if owner else record
            if value is REMOVED:
                del target[field]
            else:
                target[field] = value
        path.write_text(json.dumps(record))
    return root


def test_without_a_registered_baseline_screen_a_candidate_is_refused_with_the_commands_that_make_one(
        rig, tmp_path, capsys):
    root = tmp_path / 'screens'
    assert main(candidate_arguments(rig, 'orphan', out_root=root)) == 2
    error = capsys.readouterr().err
    assert 'no baseline screen' in error
    assert (f"python -m alignair_lab screen --name baseline-cache-s2-cosine-init0 --architecture lab-null-v1 "
            f"--init-seed 0 --steps 2 --schedule cosine --cache {rig.cache.resolve()} --out-root {root}") in error
    assert "--model-option embed_dim=4 --model-option filters=4 --model-option block_out=8" in error
    assert 'python -m alignair_lab calibrate --steps 2 --init-seeds 0,1,2 --schedule cosine' in error
    assert not (root / 'orphan').exists()


@pytest.mark.parametrize('changes', [{'init_seed': 1}, {'status': 'failed'}, {'architecture': 'conv-v1'},
                                     {'model_options': {}}, {'protocol.steps': 3}, {'protocol.schedule': None},
                                     {'protocol.cache': '0' * 64}, {'protocol.recipe': 'v2'}, {'protocol': REMOVED}])
def test_a_screen_of_another_protocol_seed_or_model_is_not_the_registered_baseline(rig, baseline, tmp_path, capsys,
                                                                                    changes):
    root = screens_with(tmp_path, baseline, 'other', changes=changes)
    assert main(candidate_arguments(rig, 'orphan', out_root=root)) == 2
    assert 'no baseline screen' in capsys.readouterr().err
    assert not (root / 'orphan').exists()


def test_two_registered_baseline_screens_are_refused_unless_the_candidate_names_one(rig, baseline, tmp_path, capsys):
    root = screens_with(tmp_path, baseline, 'base0', 'base0-again')
    assert main(candidate_arguments(rig, 'torn', out_root=root)) == 2
    error = capsys.readouterr().err
    assert 'base0, base0-again' in error and '--against-screen' in error
    assert not (root / 'torn').exists()
    assert main(candidate_arguments(rig, 'torn', '--against-screen', 'base0-again', out_root=root)) == 0
    assert json.loads((root / 'torn' / 'screen.json').read_text())['baseline']['name'] == 'base0-again'


@pytest.mark.parametrize('name,changes,problem', [
    ('absent', {}, 'no screen named absent'),
    ('base0', {'architecture': 'conv-v1'}, 'not a baseline screen'),
    ('base0', {'status': 'training lab-null-v1 for 2 steps from the record cache'}, 'has not finished'),
    ('base0', {'protocol.steps': 3}, 'steps 3, not 2')])
def test_a_named_baseline_screen_must_be_a_finished_baseline_screen_of_the_protocol(rig, baseline, tmp_path, capsys,
                                                                                    name, changes, problem):
    root = screens_with(tmp_path, baseline, 'base0', changes=changes)
    assert main(candidate_arguments(rig, 'refused', '--against-screen', name, out_root=root)) == 2
    assert problem in capsys.readouterr().err
    assert not (root / 'refused').exists()


@pytest.mark.parametrize('changes,problem', [
    ({'grad_clip': 1.0}, 'gradient clipping (grad_clip 1.0)'),
    ({'started_from': 'init_from', 'source_checkpoint_sha256': '0' * 64}, 'continued another checkpoint (init_from'),
    ({'init_seed': 1}, 'init_seed 1, not 0'),
    ({'schedule': None}, 'schedule None'),
    ({'genairr_source_sha256': '0' * 64}, 'genairr_source_sha256 000000000000')])
def test_a_baseline_screen_is_trusted_by_its_checkpoint_not_its_record(rig, baseline, tmp_path, capsys, changes,
                                                                       problem):
    root = screens_with(tmp_path, baseline, 'base0')
    copy_card(baseline / 'train' / 'model.alignair', root / 'base0' / 'train' / 'model.alignair', **changes)
    assert main(candidate_arguments(rig, 'distrusted', out_root=root)) == 2
    assert problem in capsys.readouterr().err
    assert not (root / 'distrusted').exists()


def null_band(root, protocol, *, name='calibration', **fields):
    """A null band of ``protocol`` for the rig's suites, every band 0 (any pooled Δ but 0 beyond it)."""
    record = {'kind': 'alignair_lab.null_band', 'version': 1, 'protocol': protocol, 'options': SMALL,
              'suites': ['S0'], 'max_cases': None, 'init_seeds': [0, 1, 2], 'pairs': [],
              'band': {metric: 0. for metric in ('clone_key.exact', 'junction.nt_exact', 'v.top1_in_set',
                                                 'd.top1_in_set', 'j.top1_in_set')}, **fields}
    (root / name).mkdir(parents=True)
    (root / name / 'null_band.json').write_text(json.dumps(record))
    return root / name / 'null_band.json'


def test_the_null_band_of_the_protocol_weighs_the_candidates_deltas(rig, baseline, tmp_path):
    root = screens_with(tmp_path, baseline, 'base0')
    protocol = json.loads((baseline / 'screen.json').read_text())['protocol']
    null_band(root, protocol, name='other-cases', max_cases=1)                # another protocol's: not this one
    path = null_band(root, protocol)
    assert main(candidate_arguments(rig, 'weighed', out_root=root)) == 0
    record = json.loads((root / 'weighed' / 'screen.json').read_text())
    assert record['null_band']['path'] == str(path) and record['null_band']['init_seeds'] == [0, 1, 2]
    decisions = [found['decision'] for found in record['decisions'].values() if found['pooled'] is not None]
    assert decisions and all(decision in ('beyond noise', 'within noise') for decision in decisions)
    headline = (root / 'weighed' / 'headline.txt').read_text()
    assert f'Null band: {path}' in headline and 'uncalibrated' not in headline


def test_two_null_bands_of_one_protocol_are_refused(rig, baseline, tmp_path, capsys):
    root = screens_with(tmp_path, baseline, 'base0')
    protocol = json.loads((baseline / 'screen.json').read_text())['protocol']
    null_band(root, protocol, name='first')
    null_band(root, protocol, name='second')
    assert main(candidate_arguments(rig, 'doubled', out_root=root)) == 2
    error = capsys.readouterr().err
    assert 'first/null_band.json' in error and 'second/null_band.json' in error
    assert not (root / 'doubled').exists()
