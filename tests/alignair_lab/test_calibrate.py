"""``python -m alignair_lab calibrate``: baseline screens of several init seeds, the registered one reused and the others
trained two at once, compared pair by pair into the null band of their protocol, which a candidate's headline then
weighs its Δs against. On the rig (``rig``)."""
import datetime
import json
import shutil

import pytest

from alignair_lab.cli import main
from tests.alignair_lab.rig import candidate_arguments, common, options

pytestmark = pytest.mark.integration


def calibrate_arguments(rig, root, *extra, seeds='0,1,2'):
    return ['calibrate', '--steps', '2', '--init-seeds', seeds, *options('--model-option'), *common(rig, root), *extra]


def test_calibrate_trains_the_missing_baseline_screens_and_writes_the_null_band(rig, baseline, tmp_path, capsys):
    root = tmp_path / 'screens'
    shutil.copytree(baseline, root / 'base0')                    # the registered baseline screen, reused
    assert main(calibrate_arguments(rig, root, '--parallel', '2')) == 0
    printed = capsys.readouterr().out
    path = root / 'calibration-cache-s2-cosine' / 'null_band.json'
    record = json.loads(path.read_text())
    assert record['status'] == 'done' and record['init_seeds'] == [0, 1, 2] and record['suites'] == ['S0']
    assert record['protocol'] == json.loads((baseline / 'screen.json').read_text())['protocol']
    screens = record['screens']
    assert (screens['0']['name'], screens['0']['reused']) == ('base0', True)
    trained = []
    for seed in (1, 2):
        assert (screens[str(seed)]['name'], screens[str(seed)]['reused']) == (f'baseline-cache-s2-cosine-init{seed}',
                                                                             False)
        found = json.loads((root / screens[str(seed)]['name'] / 'screen.json').read_text())
        assert (found['status'], found['role'], found['init_seed']) == ('done', 'baseline', seed)
        trained.append([datetime.datetime.fromisoformat(found[key]) for key in ('started', 'finished')])
    assert max(start for start, _ in trained) < min(end for _, end in trained)   # two at once
    assert [(found['a'], found['b']) for found in record['pairs']] == [(0, 1), (0, 2), (1, 2)]
    assert all(set(found['suites']) <= {'S0'} for found in record['pairs'])
    assert set(record['band']) >= {'clone_key.exact', 'junction.nt_exact', 'v.top1_in_set'}
    for metric, band in record['band'].items():
        assert band == max(abs(found['pooled'][metric]) for found in record['pairs'] if metric in found['pooled'])
    assert 'the null band' in printed and str(path) in printed
    assert main(candidate_arguments(rig, 'calibrated', out_root=root)) == 0
    assert json.loads((root / 'calibrated' / 'screen.json').read_text())['null_band']['path'] == str(path)


@pytest.mark.parametrize('extra,problem', [(('--init-seeds', '0'), 'two init seeds or more'),
                                           (('--init-seeds', '0,1,1'), 'each once'),
                                           (('--init-seeds', '0,-1'), 'init seed'),
                                           (('--parallel', '0'), 'parallel'),
                                           (('--model-option', 'no_such_option=1'), 'invalid model configuration'),
                                           (('--suites', 'S1'), 'suite S1 is not built')])
def test_a_calibration_that_cannot_run_is_refused_before_anything_runs(rig, tmp_path, capsys, extra, problem):
    root = tmp_path / 'screens'
    assert main([*calibrate_arguments(rig, root), *extra]) == 2
    assert problem in capsys.readouterr().err
    assert not root.exists()


def test_a_calibration_in_use_or_two_baseline_screens_of_a_seed_are_refused(rig, baseline, tmp_path, capsys):
    root = tmp_path / 'screens'
    (root / 'calibration-cache-s2-cosine').mkdir(parents=True)
    assert main(calibrate_arguments(rig, root)) == 2
    assert 'a calibration named' in capsys.readouterr().err
    for name in ('base0', 'base0-again'):
        shutil.copytree(baseline, root / name)
    assert main(calibrate_arguments(rig, root, '--name', 'other')) == 2
    assert 'base0, base0-again' in capsys.readouterr().err
    assert not (root / 'other').exists()


def test_a_screen_in_the_way_of_a_baseline_screen_to_train_is_refused(rig, tmp_path, capsys):
    root = tmp_path / 'screens'
    (root / 'baseline-cache-s2-cosine-init1').mkdir(parents=True)          # a failed run's, say
    assert main(calibrate_arguments(rig, root, seeds='1,2')) == 2
    assert 'baseline-cache-s2-cosine-init1, which exists' in capsys.readouterr().err
    assert sorted(path.name for path in root.iterdir()) == ['baseline-cache-s2-cosine-init1']
