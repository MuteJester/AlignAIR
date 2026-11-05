"""Baseline screens: screens of lab-null-v2, one per init seed, which the screens of candidates compare with.

A baseline screen trains with a cosine cooldown and compares with nothing. A candidate compares with the registered
baseline screen of its protocol (the recipe, the record cache, the steps, the schedule) and its baseline's options,
init seed 0, or with the one it names (``--against-screen``); the baseline screen's checkpoint is trusted by its
contents. The candidate's headline weighs its Δs against the protocol's null band, or says it is uncalibrated. On the
rig (``rig``), whose registered baseline screen is ``base0``.
"""
import json
import shlex
import shutil
from pathlib import Path

import pytest

from alignair.model_file import read_metadata
from alignair_lab.cli import main
from alignair_lab.recipe import MIN_LR
from alignair_lab.noise import BETTER, MIXED, WITHIN, WORSE
from alignair_lab.sizes import checkpoint_size, size_line
from tests.alignair_lab.rig import REMOVED, SMALL, candidate_arguments, copy_card, flat, write_null_band

pytestmark = pytest.mark.integration


def test_a_baseline_screen_cools_its_rate_down_and_compares_with_nothing(baseline):
    from alignair.train.checkpoints import load_training_checkpoint
    record = json.loads((baseline / 'screen.json').read_text())
    assert (record['status'], record['role'], record['architecture']) == ('done', 'baseline', 'lab-null-v2')
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
    headline = (baseline / 'headline.txt').read_text()
    assert 'Baseline screen base0' in flat(headline)
    assert record['candidate']['size'] == checkpoint_size(checkpoint)                   # its model's size
    assert size_line('model', record['candidate']['size'], record['candidate']['steps_per_second']) in flat(headline)


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
    # the secondary metric, from the two models' cached scores in the screen's assay run, weighed beside the others
    from alignair_lab.secondary import METRIC, cached_comparison
    diagnostics = screen / 'assay' / 'diagnostics'
    assert record['secondary'] == cached_comparison(diagnostics, record['candidate']['model_key'],
                                                    record['comparison']['baseline'], METRIC)
    assert METRIC in record['decisions']
    assert any(line.startswith('D short abstain') for line in (screen / 'headline.txt').read_text().splitlines())
    # the code of the product, the lab and the assay that screened it, by content
    import alignair_assay
    import alignair_lab
    from alignair_assay.core.provenance import code_identity
    assert record['code'] == {'alignair': record['comparison']['code']['this'],
                              'alignair_lab': code_identity(Path(alignair_lab.__file__).parent)['code_sha256'],
                              'alignair_assay': code_identity(Path(alignair_assay.__file__).parent)['code_sha256']}
    assert record['null_band'] is None and set(record['decisions']) >= {'clone_key.exact', 'junction.nt_exact'}
    headline = (screen / 'headline.txt').read_text()
    assert 'against the baseline screen base0 (lab-null-v2, init seed 0) at step 2' in flat(headline)
    noise = next(line for line in headline.splitlines() if line.startswith('noise '))
    assert 'Null band: none' in flat(headline) and noise.split()[1:] == ['uncalibrated'] * 5
    # the calibration it suggests makes a band this screen would use: of its protocol, options, build and cases; the
    # command whole on its own unindented line, never wrapped, so it copies as one shell command
    command = next(line for line in headline.splitlines() if line.startswith('python -m alignair_lab calibrate'))
    assert shlex.join(shlex.split(command)) == command
    assert shlex.split(command) == [
        'python', '-m', 'alignair_lab', 'calibrate', '--steps', '2', '--init-seeds', '0,1,2', '--schedule', 'cosine',
        '--cache', str(rig.cache.resolve()), '--out-root', str(rig.screens.resolve()), '--dev-build',
        str(rig.dev.resolve()), '--suites', 'S0', '--model-option', 'embed_dim=4', '--model-option', 'trunk_channels=8']
    # both models' sizes, and each one's training rate: the baseline's from its baseline screen
    base_rate = json.loads((baseline / 'screen.json').read_text())['candidate']['steps_per_second']
    assert reference['steps_per_second'] == base_rate
    assert reference['size'] == checkpoint_size(reference['checkpoint'])
    assert record['candidate']['size'] == checkpoint_size(record['candidate']['model_path'])
    assert size_line('candidate', record['candidate']['size'], record['candidate']['steps_per_second']) in \
        flat(headline)
    assert size_line('baseline', reference['size'], base_rate) in flat(headline)


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
    assert (f"python -m alignair_lab screen --name baseline-cache-s2-cosine-init0 --architecture lab-null-v2 "
            f"--init-seed 0 --steps 2 --schedule cosine --cache {rig.cache.resolve()} --out-root {root}") in error
    assert "--model-option embed_dim=4 --model-option trunk_channels=8" in error
    assert 'python -m alignair_lab calibrate --steps 2 --init-seeds 0,1,2 --schedule cosine' in error
    assert not (root / 'orphan').exists()


@pytest.mark.parametrize('changes', [{'init_seed': 1}, {'status': 'failed'}, {'architecture': 'chip-v1'},
                                     {'role': 'candidate'}, {'role': REMOVED, 'baseline': {'kind': 'milestone'}},
                                     {'model_options': {}}, {'protocol.steps': 3}, {'protocol.schedule': None},
                                     {'protocol.cache': '0' * 64}, {'protocol.recipe': 'v2'}, {'protocol': REMOVED}])
def test_a_screen_of_another_protocol_seed_or_model_is_not_the_registered_baseline(rig, baseline, tmp_path, capsys,
                                                                                    changes):
    root = screens_with(tmp_path, baseline, 'other', changes=changes)
    assert main(candidate_arguments(rig, 'orphan', out_root=root)) == 2
    assert 'no baseline screen' in capsys.readouterr().err
    assert not (root / 'orphan').exists()


def test_a_null_screen_that_compared_with_a_baseline_is_not_one(rig, baseline, tmp_path):
    """A lab-null-v2 screen run against a baseline (a sanity check, or Tier 2 against a milestone) is a candidate:
    the registered baseline screen stays the one baseline screen of the protocol."""
    from alignair_lab.baselines import baseline_screen
    root = screens_with(tmp_path, baseline, 'base0')
    screens_with(tmp_path, baseline, 'null-check', changes={'role': 'candidate', 'baseline': {'kind': 'screen'}})
    protocol = json.loads((baseline / 'screen.json').read_text())['protocol']
    assert baseline_screen(root, protocol, SMALL, schedule='cosine')['name'] == 'base0'


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
    ('base0', {'architecture': 'chip-v1'}, 'not a baseline screen'),
    ('base0', {'status': 'training lab-null-v2 for 2 steps from the record cache'}, 'has not finished'),
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


def test_the_null_band_of_the_protocol_weighs_the_candidates_deltas(rig, baseline, tmp_path):
    root = screens_with(tmp_path, baseline, 'base0')
    protocol = json.loads((baseline / 'screen.json').read_text())['protocol']
    write_null_band(rig, root, protocol, name='other-cases', max_cases=1)               # other cases: not this one
    write_null_band(rig, root, protocol, name='other-build', suite_builds={'S0': '0' * 64})   # another build of S0
    path = write_null_band(rig, root, protocol)
    assert main(candidate_arguments(rig, 'weighed', out_root=root)) == 0
    record = json.loads((root / 'weighed' / 'screen.json').read_text())
    assert record['null_band']['path'] == str(path) and record['null_band']['init_seeds'] == [0, 1, 2]
    assert 'seeds' not in record['null_band']                        # in null_band.json; the decisions hold their sums
    found = [entry for metric, entry in record['decisions'].items()
             if entry['pooled'] is not None and metric != 'd_short.abstain_ok']         # the fixture's pairs predate it
    assert found and all(entry['decision'] in (BETTER, WORSE, MIXED, WITHIN) for entry in found)
    for entry in found:     # the baseline seeds against seed 0: 0, -1 and +0.5 pp, from the pairs with seed 0
        assert (entry['k'], entry['n'], entry['baseline_mean']) == (1, 3, pytest.approx(-.005 / 3))
    headline = (root / 'weighed' / 'headline.txt').read_text()
    noise = next(line for line in headline.splitlines() if line.startswith('noise '))
    assert f'Null band: {path}' in flat(headline) and 'uncalibrated' not in noise
    assert 'other AlignAIR code' not in flat(headline)
    assert 'Warning: the calibration records no alignair_lab or alignair_assay code digest' in flat(headline)
    assert any(line.startswith('seeds ') and '-0.2±0.8 n3' in line for line in headline.splitlines())


def test_a_calibration_whose_seed_0_screen_is_another_checkpoint_leaves_the_screen_uncalibrated(rig, baseline,
                                                                                                tmp_path):
    """The calibration's values are each seed's Δ against its own seed-0 screen: they weigh a comparison only when its
    baseline is that very checkpoint (the weights part of the two model keys)."""
    root = screens_with(tmp_path, baseline, 'base0')
    protocol = json.loads((baseline / 'screen.json').read_text())['protocol']
    other = '0' * 12 + '-' + 'c' * 12
    write_null_band(rig, root, protocol, screens={'0': {'name': 'base0', 'model_key': other}})
    assert main(candidate_arguments(rig, 'other-checkpoint', out_root=root)) == 0
    record = json.loads((root / 'other-checkpoint' / 'screen.json').read_text())
    found = [entry for entry in record['decisions'].values() if entry['pooled'] is not None]
    assert found and all(entry['decision'] == 'uncalibrated' for entry in found)
    headline = (root / 'other-checkpoint' / 'headline.txt').read_text()
    assert (f"Uncalibrated: the calibration's seed-0 screen is another checkpoint: calibration {other}, this "
            f"comparison's baseline {record['comparison']['baseline']}.") in flat(headline)


def test_a_null_band_measured_under_other_code_is_applied_with_a_warning(rig, baseline, tmp_path):
    root = screens_with(tmp_path, baseline, 'base0')
    protocol = json.loads((baseline / 'screen.json').read_text())['protocol']
    from alignair_lab.screen import code_digests
    now = code_digests()
    path = write_null_band(rig, root, protocol, code_sha256='0' * 64, code={**now, 'alignair_assay': '1' * 64})
    assert main(candidate_arguments(rig, 'recoded', out_root=root)) == 0
    assert json.loads((root / 'recoded' / 'screen.json').read_text())['null_band']['path'] == str(path)
    headline = (root / 'recoded' / 'headline.txt').read_text()
    assert 'Warning: the null band was measured under other AlignAIR code (000000000000)' in flat(headline)
    # and under other assay code, as calibrate now records it: named, the lab's the same
    assert (f"Warning: the calibration was measured under other lab or assay code than this screen: alignair_assay "
            f"{'1' * 12}, this screen's {now['alignair_assay'][:12]}.") in flat(headline)


def test_a_null_band_gives_each_baseline_seeds_values_and_an_older_one_derives_them_from_its_pairs(tmp_path):
    """A null band read for a screen holds each baseline seed's values relative to the registered seed: as written, or,
    from a calibration that predates them, derived from its pairs."""
    from alignair_lab.baselines import against_calibration, null_band
    from alignair_lab.noise import seed_values
    metric = 'clone_key.exact'
    pairs = [{'a': a, 'b': b, 'pooled': {metric: delta}, 'suites': {'S0': {metric: delta}}}
             for a, b, delta in ((0, 1, .02), (0, 2, -.01), (1, 2, -.03))]
    protocol, builds = {'recipe': 'v3', 'cache': 'c' * 64, 'steps': 2, 'schedule': None}, {'S0': 'b' * 64}

    def written(**fields):
        (tmp_path / 'calibration').mkdir(exist_ok=True)
        record = {'kind': 'alignair_lab.null_band', 'version': 1, 'protocol': protocol, 'options': {}, 'suites': ['S0'],
                  'suite_builds': builds, 'max_cases': None, 'init_seeds': [0, 1, 2], 'pairs': pairs,
                  'band': {metric: .03}, 'code_sha256': 'c' * 64, **fields}
        (tmp_path / 'calibration' / 'null_band.json').write_text(json.dumps(record))
        return null_band(tmp_path, protocol, {}, ['S0'], None, builds)

    older = written()
    assert (older['relative_to'], older['seeds']) == (0, seed_values(pairs, [0, 1, 2], relative_to=0))
    stored = {seed: {'pooled': {metric: value}, 'suites': {'S0': {metric: value}}}
              for seed, value in (('0', 0.), ('1', .5), ('2', .25))}
    assert written(relative_to=0, seeds=stored)['seeds'] == stored
    # a screen against the baseline screen of init seed 1 weighs its Δs with the seeds' values against seed 1's
    against = against_calibration(written(relative_to=0, seeds=stored), 1)
    assert against['band'] == {metric: .03}
    assert {seed: found['pooled'][metric] for seed, found in against['seeds'].items()} == \
        {'0': pytest.approx(-.5), '1': 0., '2': pytest.approx(-.25)}
    assert against_calibration(older, 7)['seeds'] is None and against_calibration(None, 0) is None


def test_two_null_bands_of_one_protocol_are_refused(rig, baseline, tmp_path, capsys):
    root = screens_with(tmp_path, baseline, 'base0')
    protocol = json.loads((baseline / 'screen.json').read_text())['protocol']
    write_null_band(rig, root, protocol, name='first')
    write_null_band(rig, root, protocol, name='second')
    assert main(candidate_arguments(rig, 'doubled', out_root=root)) == 2
    error = capsys.readouterr().err
    assert 'first/null_band.json' in error and 'second/null_band.json' in error
    assert not (root / 'doubled').exists()
