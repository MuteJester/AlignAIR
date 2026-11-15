"""``python -m alignair_lab screen --against-milestone``: a candidate trained from the record cache and compared with
the baseline's milestone at the same step (Tier 2), end to end on a tiny budget, on the rig (``rig``)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from alignair.model_file import read_metadata
from alignair.train.trainer import RECIPE
from alignair_lab.cli import main
from alignair_lab.recipe import DATACONFIGS, recipe_train_args
from alignair_lab.sizes import checkpoint_size, size_line
from tests.alignair_lab.rig import REMOVED, SMALL, arguments, copy_card, flat

pytestmark = pytest.mark.integration


def test_a_screen_trains_from_the_cache_and_compares_with_the_milestone_at_its_step(rig, capsys):
    assert main(arguments(rig, 'smoke')) == 0
    printed = capsys.readouterr().out
    screen = rig.screens / 'smoke'
    record = json.loads((screen / 'screen.json').read_text())
    assert record['status'] == 'done' and record['steps'] == 2 and record['architecture'] == 'lab-null-v2'
    assert record['model_options'] == SMALL and record['suites'] == ['S0'] and record['variants'] == ['baseline']
    assert (record['schedule'], record['init_seed'], record['role']) == ('constant', 0, 'candidate')
    milestone = read_metadata(rig.milestones / 'model.step2.alignair')
    baseline = record['baseline']
    assert baseline['kind'] == 'milestone' and baseline['checkpoint'] == str(rig.milestones / 'model.step2.alignair')
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
    assert headline.strip() in printed and 'clone_key.exact' in flat(headline) and record['report'] in flat(headline)
    assert f"git {baseline['git_commit'][:12]}" in flat(headline)
    assert f"GenAIRR {baseline['genairr_version']}" in flat(headline)
    assert baseline['options'] == SMALL and f"baseline options {json.dumps(SMALL)}" in flat(headline)
    assert record['cache']['verification'] == {'code_changed': False, 'deep': False, 'records': 56}
    assert 'Record cache: producing code unchanged since the write; verified 56 records live' in flat(headline)
    assert 'Wall time' in flat(headline) and (screen / 'train.log').is_file()     # the trainer logs every 50 steps
    assert 'initial weights' not in flat(headline)              # the milestone's init seed, 0, is its own
    assert '[screen smoke] step 2/2' in printed                             # progress, past the trainer's log
    assert not (rig.dev / 'diagnostics').exists()                  # the dev build is only read
    # both models' sizes; a milestone records no training rate
    assert baseline['steps_per_second'] is None
    assert baseline['size'] == checkpoint_size(rig.milestones / 'model.step2.alignair')
    assert size_line('baseline', baseline['size'], None) in flat(headline) and 'steps/s not recorded.' in flat(headline)
    assert size_line('candidate', record['candidate']['size'], record['candidate']['steps_per_second']) in \
        flat(headline)


def test_the_null_candidate_reproduces_the_baselines_weights_on_the_cpu(rig):
    """lab-null-v2 is chip-v1, trained from the cache with a data worker: on the CPU, the baseline trained serially
    on live GenAIRR records from the same init seed, weight for weight."""
    import torch
    from alignair.model_file import load_model
    if not (rig.screens / 'smoke').exists():
        assert main(arguments(rig, 'smoke')) == 0
    candidate = load_model(rig.screens / 'smoke' / 'train' / 'model.alignair').model.state_dict()
    baseline = load_model(rig.milestones / 'model.step2.alignair').model.state_dict()
    assert candidate.keys() == baseline.keys()
    assert all(torch.equal(value, baseline[key]) for key, value in candidate.items())


def test_a_milestone_comparison_is_not_weighed_against_a_null_band(rig, baseline, tmp_path):
    """A null band measures baseline screens of a protocol; a long run's milestone is another regime of noise. (The
    band's seed-0 screen is the rig's baseline screen, ``baseline``.)"""
    from alignair_lab.baselines import protocol
    from alignair_lab.records import CachedRecordSource
    from tests.alignair_lab.rig import write_null_band
    root = tmp_path / 'screens'
    write_null_band(rig, root, protocol(CachedRecordSource(rig.cache).manifest, 2, None))   # Tier 2's: constant
    assert main([*arguments(rig, 'tier2'), '--out-root', str(root)]) == 0
    record = json.loads((root / 'tier2' / 'screen.json').read_text())
    assert record['null_band'] is None
    headline = (root / 'tier2' / 'headline.txt').read_text()
    assert 'uncalibrated (Tier 2: milestone)' in flat(headline)
    noise = next(line for line in headline.splitlines() if line.startswith('noise '))
    assert noise.split()[1:] == ['uncalibrated'] * 5


def test_a_screen_may_draw_its_initial_weights_from_its_data_seed_and_says_so_against_a_milestone(rig, tmp_path):
    """--init-seed none trains as the product does by default, to reproduce a milestone of a product run; against
    a milestone of another init seed, the headline says their initial weights and batches differ."""
    root = tmp_path / 'screens'
    assert main([*arguments(rig, 'unseeded'), '--init-seed', 'none', '--out-root', str(root)]) == 0
    record = json.loads((root / 'unseeded' / 'screen.json').read_text())
    assert record['init_seed'] is None and record['baseline']['init_seed'] == 0
    assert read_metadata(root / 'unseeded' / 'train' / 'model.alignair')['training']['train_args']['init_seed'] is None
    headline = (root / 'unseeded' / 'headline.txt').read_text()
    assert ('Note: the milestone drew its initial weights from init seed 0, this screen from none (its data seed)'
            in flat(headline))


def test_the_baseline_records_the_recipes_train_args(rig):
    train_args = read_metadata(rig.milestones / 'model.step2.alignair')['training']['train_args']
    assert {key: train_args.get(key) for key in recipe_train_args()} == recipe_train_args()


def milestone_copy(rig, tmp_path, *, step=2, saved_step=2, header=None, **changes):
    """A milestones directory holding a copy of the rig's baseline as ``model.step<step>.alignair``, its header saying
    it was saved at ``saved_step`` with ``changes`` to its train_args (``REMOVED`` removes one) and ``header``'s other
    fields."""
    copy_card(rig.milestones / 'model.step2.alignair', tmp_path / 'milestones' / f'model.step{step}.alignair',
              header=header, saved_step=saved_step, **changes)
    return tmp_path / 'milestones'


def test_a_cache_written_by_other_producing_code_is_deep_verified_and_says_so(rig, tmp_path):
    cache = tmp_path / 'cache'
    shutil.copytree(rig.cache, cache)
    manifest = json.loads((cache / 'manifest.json').read_text())
    manifest['code']['sha256'] = '0' * 64
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    assert main([*arguments(rig, 'deep'), '--cache', str(cache)]) == 0
    record = json.loads((rig.screens / 'deep' / 'screen.json').read_text())
    verification = record['cache']['verification']
    assert verification['code_changed'] and verification['deep'] and verification['records'] > 56
    assert (f"Record cache: producing code changed since the write; deep-verified {verification['records']} "
            "records live") in (rig.screens / 'deep' / 'headline.txt').read_text()


def test_a_milestone_of_another_genairr_source_than_the_caches_is_refused(rig, tmp_path, capsys):
    milestones = milestone_copy(rig, tmp_path, genairr_source_sha256='0' * 64)
    assert main([*arguments(rig, 'other-source'), '--against-milestone', str(milestones)]) == 2
    error = capsys.readouterr().err
    assert 'genairr_source_sha256 000000000000' in error and 'record cache' in error
    assert not (rig.screens / 'other-source').exists()


def test_a_milestone_without_a_genairr_source_digest_is_refused_on_another_genairr_version(rig, tmp_path, capsys):
    milestones = milestone_copy(rig, tmp_path, genairr_source_sha256=REMOVED, header={'genairr_version': '2.1.0'})
    assert main([*arguments(rig, 'other-genairr'), '--against-milestone', str(milestones)]) == 2
    error = capsys.readouterr().err
    assert "genairr_version '2.1.0', not '2.2.0'" in error and 'record cache' in error
    assert not (rig.screens / 'other-genairr').exists()


def test_a_checkpoints_genairr_digest_is_computed_as_the_caches(rig):
    """The product records the GenAIRR source digest by the assay's method, so a baseline's and a cache's compare."""
    import GenAIRR
    from alignair.genairr.identity import genairr_source_sha256
    from alignair_assay.core.provenance import tree_digest
    from alignair_lab.records import CachedRecordSource
    digest = genairr_source_sha256()
    assert digest == tree_digest(Path(GenAIRR.__file__).resolve().parent)
    assert digest == CachedRecordSource(rig.cache).manifest['genairr']['genairr_source_sha256']
    train_args = read_metadata(rig.milestones / 'model.step2.alignair')['training']['train_args']
    assert train_args['genairr_source_sha256'] == digest


def test_a_baseline_without_a_digest_is_compared_with_the_version_of_the_genairr_imported(rig):
    """A cache records the version the installed distribution's metadata names, which need not be the version of the
    GenAIRR on the path (a frozen copy); a checkpoint records the imported package's own."""
    import GenAIRR
    from alignair.errors import ConfigurationError
    from alignair_lab.records import CachedRecordSource
    from alignair_lab.screen import same_genairr
    manifest = CachedRecordSource(rig.cache).manifest
    manifest = {**manifest, 'genairr': {**manifest['genairr'], 'genairr_version': '9.9.9'}}   # an upgraded install
    same_genairr({'checkpoint': 'a milestone', 'genairr_version': GenAIRR.__version__}, manifest)
    with pytest.raises(ConfigurationError, match=f"genairr_version '9.9.9', not '{GenAIRR.__version__}'"):
        same_genairr({'checkpoint': 'a milestone', 'genairr_version': '9.9.9'}, manifest)


def test_a_milestone_of_another_capacity_is_refused_unless_the_screen_is_told_its_options(rig, capsys):
    assert main(arguments(rig, 'other-capacity', baseline_options=False)) == 2
    error = capsys.readouterr().err
    assert 'embed_dim 4, not 32' in error and 'trunk_channels 8, not 64' in error
    assert not (rig.screens / 'other-capacity').exists()


COSINE = {'policy': 'cosine-v1', 'total_steps': 2, 'min_lr': 3e-4 / 100}


@pytest.mark.parametrize('copy,problem', [
    ({'step': 3}, 'saved at step 2'),
    ({'recipe': REMOVED}, 'before recipe v3'),
    ({'indel_form': 'uniform'}, 'indel_form'),
    ({'selected_fraction': 0.5}, 'selected_fraction'),
    ({'seed': 1}, 'seed'),
    ({'lr': 1e-3}, 'lr'),
    ({'schedule': COSINE}, 'schedule'),
    ({'grad_clip': 1.0}, 'gradient clipping (grad_clip 1.0)'),
    ({'started_from': 'init_from', 'source_checkpoint_sha256': '0' * 64}, 'continued another checkpoint (init_from'),
    ({'started_from': 'resume', 'source_checkpoint_sha256': '0' * 64}, 'continued another checkpoint (resume'),
    ({'source_checkpoint_sha256': '0' * 64}, 'continued another checkpoint (source_checkpoint_sha256')])
def test_a_milestone_of_another_run_is_refused_before_anything_runs(rig, tmp_path, capsys, copy, problem):
    milestones = milestone_copy(rig, tmp_path, **copy)
    assert main([*arguments(rig, 'mismatched', steps=copy.get('step', 2)), '--against-milestone', str(milestones)]) == 2
    assert problem in capsys.readouterr().err
    assert not (rig.screens / 'mismatched').exists()


def test_a_cosine_screen_against_a_milestone_of_a_constant_rate_is_refused(rig, capsys):
    assert main([*arguments(rig, 'cooled'), '--schedule', 'cosine']) == 2
    error = capsys.readouterr().err
    assert f"schedule None, not {COSINE!r}" in error and '--schedule constant' in error
    assert not (rig.screens / 'cooled').exists()


def test_a_milestone_of_another_model_family_is_refused_before_anything_runs(rig, tmp_path, capsys):
    import GenAIRR.data as gd
    import alignair_lab.architectures  # noqa: F401  (registers lab-null-v2)
    from alignair.core import prepare_model
    from alignair.reference.reference_set import ReferenceSet
    from alignair.train.checkpoints import save_checkpoint
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), architecture='lab-null-v2', **SMALL)
    milestones = tmp_path / 'milestones'
    milestones.mkdir()
    train_args = read_metadata(rig.milestones / 'model.step2.alignair')['training']['train_args']
    save_checkpoint(str(milestones / 'model.step2.alignair'), setup.build_model(), 2,
                    dataconfigs=list(DATACONFIGS), train_args=train_args, label_schema=setup.label_schema())
    assert main([*arguments(rig, 'other-family'), '--against-milestone', str(milestones)]) == 2
    assert "architecture 'lab-null-v2', not 'chip-v1'" in capsys.readouterr().err
    assert not (rig.screens / 'other-family').exists()


def test_a_step_without_a_baseline_milestone_is_refused_before_anything_runs(rig, capsys):
    assert main(arguments(rig, 'unmatched', steps=3)) == 2
    error = capsys.readouterr().err
    assert 'no milestone at step 3' in error and 'steps 2' in error
    assert not (rig.screens / 'unmatched').exists()


def test_a_screen_the_cache_cannot_feed_is_refused_before_anything_runs(rig, tmp_path, capsys):
    milestones = milestone_copy(rig, tmp_path, step=3, saved_step=3)
    assert main([*arguments(rig, 'hungry', steps=3), '--against-milestone', str(milestones)]) == 2
    assert 'serves 128 records' in capsys.readouterr().err
    assert not (rig.screens / 'hungry').exists()


@pytest.mark.parametrize('extra,problem', [(('--architecture', 'lab-absent-v1'), 'unknown model architecture'),
                                           (('--model-option', 'no_such_option=1'), 'invalid model configuration'),
                                           (('--model-option', 'trunk_channels'), 'KEY=VALUE'),
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
