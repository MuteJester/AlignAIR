"""The screens' test rig, built once per test session in a temporary directory: a dev build of S0 (2 reads), a small
conv-v1 baseline trained 2 steps at the recipe's constant rate from init seed 0 and kept as its step-2 milestone, and
a record cache of the recipe for 2 steps. A baseline screen of the small lab-null-v1 (init seed 0, cosine schedule)
is trained in the rig's screens directory when a test first asks for it.

Every screen of these tests is of the small model (``SMALL``), on the CPU, with one data worker."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from alignair import run_training
from alignair.model_file import read_metadata, update_card
from alignair_lab.recipe import BATCH_SIZE, DATACONFIGS, POLICY, SEED, training_config
from alignair_lab.records import write_record_cache

SMALL = {'embed_dim': 4, 'filters': 4, 'block_out': 8}
REMOVED = object()


def build(root: Path) -> SimpleNamespace:
    from alignair_assay.core.layout import RunLayout
    from alignair_assay.workflow.build import build_suite
    build_suite('S0', RunLayout(root / 'dev'), reference_name='HUMAN_IGH_OGRDB', split='dev', seed=5, scale=0.02,
                workers=1)
    baseline = run_training(training_config(steps=2, model_options=SMALL, device='cpu', init_seed=0),
                            output_dir=root / 'baseline')
    (root / 'milestones').mkdir()
    shutil.copyfile(baseline.model_path, root / 'milestones' / 'model.step2.alignair')
    write_record_cache(root / 'cache', dataconfigs=DATACONFIGS, policy=POLICY, seed=SEED, steps=2,
                       batch_size=BATCH_SIZE, workers=2)
    return SimpleNamespace(root=root, dev=root / 'dev', milestones=root / 'milestones', cache=root / 'cache',
                           screens=root / 'screens')


def options(flag, values=SMALL):
    return [option for key, value in values.items() for option in (flag, f'{key}={json.dumps(value)}')]


def common(rig, out_root=None):
    """The arguments every screen of the rig shares."""
    return ['--device', 'cpu', '--data-workers', '1', '--eval-batch-size', '16', '--cache', str(rig.cache),
            '--dev-build', str(rig.dev), '--out-root', str(out_root or rig.screens)]


def arguments(rig, name, *extra, steps=2, baseline_options=True, architecture='lab-null-v1'):
    """A Tier-2 screen of the small ``architecture`` against the rig's milestone, at its constant rate, told the
    milestone's options."""
    told = options('--baseline-model-option') if baseline_options else []
    return ['screen', '--name', name, '--steps', str(steps), '--architecture', architecture,
            *options('--model-option'), *told, *common(rig), '--against-milestone', str(rig.milestones),
            '--schedule', 'constant', *extra]


def baseline_arguments(rig, name, *extra, init_seed=0, out_root=None):
    """A baseline screen of the small lab-null-v1 of ``init_seed``, at the default (cosine) schedule."""
    return ['screen', '--name', name, '--steps', '2', '--architecture', 'lab-null-v1', '--init-seed', str(init_seed),
            *options('--model-option'), *common(rig, out_root), *extra]


def candidate_arguments(rig, name, *extra, out_root=None, architecture='conv-v1'):
    """A screen of the small ``architecture`` against the registered baseline screen, told its options."""
    return ['screen', '--name', name, '--steps', '2', '--architecture', architecture, *options('--model-option'),
            *options('--baseline-model-option'), *common(rig, out_root), *extra]


def suite_build(rig, suite='S0') -> str:
    """The SHA-256 of a suite's ``build.json`` in the rig's dev build."""
    import hashlib
    return hashlib.sha256((rig.dev / 'suites' / suite / 'build.json').read_bytes()).hexdigest()


def write_null_band(rig, root: Path, protocol, *, name='calibration', band=.005, **fields) -> Path:
    """A null band of ``protocol`` for baseline screens of ``SMALL``, compared on the rig's build of S0 under this
    process's AlignAIR code, ``band`` for every headline metric, with ``fields`` changed; its path."""
    from alignair_assay.diagnostics.identity import alignair_code
    record = {'kind': 'alignair_lab.null_band', 'version': 1, 'protocol': protocol, 'options': SMALL, 'suites': ['S0'],
              'suite_builds': {'S0': suite_build(rig)}, 'max_cases': None, 'init_seeds': [0, 1, 2], 'pairs': [],
              'band': {metric: band for metric in ('clone_key.exact', 'junction.nt_exact', 'v.top1_in_set',
                                                   'd.top1_in_set', 'j.top1_in_set')},
              'code_sha256': alignair_code()['code_sha256'], **fields}
    (root / name).mkdir(parents=True)
    (root / name / 'null_band.json').write_text(json.dumps(record))
    return root / name / 'null_band.json'


def copy_card(source: Path, target: Path, *, header=None, saved_step=None, **changes):
    """A copy of the model file ``source`` at ``target``, its train_args changed by ``changes`` (``REMOVED`` removes
    one), the step its header says it was saved at by ``saved_step`` and its header's other fields by ``header``."""
    training = read_metadata(source)['training']
    train_args = {**training['train_args'], **changes}
    train_args = {key: value for key, value in train_args.items() if value is not REMOVED}
    steps = training['steps'] if saved_step is None else saved_step
    target.parent.mkdir(parents=True, exist_ok=True)
    update_card(source, target, {'training': {**training, 'steps': steps, 'train_args': train_args}, **(header or {})})
    return target
