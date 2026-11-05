"""Checkpoint publication survives failed writes without losing a completed predecessor."""
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from alignair.errors import ConfigurationError
from alignair.train import checkpoints
from alignair.train.random_state import restore_rng_state

pytestmark = pytest.mark.component


@dataclass
class Config:
    width: int = 4


def test_snapshot_is_weights_only_and_has_a_plain_config(tmp_path):
    model = torch.nn.Linear(4, 2)
    path = tmp_path/'nested'/'weights.pt'
    checkpoints.save_snapshot(path, model, Config(), 7)
    state = torch.load(path, weights_only=True)
    assert set(state) == {'model', 'cfg', 'step'}
    assert state['cfg'] == {'width': 4} and state['step'] == 7
    torch.testing.assert_close(state['model']['weight'], model.weight)


def test_failed_snapshot_preserves_previous_file_and_removes_partial_temporary(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 2)
    path = tmp_path/'weights.pt'
    checkpoints.save_snapshot(path, model, {}, 1)
    original = path.read_bytes()
    def broken_save(state, target):
        Path(target).write_bytes(b'partial')
        raise OSError('disk full')
    monkeypatch.setattr(torch, 'save', broken_save)
    with pytest.raises(OSError, match='disk full'):
        checkpoints.save_snapshot(path, model, {}, 2)
    assert path.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ['weights.pt']


def test_failed_alignair_writer_preserves_a_completed_checkpoint(tmp_path, monkeypatch):
    from alignair import model_file
    path = tmp_path/'model.alignair'
    path.write_bytes(b'completed checkpoint')
    def fail(target, *args, **kwargs):
        Path(target).write_bytes(b'incomplete header')
        raise OSError('interrupted write')
    monkeypatch.setattr(model_file, 'save_model', fail)
    with pytest.raises(OSError, match='interrupted write'):
        checkpoints.save_checkpoint(path, None, object(), {}, 8)
    assert path.read_bytes() == b'completed checkpoint'
    assert len(list(tmp_path.iterdir())) == 1


def _numbered_writer(path, cfg, model, logvars, step, opt, **kwargs):
    torch.save({'step': step}, path)


@pytest.mark.parametrize('name', ['model.alignair', 'model.pt', 'model'])
def test_rotation_retains_newest_steps_and_recovers_latest_with_literal_path_characters(tmp_path, monkeypatch, name):
    monkeypatch.setattr(checkpoints, 'save_checkpoint', _numbered_writer)
    directory = tmp_path/'run[1]'
    directory.mkdir()
    base = directory/name
    # Matching text which is not a numbered checkpoint must survive retention.
    unrelated = directory/((name.rsplit('.', 1)[0] if '.' in name else name)+'.stepnotes.alignair')
    unrelated.write_text('keep this')
    for step in (1, 4, 12):
        checkpoints.save_rotating(base, None, None, {}, step, None, keep=2)
    assert torch.load(base, weights_only=True)['step'] == 12
    latest = checkpoints.latest_checkpoint(base)
    assert torch.load(latest, weights_only=True)['step'] == 12
    assert '.step1.' not in '\n'.join(p.name for p in directory.iterdir())
    assert unrelated.read_text() == 'keep this'


@pytest.mark.parametrize('keep', [0, -1, True, 1.5])
def test_invalid_retention_cannot_write_or_prune_anything(tmp_path, keep):
    with pytest.raises(ConfigurationError, match='retention'):
        checkpoints.save_rotating(tmp_path/'model', None, None, {}, 1, None, keep=keep)
    assert list(tmp_path.iterdir()) == []


def test_latest_checkpoint_falls_back_to_base_and_ignores_unnumbered_names(tmp_path):
    base = tmp_path/'model.alignair'
    assert checkpoints.latest_checkpoint(base) is None
    base.write_bytes(b'base')
    (tmp_path/'model.stepnotes.alignair').write_bytes(b'notes')
    assert checkpoints.latest_checkpoint(base) == str(base)
    for step, extension in ((2, 'alignair'), (10, 'pt')):
        (tmp_path/f'model.step{step}.{extension}').write_bytes(b'numbered')
    assert checkpoints.latest_checkpoint(base) == str(tmp_path/'model.step10.pt')


def test_failed_base_refresh_leaves_previous_base_and_new_numbered_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, 'save_checkpoint', _numbered_writer)
    base = tmp_path/'model.alignair'
    checkpoints.save_rotating(base, None, None, {}, 1, None)
    def fail(source, target):
        Path(target).write_bytes(b'partial copy')
        raise OSError('copy interrupted')
    monkeypatch.setattr(checkpoints.shutil, 'copyfile', fail)
    with pytest.raises(OSError, match='copy interrupted'):
        checkpoints.save_rotating(base, None, None, {}, 2, None)
    assert torch.load(base, weights_only=True)['step'] == 1
    assert torch.load(checkpoints.latest_checkpoint(base), weights_only=True)['step'] == 2
    assert not list(tmp_path.glob('*.tmp'))


def test_legacy_nt_loading_requires_trust_and_normalizes_fields(tmp_path):
    from alignair.errors import ModelError
    path = tmp_path/'legacy.pt'
    torch.save({'model': {'weight': torch.tensor(2.)}, 'logvars': {}, 'step': 9}, path)
    with pytest.raises(ModelError, match='resume_trust_pickle') as caught:
        checkpoints.load_training_checkpoint(path)
    assert caught.value.context['path'] == str(path)
    state = checkpoints.load_training_checkpoint(path, trust_pickle=True)
    assert state.step == 9 and state.weight_state == {} and state.optimizer_state is None
    assert state.rng == {} and state.model_state['weight'].item() == 2.


def test_retention_failure_reports_the_problem_but_keeps_a_valid_new_checkpoint(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(checkpoints, 'save_checkpoint', _numbered_writer)
    base = tmp_path/'model.alignair'
    checkpoints.save_rotating(base, None, None, {}, 1, None, keep=1)
    remove = checkpoints.os.remove
    def fail_old(path):
        if '.step1.' in str(path):
            raise OSError('cannot delete old file')
        return remove(path)
    monkeypatch.setattr(checkpoints.os, 'remove', fail_old)
    checkpoints.save_rotating(base, None, None, {}, 2, None, keep=1)
    assert torch.load(base, weights_only=True)['step'] == 2
    assert 'Could not remove old checkpoint' in caplog.text
    assert 'cannot delete old file' in caplog.text


@pytest.mark.parametrize('mapping', [False, True])
def test_trusted_legacy_reader_retains_explicit_configuration_and_label_contract(tmp_path, mapping):
    from dataclasses import asdict
    from alignair.core import AlignAIRConfig
    cfg = AlignAIRConfig(1, 1, 1, has_d=True, short_d_penalty=False)
    schema = {'genes': {'v': ['V1'], 'd': ['D1'], 'j': ['J1']}}
    path = tmp_path / 'legacy.pt'
    torch.save({'model': {}, 'logvars': {}, 'optimizer': {}, 'config': asdict(cfg) if mapping else cfg,
                'label_schema': schema}, path)
    state = checkpoints.load_training_checkpoint(path, trust_pickle=True)
    assert state.config == cfg and state.label_schema == schema
    checkpoints.validate_continuation(state, cfg, schema)


@pytest.mark.parametrize('config', [17, {}, {'v_allele_count': 1, 'j_allele_count': 1, 'short_d_penalty': 'true'}])
def test_malformed_legacy_configuration_is_not_discarded(tmp_path, config):
    from alignair.errors import ModelError
    path = tmp_path / 'legacy.pt'
    torch.save({'model': {}, 'logvars': {}, 'config': config}, path)
    with pytest.raises(ModelError) as caught:
        checkpoints.load_training_checkpoint(path, trust_pickle=True)
    assert caught.value.context['field'] == 'config'


@pytest.mark.parametrize('changes,message', [({'config': 'other'}, 'configuration'),
    ({'label_schema': {'other': 1}}, 'label identity'),
    ({'schedule': {'policy':'old'}}, 'schedule'), ({'optimizer_state': None}, 'optimizer continuation')])
def test_incompatible_continuation_state_is_rejected_without_mutation(changes, message):
    from dataclasses import replace
    state = checkpoints.ResumeState({}, {}, {}, 4, {}, config='same', label_schema={'v': ['one']})
    invalid = replace(state, **changes)
    with pytest.raises(ConfigurationError, match=message):
        checkpoints.validate_continuation(invalid, 'same', {'v':['one']})
    assert state.config == 'same' and state.optimizer_state == {} and state.step == 4


def test_missing_scheduled_continuation_state_is_rejected():
    state = checkpoints.ResumeState({}, {}, {}, 4, {}, schedule={'policy':'cosine-v1'}, initial_lr=.001)
    with pytest.raises(ConfigurationError, match='schedule'):
        checkpoints.validate_continuation(state, None, None, schedule=state.schedule, lr=.001)
