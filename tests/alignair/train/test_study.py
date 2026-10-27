"""Study observation never changes model mode or silently adopts stale checkpoints."""
from types import SimpleNamespace

import pytest
import torch

from alignair.errors import ConfigurationError
from alignair.train import TrainingStudy
from alignair.train.study import StudySession

pytestmark = pytest.mark.component


@pytest.mark.parametrize('fields', [{'checkpoint_steps': None}, {'checkpoint_steps': (1,)},
    {'checkpoint_steps': (True,), 'checkpoint_dir': 'x'}, {'checkpoint_steps': (1,1), 'checkpoint_dir': 'x'},
    {'validation_every': -1}, {'plateau_window': True}, {'validation_every': 1}, {'evaluate': 3}])
def test_invalid_study_plan_fails_before_execution(fields):
    with pytest.raises(ConfigurationError):
        TrainingStudy(**fields)


def test_only_requested_checkpoints_are_saved_and_validation_mode_is_restored(tmp_path, capsys):
    saved, modes = [], []
    model = torch.nn.Linear(1,1).train()
    def evaluate(candidate, step):
        modes.append(candidate.training)
        candidate.eval()
        return {'objective': 4-step}
    plan = TrainingStudy(tmp_path, (2,), 1, evaluate, 2)
    session = StudySession(plan, model, lambda path, step: saved.append((path.name, step)))
    for step in (1,2,3):
        session.observe(SimpleNamespace(step=step))
        assert model.training
    assert saved == [('checkpoint-2.alignair', 2)] and modes == [False]*3
    assert session.checkpoints == {2: str(tmp_path/'checkpoint-2.alignair')}
    assert len(session.validations) == 3 and '[plateau]' in capsys.readouterr().out
    assert plan.to_dict() == {'checkpoint_steps': [2], 'validation_every': 1, 'plateau_window': 2}


@pytest.mark.parametrize('result', [None, 1])
def test_invalid_validation_result_restores_mode_and_raises(result):
    model = torch.nn.Linear(1,1).train()
    session = StudySession(TrainingStudy(validation_every=1, evaluate=lambda *_: result), model, None)
    with pytest.raises(ConfigurationError, match='metric mapping'):
        session.observe(SimpleNamespace(step=1))
    assert model.training and session.validations == []


def test_validation_exception_cannot_leak_eval_mode():
    model = torch.nn.Linear(1,1).train()
    def fail(*args):
        raise RuntimeError('evaluation failed')
    session = StudySession(TrainingStudy(validation_every=1, evaluate=fail), model, None)
    with pytest.raises(RuntimeError, match='evaluation failed'):
        session.observe(SimpleNamespace(step=1))
    assert model.training


def test_checkpoint_failure_is_not_recorded_as_successful(tmp_path):
    def fail(*args):
        raise OSError('disk full')
    session = StudySession(TrainingStudy(tmp_path, (1,)), object(), fail)
    with pytest.raises(OSError, match='disk full'):
        session.observe(SimpleNamespace(step=1))
    assert session.checkpoints == {}


@pytest.mark.parametrize('objectives', [(1,2,3), (None,None,None)])
def test_improving_or_unavailable_objectives_do_not_report_plateau(objectives, capsys):
    model = torch.nn.Linear(1,1)
    values = iter(objectives)
    session = StudySession(TrainingStudy(validation_every=1, evaluate=lambda *_: {'objective': next(values)},
                                          plateau_window=2), model, None)
    for step in (1,2,3):
        session.observe(SimpleNamespace(step=step))
    assert not capsys.readouterr().out and model.training
