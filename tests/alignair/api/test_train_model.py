"""``api.train_model`` trains the recipe ``alignair train`` trains: library flanks around the curriculum's reads
(``TrainingConfig.flanks``) unless the caller says otherwise."""
import pytest

from alignair import api
from alignair.data.flanks import FlankPolicy
from alignair.train import TrainingConfig
from alignair.train import trainer

pytestmark = pytest.mark.component


def _captured_flanks(monkeypatch, **overrides):
    seen = {}
    monkeypatch.setattr(trainer, "train", lambda *args, **kwargs: seen.update(kwargs))
    api.train_model(["HUMAN_IGH_OGRDB"], out_path="unused.alignair", steps=1, **overrides)
    return seen["flanks"]


def test_train_model_trains_with_the_default_library_flanks(monkeypatch):
    assert _captured_flanks(monkeypatch) == TrainingConfig.from_genairr("HUMAN_IGH_OGRDB").flanks == FlankPolicy()


def test_the_caller_may_train_without_flanks(monkeypatch):
    assert _captured_flanks(monkeypatch, flanks=None) is None
