"""The D presence objective (``AlignAIRConfig.d_presence_head``): the cross entropy between the model's D presence logits
(absent, too short to call, callable) and the read's class, Kendall-weighted as ``d_presence``."""
from dataclasses import replace
import math

import pytest
import torch
from torch.nn import functional as F

from alignair.errors import TrainingDataError
from alignair.tasks import build_tasks
from alignair.tasks.losses import loss_weight_names

pytestmark = pytest.mark.component

CLASSES = torch.tensor([0, 2])          # one read without a D, one with a callable D


def presence_task(task_batch, logits):
    cfg, out, targets = task_batch(protein=False, has_d=True)
    cfg = replace(cfg, d_presence_head=True)
    out["d_presence_logits"] = logits.requires_grad_()
    targets["d_presence"] = CLASSES
    return cfg, out, targets


def presence_term(cfg, out, targets, log_var=None):
    tasks = build_tasks(cfg)
    weights = tasks.make_weights()
    if log_var is not None:
        with torch.no_grad():
            weights["d_presence"].log_var.fill_(log_var)
    _, parts = tasks.loss(out, targets, weights)
    return parts["d_presence"]


def test_perfect_d_presence_logits_cost_almost_nothing(task_batch):
    term = presence_term(*presence_task(task_batch, 20.0 * F.one_hot(CLASSES, 3).float()))
    assert 0 <= term.item() < 1e-3


def test_the_d_presence_term_is_a_kendall_weighted_cross_entropy(task_batch):
    """Uniform logits cost log 3; at log-variance s the term is log(3) e^-s + s / 2, with its own weight, the last."""
    cfg, out, targets = presence_task(task_batch, torch.zeros(2, 3))
    assert loss_weight_names(cfg)[-1] == "d_presence"
    assert loss_weight_names(cfg)[:-1] == loss_weight_names(replace(cfg, d_presence_head=False))
    assert presence_term(cfg, out, targets, log_var=1.0).item() == pytest.approx(math.log(3) * math.exp(-1) + 0.5)


@pytest.mark.parametrize("missing", ["d_presence_logits", "d_presence"])
def test_the_d_presence_head_needs_its_logits_and_its_target(task_batch, missing):
    cfg, out, targets = presence_task(task_batch, torch.zeros(2, 3))
    (out if missing.endswith("_logits") else targets).pop(missing)
    with pytest.raises(TrainingDataError) as caught:
        presence_term(cfg, out, targets)
    assert missing in caught.value.context["fields"]
