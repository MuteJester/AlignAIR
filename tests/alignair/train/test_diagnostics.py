"""Capacity diagnostics: weighted evaluation, step accounting and failure boundaries."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from alignair.errors import ConfigurationError, current_error_context
from alignair.train import overfit_one_batch
from alignair.train.guards import NonFiniteLossError

pytestmark = pytest.mark.component


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, inputs):
        return self.value


def scalar_tasks(model):
    calls = []
    weights = torch.nn.ModuleDict({"offset": torch.nn.Linear(1, 1, bias=False)})
    weights["offset"].weight.data.zero_()

    def make_weights():
        calls.append("make_weights")
        return weights

    def loss(output, target, supplied):
        assert supplied is weights
        calls.append(("loss", model.training, torch.is_grad_enabled()))
        parts = {"model": (output - target).square(),
                 "weight": (supplied["offset"].weight.sum() - 2).square()}
        return sum(parts.values()), parts

    def metrics(output, target):
        calls.append(("metrics", model.training, torch.is_grad_enabled()))
        return {"absolute_error": float((output-target).abs())}

    return SimpleNamespace(make_weights=make_weights, loss=loss, metrics=metrics), weights, calls


@pytest.mark.parametrize("supplied", [True, False])
def test_one_step_has_pre_update_history_and_post_update_weighted_eval(supplied):
    model = ScalarModel()
    tasks, weights, calls = scalar_tasks(model)
    result = overfit_one_batch(model, {}, torch.tensor(1.), tasks=tasks,
                              weights=weights if supplied else None, steps=np.int64(1), lr=.1)
    # AdamW's first update from zero is +lr for each negatively signed gradient.
    assert model.value.item() == pytest.approx(.1)
    assert weights["offset"].weight.item() == pytest.approx(.1)
    assert result["initial_loss"] == 5.
    assert result["history"] == [{"total": 5., "model": 1., "weight": 4.}]
    assert result["final_loss"] == pytest.approx(.9**2 + 1.9**2)
    assert result["absolute_error"] == pytest.approx(.9)
    assert calls == ([] if supplied else ["make_weights"]) + [
        ("loss", True, True), ("loss", False, False), ("metrics", False, False)]
    assert not model.training
    assert current_error_context() == {}


def test_requested_steps_are_actual_updates_and_history_is_detached():
    model = ScalarModel()
    tasks, _, _ = scalar_tasks(model)
    result = overfit_one_batch(model, {}, torch.tensor(1.), tasks=tasks, steps=12, lr=.03, grad_clip=1.)
    assert len(result["history"]) == 12
    assert all(isinstance(value, float) for record in result["history"] for value in record.values())
    assert result["history"][-1]["total"] < result["history"][0]["total"]
    assert result["final_loss"] < result["initial_loss"]


@pytest.mark.parametrize("steps", [True, False, 0, -1, 1.5, "2", None])
def test_invalid_step_count_fails_before_model_or_task_changes(steps):
    model = ScalarModel().eval()
    tasks, weights, calls = scalar_tasks(model)
    with pytest.raises(ConfigurationError) as caught:
        overfit_one_batch(model, {}, 1., tasks=tasks, steps=steps)
    assert caught.value.context["stage"] == "capacity_proof"
    assert caught.value.context["field"] == "steps"
    assert not model.training and model.value.item() == 0
    assert weights["offset"].weight.item() == 0 and calls == []


def test_explicit_tasks_require_metrics_before_training_state_allocation():
    model = ScalarModel()
    tasks = SimpleNamespace(loss=lambda *_: None, make_weights=lambda: pytest.fail("allocated weights"))
    with pytest.raises(ConfigurationError, match="metrics"):
        overfit_one_batch(model, {}, {}, tasks=tasks)


def test_gene_selection_cannot_silently_override_custom_tasks():
    model = ScalarModel()
    tasks, _, calls = scalar_tasks(model)
    with pytest.raises(ConfigurationError, match="select genes"):
        overfit_one_batch(model, {}, {}, tasks=tasks, genes=("v",))
    assert calls == []


def test_gene_selection_requires_a_model_config():
    with pytest.raises(ConfigurationError, match="AlignAIRConfig"):
        overfit_one_batch(ScalarModel(), {}, {}, genes=("v",))


def test_nonfinite_training_loss_has_step_context_without_updating_model():
    model = ScalarModel()
    tasks, _, _ = scalar_tasks(model)
    tasks.loss = lambda output, *_: (output + float("nan"), {})
    with pytest.raises(NonFiniteLossError) as caught:
        overfit_one_batch(model, {}, {}, tasks=tasks, steps=2)
    assert caught.value.context["step"] == 1
    assert model.value.item() == 0
    assert current_error_context() == {}


@pytest.mark.parametrize("kind", ["total", "part"])
def test_final_eval_rejects_nonfinite_losses_with_evaluation_context(kind):
    model = ScalarModel()
    tasks, _, _ = scalar_tasks(model)
    original = tasks.loss

    def loss(*args):
        total, parts = original(*args)
        if not model.training:
            if kind == "total":
                total = total + float("nan")
            else:
                parts["model"] = parts["model"] + float("inf")
        return total, parts

    tasks.loss = loss
    with pytest.raises(NonFiniteLossError) as caught:
        overfit_one_batch(model, {}, torch.tensor(1.), tasks=tasks, steps=1)
    assert caught.value.context["stage"] == "capacity_evaluation"
    assert caught.value.context["step"] == 1
    assert current_error_context() == {}
