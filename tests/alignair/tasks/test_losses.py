"""Independent expectations for objective reductions, partial labels, and supervision."""
from dataclasses import replace
import math

import pytest
import torch

from alignair.errors import ConfigurationError, TrainingDataError
from alignair.tasks import build_tasks
from alignair.tasks.losses import masked_soft_gaussian
from alignair.tasks.functional import _soft_ce
from alignair.core import AlignAIR
from alignair.core.boundaries import boundary_masks
from alignair.train.trainer import train_step

pytestmark = pytest.mark.component


def test_auxiliary_and_state_objectives_train_their_heads_and_weights(task_batch):
    cfg, out, targets = task_batch(auxiliary=True)
    tasks = build_tasks(cfg)
    weights = tasks.make_weights()
    total, parts = tasks.loss(out, targets, weights)
    assert {"mutation", "indel", "productive", "state", "chain_type"} <= parts.keys()
    assert torch.isfinite(total)
    total.backward()
    assert out["mutation_rate"].grad is not None and out["state_logits"].grad is not None
    assert all(weight.log_var.grad is not None for weight in weights.values())


def test_productive_objective_leaves_out_rows_whose_productivity_is_unsupervised(task_batch):
    cfg, out, targets = task_batch(batch_size=3, auxiliary=True)
    out["productive"] = torch.tensor([[.9], [.2], [.6]], requires_grad=True)
    targets["productive"] = torch.tensor([[1.], [0.], [0.]])
    targets["productive_supervise"] = torch.tensor([True, True, False])
    tasks = build_tasks(cfg)
    weights = tasks.make_weights()
    _, parts = tasks.loss(out, targets, weights)
    _, kept_out, kept = task_batch(batch_size=2, auxiliary=True)
    kept_out["productive"] = torch.tensor([[.9], [.2]])
    kept["productive"] = torch.tensor([[1.], [0.]])
    torch.testing.assert_close(parts["productive"], tasks.loss(kept_out, kept, weights)[1]["productive"])
    expected = -(math.log(.9) + math.log(.8)) / 2
    torch.testing.assert_close(parts["productive"].detach(),
                               (expected * weights["productive"]() + weights["productive"].penalty()).detach())
    parts["productive"].backward()
    assert out["productive"].grad[2].item() == 0 and out["productive"].grad[:2].abs().min() > 0


def test_productive_objective_without_a_supervised_row_is_zero_with_zero_weight_gradient(task_batch):
    cfg, out, targets = task_batch(auxiliary=True)
    targets["productive_supervise"] = torch.zeros(2, dtype=torch.bool)
    tasks = build_tasks(cfg)
    weights = tasks.make_weights()
    total, parts = tasks.loss(out, targets, weights)
    assert parts["productive"].item() == 0
    total.backward()
    assert weights["productive"].log_var.grad.item() == 0
    torch.testing.assert_close(out["productive"].grad, torch.zeros_like(out["productive"]))


def test_state_head_without_any_label_has_zero_loss_and_weight_gradient(task_batch):
    cfg, out, targets = task_batch(auxiliary=True)
    targets["state_labels"].fill_(-100)
    tasks = build_tasks(cfg)
    weights = tasks.make_weights()
    total, parts = tasks.loss(out, targets, weights)
    assert parts["state"].item() == 0
    total.backward()
    assert weights["state"].log_var.grad.item() == 0
    torch.testing.assert_close(out["state_logits"].grad, torch.zeros_like(out["state_logits"]))


def test_loss_weights_exist_only_for_enabled_weighted_tasks(task_batch):
    cfg, _, _ = task_batch()
    alignment = {"v_start", "v_end", "v_classification", "j_start", "j_end", "j_classification"}
    assert set(build_tasks(cfg).make_weights()) == alignment
    mixed = replace(cfg, use_meta_heads=True, state_head=True)
    assert set(build_tasks(mixed).make_weights()) == alignment | {"mutation", "indel", "productive", "state"}


def test_gaussian_width_parameter_changes_target_concentration():
    mask = torch.ones(1, 16, dtype=torch.bool)
    narrow = masked_soft_gaussian(torch.tensor([7.]), 16, mask, sigma=.5)
    broad = masked_soft_gaussian(torch.tensor([7.]), 16, mask, sigma=3.)
    assert narrow[0, 7] > broad[0, 7]
    torch.testing.assert_close(narrow.sum(), torch.tensor(1.))


def test_terminal_gaussian_target_retains_mass_and_is_normalized_over_end_slots():
    tokens = torch.zeros(1, 16, dtype=torch.long)
    tokens[:, :8] = 1
    start_mask, end_mask = boundary_masks(tokens)
    endpoint = torch.tensor([8.])
    target = masked_soft_gaussian(endpoint, 16, end_mask)
    assert target.sum().item() == pytest.approx(1.)
    assert target.argmax(-1).item() == 8
    assert target[0, 8] > .4 and target[0, 0] == 0 and target[0, 9:].sum() == 0
    # Reusing the start mask would erase the true exclusive endpoint.
    assert masked_soft_gaussian(endpoint, 16, start_mask)[0, 8] == 0


def test_fixed_terminal_truth_rewards_predictions_closer_to_the_endpoint(task_batch):
    cfg, outputs, targets = task_batch(batch_size=1)
    targets['v_end'].fill_(8.)
    tasks = build_tasks(cfg)
    weights = tasks.make_weights()
    losses = []
    for position in (8, 7, 6, 4):
        logits = torch.zeros(1, 16)
        logits[0, position] = 10.
        total, _ = tasks.loss({**outputs, 'v_end_logits': logits}, targets, weights)
        losses.append(total.item())
    assert all(closer < farther for closer, farther in zip(losses, losses[1:]))


def test_unmasked_soft_cross_entropy_has_expected_entropy_and_gradient():
    logits = torch.zeros(1, 4, requires_grad=True)
    target = torch.tensor([[.1, .2, .3, .4]])
    loss = _soft_ce(logits, target)
    assert loss.item() == pytest.approx(math.log(4))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[.15, .05, -.05, -.15]]))


def test_shared_optimizer_step_trains_the_allele_heads(task_batch):
    cfg, _, targets = task_batch()
    cfg = replace(cfg, max_seq_length=128, embed_dim=4, filters=4, block_out=8)
    model = AlignAIR(cfg)
    tasks = build_tasks(cfg)
    weights = tasks.make_weights()
    optimizer = torch.optim.AdamW([*model.parameters(), *weights.parameters()], lr=.001)
    before = model.branches["v"].cls_head.weight.detach().clone()
    tokens = torch.zeros(2, 128, dtype=torch.long)
    tokens[:, :8] = 1
    inputs = {"tokenized_sequence": tokens}
    total, parts, norm = train_step(model, inputs, targets, cfg, weights, optimizer, tasks=tasks)
    assert math.isfinite(total) and math.isfinite(norm)
    assert {"segmentation", "classification"} <= parts.keys()
    assert not torch.equal(before, model.branches["v"].cls_head.weight.detach())


def test_allele_objective_learns_terminal_exclusive_end_slots(task_batch):
    cfg, out, targets = task_batch()
    cfg = replace(cfg, boundary_mode="half_open")
    tokens = torch.zeros(2, 16, dtype=torch.long)
    tokens[:, :8] = 1
    out["start_mask"], out["end_mask"] = boundary_masks(tokens)
    out.pop('position_mask', None)
    targets["v_end"].fill_(8.)
    tasks = build_tasks(cfg)
    total, _ = tasks.loss(out, targets, tasks.make_weights())
    total.backward()
    assert (out["v_end_logits"].grad[:, 8] < 0).all()
    assert (out['v_end_logits'].grad[:, 9:] == 0).all()
    assert (out['v_end_logits'].grad[:, 0] == 0).all()


@pytest.mark.parametrize("failure", ["missing", "mask", "weights"])
def test_task_contract_errors_are_classified_and_located(task_batch, failure):
    cfg, out, targets = task_batch()
    tasks = build_tasks(cfg)
    weights = tasks.make_weights()
    expected = TrainingDataError
    if failure == "missing":
        del targets["v_start"]
    elif failure == "mask":
        targets["v_valid"] = torch.ones(2, 1)
    else:
        weights, expected = None, ConfigurationError
    with pytest.raises(expected) as caught:
        tasks.loss(out, targets, weights)
    if expected is TrainingDataError:
        assert "field" in caught.value.context or "fields" in caught.value.context


@pytest.mark.parametrize("genes", [(), ("v", "v"), ("unknown",)])
def test_invalid_gene_plans_fail_before_training(task_batch, genes):
    cfg, _, _ = task_batch()
    with pytest.raises(ConfigurationError, match="task genes"):
        build_tasks(cfg, genes=genes)


def test_hierarchical_plan_rejects_an_implicit_gene_subset(task_batch):
    cfg, _, _ = task_batch()
    with pytest.raises(ConfigurationError, match="configured gene set"):
        build_tasks(cfg, genes=("v",))
