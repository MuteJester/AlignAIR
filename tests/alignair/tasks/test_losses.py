"""Independent expectations for objective reductions, partial labels, and supervision."""
from dataclasses import replace
import math

import pytest
import torch

from alignair.errors import ConfigurationError, TrainingDataError
from alignair.tasks import build_tasks
from alignair.tasks.losses import categorical_loss, masked_soft_gaussian
from alignair.tasks.functional import _soft_ce
from alignair.core import AlignAIR
from alignair.core.boundaries import boundary_masks
from alignair.train.trainer import train_step

pytestmark = pytest.mark.component


@pytest.mark.parametrize('has_d', [False, True])
def test_uniform_categorical_objective_has_hand_calculated_loss_and_gradient(task_batch, has_d):
    cfg, out, targets = task_batch(batch_size=1, has_d=has_d)
    out["v_class_logits"] = torch.tensor([[math.log(3), 0.]], requires_grad=True)
    tasks = build_tasks(cfg)
    total, parts = tasks.loss(out, targets)
    assert total.ndim == 0 and set(parts) == {'segmentation', 'classification'}
    assert float(parts["segmentation"].detach()) == pytest.approx(math.log(8))
    gene_count = 3 if has_d else 2
    expected = (math.log(4 / 3) + (gene_count-1) * math.log(2)) / gene_count
    assert float(parts["classification"].detach()) == pytest.approx(expected)
    assert float(total.detach()) == pytest.approx(math.log(8) + expected)
    total.backward()
    torch.testing.assert_close(out["v_class_logits"].grad, torch.tensor([[-.25, .25]]) / gene_count)
    for gene in (('d', 'j') if has_d else ('j',)):
        torch.testing.assert_close(out[f'{gene}_class_logits'].grad, torch.tensor([[-.5, .5]]) / gene_count)


def test_compatible_set_containing_every_class_has_zero_class_gradient(task_batch):
    cfg, out, targets = task_batch(batch_size=1)
    targets["v_compatible"] = torch.ones(1, 2)
    total, parts = build_tasks(cfg).loss(out, targets)
    assert float(parts["classification"].detach()) == pytest.approx(math.log(2) / 2)
    total.backward()
    torch.testing.assert_close(out["v_class_logits"].grad, torch.zeros(1, 2))


def test_excluded_empty_compatible_sets_are_finite_with_zero_gradients(task_batch):
    cfg, out, targets = task_batch()
    for gene in ("v", "j"):
        targets[f"{gene}_valid"] = torch.tensor([1., 0.])
        targets[f"{gene}_compatible"] = torch.tensor([[1., 0.], [0., 0.]])
    total, _ = build_tasks(cfg).loss(out, targets)
    assert torch.isfinite(total)
    total.backward()
    for gene in ("v", "j"):
        torch.testing.assert_close(out[f"{gene}_class_logits"].grad[1], torch.zeros(2))
        assert torch.isfinite(out[f"{gene}_class_logits"].grad).all()
        for boundary in ("start", "end"):
            torch.testing.assert_close(out[f"{gene}_{boundary}_logits"].grad[1], torch.zeros(16))


def test_fully_unsupervised_categorical_batch_is_zero_and_differentiable(task_batch):
    cfg, out, targets = task_batch()
    for gene in ("v", "j"):
        targets[f"{gene}_valid"] = torch.zeros(2)
        targets[f"{gene}_compatible"] = torch.zeros(2, 2)
    total, parts = build_tasks(cfg).loss(out, targets)
    assert total.requires_grad and total.item() == 0
    assert all(value.item() == 0 for value in parts.values())
    total.backward()
    torch.testing.assert_close(out["v_class_logits"].grad, torch.zeros(2, 2))


def test_missing_class_sentinel_excludes_only_classification_support(task_batch):
    cfg, out, targets = task_batch()
    targets["v_class"] = torch.tensor([0, -1])
    total, parts = build_tasks(cfg).loss(out, targets)
    assert parts["classification"].item() == pytest.approx(math.log(2))
    total.backward()
    torch.testing.assert_close(out["v_class_logits"].grad[1], torch.zeros(2))


@pytest.mark.parametrize("protein", [False, True])
def test_auxiliary_and_state_objectives_work_for_both_score_families(task_batch, protein):
    cfg, out, targets = task_batch(protein=protein, auxiliary=True)
    tasks = build_tasks(cfg)
    weights = tasks.make_weights()
    total, parts = tasks.loss(out, targets, weights)
    assert {"mutation", "indel", "productive", "state", "chain_type"} <= parts.keys()
    assert torch.isfinite(total)
    total.backward()
    assert out["mutation_rate"].grad is not None and out["state_logits"].grad is not None
    assert all(weight.log_var.grad is not None for weight in weights.values())


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
    assert not build_tasks(cfg).make_weights()
    mixed = replace(cfg, use_meta_heads=True, state_head=True)
    assert set(build_tasks(mixed).make_weights()) == {"mutation", "indel", "productive", "state"}
    multilabel = replace(cfg, classification_mode="multilabel")
    assert set(build_tasks(multilabel).make_weights()) == {
        "v_start", "v_end", "v_classification", "j_start", "j_end", "j_classification"}


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
    losses = []
    for position in (8, 7, 6, 4):
        logits = torch.zeros(1, 16)
        logits[0, position] = 10.
        total, _ = tasks.loss({**outputs, 'v_end_logits': logits}, targets)
        losses.append(total.item())
    assert all(closer < farther for closer, farther in zip(losses, losses[1:]))


def test_confident_correct_categorical_classes_have_near_zero_loss_and_finite_gradients(task_batch):
    _, outputs, targets = task_batch(batch_size=1)
    for gene in ('v', 'j'):
        outputs[f'{gene}_class_logits'] = torch.tensor([[10., -10.]], requires_grad=True)
    total, parts = categorical_loss(outputs, targets)
    assert parts['classification'].item() < 1e-6
    total.backward()
    for gene in ('v', 'j'):
        gradient = outputs[f'{gene}_class_logits'].grad
        assert torch.isfinite(gradient).all() and gradient.abs().max() < 1e-6


def test_unmasked_soft_cross_entropy_has_expected_entropy_and_gradient():
    logits = torch.zeros(1, 4, requires_grad=True)
    target = torch.tensor([[.1, .2, .3, .4]])
    loss = _soft_ce(logits, target)
    assert loss.item() == pytest.approx(math.log(4))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[.15, .05, -.05, -.15]]))


@pytest.mark.parametrize("protein", [False, True])
def test_shared_optimizer_step_trains_either_score_family(task_batch, protein):
    cfg, _, targets = task_batch(protein=protein)
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


@pytest.mark.parametrize('protein', [False, True])
def test_both_score_families_learn_terminal_exclusive_end_slots(task_batch, protein):
    cfg, out, targets = task_batch(protein=protein)
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


@pytest.mark.parametrize("failure", ["missing", "empty_set", "set_shape", "set_values", "mask", "label",
                                     "label_dtype", "weights", "slots"])
def test_task_contract_errors_are_classified_and_located(task_batch, failure):
    cfg, out, targets = task_batch()
    expected = TrainingDataError
    if failure == "missing":
        del targets["v_start"]
    elif failure == "empty_set":
        targets["v_compatible"] = torch.zeros(2, 2)
    elif failure == "set_shape":
        targets["v_compatible"] = torch.ones(2, 3)
    elif failure == "set_values":
        targets["v_compatible"] = torch.full((2, 2), .5)
    elif failure == "mask":
        targets["v_valid"] = torch.ones(2, 1)
    elif failure == "label":
        targets["v_class"] = torch.tensor([0, 7])
    elif failure == "label_dtype":
        targets["v_class"] = torch.zeros(2)
    elif failure == "weights":
        cfg = replace(cfg, classification_mode="multilabel")
        for gene in ("v", "j"):
            out[f"{gene}_allele"] = out.pop(f"{gene}_class_logits").sigmoid()
            targets[f"{gene}_allele"] = torch.ones(2, 2)
        expected = ConfigurationError
    else:
        del out["start_mask"], out["end_mask"]
    with pytest.raises(expected) as caught:
        build_tasks(cfg).loss(out, targets)
    if expected is TrainingDataError:
        assert "field" in caught.value.context or "fields" in caught.value.context


@pytest.mark.parametrize("genes", [(), ("v", "v"), ("unknown",)])
def test_invalid_gene_plans_fail_before_training(task_batch, genes):
    cfg, _, _ = task_batch()
    with pytest.raises(ConfigurationError, match="task genes"):
        build_tasks(cfg, genes=genes)


def test_hierarchical_plan_rejects_an_implicit_gene_subset(task_batch):
    cfg, _, _ = task_batch(protein=False)
    with pytest.raises(ConfigurationError, match="configured gene set"):
        build_tasks(cfg, genes=("v",))


def test_unknown_partial_classes_keep_boundary_gradients_and_metrics(task_batch):
    cfg, out, targets = task_batch(batch_size=1)
    for gene in ('v','j'):
        targets[f'{gene}_compatible'] = torch.zeros(1,2)
        targets[f'{gene}_class_supervise'] = torch.tensor([False])
    tasks = build_tasks(cfg)
    total,parts = tasks.loss(out,targets)
    assert torch.isfinite(total) and parts['classification'].item() == 0
    total.backward()
    for gene in ('v','j'):
        torch.testing.assert_close(out[f'{gene}_class_logits'].grad,torch.zeros(1,2))
        assert out[f'{gene}_end_logits'].grad.abs().sum() > 0
    report = tasks.metric_report(out,targets)
    assert 'v_class_acc' not in report.counts
    assert report.counts['v_end_mae'] == 1


@pytest.mark.parametrize('mask',[torch.tensor([.5,1.]),torch.ones(2,1),[True,False]])
def test_class_only_mask_must_be_a_binary_row_vector(task_batch,mask):
    cfg,out,targets=task_batch()
    targets['v_class_supervise']=mask
    with pytest.raises(TrainingDataError,match='binary row mask'):
        build_tasks(cfg).loss(out,targets)
