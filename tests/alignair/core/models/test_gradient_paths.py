"""Protect task-gradient routing across the shared model's supported input/head profiles."""
import pytest
import torch
from torch.nn import functional as F

from alignair.core import AlignAIR, build_model_config, evaluation_mode
from alignair.core.layers import SoftCutoutLayer
from alignair.tasks import build_tasks
from alignair.xray.dynamics import gradient_conflict

pytestmark = pytest.mark.component


@pytest.fixture(params=[('nucleotide', 'alleles'), ('protein', 'alleles'), ('protein', 'protein_classes')])
def alignment_case(request):
    space, labels = request.param
    cfg = build_model_config(4, 3, 2, input_space=space, label_space=labels, has_d=True,
                             max_seq_length=192, embed_dim=4, filters=4, block_out=8,
                             use_meta_heads=True, state_head=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        model = AlignAIR(cfg).eval()
        tokens = torch.zeros(2, cfg.max_seq_length, dtype=torch.long)
        for row, length in enumerate((96, 80)):
            tokens[row, :length] = torch.randint(1, cfg.vocab_size, (length,))
        # Keep the predicted interval away from the minimum-span clamp. Test the clamp separately.
        with torch.no_grad():
            positions = torch.arange(cfg.max_seq_length, dtype=torch.float32)
            for branch in model.branches.values():
                branch.start_head.bias.copy_(-((positions - 12) / 4).square())
                branch.end_head.bias.copy_(-((positions - 60) / 4).square())
    inputs = {'tokenized_sequence': tokens, 'orientation': torch.zeros(2, dtype=torch.long)}
    targets = {'mutation_rate': torch.zeros(2, 1), 'indel_count': torch.zeros(2, 1),
               'productive': torch.ones(2, 1), 'orientation': inputs['orientation']}
    for spec in cfg.gene_specs:
        targets[f'{spec.name}_start'] = torch.full((2, 1), 10.)
        targets[f'{spec.name}_end'] = torch.full((2, 1), 64.)
        # Isolate V without changing the task's configured gene set.
        targets[f'{spec.name}_supervise'] = torch.full((2,), spec.name == 'v')
        if cfg.classification_mode == 'categorical':
            targets[f'{spec.name}_class'] = torch.tensor([0, 1])
        else:
            targets[f'{spec.name}_allele'] = F.one_hot(torch.tensor([0, 1]), spec.allele_count).float()
    tasks = build_tasks(cfg)
    outputs = model(inputs)
    _, parts = tasks.loss(outputs, targets, tasks.make_weights())
    return model, inputs, parts


@pytest.mark.parametrize('objective', ['classification', 'segmentation'])
def test_task_gradients_reach_expected_components_and_stay_out_of_unrelated_heads(alignment_case, objective):
    model, _, parts = alignment_case
    parameters = dict(model.named_parameters())
    gradients = dict(zip(parameters, torch.autograd.grad(parts[objective], tuple(parameters.values()),
                                                       allow_unused=True)))
    assert all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients.values())
    reached = ['embedding.', 'branches.v.seg_tower.', 'branches.v.start_head.', 'branches.v.end_head.']
    excluded = ['branches.d.', 'branches.j.', 'orientation_head.', 'meta_tower.', 'meta_heads.', 'state_branch.']
    classifier = ['branches.v.cls_tower.', 'branches.v.cls_mid.', 'branches.v.cls_head.']
    (reached if objective == 'classification' else excluded).extend(classifier)
    for prefix in reached:
        assert any(gradient is not None and torch.count_nonzero(gradient) > 0
                   for name, gradient in gradients.items() if name.startswith(prefix)), prefix
    for prefix in excluded:
        assert all(gradient is None or torch.count_nonzero(gradient) == 0
                   for name, gradient in gradients.items() if name.startswith(prefix)), prefix
    assert all(parameter.grad is None for parameter in model.parameters())


def test_existing_gradient_probe_measures_shared_embedding_without_accumulating_gradients(alignment_case):
    model, _, parts = alignment_case
    report = gradient_conflict({name: parts[name] for name in ('segmentation', 'classification')},
                               list(model.embedding.parameters()))
    assert all(norm > 0 for norm in report['grad_norm'].values())
    cosine = report['cosine']['segmentation|classification']
    assert -1.000001 <= cosine <= 1.000001
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize('teacher_forced', [False, True])
@pytest.mark.parametrize('alignment_case', [('nucleotide', 'alleles')], indirect=True)
def test_classification_does_not_train_the_hard_orientation_decision(alignment_case, teacher_forced):
    model, _, _ = alignment_case
    tokens = torch.ones(2, model.cfg.max_seq_length, dtype=torch.long)
    inputs = {'tokenized_sequence': tokens}
    if teacher_forced:
        inputs['orientation'] = torch.zeros(2, dtype=torch.long)
    outputs = model(inputs)
    parameters = tuple(model.orientation_head.parameters())
    classification = F.binary_cross_entropy(outputs['v_allele'], torch.zeros_like(outputs['v_allele']))
    gradients = torch.autograd.grad(classification, parameters, allow_unused=True, retain_graph=True)
    assert all(gradient is None for gradient in gradients)
    orientation = F.cross_entropy(outputs['orientation_logits'], torch.zeros(2, dtype=torch.long))
    gradients = torch.autograd.grad(orientation, parameters)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_minimum_span_clamp_blocks_the_end_gradient_from_classification():
    start = torch.tensor([[12.]], requires_grad=True)
    end = torch.tensor([[10.]], requires_grad=True)
    mask = SoftCutoutLayer(32)(start, end)
    start_gradient, end_gradient = torch.autograd.grad(mask.square().sum(), (start, end))
    assert torch.isfinite(start_gradient).all() and torch.count_nonzero(start_gradient) > 0
    assert torch.equal(end_gradient, torch.zeros_like(end))


def test_boundary_comparison_reuses_canonical_features_and_preserves_model_state(alignment_case):
    model, inputs, _ = alignment_case
    model.train()
    model.branches['v'].cls_head.eval()
    modes = [module.training for module in model.modules()]
    state = {name: value.clone() for name, value in model.state_dict().items()}
    with evaluation_mode(model):
        predicted = model(inputs)
        embeddings = model.embedding(inputs['tokenized_sequence'])
        branch = model.branches['v']
        replayed = branch.classify(embeddings, predicted['v_start'], predicted['v_end'])
        torch.testing.assert_close(replayed, predicted[model.cfg.classification_key('v')], rtol=0, atol=0)
        known = branch.classify(embeddings, embeddings.new_full((2, 1), 10),
                                embeddings.new_full((2, 1), 64))
        assert known.shape == replayed.shape and torch.isfinite(known).all()
    assert [module.training for module in model.modules()] == modes
    assert all(torch.equal(value, state[name]) for name, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
