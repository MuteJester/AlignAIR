"""The region objective (``AlignAIRConfig.region_head``): the cross entropy between a model's per-position region logits
(pre, V, N1, D, N2, J, post) and its region labels over the read's positions only, Kendall-weighted as ``regions``. The
head reads a trunk's per-position features, so a model without a trunk refuses the flag."""
from dataclasses import replace
import math

import pytest
import torch
from torch.nn import functional as F

from alignair.core import AlignAIR, AlignAIRConfig
from alignair.core.boundaries import boundary_masks
from alignair.errors import ConfigurationError, TrainingDataError
from alignair.neural.heads.region import READ_REGIONS
from alignair.tasks import build_tasks
from alignair.tasks.losses import loss_weight_names

pytestmark = pytest.mark.component

READ = 8          # the task batch's reads fill the first 8 of its 16 positions
CLASSES = len(READ_REGIONS)


def read_labels(batch_size=2, seed=0):
    """Random region classes on the read's positions, the padding value on the rest."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.full((batch_size, 16), -100)
    labels[:, :READ] = torch.randint(0, CLASSES, (batch_size, READ), generator=generator)
    return labels


def region_task(task_batch, labels, logits, boundary_mode='residue'):
    """The task batch of a nucleotide model with a D and the region head, given ``logits`` and ``labels``."""
    cfg, out, targets = task_batch(has_d=True)
    cfg = replace(cfg, region_head=True, boundary_mode=boundary_mode)
    if boundary_mode == 'half_open':
        tokens = torch.zeros(labels.shape[0], 16, dtype=torch.long)
        tokens[:, :READ] = 1
        del out['position_mask']
        out['start_mask'], out['end_mask'] = boundary_masks(tokens, 'half_open')
    out['region_logits'] = logits
    targets['region_labels'] = labels
    return cfg, out, targets


def regions_term(cfg, out, targets, log_var=None):
    tasks = build_tasks(cfg)
    weights = tasks.make_weights()
    if log_var is not None:
        with torch.no_grad():
            weights['regions'].log_var.fill_(log_var)
    total, parts = tasks.loss(out, targets, weights)
    return total, parts['regions']


def test_perfect_region_logits_cost_almost_nothing(task_batch):
    labels = read_labels()
    logits = (20.0 * F.one_hot(labels.clamp(min=0), CLASSES).float()).requires_grad_()
    _, term = regions_term(*region_task(task_batch, labels, logits))
    assert 0 <= term.item() < 1e-3


def test_the_regions_term_is_a_kendall_weighted_cross_entropy(task_batch):
    """Uniform logits cost log(7) a position; at log-variance s the term is log(7) e^-s + s / 2."""
    cfg, out, targets = region_task(task_batch, read_labels(), torch.zeros(2, 16, CLASSES, requires_grad=True))
    assert 'regions' in loss_weight_names(cfg)
    _, term = regions_term(cfg, out, targets, log_var=1.0)
    assert term.item() == pytest.approx(math.log(CLASSES) * math.exp(-1) + 0.5)


@pytest.mark.parametrize('boundary_mode', ['residue', 'half_open'])
def test_padding_positions_carry_no_region_loss(task_batch, boundary_mode):
    """Whatever the pad positions' logits and labels hold, even a real class, the term and the total are those of the
    read's positions alone, and no gradient reaches a pad logit: the term reads the model's read positions (the
    position mask, or the half-open start mask)."""
    labels = read_labels()
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(2, 16, CLASSES, generator=generator)
    clean = logits.clone().requires_grad_()
    noisy = logits.clone()
    noisy[:, READ:] = 50 * torch.randn(2, 16 - READ, CLASSES, generator=generator)
    noisy.requires_grad_()
    labelled = labels.clone()
    labelled[:, READ:] = 3
    clean_total, clean_term = regions_term(*region_task(task_batch, labels, clean, boundary_mode))
    noisy_total, noisy_term = regions_term(*region_task(task_batch, labelled, noisy, boundary_mode))
    expected = F.cross_entropy(logits[:, :READ].reshape(-1, CLASSES), labels[:, :READ].reshape(-1))
    assert clean_term.item() == pytest.approx(expected.item())
    assert noisy_term.item() == pytest.approx(expected.item()) and noisy_total.item() == pytest.approx(clean_total.item())
    noisy_term.backward()
    assert not noisy.grad[:, READ:].any() and noisy.grad[:, :READ].abs().sum() > 0


def test_a_batch_without_read_positions_has_a_zero_regions_term(task_batch):
    labels = torch.full((2, 16), -100)
    logits = torch.randn(2, 16, CLASSES, requires_grad=True)
    total, term = regions_term(*region_task(task_batch, labels, logits))
    assert term.item() == 0 and torch.isfinite(total)
    term.backward()
    assert not logits.grad.any()


@pytest.mark.parametrize('missing', ['region_logits', 'region_labels'])
def test_the_region_head_needs_its_logits_and_its_labels(task_batch, missing):
    cfg, out, targets = region_task(task_batch, read_labels(), torch.zeros(2, 16, CLASSES))
    out.pop(missing, None)
    targets.pop(missing, None)
    tasks = build_tasks(cfg)
    with pytest.raises(TrainingDataError, match=missing):
        tasks.loss(out, targets, tasks.make_weights())


def test_region_head_without_a_trunk_is_refused():
    """conv-v1 has no trunk, so it has no per-position features for the region head to read."""
    cfg = AlignAIRConfig(v_allele_count=2, j_allele_count=2, d_allele_count=2, max_seq_length=128, embed_dim=4,
                         filters=4, block_out=8, region_head=True)
    with pytest.raises(ConfigurationError, match='trunk') as caught:
        AlignAIR(cfg)
    assert caught.value.context['component'] == 'trunk_heads'
    assert caught.value.context['fields'] == ['region_head']


def test_the_region_head_flag_is_a_boolean():
    with pytest.raises(ConfigurationError) as caught:
        AlignAIRConfig(v_allele_count=2, j_allele_count=2, region_head=1)
    assert caught.value.context['field'] == 'region_head'
