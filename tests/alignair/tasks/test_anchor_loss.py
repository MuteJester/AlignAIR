"""The anchor objective (``AlignAIRConfig.anchor_heads``): for each of the V and J anchors, a soft-label cross entropy
over the read's codon starts (the positions where a whole codon fits in the read, the slots an anchor can take) plus
the null slot L. An observed anchor's target is a Gaussian (sigma 1) over the codon starts around it, renormalised,
with no weight on the null slot; an anchor that is not observed is a one-hot on the null slot. The read's last two
positions and the pads are left out. Each anchor has its own Kendall weight (``v_anchor``, ``j_anchor``). The
productive head and its term are dropped: productivity is derived from the anchors."""
from dataclasses import replace
import math

import pytest
import torch

from alignair.core.boundaries import boundary_masks
from alignair.core.components import build_meta_heads
from alignair.errors import TrainingDataError
from alignair.tasks import build_tasks
from alignair.tasks.losses import loss_weight_names

pytestmark = pytest.mark.component

READ, WIDTH = 8, 16              # the task batch's reads fill the first 8 of its 16 positions; the null slot is 16
STARTS = READ - 2                # a whole codon fits at the read positions 0..5
ANCHORS = {"v_anchor": torch.tensor([3, WIDTH]), "j_anchor": torch.tensor([4, 5])}


def expected_target(anchor):
    """The target a reader can check by hand: exp(-(k - a)^2 / 2) on the codon starts k, renormalised, 0 on the null
    slot; a one-hot on the null slot when the anchor is the null slot."""
    target = torch.zeros(WIDTH + 1)
    if anchor == WIDTH:
        target[WIDTH] = 1.
        return target
    positions = torch.arange(STARTS, dtype=torch.float32)
    target[:STARTS] = torch.exp(-0.5 * (positions - anchor) ** 2)
    return target / target.sum()


def anchor_task(task_batch, logits=None, anchors=ANCHORS, boundary_mode="residue", meta=False):
    cfg, out, targets = task_batch(protein=False, has_d=True, auxiliary=meta)
    cfg = replace(cfg, anchor_heads=True, boundary_mode=boundary_mode)
    if boundary_mode == "half_open":
        tokens = torch.zeros(2, WIDTH, dtype=torch.long)
        tokens[:, :READ] = 1
        del out["position_mask"]
        out["start_mask"], out["end_mask"] = boundary_masks(tokens, "half_open")
    for key, anchor in anchors.items():
        out[f"{key}_logits"] = (torch.zeros(2, WIDTH + 1) if logits is None else logits[key]).requires_grad_()
        targets[key] = anchor
    return cfg, out, targets


def anchor_terms(cfg, out, targets, log_vars=None):
    tasks = build_tasks(cfg)
    weights = tasks.make_weights()
    for key, value in (log_vars or {}).items():
        with torch.no_grad():
            weights[key].log_var.fill_(value)
    total, parts = tasks.loss(out, targets, weights)
    return total, parts


def test_perfect_anchor_logits_cost_almost_nothing(task_batch):
    """Logits that are the target's own log-probabilities cost the target's entropy and nothing more: the soft
    target's spread (sigma 1), and 0 for a one-hot null. A logit peaked one position off costs more."""
    targets = {key: torch.stack([expected_target(int(a)) for a in anchor]) for key, anchor in ANCHORS.items()}
    logits = {key: target.clamp_min(1e-12).log() for key, target in targets.items()}
    _, parts = anchor_terms(*anchor_task(task_batch, logits))
    for key, target in targets.items():
        entropy = -(target * target.clamp_min(1e-12).log()).sum(-1).mean()
        assert 0 <= parts[key].item() - entropy.item() < 1e-3, key
    assert -(targets["v_anchor"][1] * targets["v_anchor"][1].clamp_min(1e-12).log()).sum() == 0   # the null row
    off = {key: torch.roll(value, 1, dims=-1) for key, value in logits.items()}
    _, worse = anchor_terms(*anchor_task(task_batch, off))
    assert all(worse[key] > parts[key] + 0.2 for key in ANCHORS)      # J's anchors sit at the read's edge


def test_the_anchor_terms_are_two_kendall_weighted_cross_entropies(task_batch):
    """Uniform logits cost log(7) a read (its 6 codon starts and the null slot), whatever the target; at log-variance
    s the term is log(7) e^-s + s / 2, with each anchor's own s."""
    cfg, out, targets = anchor_task(task_batch)
    assert loss_weight_names(cfg)[-2:] == ("v_anchor", "j_anchor")
    _, parts = anchor_terms(cfg, out, targets, log_vars={"v_anchor": 1.0, "j_anchor": -0.5})
    assert parts["v_anchor"].item() == pytest.approx(math.log(STARTS + 1) * math.exp(-1) + 0.5)
    assert parts["j_anchor"].item() == pytest.approx(math.log(STARTS + 1) * math.exp(0.5) - 0.25)


@pytest.mark.parametrize("boundary_mode", ["residue", "half_open"])
def test_padding_positions_carry_no_anchor_loss(task_batch, boundary_mode):
    """Whatever the logits of the pads and of the read's last two positions hold, the terms are those of the codon
    starts and the null slot, and no gradient reaches the others (the read is the position mask, or the half-open
    start mask)."""
    generator = torch.Generator().manual_seed(0)
    logits = {key: torch.randn(2, WIDTH + 1, generator=generator) for key in ANCHORS}
    noisy = {key: value.clone() for key, value in logits.items()}
    for value in noisy.values():
        value[:, STARTS:WIDTH] = 50.0
    cfg, out, targets = anchor_task(task_batch, logits, boundary_mode=boundary_mode)
    total, parts = anchor_terms(cfg, out, targets)
    _, noisy_parts = anchor_terms(*anchor_task(task_batch, noisy, boundary_mode=boundary_mode))
    for key in ANCHORS:
        assert noisy_parts[key].item() == pytest.approx(parts[key].item())
    total.backward()
    for key in ANCHORS:
        gradient = out[f"{key}_logits"].grad
        assert not gradient[:, STARTS:WIDTH].any() and gradient[:, :STARTS].any() and gradient[:, WIDTH].any(), key


def test_anchor_heads_drop_the_productive_head_and_term(task_batch):
    """With the anchor heads, the meta heads have no productive head, the weights no productive weight, and the loss
    no productive term and no need of a productive output; without them, all three are as before."""
    cfg, out, targets = anchor_task(task_batch, meta=True)
    off = replace(cfg, anchor_heads=False)
    assert "productive" in build_meta_heads(off) and "productive" not in build_meta_heads(cfg)
    assert set(build_meta_heads(off)) - set(build_meta_heads(cfg)) == {"productive"}
    assert "productive" in loss_weight_names(off) and "productive" not in loss_weight_names(cfg)
    assert loss_weight_names(cfg) == tuple(k for k in loss_weight_names(off) if k != "productive") + (
        "v_anchor", "j_anchor")
    del out["productive"]
    _, parts = anchor_terms(cfg, out, targets)
    assert "productive" not in parts and {"v_anchor", "j_anchor", "mutation", "indel"} <= set(parts)


@pytest.mark.parametrize("missing", ["v_anchor_logits", "j_anchor_logits", "v_anchor", "j_anchor"])
def test_the_anchor_heads_need_their_logits_and_their_targets(task_batch, missing):
    cfg, out, targets = anchor_task(task_batch)
    (out if missing.endswith("_logits") else targets).pop(missing)
    with pytest.raises(TrainingDataError) as caught:
        anchor_terms(cfg, out, targets)
    assert missing in caught.value.context["fields"]
