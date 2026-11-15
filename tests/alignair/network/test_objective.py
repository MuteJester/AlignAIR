"""The chip's objective: every unit trained on its own GenAIRR truth, each loss divided by its chance level."""
import math

import pytest
import torch

from alignair.tasks.objective import build_tasks

from alignair.network.objective import (
    LOSS_WEIGHTS, ChipObjective, parse_loss, scan_loss, shm_labels)
from alignair.core.regions import REGIONS
from alignair.network.parser import crf
from alignair.network.scan import OUT

pytestmark = pytest.mark.component

V, N1, J = REGIONS.index("V"), REGIONS.index("N1"), REGIONS.index("J")


def _path(length, v_end, j_start):
    path = torch.full((length,), N1)
    path[:v_end], path[j_start:] = V, J
    return path


def _misplaced_parse_loss(length):
    """The parse loss of a read V [0, L-100), N1 5 bases, J to the end, whose parser is sure of a path with both
    boundaries 2 bases late."""
    true = _path(length, length - 100, length - 95)
    believed = _path(length, length - 98, length - 93)
    emissions = torch.full((1, length, len(REGIONS)), -10.)
    emissions[0, torch.arange(length), believed] = 10.
    mask = torch.ones(1, length, dtype=torch.bool)
    allowed = torch.zeros(1, length, len(REGIONS), dtype=torch.bool)
    allowed[0, torch.arange(length), true] = True
    output = crf(emissions, torch.zeros(1, length, len(REGIONS), len(REGIONS)), mask)
    return float(parse_loss(output, allowed, true.unsqueeze(0), mask))


def test_the_parse_loss_counts_boundaries_not_bases():
    short, long = _misplaced_parse_loss(200), _misplaced_parse_loss(500)
    assert short >= 0.5 and long >= 0.5           # a boundary 2 bases off is plainly visible
    assert abs(long - short) / short < 0.25         # and a longer read does not dilute it

PARTS = {"orientation", "parse", "v_anchor", "j_anchor", "shm", "coord", "scan_v", "scan_d", "scan_j", "d_presence"}


def _full_bank(batch):
    """The batch with every allele in its bank, so every scan term is trained."""
    inputs = dict(batch.inputs)
    for gene in "vdj":
        inputs[f"bank_mask_{gene}"] = torch.ones_like(inputs[f"bank_mask_{gene}"])
    return inputs


def test_build_tasks_gives_the_chip_objective(chip):
    _, _, setup = chip
    assert isinstance(build_tasks(setup.config), ChipObjective)


def test_every_part_is_near_one_at_initialization(chip, monkeypatch):
    """The model's own predictions: no read is given its true columns (a taught read's scan term is near 0)."""
    from alignair.network import chip as chip_core
    monkeypatch.setattr(chip_core, "P_TEACHER", 0.0)
    model, batch, setup = chip
    model.train()
    total, parts = ChipObjective(setup.config).loss(model(_full_bank(batch)), batch.targets)
    assert set(parts) == PARTS and set(LOSS_WEIGHTS) == PARTS
    for name, value in parts.items():
        assert 0.3 <= float(value.detach()) <= 2.0, (name, float(value.detach()))
    assert torch.allclose(total, sum(parts.values()))


def test_scan_loss_counts_any_allele_in_the_true_set():
    log_posterior = torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.4]]))
    true = torch.tensor([[False, True, False, True]])
    loss, rows = scan_loss(log_posterior, true, torch.tensor([True]))
    assert rows.tolist() == [True]
    assert torch.allclose(loss, torch.tensor([-math.log(0.6) / math.log(4)]))


def test_a_one_allele_bank_skips_the_scan_term():
    log_posterior = torch.tensor([[0., OUT, OUT]])
    _, rows = scan_loss(log_posterior, torch.tensor([[True, False, False]]), torch.tensor([True]))
    assert rows.tolist() == [False]


def test_deletion_labels_and_non_germline_bases_are_ignored_by_the_shm_term():
    state = torch.tensor([[0, 1, 2, 3, 0, -100]])            # germline, substitution, insertion, deletion, ..., pad
    region = torch.tensor([[REGIONS.index("V")] * 4 + [REGIONS.index("N1"), -100]])
    assert shm_labels(state, region).tolist() == [[0, 1, 2, -100, -100, -100]]


def test_one_training_step_lowers_the_loss(chip):
    model, batch, setup = chip
    objective = ChipObjective(setup.config)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    inputs = _full_bank(batch)
    losses = []
    for _ in range(10):
        torch.manual_seed(0)
        total, _ = objective.loss(model(inputs), batch.targets)
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        losses.append(float(total.detach()))
    assert losses[-1] < losses[0]


def test_metric_report_has_the_unit_metrics(chip):
    model, batch, setup = chip
    model.eval()
    report = ChipObjective(setup.config).metric_report(model(_full_bank(batch)), batch.targets).means()
    for name in ("v_allele_top1", "parse_v_end_exact", "anchor_v_exact", "shm_brier", "coord_v_top1", "scan_v_top1",
                 "bank_v_size", "d_presence_acc"):
        assert name in report and math.isfinite(report[name]), name
