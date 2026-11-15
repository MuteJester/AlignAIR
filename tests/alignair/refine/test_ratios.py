"""The refine stages read one set of per-base evidence ratios, and the chip and the refine stages one boundary
expectation."""
import math

import pytest
import torch

from alignair.network.parser import IMPOSSIBLE, expected_slot
from alignair.refine.ratios import OUTSIDE, RATE, pair_ratios

pytestmark = pytest.mark.component


def test_a_base_pair_weighs_match_against_mismatch_at_the_read_mutation_rate():
    match, mismatch = pair_ratios(torch.tensor([[0.25], [0.01]]))
    assert torch.allclose(match, torch.tensor([[math.log(3.)], [math.log(3.96)]]))
    assert torch.allclose(mismatch, torch.tensor([[math.log(1 / 3)], [math.log(0.04 / 3)]]))
    assert RATE == (0.01, 0.25) and OUTSIDE == -8.0


def test_a_boundary_is_the_expected_slot_over_the_slots_a_read_has():
    logits = torch.tensor([[0., 0., 5., 0.]])
    slots = torch.tensor([[True, True, False, True]])
    assert expected_slot(logits, slots).tolist() == [[pytest.approx(4 / 3)]]
    assert IMPOSSIBLE == -1e4
