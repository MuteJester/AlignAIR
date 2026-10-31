"""The chip's SHM reader: each base's mutation and insertion probability and its column in each gene's frame."""
import pytest
import torch

from alignair.reference.frames import FRAME_WIDTHS
from alignair_lab.architectures.chip.reader import EDIT_STATES, ShmReader, top_coordinates

pytestmark = pytest.mark.component


def test_shapes():
    reader = ShmReader(channels=8)
    out = reader(torch.randn(2, 10, 8), torch.ones(2, 10, dtype=torch.bool))
    assert EDIT_STATES == ("germline", "substitution", "insertion")
    assert out.edit_logits.shape == (2, 10, 3)
    assert {gene: logits.shape for gene, logits in out.coord_logits.items()} == {
        gene: (2, 10, width) for gene, width in FRAME_WIDTHS.items()}


def test_m_and_inserted_are_probabilities():
    out = ShmReader(channels=8)(torch.randn(2, 10, 8), torch.ones(2, 10, dtype=torch.bool))
    probabilities = out.edit_logits.softmax(-1)
    assert torch.allclose(out.m, probabilities[..., 1]) and torch.allclose(out.inserted, probabilities[..., 2])


def test_top_coordinates_are_the_k_most_probable_with_their_softmax_probabilities():
    logits = torch.tensor([[[0., 3., 1., 2.]]])
    columns, probs = top_coordinates(logits, k=3)
    assert columns.tolist() == [[[1, 3, 2]]]
    assert torch.allclose(probs, logits.softmax(-1)[..., [1, 3, 2]])
