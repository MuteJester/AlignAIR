"""The chip's SHM reader: each base's mutation and insertion probability and its column in each gene's frame."""
import pytest
import torch

from alignair.reference.frames import FRAME_WIDTHS
from alignair.network.reader import READ_EDITS, ShmReader, top_coordinates

pytestmark = pytest.mark.component


def test_shapes():
    reader = ShmReader(channels=8)
    out = reader(torch.randn(2, 10, 8), torch.ones(2, 10, dtype=torch.bool))
    assert READ_EDITS == ("germline", "substitution", "insertion")
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


def test_a_reader_asked_for_no_coordinates_reads_only_the_edit_states():
    reader = ShmReader(channels=8)
    features, mask = torch.randn(2, 10, 8), torch.ones(2, 10, dtype=torch.bool)
    out = reader(features, mask, coordinates=False)
    assert out.coord_logits == {} and torch.equal(out.edit_logits, reader(features, mask).edit_logits)


def test_top_columns_read_each_reads_window_of_supported_bases_and_nothing_outside_it():
    """From each read's first to its last supported base (an unsupported base inside included) the columns and their
    probabilities are the whole read's; outside the window, and in a read with no supported base, the probabilities
    are 0 (no weight in the scan)."""
    torch.manual_seed(0)
    reader = ShmReader(channels=8)
    features = torch.randn(3, 12, 8)
    support = torch.zeros(3, 12, dtype=torch.bool)
    support[0, [3, 4, 6, 7]] = True                     # base 5 inside the window, unsupported
    support[1, [0, 11]] = True                          # the whole read
    whole_columns, whole_probs = top_coordinates(reader(features, torch.ones(3, 12, dtype=torch.bool))
                                                 .coord_logits["d"], k=3)
    columns, probs = reader.top_columns(features, "d", support, k=3)
    window = torch.zeros(3, 12, dtype=torch.bool)
    window[0, 3:8] = True
    window[1] = True
    assert torch.equal(columns[window], whole_columns[window])
    assert torch.allclose(probs[window], whole_probs[window], atol=1e-6)
    assert (probs[~window] == 0).all()
