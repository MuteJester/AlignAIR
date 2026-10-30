"""The longest read a model sees: the padded width, less the end slot a half-open model reserves."""
import pytest

from alignair.core.boundaries import read_window

pytestmark = pytest.mark.component


def test_a_half_open_window_leaves_the_terminal_slot():
    assert read_window(576, "residue") == 576
    assert read_window(576, "half_open") == 575
