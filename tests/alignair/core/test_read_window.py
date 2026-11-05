"""The longest read a model sees: the padded width, less the terminal end slot of a half-open segment end."""
import pytest

from alignair.core.boundaries import read_window

pytestmark = pytest.mark.component


def test_the_window_leaves_the_terminal_slot():
    assert read_window(576) == 575
