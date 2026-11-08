"""The longest read a model sees: the padded width, less the terminal end slot of a half-open segment end."""
import pytest

from alignair.core.boundaries import read_window

pytestmark = pytest.mark.component


def test_the_window_leaves_the_terminal_slot():
    assert read_window(576) == 575


def test_a_forward_width_is_rounded_up_to_its_bucket_and_capped_at_the_window():
    from alignair.core.boundaries import WIDTH_BUCKET, width_bucket
    assert WIDTH_BUCKET == 32
    assert [width_bucket(n, 576) for n in (1, 32, 33, 64, 65, 545, 560, 576)] == [32, 32, 64, 64, 96, 576, 576, 576]
    assert width_bucket(40, 50) == 50
