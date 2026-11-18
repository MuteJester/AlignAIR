"""The cache writer's time left: each stream's remaining records at its own rate (a stream not started yet at the mean
rate of the streams of its kind, selected or unselected, that have written records), shared among the writers, and
never less than the slowest running stream's own time left. A single records-per-second figure cannot do this:
selected streams are about 20 times slower to simulate than unselected ones. Streams of a kind no stream has measured
yet are left out, and the estimate is then a lower bound."""
import pytest

from alignair_lab.records import seconds_left

pytestmark = pytest.mark.component


def test_the_slowest_running_stream_bounds_the_time_left():
    # two selected streams at 10 records/s with 1,000 records left each; two unselected streams finished
    left = seconds_left(done=[500, 500, 900, 900], totals=[1500, 1500, 900, 900], running=[50., 50., 3., 3.],
                        selected=[True, True, False, False], finished=[False, False, True, True], workers=16)
    assert left == (pytest.approx(100.), True)


def test_queued_streams_share_the_writers_at_the_rate_of_their_kind():
    # one writer: a running selected stream (10 records/s, 100 s left), a queued selected stream of 1,000 records
    # (100 s at its kind's rate) and a queued unselected stream of 1,000 records
    arguments = dict(totals=[2000, 1000, 1000, 1000], selected=[True, True, False, False], workers=1)
    # no unselected stream has written records: the queued one is left out, and the estimate is a lower bound
    assert seconds_left(done=[1000, 0, 0, 0], running=[100., 0., 0., 0.], finished=[False, False, False, False],
                        **arguments) == (pytest.approx(200.), False)
    # a finished unselected stream wrote 1,000 records in 2 s: the queued one takes 2 s more
    assert seconds_left(done=[1000, 0, 0, 1000], running=[100., 0., 0., 2.], finished=[False, False, False, True],
                        **arguments) == (pytest.approx(202.), True)


def test_nothing_left_is_zero_and_no_rate_yet_is_unknown():
    assert seconds_left(done=[5], totals=[5], running=[1.], selected=[True], finished=[True], workers=2) == (0., True)
    assert seconds_left(done=[0], totals=[5], running=[0.], selected=[True], finished=[False], workers=2) == \
        (None, False)
