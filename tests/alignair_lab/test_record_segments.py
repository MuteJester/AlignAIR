"""How the cache writer cuts streams into segments, each a range of records drawn from its own seed: the slow selected
streams are cut so that every writer stays busy to the end, the fast unselected ones are written whole, chunk numbers
run on across a stream's segments, and the selected segments are handed out first."""
import os

import pytest

from alignair_lab.records import (MIN_SEGMENT_RECORDS, SEGMENTS_PER_WRITER, Segment, default_writers, physical_cores,
                                   segment_plan)

pytestmark = pytest.mark.component


def test_segments_of_a_given_size_cut_every_stream_and_number_its_chunks_on():
    plan = segment_plan([10, 7], [False, False], workers=2, chunk_records=4, segment_records=5)
    assert plan == [Segment(0, 0, 5, 0, False, True), Segment(0, 5, 10, 2, True, True),
                    Segment(1, 0, 5, 0, False, True), Segment(1, 5, 7, 2, True, True)]


def test_only_selected_streams_are_cut_and_their_segments_come_first():
    total = 4096 * 3
    plan = segment_plan([1000, total], [False, True], workers=2, chunk_records=4096)
    size = total // (SEGMENTS_PER_WRITER * 2)                          # 1,024 records: 12 segments for 2 writers
    assert [segment.stream for segment in plan] == [1] * 12 + [0]
    assert [(segment.start, segment.stop) for segment in plan[:12]] == [(k * size, (k + 1) * size) for k in range(12)]
    assert [segment.first_chunk for segment in plan[:12]] == list(range(12)) and plan[11].last
    assert plan[12] == Segment(0, 0, 1000, 0, True, False)             # written whole, as before


def test_segments_are_at_least_the_minimum_and_at_most_a_chunk():
    small = segment_plan([300], [True], workers=16, chunk_records=4096)
    assert [(segment.start, segment.stop) for segment in small] == [(0, MIN_SEGMENT_RECORDS),
                                                                     (MIN_SEGMENT_RECORDS, 2 * MIN_SEGMENT_RECORDS),
                                                                     (2 * MIN_SEGMENT_RECORDS, 300)]
    large = segment_plan([100_000], [True], workers=1, chunk_records=4096)
    assert {segment.stop - segment.start for segment in large[:-1]} == {4096}


CPUINFO = ''.join(f'processor\t: {n}\nphysical id\t: {socket}\ncore id\t\t: {core}\n\n'
                  for n, (socket, core) in enumerate((s, c) for s in (0, 1) for c in (0, 1, 0, 1)))


def test_the_default_is_a_writer_per_physical_core():
    """Two sockets of two cores of two hardware threads: 4 writers. GenAIRR's simulation gains nothing from a core's
    second thread (a 1,000-step v3 cache: 2.9 min with 8 writers on 8 cores, 4.4 min with 16 on their 16 threads)."""
    assert physical_cores(CPUINFO) == 4
    assert physical_cores('') == (os.cpu_count() or 1)                  # no core ids: every logical CPU
    assert default_writers(100) == physical_cores() and default_writers(1) == 1


def test_without_a_selected_stream_every_stream_is_written_whole():
    assert segment_plan([10, 7], [False, False], workers=4, chunk_records=4) == [
        Segment(0, 0, 10, 0, True, False), Segment(1, 0, 7, 0, True, False)]
