"""On the golden conv-v1 rows (``golden_rows``: reads with V indels, V starting inside its germline), each assembled D
and J sequence alignment is the read's own bases at that segment's coordinates, whatever the V before it."""
import json

import pytest

from tests.alignair.predict.golden_rows import FIXTURE, golden_model, predict_rows

pytestmark = pytest.mark.integration


def test_every_d_and_j_sequence_alignment_is_the_reads_own_segment():
    rows = predict_rows(*golden_model(), json.loads(FIXTURE.read_text())["reads"])
    checked = 0
    for row in rows:
        for g in "dj":
            start, aligned = row.get(f"{g}_sequence_start"), row.get(f"{g}_sequence_alignment")
            if start is not None and aligned and not row.get("rev_comp"):
                assert aligned == row["sequence"][start:row[f"{g}_sequence_end"]], (row.get("sequence_id"), g)
                checked += 1
    assert checked >= 10


def test_every_d_and_j_germline_alignment_takes_its_sequence_alignments_columns():
    rows = predict_rows(*golden_model(), json.loads(FIXTURE.read_text())["reads"])
    checked = 0
    for row in rows:
        for g in "dj":
            aligned, germline = row.get(f"{g}_sequence_alignment"), row.get(f"{g}_germline_alignment")
            if aligned and germline:
                assert len(germline) == len(aligned), (row.get("sequence_id"), g, aligned, germline)
                checked += 1
    assert checked >= 10
