"""AIRR rows of a model with a D presence head, written by AlignAIR's AIRR writer and read back by the assay's AIRR
reader: a file mixing rows whose D was dropped (absent: no D field at all), kept without a call (short) and called has
one header, the full schema, and the absent row's D cells are empty, so the reader sees no D and flags nothing."""
import csv
import itertools

import pytest

pytest.importorskip("GenAIRR")

from alignair_assay.airr.reader import load_airr_tsv  # noqa: E402

pytestmark = pytest.mark.integration

D_FIELDS = ("d_call", "d_sequence_start", "d_sequence_end", "d_germline_start", "d_germline_end", "d_cigar")


def _product_read(record, d_presence):
    """The typed read AlignAIR's pipeline would give for a GenAIRR record read perfectly (forward, its own calls,
    coordinates and CIGARs), its D gated by the D presence class ``d_presence``."""
    from alignair.predict.state import GermlineAlignment
    from tests.support.read_results import make_read
    calls = {g: record[f"{g}_call"].split(",")[0] for g in "vdj"}
    alignments = {g: GermlineAlignment(calls[g], record[f"{g}_sequence_start"], record[f"{g}_sequence_end"],
                                       record[f"{g}_germline_start"], record[f"{g}_germline_end"], record[f"{g}_cigar"])
                  for g in "vdj"}
    return make_read(record["sequence"].upper(), calls=calls, alignments=alignments, genes=("v", "d", "j"),
                     d_presence=d_presence)


def test_a_file_mixing_rows_with_and_without_d_fields_has_one_header_and_empty_d_cells(tmp_path):
    import GenAIRR.data as gd
    from alignair.core.d_presence import D_CLASSES
    from alignair.io.airr import COLUMNS, write_airr
    from alignair.predict.airr import build_airr
    from alignair.predict.results import light_record
    from alignair.reference.reference_set import ReferenceSet
    from alignair.train.gym import Curriculum
    from alignair.train.gym.streams import stream_records
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    found = [r for r in itertools.islice(stream_records(gd.HUMAN_IGH_OGRDB, dict(Curriculum().params(0.3)), 5), 20)
             if all(r.get(f"{g}_cigar") for g in "vdj")][:3]
    records = [light_record(_product_read(r, D_CLASSES[klass])) for r, klass in zip(found, (0, 1, 2))]
    assert not any(key in records[0] for key in D_FIELDS[1:]) and records[0]["d_call"] == ""
    rows = build_airr(records, reference, chain="heavy")
    path = tmp_path / "gated.tsv"
    ids = ["absent", "short", "callable"]
    write_airr(str(path), ids, [r["sequence"] for r in records], rows)
    with open(path, newline="") as handle:
        table = list(csv.DictReader(handle, delimiter="\t"))
    assert list(table[0]) == list(COLUMNS) and all(len(row) == len(COLUMNS) for row in table)
    assert all(table[0][field] == "" for field in D_FIELDS)
    assert table[1]["d_call"] == "" and table[1]["d_sequence_start"] != ""
    assert table[2]["d_call"] == found[2]["d_call"].split(",")[0]
    predictions, stats = load_airr_tsv(path)
    assert stats["invalid_rows"] == sum(bool(p.invalid) for p in predictions.values())
    assert not any(message.startswith("d") for p in predictions.values() for message in p.invalid)
    absent, short, called = (predictions[i].segments["d"] for i in ids)
    assert (absent.call, absent.seq_start, absent.seq_end, absent.cigar) == (None, None, None, None)
    assert short.call is None and short.seq_start == found[1]["d_sequence_start"]
    assert called.call == found[2]["d_call"].split(",")[0] and called.seq_end == found[2]["d_sequence_end"]
    assert [rows[k]["d_presence"] for k in range(3)] == list(D_CLASSES)
