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


def _product_record(record):
    """The light record AlignAIR's pipeline would give for a GenAIRR record read perfectly (forward, its own calls,
    coordinates and CIGARs)."""
    sequence = record["sequence"].upper()
    light = {"sequence": sequence, "input_sequence": sequence, "orientation_id": 0, "productive": True,
             "indel_count": 0.0, "mutation_rate": 0.0}
    for g in "vdj":
        call = record[f"{g}_call"].split(",")[0]
        light.update({f"{g}_call": call, f"{g}_calls": [call], f"{g}_likelihoods": [1.0]})
        for key in ("sequence_start", "sequence_end", "germline_start", "germline_end", "cigar"):
            light[f"{g}_{key}"] = record[f"{g}_{key}"]
    return light


def test_a_file_mixing_rows_with_and_without_d_fields_has_one_header_and_empty_d_cells(tmp_path):
    import GenAIRR.data as gd
    from alignair.core.d_presence import D_CLASSES
    from alignair.io.airr import COLUMNS, write_airr
    from alignair.predict.airr import build_airr
    from alignair.predict.stages import gate_d as _gate_d
    from alignair.reference.reference_set import ReferenceSet
    from alignair.train.gym import Curriculum
    from alignair.train.gym.streams import stream_records
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    found = [r for r in itertools.islice(stream_records(gd.HUMAN_IGH_OGRDB, dict(Curriculum().params(0.3)), 5), 20)
             if all(r.get(f"{g}_cigar") for g in "vdj")][:3]
    records = [_product_record(r) for r in found]
    for record, klass in zip(records, (0, 1, 2)):                  # absent, short, callable
        _gate_d(record, klass)
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
