"""Tests for the `alignair predict-aa` CLI verb (Task 8, design spec §10): file parsing here (not the
façade), JSONL output + `<out>.run.json` provenance sidecar."""
from __future__ import annotations

import json

import pytest

from alignair.aa.product.cli import _read_records, run
from alignair.cli.main import build_parser

pytestmark = pytest.mark.component

_VH_1 = ("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAV"
        "YYCSRWGGDGFYAMDYWGQGTLVTVSS")
_VH_2 = ("EVQLVESGGGLVQPGGSLRLSCAASGYTFTNYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAV"
        "YYCAKYPHYYGSSHWYFDVWGQGTLVTVSS")


def _write_fasta(path, records):
    with open(path, "w") as f:
        for seq_id, seq in records:
            f.write(f">{seq_id}\n{seq}\n")


def _parse_args(argv):
    return build_parser().parse_args(argv)


# --------------------------------------------------------------------------------- _read_records


def test_read_records_parses_fasta_headers_and_sequences(tmp_path):
    fasta = tmp_path / "reads.fasta"
    _write_fasta(fasta, [("read-1", _VH_1), ("read-2", _VH_2)])

    records = _read_records(str(fasta))

    assert records == [{"sequence_id": "read-1", "sequence": _VH_1},
                       {"sequence_id": "read-2", "sequence": _VH_2}]


def test_read_records_does_not_apply_nucleotide_cleaning(tmp_path):
    """Regression guard: a naive reuse of `alignair.io.read_sequences` would rewrite most AA letters
    to 'N' (nucleotide-alphabet cleaning) and likely drop the record as >20% ambiguous. The AA reader
    must return the amino-acid sequence byte-for-byte."""
    fasta = tmp_path / "one.fasta"
    _write_fasta(fasta, [("vh", _VH_1)])

    [record] = _read_records(str(fasta))
    assert record["sequence"] == _VH_1


def test_read_records_plain_text_one_sequence_per_line_gets_no_id(tmp_path):
    txt = tmp_path / "reads.txt"
    txt.write_text(f"{_VH_1}\n{_VH_2}\n")

    records = _read_records(str(txt))
    assert [r["sequence"] for r in records] == [_VH_1, _VH_2]
    assert all(r["sequence_id"] is None for r in records)


def test_read_records_stdin(monkeypatch):
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO(f">only\n{_VH_1}\n"))
    records = _read_records("-")
    assert records == [{"sequence_id": "only", "sequence": _VH_1}]


def test_read_records_ignores_leading_blank_lines_before_first_header(tmp_path):
    """A FASTA with blank lines before the first >header should not create an empty record."""
    fasta = tmp_path / "reads.fasta"
    with open(fasta, "w") as f:
        f.write(f"\n\n>read-1\n{_VH_1}\n>read-2\n{_VH_2}\n")

    records = _read_records(str(fasta))
    assert len(records) == 2
    assert records[0] == {"sequence_id": "read-1", "sequence": _VH_1}
    assert records[1] == {"sequence_id": "read-2", "sequence": _VH_2}


# --------------------------------------------------------------------------------- CLI end-to-end


def test_predict_aa_parser_registered():
    parser = build_parser()
    args = parser.parse_args(["predict-aa", "-", "--out", "/tmp/whatever.jsonl"])
    assert args.command == "predict-aa"
    assert args.model == "alignair-aa-igh-human@2.0.0"
    assert args.coverage_group is None
    assert args.device == "auto"


def test_cli_writes_jsonl_and_run_metadata_sidecar(tmp_path, capsys):
    fasta = tmp_path / "reads.fasta"
    _write_fasta(fasta, [("read-1", _VH_1), ("read-2", _VH_2)])
    out = tmp_path / "out.jsonl"

    args = _parse_args(["predict-aa", str(fasta), "--out", str(out)])
    rc = run(args)
    assert rc == 0

    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert [p["sequence_id"] for p in parsed] == ["read-1", "read-2"]
    for p in parsed:
        assert set(p["segments"]) == {"V", "D", "J"}
        assert p["provenance"]["model_id"] == "alignair-aa-igh-human"

    run_json_path = str(out) + ".run.json"
    import os
    assert os.path.exists(run_json_path)
    meta = json.loads(open(run_json_path).read())
    for key in ("model_id", "model_version", "weights_sha256", "calibration_sha256",
               "class_map_db_hash", "alignair_version", "calibration_scope",
               "n_input", "n_predicted", "n_errors", "command", "timestamp"):
        assert key in meta, key
    assert meta["model_id"] == "alignair-aa-igh-human"
    assert meta["weights_sha256"]
    assert "simulator" in meta["calibration_scope"].lower()
    assert meta["n_input"] == 2
    assert meta["n_predicted"] == 2
    assert meta["n_errors"] == 0
    assert meta["command"]["input"] == str(fasta)
    assert meta["command"]["out"] == str(out)

    captured = capsys.readouterr()
    assert "predicted 2 sequences" in captured.out


def test_cli_no_run_metadata_skips_sidecar(tmp_path):
    fasta = tmp_path / "reads.fasta"
    _write_fasta(fasta, [("read-1", _VH_1)])
    out = tmp_path / "out.jsonl"

    args = _parse_args(["predict-aa", str(fasta), "--out", str(out), "--no-run-metadata"])
    rc = run(args)
    assert rc == 0

    import os
    assert os.path.exists(out)
    assert not os.path.exists(str(out) + ".run.json")


def test_cli_invalid_record_is_skipped_not_fatal(tmp_path):
    """A malformed record (stop codon '*') must not abort the whole batch (on_invalid='skip' inside
    the CLI) -- it becomes a structured per-record error line and every other record still predicts."""
    fasta = tmp_path / "reads.fasta"
    _write_fasta(fasta, [("good", _VH_1), ("bad", "EV*QL")])
    out = tmp_path / "out.jsonl"

    args = _parse_args(["predict-aa", str(fasta), "--out", str(out)])
    rc = run(args)
    assert rc == 0

    parsed = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert len(parsed) == 2
    assert "provenance" in parsed[0]
    assert parsed[1] == {"sequence_id": "bad", "error": "invalid_character", "detail": parsed[1]["detail"]}

    meta = json.loads(open(str(out) + ".run.json").read())
    assert meta["n_predicted"] == 1
    assert meta["n_errors"] == 1


def test_cli_empty_input_returns_error_and_writes_nothing(tmp_path):
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("")
    out = tmp_path / "out.jsonl"

    args = _parse_args(["predict-aa", str(fasta), "--out", str(out)])
    rc = run(args)
    assert rc == 1
    assert not out.exists()


def test_cli_uses_coverage_group_flag(tmp_path):
    fasta = tmp_path / "reads.fasta"
    _write_fasta(fasta, [("read-1", _VH_1)])
    out = tmp_path / "out.jsonl"

    args = _parse_args(["predict-aa", str(fasta), "--out", str(out), "--coverage-group", "full"])
    rc = run(args)
    assert rc == 0

    [parsed] = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert parsed["applied_calibration_group"]["group"] == "full"
    assert parsed["applied_calibration_group"]["source"] == "supplied"


def test_cli_on_invalid_flag_defaults_to_skip(tmp_path):
    """--on-invalid defaults to 'skip', so invalid records emit error dicts, not fatal errors."""
    fasta = tmp_path / "reads.fasta"
    _write_fasta(fasta, [("good", _VH_1), ("bad", "EV*QL")])
    out = tmp_path / "out.jsonl"

    args = _parse_args(["predict-aa", str(fasta), "--out", str(out)])
    # Should not raise; on_invalid="skip" is default
    assert args.on_invalid == "skip"
    rc = run(args)
    assert rc == 0

    parsed = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert len(parsed) == 2
    assert "provenance" in parsed[0]
    assert parsed[1] == {"sequence_id": "bad", "error": "invalid_character", "detail": parsed[1]["detail"]}


def test_cli_on_invalid_error_raises_on_bad_record(tmp_path):
    """--on-invalid error causes the CLI to fail when an invalid record is encountered."""
    fasta = tmp_path / "reads.fasta"
    _write_fasta(fasta, [("good", _VH_1), ("bad", "EV*QL")])
    out = tmp_path / "out.jsonl"

    args = _parse_args(["predict-aa", str(fasta), "--out", str(out), "--on-invalid", "error"])
    assert args.on_invalid == "error"

    # With on_invalid="error", the predictor should raise ValueError on the bad record
    import pytest
    with pytest.raises(ValueError, match="stop character"):
        run(args)


def test_cli_on_invalid_skip_writes_error_records_to_output(tmp_path):
    """--on-invalid skip writes per-record error dicts to output, allowing the batch to continue."""
    fasta = tmp_path / "reads.fasta"
    _write_fasta(fasta, [("r1", _VH_1), ("r2", "EV*QL"), ("r3", _VH_2)])
    out = tmp_path / "out.jsonl"

    args = _parse_args(["predict-aa", str(fasta), "--out", str(out), "--on-invalid", "skip"])
    rc = run(args)
    assert rc == 0

    parsed = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert len(parsed) == 3
    assert "provenance" in parsed[0]
    assert parsed[1] == {"sequence_id": "r2", "error": "invalid_character", "detail": parsed[1]["detail"]}
    assert "provenance" in parsed[2]

    meta = json.loads(open(str(out) + ".run.json").read())
    assert meta["command"]["on_invalid"] == "skip"
    assert meta["n_predicted"] == 2
    assert meta["n_errors"] == 1
