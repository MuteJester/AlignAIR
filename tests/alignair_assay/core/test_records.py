"""The record helpers every workflow step shares (``core.records``): the canonical-JSON digest, strict JSON reads and
atomic writes. The digest keys every cache the assay keeps, so a known record must hash as it always has."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from alignair_assay.core import records
from alignair_assay.core.jsonl import read_header, read_jsonl
from alignair_assay.core.records import digest, read_json, write_json, write_jsonl, write_text

REPO = Path(__file__).resolve().parents[3]
# A record with nested keys out of order, a float, a None, a bool and a non-ASCII character, and its digest as the
# workflow helpers computed it before they were promoted (``real_steps.digest``, ``build._digest``,
# ``orchestrate._digest``, all equal).
KNOWN = {"inputs": {"reads_sha256": "a" * 64, "reference_digest": "4d70015675eb"}, "code_sha256": "c" * 64,
         "list": [1, 2.5, None, True], "name": "IGHV1-69*01 é"}
KNOWN_DIGEST = "f6bf21c7c5fcb78deaf307c41bb063f61eb5581ffb019ec6e85c3f56474f8d5f"


def test_a_known_record_hashes_as_before():
    assert digest(KNOWN) == KNOWN_DIGEST
    reordered = json.loads(json.dumps({key: KNOWN[key] for key in reversed(list(KNOWN))}))
    assert digest(reordered) == KNOWN_DIGEST  # key order never matters


@pytest.mark.parametrize("path", [".private/assay/p2b_dev/suites/S2/build.json",
                                  ".private/assay/p2_run_genairr_fix/manifest.json",
                                  ".private/assay/p3a_dev/real/manifest.json"])
def test_stored_records_still_match_their_hash(path):
    # the P2 build and run and the P3a dev run were hashed by the old helpers: the promoted digest reproduces them,
    # so their caches stay valid
    record = REPO / path
    if not record.is_file():
        pytest.skip(f"{path} is not here")
    stored = json.loads(record.read_text(encoding="utf-8"))
    assert digest(stored["identity"]) == stored["config_hash"]


def test_read_json_takes_a_json_object_or_nothing(tmp_path):
    path = tmp_path / "record.json"
    assert read_json(path) is None  # missing
    path.write_text("{not json", encoding="utf-8")
    assert read_json(path) is None
    path.write_text("[1, 2]", encoding="utf-8")
    assert read_json(path) is None  # not an object
    path.write_text('{"a": 1}', encoding="utf-8")
    assert read_json(path) == {"a": 1}


def test_writes_are_atomic_through_a_temporary_file_of_their_own(tmp_path, monkeypatch):
    path = tmp_path / "sub" / "record.json"
    write_json(path, {"b": 1, "a": [1.5]})
    assert path.read_text(encoding="utf-8") == '{\n  "b": 1,\n  "a": [\n    1.5\n  ]\n}\n'
    assert oct(path.stat().st_mode & 0o777) == "0o644"
    with pytest.raises(ValueError):
        write_json(path, {"a": float("nan")})  # strict JSON
    assert json.loads(path.read_text(encoding="utf-8")) == {"b": 1, "a": [1.5]}  # the old file stands
    write_json(path, {"a": float("nan")}, allow_nan=True)
    assert path.read_text(encoding="utf-8") == '{\n  "a": NaN\n}\n'
    # another writer's temporary file (the old fixed ``.<name>.part``) is never touched
    other = path.with_name(f".{path.name}.part")
    other.write_text("another writer", encoding="utf-8")
    write_text(path, "text\n")
    assert other.read_text(encoding="utf-8") == "another writer" and path.read_text(encoding="utf-8") == "text\n"
    # a failed write leaves the file as it was and no temporary file behind
    monkeypatch.setattr(records.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        write_text(path, "lost\n")
    assert path.read_text(encoding="utf-8") == "text\n"
    assert sorted(item.name for item in path.parent.iterdir()) == [f".{path.name}.part", path.name]


def test_concurrent_writers_never_share_a_temporary_file(tmp_path):
    path = tmp_path / "record.json"
    errors = []

    def write(value: int) -> None:
        try:
            for _ in range(50):
                write_json(path, {"writer": value})
        except Exception as error:  # noqa: BLE001 (any failure is the test's)
            errors.append(error)

    threads = [threading.Thread(target=write, args=(value,)) for value in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == [] and json.loads(path.read_text(encoding="utf-8"))["writer"] in range(4)
    assert [item.name for item in tmp_path.iterdir()] == ["record.json"]
    assert os.access(path, os.R_OK)


def test_core_jsonl_writes_the_same_rows_as_the_same_bytes_at_any_time(tmp_path, monkeypatch):
    # A suite's cases and truth (``core.jsonl.write_jsonl``) built twice from the same rows must be byte-identical,
    # whenever they are written: the gzip header holds no time, so a file's SHA-256 is a digest of its content (the
    # audit's reproducibility check compares two builds' files byte for byte).
    import gzip

    from alignair_assay.core import jsonl

    rows = [{"case_id": "S1:0", "sequence": "ACGT"}, {"case_id": "S1:1", "sequence": "TTGA"}]
    written = []
    for clock in (1_000_000_000.0, 2_000_000_000.0):
        monkeypatch.setattr(gzip.time, "time", lambda clock=clock: clock)
        path = tmp_path / str(int(clock)) / "cases.jsonl.gz"
        path.parent.mkdir()
        jsonl.write_jsonl(rows, path, header={"kind": "k"})
        written.append(path.read_bytes())
        assert jsonl.read_jsonl(path) == rows and jsonl.read_header(path) == {"kind": "k"}
    assert written[0] == written[1]
    plain = tmp_path / "rows.jsonl"
    jsonl.write_jsonl(rows, plain)
    assert plain.read_text(encoding="utf-8") == ('{"case_id":"S1:0","sequence":"ACGT"}\n'
                                                 '{"case_id":"S1:1","sequence":"TTGA"}\n')
    with pytest.raises(ValueError):
        jsonl.write_jsonl([{"m": float("nan")}], plain)


def test_json_lines_are_written_atomically_and_compressed_reproducibly(tmp_path, monkeypatch):
    path = tmp_path / "sub" / "cases.jsonl.gz"
    rows = [{"case_id": "a", "scores": {"m": 1.0, "n": None}}, {"case_id": "b", "scores": {}}]
    write_jsonl(path, rows, header={"kind": "k"})
    assert read_jsonl(path) == rows and read_header(path) == {"kind": "k"}
    assert path.read_bytes()[:2] == b"\x1f\x8b" and oct(path.stat().st_mode & 0o777) == "0o644"
    # the same rows are the same bytes (no time or temporary name in the gzip header), so the file's digest is a
    # digest of its content
    first = path.read_bytes()
    write_jsonl(path, iter(rows), header={"kind": "k"})
    assert path.read_bytes() == first
    plain = tmp_path / "cases.jsonl"
    write_jsonl(plain, rows)
    assert plain.read_text(encoding="utf-8") == ('{"case_id":"a","scores":{"m":1.0,"n":null}}\n'
                                                 '{"case_id":"b","scores":{}}\n')
    # strict JSON; a failed write leaves the file as it was and no temporary file behind
    with pytest.raises(ValueError):
        write_jsonl(path, [{"m": float("nan")}])
    monkeypatch.setattr(records.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        write_jsonl(path, [{"lost": 1}])
    assert path.read_bytes() == first and [item.name for item in path.parent.iterdir()] == [path.name]
