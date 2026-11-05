"""Raw sequence syntax preserves content and identity before molecule-specific validation."""
import gzip
import io
import zlib

import pytest

from alignair.errors import ConfigurationError, InputError, OperationIOError, current_error_context
from alignair.io import RawSequenceRecord, detect_sequence_format, iter_sequence_records

pytestmark = pytest.mark.component


@pytest.mark.parametrize("suffix,text,expected", [
    (".fa", ">same description\nevQL\nVE X\n>same\n*\n", [("same", "evQLVE X"), ("same", "*")]),
    (".fq", "@protein description\r\nevQL*X\r\n+\r\n@++!!!\r\n", [("protein", "evQL*X")]),
    (".csv", 'sequence_id,sequence\nsame,"evQL*X"\nsame,acgt\n', [("same", "evQL*X"), ("same", "acgt")]),
    (".tsv", "id\tsequence\np\tevQL*X\n", [("p", "evQL*X")]),
    (".txt", "\n evQL*X \n\n acgt \n", [(None, " evQL*X "), (None, " acgt ")]),
])
@pytest.mark.parametrize("compressed", [False, True])
def test_formats_preserve_case_alphabet_whitespace_and_duplicate_ids(tmp_path, suffix, text, expected, compressed):
    path = tmp_path / ("reads" + suffix + (".gz" if compressed else ""))
    path.write_bytes(gzip.compress(text.encode()) if compressed else text.encode())
    records = list(iter_sequence_records(path))
    assert records == expected
    assert all(isinstance(record, RawSequenceRecord) for record in records)


def test_fasta_retains_empty_and_unnamed_records_for_downstream_accounting(tmp_path):
    path = tmp_path / "records.fa"
    path.write_text("\n>\n\n>same\n>same\n ac \n de \n>\n")
    assert list(iter_sequence_records(path)) == [(None, ""), ("same", ""), ("same", " ac  de "), (None, "")]


@pytest.mark.parametrize("text", ["", "\n \r\n\t\n"])
@pytest.mark.parametrize("format", ["auto", "fasta", "fastq", "table", "txt"])
def test_empty_sources_have_no_records(tmp_path, text, format):
    path = tmp_path / "empty"
    path.write_text(text)
    assert list(iter_sequence_records(path, format=format)) == []


@pytest.mark.parametrize("head,expected", [(">id\n", "fasta"), ("@id\n", "fastq"),
                                          ("id,sequence\n", "table"), ("ACGT\n", "txt")])
def test_detection_without_suffix(head, expected):
    assert detect_sequence_format("-", head) == expected


def test_suffix_and_explicit_format_override_content_detection(tmp_path):
    path = tmp_path / "reads.FASTA.GZ"
    path.write_bytes(gzip.compress(b"ACGT\n"))
    with pytest.raises(InputError, match="before its FASTA header") as caught:
        list(iter_sequence_records(path))
    assert caught.value.context == {"stage": "sequence_file_parsing", "path": str(path), "format": "fasta", "line": 1}
    assert list(iter_sequence_records(path, format="txt")) == [(None, "ACGT")]


def test_csv_quoting_custom_columns_and_leading_blanks(tmp_path):
    path = tmp_path / "reads.csv.gz"
    path.write_bytes(gzip.compress(b'\n\nlabel,protein\n"sample,one","ac\nde"\n,\n'))
    assert list(iter_sequence_records(path, seq_column="protein", id_column="label")) == [("sample,one", "ac\nde"), ("", "")]


@pytest.mark.parametrize("text,kwargs,message,line", [
    ("id,protein\np,AC\n", {"seq_column": "missing"}, "sequence column 'missing'", 1),
    ("id,protein\np,AC\n", {}, "no sequence column", 1),
    ("id,sequence\np,AC\n", {"id_column": "missing"}, "id column 'missing'", 1),
    ("\n\nid,sequence\np\n", {}, "missing sequence field", 4),
    ('id,sequence\np,"AC\n', {}, "malformed sequence table", 2),
    ('"id,sequence\n', {}, "malformed sequence table", 1),
])
def test_table_structure_errors_have_source_location(tmp_path, text, kwargs, message, line):
    path = tmp_path / "reads.csv"
    path.write_text(text)
    with pytest.raises(InputError, match=message) as caught:
        list(iter_sequence_records(path, **kwargs))
    assert caught.value.context["path"] == str(path)
    assert caught.value.context["format"] == "table" and caught.value.context["line"] == line


@pytest.mark.parametrize("text,message,line", [
    ("\n\nwrong\nAC\n+\n!!\n", "expected '@'", 3),
    ("@id\nAC\n", "truncated record", 1),
    ("@id\nAC\nwrong\n!!\n", "expected '\\+'", 3),
    ("@id\nAC\n+\n!\n", "sequence length 2 != quality length 1", 4),
    ("@id\nAC\n+\n!!\nwrong\n", "expected '@'", 5),
])
def test_fastq_structure_errors_have_physical_line_numbers(tmp_path, text, message, line):
    path = tmp_path / "reads.fq"
    path.write_text(text)
    with pytest.raises(InputError, match=message) as caught:
        list(iter_sequence_records(path))
    assert caught.value.context["format"] == "fastq" and caught.value.context["line"] == line
    assert current_error_context() == {}


def test_unnamed_and_empty_fastq_records_reach_validation(tmp_path):
    path = tmp_path / "reads.fq"
    path.write_text("@\n\n+\n\n")
    assert list(iter_sequence_records(path)) == [(None, "")]


class StreamingInput(io.StringIO):
    def read(self, *args):
        pytest.fail("read the whole source")

    def readlines(self, *args):
        pytest.fail("materialized source lines")


@pytest.mark.parametrize("failure", [False, True])
def test_stdin_streams_once_and_remains_open_on_close_or_failure(monkeypatch, failure):
    source = StreamingInput("\n>first\nac\n>second\nde\n>third\nfg\n" if not failure else "@bad\nAC\n")
    monkeypatch.setattr("sys.stdin", source)
    records = iter_sequence_records("-")
    if failure:
        with pytest.raises(InputError):
            next(records)
    else:
        assert next(records) == ("first", "ac")
        assert source.tell() < len(source.getvalue())
        records.close()
    assert not source.closed


@pytest.mark.parametrize("finish", ["exhaust", "close", "failure"])
def test_owned_file_closes_on_exhaustion_early_close_and_parse_failure(monkeypatch, finish):
    source = StreamingInput("@bad\nAC\n" if finish == "failure" else ">one\nAC\n>two\nDE\n")
    calls = []
    def open_source(*args, **kwargs):
        calls.append(args[0])
        return source
    monkeypatch.setattr("builtins.open", open_source)
    records = iter_sequence_records("owned.fa" if finish != "failure" else "owned.fq")
    if finish == "exhaust":
        assert len(list(records)) == 2
    elif finish == "close":
        next(records)
        records.close()
    else:
        with pytest.raises(InputError):
            next(records)
    assert source.closed and len(calls) == 1


@pytest.mark.parametrize("kwargs", [{"format": "unknown"}, {"format": None}, {"format": []}])
def test_invalid_options_fail_before_opening_source(monkeypatch, kwargs):
    def forbidden(*args, **kw): pytest.fail("opened source before validating options")
    monkeypatch.setattr("builtins.open", forbidden)
    with pytest.raises(ConfigurationError):
        list(iter_sequence_records("unused", **kwargs))


def test_byte_paths_are_rejected():
    with pytest.raises(ConfigurationError, match="path must be text"):
        list(iter_sequence_records(b"reads.fa"))


def test_io_failures_are_classified_and_chained(tmp_path):
    missing = tmp_path / "missing.fa"
    with pytest.raises(OperationIOError) as caught:
        list(iter_sequence_records(missing))
    assert isinstance(caught.value.__cause__, FileNotFoundError)
    assert caught.value.context["path"] == str(missing)


# ``mtime=0``: a fixed gzip header, so the parameter (and the test id) is the same in every process that collects it
@pytest.mark.parametrize("name,payload,cause", [("bad.fa", b"\xff", UnicodeError),
                                               ("bad.fa.gz", b"not gzip", gzip.BadGzipFile),
                                               ("deflate.fa.gz", b"\x1f\x8b\x08\x00" + b"\x00" * 6 + b"\x07" + b"\x00" * 8, zlib.error),
                                               ("short.fa.gz", gzip.compress(b">id\nAC\n", mtime=0)[:-5], EOFError)],
                         ids=["bad_text", "not_gzip", "bad_deflate", "truncated_gzip"])
def test_corrupt_text_or_gzip_is_an_input_error(tmp_path, name, payload, cause):
    path = tmp_path / name
    path.write_bytes(payload)
    with pytest.raises(InputError, match="cannot decode") as caught:
        list(iter_sequence_records(path))
    assert isinstance(caught.value.__cause__, cause)
