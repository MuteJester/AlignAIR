"""Input validation policies for the predict reader — no silent truncation, strict FASTQ,
empty rejection, and duplicate-ID disambiguation with preserved order."""
import pytest

from alignair.io.sequence_reader import read_sequences, validate, validate_sequence

pytestmark = pytest.mark.component


def test_validate_sequence_reasons():
    assert validate_sequence("")[1] == "empty"
    assert validate_sequence("   ")[1] == "empty"
    assert validate_sequence("acgt")[0] == "ACGT"                 # uppercased, ok
    assert validate_sequence("ACGTR")[0] == "ACGTN"              # IUPAC ambiguity -> N
    assert validate_sequence("ACGT", max_len=3)[1] == "too_long"  # length policy
    assert validate_sequence("XXXXZZ")[1] == "ambiguous"        # >20% unusable
    # back-compat wrapper still returns just the cleaned string (or None)
    assert validate("acgt") == "ACGT" and validate("") is None


def test_reader_disambiguates_duplicate_ids_preserving_order(tmp_path):
    p = tmp_path / "d.fasta"
    p.write_text(">r1\nACGT\n>r1\nTGCA\n>r2\nGGGG\n")
    ids, seqs, info = read_sequences(str(p))
    assert seqs == ["ACGT", "TGCA", "GGGG"]           # order preserved
    assert len(set(ids)) == 3                          # ids made unique
    assert ids[0] == "r1" and ids[1] != "r1" and ids[2] == "r2"


def test_fastq_strict_rejects_malformed(tmp_path):
    p = tmp_path / "bad.fastq"
    # sequence and quality lengths differ -> malformed record
    p.write_text("@r1\nACGT\n+\n!!\n")
    with pytest.raises(ValueError, match="line|length|malformed|FASTQ"):
        read_sequences(str(p))


def test_fastq_valid_parses(tmp_path):
    p = tmp_path / "ok.fastq"
    p.write_text("@r1\nACGT\n+\nIIII\n@r2\nTTTT\n+\nIIII\n")
    ids, seqs, info = read_sequences(str(p))
    assert seqs == ["ACGT", "TTTT"] and ids == ["r1", "r2"]


def test_table_input_custom_sequence_and_id_columns(tmp_path):
    p = tmp_path / "d.tsv"
    p.write_text("contig_id\tnt\tbarcode\nc1\tACGTACGT\tAAAA\nc2\tTTTTGGGG\tCCCC\n")
    ids, seqs, info = read_sequences(str(p), seq_column="nt", id_column="contig_id")
    assert ids == ["c1", "c2"] and seqs == ["ACGTACGT", "TTTTGGGG"]


def test_collect_rejects_records_reason_and_sequence(tmp_path):
    p = tmp_path / "mix.fasta"
    p.write_text(">good\nACGTACGT\n>bad\nXXXXZZZZ\n>empty\n\n")   # bad = ambiguous, empty = blank
    ids, seqs, info = read_sequences(str(p), collect_rejects=True)
    assert seqs == ["ACGTACGT"]
    reasons = {r["sequence_id"]: r["reason"] for r in info["rejects"]}
    assert reasons["bad"] == "ambiguous" and reasons["empty"] == "empty"


def test_stdin_is_streamed_not_slurped(monkeypatch):
    """`--input -` reads stdin as a single streaming pass (no `.read()` of the whole stream) — the
    bounded-memory guarantee must hold for piped input."""
    import io as _io

    from alignair.io import sequence_reader as sr

    class _NoSlurpStdin(_io.StringIO):
        def read(self, *a, **k):                       # a full-stream .read() would break streaming
            raise AssertionError("stdin was slurped instead of streamed")

    monkeypatch.setattr("sys.stdin", _NoSlurpStdin(">r1\nACGTACGT\n>r2\nTTTTGGGG\n"))
    ids, seqs, info = sr.read_sequences("-")
    assert seqs == ["ACGTACGT", "TTTTGGGG"] and ids == ["r1", "r2"]


def test_sniff_requires_seq_column_for_custom_table():
    """An extensionless delimited header (piped stdin) with a non-standard column is txt UNLESS the
    caller names the sequence column, in which case it is a table."""
    from alignair.io import detect_sequence_format
    head = "contig_id,nt,barcode\n"
    assert detect_sequence_format("-", head) == "txt"                       # no known seq-col keyword, no hint
    assert detect_sequence_format("-", head, seq_column="nt") == "table"    # explicit --sequence-column disambiguates


def test_stdin_custom_column_table_is_detected(monkeypatch):
    """A CSV piped over stdin with a non-standard --sequence-column reads as a table, not one-seq-per-line
    txt — so `cat reads.csv | alignair predict --input - --sequence-column nt` works."""
    import io as _io

    from alignair.io import sequence_reader as sr
    monkeypatch.setattr("sys.stdin",
                        _io.StringIO("contig_id,nt,barcode\nc1,ACGTACGT,AAAA\nc2,TTTTGGGG,CCCC\n"))
    ids, seqs, info = sr.read_sequences("-", seq_column="nt", id_column="contig_id")
    assert seqs == ["ACGTACGT", "TTTTGGGG"] and ids == ["c1", "c2"]


def test_unnamed_and_empty_fasta_records_are_accounted_for_by_nucleotide_policy(tmp_path):
    path = tmp_path / 'reads.fa'
    path.write_text('>\n>\nacgtr\n>same\nXXXXZZ\n>same\nACGT\n>same\nTGCA\n')
    ids, sequences, info = read_sequences(path, collect_rejects=True)
    assert ids == ['seq1', 'same', 'same_dup1']
    assert sequences == ['ACGTN', 'ACGT', 'TGCA']
    assert info['n_read'] == 5 and info['n_dropped'] == 2
    assert [(row['position'], row['reason']) for row in info['rejects']] == [(0, 'empty'), (2, 'ambiguous')]


def test_closing_nucleotide_chunk_iterator_closes_owned_source(monkeypatch):
    import io
    from alignair.io.sequence_reader import iter_sequences
    source = io.StringIO('>one\nAC\n>two\nGT\n')
    monkeypatch.setattr('builtins.open', lambda *args, **kwargs: source)
    chunks = iter_sequences('reads.fa', chunk_size=1)
    assert next(chunks) == (['one'], ['AC'], 0)
    chunks.close()
    assert source.closed


@pytest.mark.parametrize('chunk_size', [1, 2, 3, 20])
def test_source_identity_survives_suffix_and_fallback_collisions_across_chunks(tmp_path, chunk_size):
    from alignair.io import iter_sequence_batches
    path = tmp_path / 'reads.fa'
    path.write_text('>read\nAC\n>bad\nXXXXX\n>read\nGT\n>read_dup1\nTG\n>\nAC\n>seq4\nCG\n')
    batches = list(iter_sequence_batches(path, chunk_size=chunk_size))
    records = [record for batch in batches for record in batch.records]
    assert [record.sequence_id for record in records] == ['read', 'read_dup1', 'read_dup1_dup1', 'seq4', 'seq4_dup1']
    assert [record.source_id for record in records] == ['read', 'read', 'read_dup1', None, 'seq4']
    assert [record.source_index for record in records] == [0, 2, 3, 4, 5]
    assert sum(batch.n_dropped for batch in batches) == 1
    assert [key for batch in batches for key in batch.source_ids] == ['read', 'read', 'read_dup1', None, 'seq4']


def test_empty_table_ids_receive_fallback_output_names(tmp_path):
    path = tmp_path / 'reads.csv'
    path.write_text('id,sequence\n,ACGT\nseq0,ACGT\n')
    assert read_sequences(path)[0] == ['seq0', 'seq0_dup1']


@pytest.mark.parametrize('chunk_size', [0, -1, True, 1.5, None])
def test_invalid_chunk_sizes_fail_before_opening_source(monkeypatch, chunk_size):
    from alignair.errors import ConfigurationError
    def forbidden(*args, **kwargs): pytest.fail('opened invalid request')
    monkeypatch.setattr('builtins.open', forbidden)
    from alignair.io.sequence_reader import iter_sequences
    with pytest.raises(ConfigurationError, match='chunk_size'):
        next(iter_sequences('unused', chunk_size=chunk_size))
