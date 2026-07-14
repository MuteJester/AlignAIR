"""P0-8: input validation policies for the predict reader — no silent truncation, strict FASTQ,
empty rejection, and duplicate-ID disambiguation with preserved order."""
import pytest

from alignair.io.sequence_reader import read_sequences, validate, validate_sequence
from alignair.predict.pipeline import apply_input_policy


def test_validate_sequence_reasons():
    assert validate_sequence("")[1] == "empty"
    assert validate_sequence("   ")[1] == "empty"
    assert validate_sequence("acgt")[0] == "ACGT"                 # uppercased, ok
    assert validate_sequence("ACGTR")[0] == "ACGTN"              # IUPAC ambiguity -> N
    assert validate_sequence("ACGT", max_len=3)[1] == "too_long"  # length policy
    assert validate_sequence("XXXXZZ")[1] == "ambiguous"        # >20% unusable
    # back-compat wrapper still returns just the cleaned string (or None)
    assert validate("acgt") == "ACGT" and validate("") is None


def test_apply_input_policy_crops_over_length_and_flags():
    seqs = ["ACGTACGT", "ACGTACGTAC"]        # second is 10, window is 8
    out, cropped = apply_input_policy(seqs, max_len=8)
    assert out == ["ACGTACGT", "ACGTACGT"]   # cropped CONSISTENTLY to the window (not silently)
    assert cropped == [False, True]


def test_apply_input_policy_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        apply_input_policy(["ACGT", "", "ACGT"], max_len=576)


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


def test_metadata_join_carries_10x_columns(tmp_path):
    m = tmp_path / "annot.csv"
    m.write_text("contig_id,barcode,umi_count\nc1,AAAA-1,7\nc2,CCCC-1,3\n")
    from alignair.io.sequence_reader import load_metadata
    meta, kept = load_metadata(str(m), id_column="contig_id")
    assert "barcode" in kept and meta["c1"]["umi_count"] == "7"


def test_10x_metadata_normalized_to_airr_names(tmp_path):
    """A stock 10x annotations file gains AIRR-standard cell_id/umi_count/c_call (raw columns kept)."""
    m = tmp_path / "filtered_contig_annotations.csv"
    m.write_text("barcode,contig_id,reads,umis,chain,c_gene\n"
                 "AAAA-1,c1,120,7,IGH,IGHM\n")
    from alignair.io.sequence_reader import load_metadata
    meta, kept = load_metadata(str(m), id_column="contig_id", normalize_10x=True)
    assert meta["c1"]["cell_id"] == "AAAA-1"          # barcode -> cell_id
    assert meta["c1"]["umi_count"] == "7"             # umis -> umi_count
    assert meta["c1"]["c_call"] == "IGHM"             # c_gene -> c_call
    assert meta["c1"]["barcode"] == "AAAA-1"          # raw 10x column preserved
    for c in ("cell_id", "umi_count", "c_call"):
        assert c in kept
