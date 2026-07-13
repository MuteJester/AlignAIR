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
