"""Robustness/fuzz: malformed inputs produce clean, typed errors (or a safe negative),
never an uncaught crash or silent garbage."""
import numpy as np
import pytest


def test_malformed_genotype_yaml_errors(tmp_path):
    from alignair.genotype.constraint import load_genotype
    p = tmp_path / "bad.yaml"
    p.write_text("v: [IGHV1-2*01\n  : : broken")            # not valid YAML/JSON
    with pytest.raises(Exception):                          # a parse error, not a crash we ignore
        load_genotype(str(p))


def test_container_detection_on_garbage(tmp_path):
    from alignair.model_file import container
    p = tmp_path / "garbage.alignair"
    p.write_bytes(b"this is not an alignair container at all\n" * 20)
    assert container.is_alignair_file(str(p)) is False      # detected as non-alignair, no crash


def test_read_metadata_on_garbage_errors(tmp_path):
    from alignair import model_file as mf
    p = tmp_path / "garbage.alignair"
    p.write_bytes(b"\x00\x01\x02\x03 not a real container " * 8)
    with pytest.raises(Exception):                          # clean error, not a segfault/hang
        mf.read_metadata(str(p))


def test_truncated_fastq_errors_with_line_number(tmp_path):
    from alignair.io.sequence_reader import read_sequences
    p = tmp_path / "trunc.fastq"
    p.write_text("@r1\nACGT\n")                             # missing '+' and quality lines
    with pytest.raises(ValueError, match="FASTQ|line|truncated"):
        read_sequences(str(p))


def test_empty_input_file_is_reported_not_crashed(tmp_path):
    from alignair.io.sequence_reader import read_sequences
    p = tmp_path / "empty.fasta"
    p.write_text("")
    ids, seqs, info = read_sequences(str(p))
    assert seqs == [] and ids == []                         # empty, handled gracefully


def test_nonfinite_probabilities_are_caught():
    from alignair.predict.pipeline import _assert_finite
    with pytest.raises(ValueError, match="non-finite"):
        _assert_finite({"v": np.array([[0.5, np.nan, 0.3]])}, "fuzz")
    with pytest.raises(ValueError, match="non-finite"):
        _assert_finite({"j": np.array([[np.inf, 0.1]])}, "fuzz")


def test_raw_predictions_validated_before_use():
    """A damaged checkpoint emitting NaN outputs is rejected right after clean(), before it
    can become a plausible argmax call or a clipped-NaN coordinate."""
    from alignair.predict.pipeline import _validate_predictions
    from alignair.predict.state import Predictions
    bad_probs = Predictions(allele={"v": np.array([[np.nan, 0.2]])}, start={"v": np.array([0.0])},
                            end={"v": np.array([1.0])}, mutation_rate=np.zeros(1),
                            indel_count=np.zeros(1), productive=np.ones(1))
    with pytest.raises(ValueError, match="non-finite allele"):
        _validate_predictions(bad_probs)
    bad_coord = Predictions(allele={"v": np.array([[0.9, 0.1]])}, start={"v": np.array([np.inf])},
                            end={"v": np.array([1.0])}, mutation_rate=np.zeros(1),
                            indel_count=np.zeros(1), productive=np.ones(1))
    with pytest.raises(ValueError, match="non-finite start"):
        _validate_predictions(bad_coord)


def test_oversized_sequence_is_cropped_not_truncated_silently():
    from alignair.predict.pipeline import apply_input_policy
    long_read = "ACGT" * 500                                # 2000 nt, window 576
    seqs, cropped = apply_input_policy([long_read], max_len=576)
    assert len(seqs[0]) == 576 and cropped == [True] # cropped + FLAGGED, never silent
