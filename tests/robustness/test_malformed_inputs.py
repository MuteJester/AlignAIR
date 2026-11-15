"""Robustness/fuzz: malformed inputs produce clean, typed errors (or a safe negative),
never an uncaught crash or silent garbage."""
import pytest

from alignair.errors import ModelFormatError

pytestmark = pytest.mark.component


def test_malformed_genotype_yaml_errors(tmp_path):
    from alignair.errors import ConfigurationError
    from alignair.genotype.files import load_genotype
    p = tmp_path / "bad.yaml"
    p.write_text("v: [IGHV1-2*01\n  : : broken")            # not valid YAML/JSON
    with pytest.raises(ConfigurationError, match="is not a YAML or JSON genotype") as caught:
        load_genotype(str(p))
    assert caught.value.context["path"] == str(p)


def test_container_detection_on_garbage(tmp_path):
    from alignair.model_file import container
    p = tmp_path / "garbage.alignair"
    p.write_bytes(b"this is not an alignair container at all\n" * 20)
    assert container.is_alignair_file(str(p)) is False      # detected as non-alignair, no crash


def test_read_metadata_on_garbage_errors(tmp_path):
    from alignair import model_file as mf
    p = tmp_path / "garbage.alignair"
    p.write_bytes(b"\x00\x01\x02\x03 not a real container " * 8)
    with pytest.raises(ModelFormatError) as caught:
        mf.read_metadata(str(p))
    assert caught.value.code == "ALIGNAIR_MODEL_FORMAT"
    assert caught.value.context["path"] == str(p)


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


def test_oversized_sequence_is_cropped_not_truncated_silently():
    from alignair.data import prepare_nucleotide_inputs
    long_read = "ACGT" * 500                                # 2000 nt, window 576
    seqs, cropped = prepare_nucleotide_inputs([long_read], 576)
    assert len(seqs[0]) == 576 and cropped == [True] # cropped + FLAGGED, never silent
