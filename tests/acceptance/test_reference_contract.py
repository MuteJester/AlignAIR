"""The reference is the chip's input: a novel allele given in it is called, under its own name, with no retraining;
the model's own alleles given as a FASTA give the rows of no reference."""
import csv
import json

import pytest

from alignair.cli.main import main
from alignair.model_file import load_model
from tests.alignair.predict.golden_rows import FIXTURE, TEST_CHIP

pytestmark = pytest.mark.integration

NOVEL = "IGHV1-NOVEL*01"


def test_a_novel_allele_in_the_reference_is_called(tmp_path):
    fixture = json.loads(FIXTURE.read_text())
    read, row = next((read, row) for read, row in zip(fixture["reads"], fixture["rows"])
                     if row["v_call"] and "," not in row["v_call"])
    reference = load_model(str(TEST_CHIP)).reference
    lines = []
    for gene in ("V", "D", "J"):
        alleles = reference.gene(gene)
        for name, sequence in zip(alleles.names, alleles.sequences):
            lines += [f">{NOVEL if name == row['v_call'] else name}", sequence]
    germlines, reads, out = tmp_path / "germlines.fasta", tmp_path / "reads.fasta", tmp_path / "rows.tsv"
    germlines.write_text("\n".join(lines) + "\n")
    reads.write_text(f">r\n{read}\n")
    assert main(["predict", "--model", str(TEST_CHIP), "--input", str(reads), "--out", str(out),
                 "--reference", str(germlines), "--offline", "--quiet", "--no-run-metadata"]) == 0
    with open(out, newline="") as rows:
        (called,) = list(csv.DictReader(rows, delimiter="\t"))
    assert called["v_call"].split(",")[0] == NOVEL


def test_the_models_own_alleles_as_a_fasta_give_the_rows_of_no_reference(tmp_path, capsys):
    from alignair.model_file import serialize
    reads = json.loads(FIXTURE.read_text())["reads"][:6]
    fasta, germlines = tmp_path / "reads.fasta", tmp_path / "germlines.fasta"
    fasta.write_text("".join(f">r{i}\n{read}\n" for i, read in enumerate(reads)))
    germlines.write_text(serialize.reference_fasta(load_model(str(TEST_CHIP)).reference))
    common = ["predict", "--model", str(TEST_CHIP), "--input", str(fasta), "--offline", "--quiet", "--no-run-metadata",
              "--device", "cpu", "--threads", "1"]
    assert main([*common, "--out", str(tmp_path / "plain.tsv")]) == 0
    capsys.readouterr()
    assert main([*common, "--out", str(tmp_path / "own.tsv"), "--reference", str(germlines)]) == 0
    assert "warning" not in capsys.readouterr().err

    def rows(path):
        with open(path, newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))
    assert rows(tmp_path / "own.tsv") == rows(tmp_path / "plain.tsv")
