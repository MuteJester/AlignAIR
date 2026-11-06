"""The reference is the chip's input: a novel allele given in it is called, under its own name, with no retraining."""
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
