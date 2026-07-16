"""The CLI and the Python API produce identical predictions on the same fixture (the CLI is a
thin client of the Aligner object). Skipped when the shipped IGH model is absent."""
import csv
import os

import pytest

_IGH = ".private/models/alignair_igh_v1.cal.alignair"

pytestmark = pytest.mark.skipif(not os.path.exists(_IGH), reason="shipped IGH model not present")

_SEQS = ["CAGGTGCAGCTGGTGCAGTCTGGGGCTGAGGTGAAGAAGCCTGGGGCCTCAGTGAAGGTCTCCTGCAAGGCT",
         "GAGGTGCAGCTGGTGGAGTCTGGGGGAGGCTTGGTACAGCCTGGGGGGTCCCTGAGACTCTCCTGTGCAGCC"]


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_cli_matches_python_api(tmp_path):
    from alignair import Aligner
    from alignair.cli.main import main

    fasta = tmp_path / "in.fasta"
    fasta.write_text("".join(f">r{i}\n{s}\n" for i, s in enumerate(_SEQS)))
    cli_out = tmp_path / "cli.tsv"
    rc = main(["predict", "--model", _IGH, "--input", str(fasta), "--out", str(cli_out),
               "--offline", "--no-run-metadata", "--quiet"])
    assert rc == 0

    api_out = tmp_path / "api.tsv"
    Aligner.from_pretrained(_IGH, device="cpu").predict(_SEQS).write_airr(str(api_out))

    cli_rows, api_rows = _read(str(cli_out)), _read(str(api_out))
    assert len(cli_rows) == len(api_rows) == len(_SEQS)
    for c, a in zip(cli_rows, api_rows):
        for col in ("v_call", "d_call", "j_call", "v_sequence_start", "j_sequence_end",
                    "productive", "locus", "junction"):
            assert c.get(col) == a.get(col), (col, c.get(col), a.get(col))
