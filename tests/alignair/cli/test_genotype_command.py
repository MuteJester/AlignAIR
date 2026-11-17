"""``alignair genotype``: a donor's V genotype from an AIRR table by the frequency rule the predict pass uses
(``genotype.infer``): per gene the most frequent alleles of its single-allele V calls until they explain
``--explain`` of them, a gene with fewer than ``--min-reads`` such calls left open; written as an AIRR GenotypeSet."""
import csv
import json

import pytest

from alignair.cli.main import main
from alignair.genotype.airr import validate

pytestmark = pytest.mark.component


def _table(path, calls):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sequence_id", "v_call", "j_call"])
        writer.writerows([[f"r{i}", call, "IGHJ4*02"] for i, call in enumerate(calls)])


def test_the_genotype_keeps_the_alleles_that_explain_a_gene_and_leaves_a_thin_gene_open(tmp_path, capsys):
    calls = ["IGHV1-2*02"] * 20 + ["IGHV1-2*04"] * 8 + ["IGHV1-2*02,IGHV1-2*04"] * 2 + ["IGHV3-23*01"] * 5
    table, out = tmp_path / "pred.tsv", tmp_path / "genotype.json"
    _table(table, calls)
    assert main(["genotype", str(table), "--out", str(out)]) == 0
    genotype_set = json.loads(out.read_text())
    assert validate(genotype_set) == []
    alleles = {allele["label"] for entry in genotype_set["genotype_class_list"]
               for allele in entry["documented_alleles"]}
    assert alleles == {"IGHV1-2*02", "IGHV1-2*04"}
    printed = capsys.readouterr().out
    assert "IGHV1-2: IGHV1-2*02, IGHV1-2*04" in printed and "IGHV3-23" in printed


def test_the_rule_is_set_by_its_options(tmp_path):
    calls = ["IGHV1-2*02"] * 20 + ["IGHV1-2*04"] * 8 + ["IGHV3-23*01"] * 5
    table, out = tmp_path / "pred.tsv", tmp_path / "genotype.json"
    _table(table, calls)
    assert main(["genotype", str(table), "--out", str(out), "--explain", "0.7", "--min-reads", "5"]) == 0
    alleles = {allele["label"] for entry in json.loads(out.read_text())["genotype_class_list"]
               for allele in entry["documented_alleles"]}
    assert alleles == {"IGHV1-2*02", "IGHV3-23*01"}


def test_a_table_without_v_call_is_refused(tmp_path):
    table = tmp_path / "pred.tsv"
    table.write_text("sequence_id\tj_call\nr0\tIGHJ4*02\n")
    assert main(["genotype", str(table), "--out", str(tmp_path / "g.json")]) != 0
