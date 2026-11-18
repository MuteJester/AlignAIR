"""The genotype pass: the donor's V genotype inferred from a repertoire's rows (TIgGER-like frequency rule on the
reads whose V tie list is a single allele), and the rows whose call it moves."""
import pytest

from alignair.genotype.infer import GenotypeEvidence, allowed_v

pytestmark = pytest.mark.component


def _evidence(*calls):
    evidence = GenotypeEvidence()
    for index, (v_call, times) in enumerate(calls):
        for _ in range(times):
            evidence.observe(index, v_call)
    return evidence


def test_a_gene_keeps_the_alleles_that_explain_most_of_its_single_allele_reads():
    evidence = _evidence(("IGHV1-2*02", 18), ("IGHV1-2*04", 2),
                         ("IGHV3-23*01", 10), ("IGHV3-23*04", 9), ("IGHV3-23*05", 1))
    assert evidence.genotype() == {"IGHV1-2": ["IGHV1-2*02"], "IGHV3-23": ["IGHV3-23*01", "IGHV3-23*04"]}


def test_a_gene_with_too_few_reads_is_left_open():
    assert _evidence(("IGHV4-34*01", 9)).genotype() == {}
    assert _evidence(("IGHV4-34*01", 10)).genotype() == {"IGHV4-34": ["IGHV4-34*01"]}


def test_reads_whose_call_ties_alleles_do_not_count_toward_the_genotype():
    evidence = _evidence(("IGHV1-2*04,IGHV1-2*02", 50), ("IGHV1-2*02", 10))
    assert evidence.genotype() == {"IGHV1-2": ["IGHV1-2*02"]}


def test_reads_without_a_v_call_are_ignored():
    evidence = _evidence(("", 30), ("IGHV1-2*02", 10))
    assert evidence.genotype() == {"IGHV1-2": ["IGHV1-2*02"]}


def test_a_row_moves_when_its_call_is_outside_the_genotype_and_a_tied_allele_is_allowed():
    evidence = GenotypeEvidence()
    for index in range(10):
        evidence.observe(100 + index, "IGHV1-2*02")
    rows = {0: "IGHV1-2*04,IGHV1-2*02",          # call excluded, a tied genotype allele: moves
            1: "IGHV1-2*02,IGHV1-2*04",          # call in the genotype: stays
            2: "IGHV1-2*04",                     # call excluded but nothing else tied: stays
            3: "IGHV4-34*02,IGHV4-34*01",        # a gene without evidence is open: stays
            4: "IGHV1-2*04,IGHV4-34*01"}         # call excluded, tied with an open gene's allele: moves
    for index, v_call in rows.items():
        evidence.observe(index, v_call)
    assert evidence.moved(evidence.genotype()) == [0, 4]


def test_the_allowed_v_alleles_are_the_genotype_where_a_gene_has_evidence_and_every_allele_elsewhere():
    names = ["IGHV1-2*02", "IGHV1-2*04", "IGHV3-23*01", "IGHV3-23*04", "IGHV4-34*01"]
    genotype = {"IGHV1-2": ["IGHV1-2*02"], "IGHV3-23": ["IGHV3-23*01", "IGHV3-23*04"]}
    assert allowed_v(genotype, names) == ["IGHV1-2*02", "IGHV3-23*01", "IGHV3-23*04", "IGHV4-34*01"]


def test_an_inferred_genotype_is_written_as_an_airr_genotype_set(tmp_path):
    import json

    from alignair.genotype.airr import validate, write_genotype_set
    path = tmp_path / "genotype.json"
    write_genotype_set(path, {"V": ["IGHV1-2*02", "IGHV1-2*04"], "J": ["IGHJ4*02"]}, "IGH")
    genotype_set = json.loads(path.read_text())
    assert validate(genotype_set) == []
    (genotype,) = genotype_set["genotype_class_list"]
    assert genotype["locus"] == "IGH" and [a["label"] for a in genotype["documented_alleles"]] == [
        "IGHV1-2*02", "IGHV1-2*04", "IGHJ4*02"]
    assert genotype_set["receptor_genotype_set_id"] == "alignair-genotype-pass"
