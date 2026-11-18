"""``alignair predict --infer-genotype``: the donor's V genotype inferred from the reads' own calls, the reads it moves
re-called under it, every other row as a plain run writes it, and the genotype written as an AIRR GenotypeSet. On the
test chip, with the golden reads and the tie reads (short amplicons whose V the chip ties with sibling alleles)."""
import csv
import json

import pytest

from alignair.genotype.infer import GenotypeEvidence, gene_of

from tests.alignair.predict.golden_rows import FIXTURE, TEST_CHIP, TIE_READS

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def model_file():
    return str(TEST_CHIP)


@pytest.fixture(scope="module")
def fasta(tmp_path_factory):
    reads = (json.loads(FIXTURE.read_text())["reads"] + json.loads(TIE_READS.read_text())["reads"]) * 2
    path = tmp_path_factory.mktemp("reads") / "reads.fasta"
    path.write_text("".join(f">r{i}\n{read}\n" for i, read in enumerate(reads)))
    return str(path)


def _predict(model_file, fasta, out, *extra):
    from alignair.cli.main import main
    return main(["predict", "--model", model_file, "--input", fasta, "--out", str(out), "--offline", "--quiet",
                 "--no-run-metadata", "--device", "cpu", "--batch-size", "4", "--chunk-size", "7", *extra])


def _rows(path):
    with open(path) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _labels(path):
    genotype = json.loads(path.read_text())
    return {allele["label"] for entry in genotype["genotype_class_list"] for allele in entry["documented_alleles"]}


@pytest.fixture(scope="module")
def plain(model_file, fasta, tmp_path_factory):
    out = tmp_path_factory.mktemp("plain") / "plain.tsv"
    assert _predict(model_file, fasta, out) == 0
    return _rows(out)


@pytest.fixture(scope="module")
def tie(plain):
    """``(gene, call, other)``: the first plain row whose V call ties another allele of its own gene."""
    for row in plain:
        call, *tied = row["v_call"].split(",")
        same = [allele for allele in tied if gene_of(allele) == gene_of(call)]
        if same:
            return gene_of(call), call, same[0]
    pytest.fail("no plain row ties its V call with an allele of the same gene")


def _excluding(tie):
    """The genotype that keeps only ``other`` of the tie's gene, and which alleles it allows (other genes are open)."""
    gene, _, other = tie
    return {gene: [other]}, lambda allele: gene_of(allele) != gene or allele == other


def test_a_genotype_that_allows_every_call_moves_no_row(model_file, fasta, plain, tmp_path, monkeypatch):
    called = {}
    for row in plain:
        for allele in filter(None, row["v_call"].split(",")):
            called.setdefault(gene_of(allele), set()).add(allele)
    genotype = {gene: sorted(alleles) for gene, alleles in called.items()}
    monkeypatch.setattr(GenotypeEvidence, "genotype", lambda self, **rule: genotype)
    out = tmp_path / "pass.tsv"
    assert _predict(model_file, fasta, out, "--infer-genotype") == 0
    assert _rows(out) == plain
    assert _labels(tmp_path / "pass.tsv.genotype.json") == set().union(*called.values())


def test_the_rows_a_genotype_moves_are_re_called_under_it_and_the_rest_are_unchanged(model_file, fasta, plain, tie,
                                                                                    tmp_path, monkeypatch):
    genotype, allowed = _excluding(tie)
    monkeypatch.setattr(GenotypeEvidence, "genotype", lambda self, **rule: genotype)
    out, genotype_out = tmp_path / "pass.tsv", tmp_path / "donor.json"
    assert _predict(model_file, fasta, out, "--infer-genotype", "--genotype-out", str(genotype_out)) == 0
    rows = _rows(out)
    moving = [i for i, row in enumerate(plain) if row["v_call"] and not allowed(row["v_call"].split(",")[0])
              and any(map(allowed, row["v_call"].split(",")[1:]))]
    assert moving and len(rows) == len(plain)
    for i, (row, before) in enumerate(zip(rows, plain)):
        if i in moving:
            assert row["sequence_id"] == before["sequence_id"]
            assert allowed(row["v_call"].split(",")[0])            # the call is one the genotype allows
        else:
            assert row == before
    assert _labels(genotype_out) == {tie[2]}


@pytest.mark.parametrize("extra", [["--genotype", "donor.json"], ["--input", "-"], ["--out", "-"]])
def test_the_pass_is_refused_with_a_given_genotype_or_a_stream(model_file, fasta, tmp_path, extra):
    from alignair.cli.main import main
    args = {"--input": fasta, "--out": str(tmp_path / "x.tsv")}
    flags = []
    for key, value in zip(extra[::2], extra[1::2]):
        if key in args:
            args[key] = value
        else:
            flags += [key, value]
    argv = ["predict", "--model", model_file, "--offline", "--quiet", "--no-run-metadata", "--device", "cpu",
            "--infer-genotype", *[x for pair in args.items() for x in pair], *flags]
    assert main(argv) != 0


def test_the_run_metadata_records_the_pass_and_counts_the_rows_written(model_file, fasta, plain, tie, tmp_path,
                                                                       monkeypatch):
    from collections import Counter

    from alignair.cli.main import main
    genotype, _ = _excluding(tie)
    monkeypatch.setattr(GenotypeEvidence, "genotype", lambda self, **rule: genotype)
    out = tmp_path / "pass.tsv"
    assert main(["predict", "--model", model_file, "--input", fasta, "--out", str(out), "--offline", "--quiet",
                 "--device", "cpu", "--batch-size", "4", "--chunk-size", "7", "--infer-genotype"]) == 0
    run = json.loads((tmp_path / "pass.tsv.run.json").read_text())
    rows = _rows(out)
    moved = sum(row != before for row, before in zip(rows, plain))
    assert run["genotype_pass"]["moved_rows"] >= moved > 0
    assert run["genotype_pass"]["genotype"] == genotype
    statuses = Counter(row["airr_assembly_status"] for row in rows)
    assert run["counts"]["complete_assemblies"] == statuses["complete"]
    assert run["counts"]["partial_assemblies"] == statuses["partial"]
    assert sum(run["airr_assembly_partial_reasons"].values()) == statuses["partial"]


def test_the_merged_table_ends_its_lines_with_a_newline_alone(tmp_path):
    """The genotype pass rewrites the table it re-called rows into; it ends lines as the AIRR writer does (\\n)."""
    from alignair.cli.genotype_pass import merge
    first, second, out = tmp_path / "first.tsv", tmp_path / "second.tsv", tmp_path / "out.tsv"
    first.write_text("sequence_id\tv_call\nr0\tA\nr1\tB\n")
    second.write_text("sequence_id\tv_call\nr1\tC\n")
    merge(str(first), str(second), [1], str(out))
    assert out.read_bytes() == b"sequence_id\tv_call\nr0\tA\nr1\tC\n"
