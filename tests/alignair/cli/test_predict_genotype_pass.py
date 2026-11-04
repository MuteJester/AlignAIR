"""``alignair predict --infer-genotype``: the donor's V genotype inferred from the reads' own calls, the reads it moves
re-called under it, every other row as a plain run writes it, and the genotype written as an AIRR GenotypeSet."""
import csv
import json

import pytest

from alignair.genotype.lean import GenotypeEvidence

from tests.alignair.predict.golden_rows import FIXTURE, golden_model

pytestmark = pytest.mark.integration

CALL = "IGHVF10-G52*04"              # the golden model's V call for every read, mostly with tied alleles


@pytest.fixture(scope="module")
def model_file(tmp_path_factory):
    from alignair.model_file import save_model
    model, _ = golden_model()
    path = tmp_path_factory.mktemp("model") / "golden.alignair"
    save_model(str(path), model, dataconfigs=["HUMAN_IGH_OGRDB"], training={}, include_trusted_pickle=False)
    return str(path)


@pytest.fixture(scope="module")
def fasta(tmp_path_factory):
    reads = json.loads(FIXTURE.read_text())["reads"] * 2
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


@pytest.fixture(scope="module")
def plain(model_file, fasta, tmp_path_factory):
    out = tmp_path_factory.mktemp("plain") / "plain.tsv"
    assert _predict(model_file, fasta, out) == 0
    return _rows(out)


def test_a_genotype_that_allows_every_call_moves_no_row(model_file, fasta, plain, tmp_path):
    out = tmp_path / "pass.tsv"
    assert _predict(model_file, fasta, out, "--infer-genotype", "--genotype-min-reads", "1") == 0
    assert _rows(out) == plain
    genotype = json.loads((tmp_path / "pass.tsv.genotype.json").read_text())
    labels = [a["label"] for a in genotype["genotype_class_list"][0]["documented_alleles"]]
    assert labels == [CALL]


def test_the_rows_a_genotype_moves_are_re_called_under_it_and_the_rest_are_unchanged(model_file, fasta, plain,
                                                                                    tmp_path, monkeypatch):
    other = "IGHVF10-G52*05"
    monkeypatch.setattr(GenotypeEvidence, "genotype", lambda self, **rule: {"IGHVF10-G52": [other]})
    out, genotype_out = tmp_path / "pass.tsv", tmp_path / "donor.json"
    assert _predict(model_file, fasta, out, "--infer-genotype", "--genotype-out", str(genotype_out)) == 0
    rows = _rows(out)

    def allowed(allele):                                     # every other gene has no evidence, so stays open
        return not allele.startswith("IGHVF10-G52*") or allele == other
    moving = [i for i, row in enumerate(plain) if any(map(allowed, row["v_call"].split(",")[1:]))]
    assert moving and len(rows) == len(plain)
    for i, (row, before) in enumerate(zip(rows, plain)):
        if i in moving:
            assert row["sequence_id"] == before["sequence_id"]
            assert row["v_call"].split(",")[0] != CALL          # the excluded allele is no longer the call
        else:
            assert row == before
    labels = [a["label"] for a in json.loads(genotype_out.read_text())["genotype_class_list"][0]["documented_alleles"]]
    assert labels == [other]


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


def test_the_run_metadata_records_the_pass_and_counts_the_rows_written(model_file, fasta, plain, tmp_path,
                                                                       monkeypatch):
    from collections import Counter

    from alignair.cli.main import main
    monkeypatch.setattr(GenotypeEvidence, "genotype", lambda self, **rule: {"IGHVF10-G52": ["IGHVF10-G52*05"]})
    out = tmp_path / "pass.tsv"
    assert main(["predict", "--model", model_file, "--input", fasta, "--out", str(out), "--offline", "--quiet",
                 "--device", "cpu", "--batch-size", "4", "--chunk-size", "7", "--infer-genotype"]) == 0
    run = json.loads((tmp_path / "pass.tsv.run.json").read_text())
    rows = _rows(out)
    moved = sum(row != before for row, before in zip(rows, plain))
    assert run["genotype_pass"]["moved_rows"] >= moved > 0
    assert run["genotype_pass"]["genotype"] == {"IGHVF10-G52": ["IGHVF10-G52*05"]}
    statuses = Counter(row["airr_assembly_status"] for row in rows)
    assert run["counts"]["complete_assemblies"] == statuses["complete"]
    assert run["counts"]["partial_assemblies"] == statuses["partial"]
    assert sum(run["airr_assembly_partial_reasons"].values()) == statuses["partial"]
