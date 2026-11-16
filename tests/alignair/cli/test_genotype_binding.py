"""A genotype run binds its bank once: ``alignair predict --genotype`` (and pass 2 of ``--infer-genotype``) binds the
model to the genotype's subset of the reference once, before streaming, not once per chunk; the rows are those of the
API's genotype call, threaded or not, and a genotype may name a novel allele given in ``--reference``."""
import csv
import json

import pytest

from alignair.genotype.infer import GenotypeEvidence, gene_of
from alignair.model_file import load_model

from tests.alignair.predict.golden_rows import FIXTURE, TEST_CHIP, TIE_READS

pytestmark = pytest.mark.integration

NOVEL = "IGHV1-NOVEL*01"


@pytest.fixture(scope="module")
def golden():
    """``(reads, rows)``: the first six golden reads and the test chip's rows of them."""
    fixture = json.loads(FIXTURE.read_text())
    return fixture["reads"][:6], fixture["rows"][:6]


@pytest.fixture
def fasta(golden, tmp_path):
    path = tmp_path / "reads.fasta"
    path.write_text("".join(f">r{i}\n{read}\n" for i, read in enumerate(golden[0])))
    return path


@pytest.fixture
def binds(monkeypatch):
    """The references the model is bound to other than the one its file carries, in order."""
    from alignair.core import AlignAIR
    own, bind, seen = load_model(str(TEST_CHIP)).reference.gene("V").names, AlignAIR.bind_reference, []

    def counting(self, reference):
        if reference.gene("V").names != own:
            seen.append(reference)
        return bind(self, reference)
    monkeypatch.setattr(AlignAIR, "bind_reference", counting)
    return seen


def _predict(fasta, out, *extra):
    from alignair.cli.main import main
    return main(["predict", "--model", str(TEST_CHIP), "--input", str(fasta), "--out", str(out), "--offline",
                 "--quiet", "--no-run-metadata", "--device", "cpu", "--batch-size", "4", "--chunk-size", "2", *extra])


def _rows(path):
    with open(path) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _genotype(golden, tmp_path):
    """A genotype of the V alleles the golden rows call first but one, so the reads called with that one must move,
    written as JSON; returns ``(path, genotype)``."""
    genotype = {"V": sorted({row["v_call"].split(",")[0] for row in golden[1] if row["v_call"]})[1:]}
    path = tmp_path / "donor.json"
    path.write_text(json.dumps(genotype))
    return path, genotype


def path_genotype(path):
    return json.loads(path.read_text())["V"]


def test_a_genotype_run_binds_its_bank_once(golden, fasta, binds, tmp_path):
    from alignair.aligner import PredictionResult
    from alignair import Aligner
    path, genotype = _genotype(golden, tmp_path)
    assert _predict(fasta, tmp_path / "cli.tsv", "--genotype", str(path)) == 0
    assert len(binds) == 1 and set(binds[0].gene("V").names) == set(genotype["V"])
    loaded = load_model(str(TEST_CHIP))
    # the API on the CLI's chunks of two: a read's row depends on the batch it is predicted in
    records = [record for i in range(0, 6, 2) for record in Aligner.from_model(loaded.model, loaded.reference, device="cpu").with_genotype(genotype).predict(golden[0][i:i + 2], batch_size=4).records]
    PredictionResult(records, ids=[f"r{i}" for i in range(len(records))]).write_airr(str(tmp_path / "api.tsv"))
    cli, api = _rows(tmp_path / "cli.tsv"), _rows(tmp_path / "api.tsv")
    common = sorted(set(cli[0]) & set(api[0]))
    assert len(cli) == 6 and "v_call" in common
    assert [{key: row[key] for key in common} for row in cli] == [{key: row[key] for key in common} for row in api]
    assert any(row["v_call"] for row in cli)
    assert all(set(filter(None, row["v_call"].split(","))) <= set(genotype["V"]) for row in cli)


def test_a_genotype_may_name_a_novel_allele_of_the_reference(golden, fasta, tmp_path):
    called = next(row["v_call"] for row in golden[1] if row["v_call"] and "," not in row["v_call"])
    reference = load_model(str(TEST_CHIP)).reference
    lines = []
    for gene in ("V", "D", "J"):
        alleles = reference.gene(gene)
        for name, sequence in zip(alleles.names, alleles.sequences):
            lines += [f">{NOVEL if name == called else name}", sequence]
    germlines, genotype = tmp_path / "germlines.fasta", tmp_path / "donor.json"
    germlines.write_text("\n".join(lines) + "\n")
    genotype.write_text(json.dumps({"V": [NOVEL]}))
    assert _predict(fasta, tmp_path / "rows.tsv", "--reference", str(germlines), "--genotype", str(genotype)) == 0
    calls = [row["v_call"] for row in _rows(tmp_path / "rows.tsv") if row["v_call"]]
    assert calls and set(calls) == {NOVEL}


def test_threaded_genotype_rows_equal_one_process(golden, fasta, tmp_path):
    path, _ = _genotype(golden, tmp_path)
    assert _predict(fasta, tmp_path / "one.tsv", "--genotype", str(path), "--threads", "1") == 0
    assert _predict(fasta, tmp_path / "two.tsv", "--genotype", str(path), "--threads", "2") == 0
    assert _predict(fasta, tmp_path / "plain.tsv", "--threads", "1") == 0
    one, two = _rows(tmp_path / "one.tsv"), _rows(tmp_path / "two.tsv")
    assert len(one) == 6 and two == one and one != _rows(tmp_path / "plain.tsv")
    assert any(row["v_call"] for row in two)
    assert all(set(filter(None, row["v_call"].split(","))) <= set(path_genotype(path)) for row in two)


def _tie_genotype(tmp_path, monkeypatch):
    """Golden and tie reads, and an inferred genotype that keeps, for every gene whose call a plain row ties with a
    sibling of its own gene, only that sibling, so pass 2 re-calls every such row; returns ``(fasta, plain rows, the
    genotype)``."""
    reads = (json.loads(FIXTURE.read_text())["reads"] + json.loads(TIE_READS.read_text())["reads"]) * 2
    fasta = tmp_path / "reads.fasta"
    fasta.write_text("".join(f">r{i}\n{read}\n" for i, read in enumerate(reads)))
    assert _predict(fasta, tmp_path / "plain.tsv", "--threads", "1") == 0
    plain = _rows(tmp_path / "plain.tsv")
    genotype = {}
    for call, *tied in (row["v_call"].split(",") for row in plain if row["v_call"]):
        sibling = next((allele for allele in tied if gene_of(allele) == gene_of(call)), None)
        if sibling is not None:
            genotype.setdefault(gene_of(call), [sibling])
    monkeypatch.setattr(GenotypeEvidence, "genotype", lambda self, **rule: genotype)
    return fasta, plain, genotype


def test_the_genotype_pass_binds_its_bank_once(binds, tmp_path, monkeypatch):
    fasta, plain, genotype = _tie_genotype(tmp_path, monkeypatch)
    assert binds == []
    assert _predict(fasta, tmp_path / "pass.tsv", "--threads", "1", "--infer-genotype") == 0
    moved = sum(row != before for row, before in zip(_rows(tmp_path / "pass.tsv"), plain))
    assert moved > 2                                                # pass 2 re-calls more than one chunk
    assert len(binds) == 1 and {allele for alleles in genotype.values() for allele in alleles} <= set(
        binds[0].gene("V").names)


def test_a_threaded_genotype_pass_gives_the_one_process_rows(tmp_path, monkeypatch):
    fasta, plain, _ = _tie_genotype(tmp_path, monkeypatch)
    assert _predict(fasta, tmp_path / "one.tsv", "--threads", "1", "--infer-genotype") == 0
    assert _predict(fasta, tmp_path / "two.tsv", "--threads", "2", "--infer-genotype") == 0
    one = _rows(tmp_path / "one.tsv")
    assert one != plain and _rows(tmp_path / "two.tsv") == one


def test_bound_to_binds_a_copy_and_leaves_the_aligner_as_it_was():
    from alignair.aligner import Aligner
    from alignair.genotype import genotype_bank
    aligner = Aligner.from_pretrained(str(TEST_CHIP), device="cpu", offline=True)
    names = aligner.reference.gene("V").names
    bank = genotype_bank(aligner.reference, {"V": names[:3]})
    bound = aligner.bound_to(bank)
    assert bound.reference.gene("V").names == names[:3]
    assert bound.model is not aligner.model and bound.model_path is None
    assert bound.label_schema["genes"]["v"] == names[:3] and aligner.label_schema["genes"]["v"] == list(names)
    read = json.loads(FIXTURE.read_text())["reads"][0]
    assert bound.predict_raw([read])["v_allele"].shape[1] == 3                 # the copy's model reads the bank
    assert aligner.predict_raw([read])["v_allele"].shape[1] == len(names)       # the original's is as it was


def test_a_genotype_naming_every_allele_gives_the_rows_of_none(tmp_path):
    """Named in any order, the whole reference is the bank it already was: the rows are the plain run's."""
    reads = (json.loads(FIXTURE.read_text())["reads"] + json.loads(TIE_READS.read_text())["reads"])
    fasta, genotype = tmp_path / "reads.fasta", tmp_path / "donor.json"
    fasta.write_text("".join(f">r{i}\n{read}\n" for i, read in enumerate(reads)))
    genotype.write_text(json.dumps({"V": load_model(str(TEST_CHIP)).reference.gene("V").names[::-1]}))
    assert _predict(fasta, tmp_path / "plain.tsv", "--threads", "1") == 0
    assert _predict(fasta, tmp_path / "every.tsv", "--threads", "1", "--genotype", str(genotype)) == 0
    assert _rows(tmp_path / "every.tsv") == _rows(tmp_path / "plain.tsv")


def test_bound_to_annotates_a_reference_against_the_aligners_own(tmp_path):
    """Binding an aligner to its own alleles given as a FASTA (no anchors, no gaps) gives its rows, as a reference
    given at load does."""
    from alignair.aligner import Aligner
    from alignair.model_file import serialize
    from alignair.reference import anchor_report
    from alignair.reference.reference_set import ReferenceSet
    aligner = Aligner.from_pretrained(str(TEST_CHIP), device="cpu", offline=True)
    path = tmp_path / "own.fasta"
    path.write_text(serialize.reference_fasta(aligner.reference))
    bound = aligner.bound_to(ReferenceSet.from_fasta(str(path)))
    assert anchor_report(bound.reference).complete()
    reads = json.loads(FIXTURE.read_text())["reads"][:6]
    keys = ("v_call", "d_call", "j_call", "junction", "v_germline_start", "j_germline_end")
    assert [{k: r[k] for k in keys} for r in bound.predict(reads, batch_size=4).to_dicts()] == [
        {k: r[k] for k in keys} for r in aligner.predict(reads, batch_size=4).to_dicts()]
