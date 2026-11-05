"""``alignair predict --threads N`` on the CPU predicts in N single-threaded worker processes and writes the rows one
process on one thread writes; a thread count below 1 is refused."""
import csv
import json

import pytest

from tests.alignair.predict.golden_rows import FIXTURE, TEST_CHIP

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def model_file():
    return str(TEST_CHIP)


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


def test_threads_on_the_cpu_write_the_rows_one_thread_writes(model_file, fasta, tmp_path):
    assert _predict(model_file, fasta, tmp_path / "one.tsv", "--threads", "1") == 0
    assert _predict(model_file, fasta, tmp_path / "three.tsv", "--threads", "3") == 0
    one, three = _rows(tmp_path / "one.tsv"), _rows(tmp_path / "three.tsv")
    assert len(one) == 20 and three == one


def test_a_thread_count_below_one_is_refused(model_file, fasta, tmp_path):
    assert _predict(model_file, fasta, tmp_path / "none.tsv", "--threads", "0") != 0


@pytest.mark.parametrize("threads, starts", [("3", True), ("1", False)])
def test_predict_starts_the_workers_forkserver_first_only_when_it_will_use_workers(model_file, fasta, tmp_path,
                                                                                   monkeypatch, threads, starts):
    """Before the model loads, so the forkserver's imports run alongside the loading instead of after it."""
    from alignair.aligner import Aligner
    from alignair.predict import parallel
    events, load = [], Aligner.from_pretrained.__func__
    monkeypatch.setattr(parallel, "prestart", lambda: events.append("prestart"))
    monkeypatch.setattr(Aligner, "from_pretrained",
                        classmethod(lambda cls, *a, **k: events.append("load") or load(cls, *a, **k)))
    assert _predict(model_file, fasta, tmp_path / "out.tsv", "--threads", threads) == 0
    assert events[:2] == (["prestart", "load"] if starts else ["load"]) and ("prestart" in events) == starts


def test_threads_without_the_lab_on_the_path_give_the_one_process_rows(model_file, fasta, tmp_path, monkeypatch):
    """The workers load chip-v1 from the product alone: no ALIGNAIR_ARCHITECTURE_MODULES."""
    monkeypatch.delenv("ALIGNAIR_ARCHITECTURE_MODULES", raising=False)
    assert _predict(model_file, fasta, tmp_path / "one.tsv", "--threads", "1") == 0
    assert _predict(model_file, fasta, tmp_path / "two.tsv", "--threads", "2") == 0
    assert _rows(tmp_path / "two.tsv") == _rows(tmp_path / "one.tsv")


def test_a_genotype_keeps_the_chips_v_calls_inside_it(model_file, fasta, tmp_path):
    """--genotype masks the chip's bank: every V call names only the genotype's alleles (or no V at all)."""
    from alignair.model_file import load_model
    names = load_model(model_file).reference.gene("V").names
    genotype = tmp_path / "donor.json"
    genotype.write_text(json.dumps({"v": names[:2]}))
    assert _predict(model_file, fasta, tmp_path / "geno.tsv", "--genotype", str(genotype)) == 0
    rows = _rows(tmp_path / "geno.tsv")
    assert rows and all(set(filter(None, row["v_call"].split(","))) <= set(names[:2]) for row in rows)
