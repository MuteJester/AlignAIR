"""``alignair predict --threads N`` on the CPU predicts in N single-threaded worker processes and writes the rows one
process on one thread writes; a thread count below 1 is refused."""
import csv
import json

import pytest

from tests.alignair.predict.golden_rows import FIXTURE, golden_model

pytestmark = pytest.mark.integration


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


def test_threads_on_the_cpu_write_the_rows_one_thread_writes(model_file, fasta, tmp_path):
    assert _predict(model_file, fasta, tmp_path / "one.tsv", "--threads", "1") == 0
    assert _predict(model_file, fasta, tmp_path / "three.tsv", "--threads", "3") == 0
    one, three = _rows(tmp_path / "one.tsv"), _rows(tmp_path / "three.tsv")
    assert len(one) == 20 and three == one


def test_a_thread_count_below_one_is_refused(model_file, fasta, tmp_path):
    assert _predict(model_file, fasta, tmp_path / "none.tsv", "--threads", "0") != 0
