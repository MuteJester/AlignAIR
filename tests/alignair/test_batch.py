"""Multi-sample manifest mode (`alignair batch`)."""
import csv
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("GenAIRR")

from alignair import cli
from alignair.serialization.dnalignair_bundle import save_dnalignair_bundle
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR


@pytest.fixture(scope="module")
def igh_bundle(tmp_path_factory):
    torch.manual_seed(0)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64))
    d = tmp_path_factory.mktemp("bundle")
    save_dnalignair_bundle(d, model=model, dataconfigs=["HUMAN_IGH_OGRDB"], locus="IGH")
    return str(d)


def _fasta(path, n=2):
    path.write_text("".join(f">{path.stem}_r{i}\n{'ACGTACGT' * 20}\n" for i in range(n)))
    return path


def _summary(out_dir):
    return list(csv.DictReader((Path(out_dir) / "manifest_summary.tsv").open(), delimiter="\t"))


def test_batch_runs_all_samples(igh_bundle, tmp_path):
    s1, s2 = _fasta(tmp_path / "S1.fasta"), _fasta(tmp_path / "S2.fasta", n=3)
    man = tmp_path / "manifest.tsv"
    man.write_text(f"sample_id\tinput\nS1\t{s1}\nS2\t{s2}\n")
    out = tmp_path / "results"
    cli.main(["batch", "--manifest", str(man), "-o", str(out), "--model", igh_bundle,
              "--device", "cpu", "--quiet"])
    assert (out / "S1.tsv").exists() and (out / "S2.tsv").exists()
    rows = {r["sample_id"]: r for r in _summary(out)}
    assert rows["S1"]["status"] == "ok" and int(rows["S1"]["n_aligned"]) == 2
    assert rows["S2"]["status"] == "ok" and int(rows["S2"]["n_aligned"]) == 3
    js = json.loads((out / "manifest_summary.json").read_text())
    assert {s["sample_id"] for s in js} == {"S1", "S2"}


def test_batch_one_model_load_shared(igh_bundle, tmp_path, monkeypatch):
    s1, s2 = _fasta(tmp_path / "A.fasta"), _fasta(tmp_path / "B.fasta")
    man = tmp_path / "m.tsv"
    man.write_text(f"sample_id\tinput\nA\t{s1}\nB\t{s2}\n")
    calls = {"n": 0}
    import alignair.api as api
    real = api.load_model
    monkeypatch.setattr(api, "load_model", lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), real(*a, **k))[1])
    # cli imports load_model from .api into module namespace; patch there too
    monkeypatch.setattr(cli, "load_model", api.load_model, raising=False)
    cli.main(["batch", "--manifest", str(man), "-o", str(tmp_path / "o"), "--model", igh_bundle,
              "--device", "cpu", "--quiet"])
    assert calls["n"] == 1                       # model loaded exactly once for the whole cohort


def test_batch_per_sample_metadata_preserved(igh_bundle, tmp_path):
    s1 = _fasta(tmp_path / "S1.fasta", n=2)      # ids: S1_r0, S1_r1
    meta = tmp_path / "S1_meta.csv"
    meta.write_text("sequence_id,sample_id,tissue\nS1_r0,S1,blood\nS1_r1,S1,blood\n")
    man = tmp_path / "m.tsv"
    man.write_text(f"sample_id\tinput\tmetadata\nS1\t{s1}\t{meta}\n")
    out = tmp_path / "res"
    cli.main(["batch", "--manifest", str(man), "-o", str(out), "--model", igh_bundle,
              "--device", "cpu", "--quiet", "--keep-columns", "sample_id,tissue"])
    rows = list(csv.DictReader((out / "S1.tsv").open(), delimiter="\t"))
    assert all(r["tissue"] == "blood" and r["sample_id"] == "S1" for r in rows)


def test_batch_records_failed_sample_but_continues(igh_bundle, tmp_path):
    good = _fasta(tmp_path / "good.fasta")
    man = tmp_path / "m.tsv"
    man.write_text(f"sample_id\tinput\nGOOD\t{good}\nBAD\t{tmp_path / 'nope.fasta'}\n")
    out = tmp_path / "res"
    cli.main(["batch", "--manifest", str(man), "-o", str(out), "--model", igh_bundle,
              "--device", "cpu", "--quiet"])           # exit 0: at least one sample ok
    rows = {r["sample_id"]: r for r in _summary(out)}
    assert rows["GOOD"]["status"] == "ok"
    assert rows["BAD"]["status"] == "error" and rows["BAD"]["error"]
    assert (out / "GOOD.tsv").exists()


def test_batch_all_failed_exits_nonzero(igh_bundle, tmp_path):
    man = tmp_path / "m.tsv"
    man.write_text(f"sample_id\tinput\nX\t{tmp_path / 'a.fasta'}\nY\t{tmp_path / 'b.fasta'}\n")
    with pytest.raises(SystemExit):
        cli.main(["batch", "--manifest", str(man), "-o", str(tmp_path / "o"),
                  "--model", igh_bundle, "--device", "cpu", "--quiet"])


def test_batch_relative_paths_resolve_against_manifest_dir(igh_bundle, tmp_path):
    sub = tmp_path / "run"
    sub.mkdir()
    _fasta(sub / "S1.fasta")
    man = sub / "manifest.tsv"
    man.write_text("sample_id\tinput\nS1\tS1.fasta\n")     # bare filename, next to the manifest
    out = tmp_path / "res"
    cli.main(["batch", "--manifest", str(man), "-o", str(out), "--model", igh_bundle,
              "--device", "cpu", "--quiet"])
    assert _summary(out)[0]["status"] == "ok"


def test_batch_manifest_missing_columns_errors(igh_bundle, tmp_path):
    man = tmp_path / "m.tsv"
    man.write_text("sample_id\tfastq\nS1\tx.fasta\n")      # no 'input' column
    with pytest.raises(SystemExit, match="input"):
        cli.main(["batch", "--manifest", str(man), "-o", str(tmp_path / "o"),
                  "--model", igh_bundle, "--device", "cpu", "--quiet"])
