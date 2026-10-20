"""The alignair-api runner's subprocess (``diagnostics.api_predict``), run in process on a small shaped model saved
as a ``.alignair`` file: its rows are the CLI's on the same file, a genotype constrains its calls, and a failure on
the genotype path is told apart from any other failure."""
import csv
import json

import pytest

pytest.importorskip("GenAIRR")

import torch  # noqa: E402
from alignair.cli.main import main as alignair_main  # noqa: E402
from alignair.model_file import save_model  # noqa: E402

from alignair_assay.core.reference import Reference  # noqa: E402
from alignair_assay.diagnostics.api_predict import CHUNK_SIZE, GENOTYPE_FAILED, GENOTYPE_NOTE, main  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory, make_shaped_model):
    """The shaped IGH model (real-sized V, D and J spans, so the germline reader and the AIRR assembly run)."""
    model, _ = make_shaped_model("igh")
    path = tmp_path_factory.mktemp("api_predict") / "shaped.alignair"
    save_model(str(path), model, dataconfigs=["HUMAN_IGH_OGRDB"], training={}, include_trusted_pickle=False)
    return path


@pytest.fixture
def inputs(tmp_path, s0_reads, reference):
    """S0's reads as a FASTA, one of them lower case and one id given twice (the reader's cleaning and its
    duplicate ids), and the truth reference as a FASTA."""
    records = [(name, sequence) for name, sequence in s0_reads]
    records[3] = (records[3][0], records[3][1].lower())
    records[7] = (records[0][0], records[7][1])
    fasta = tmp_path / "reads.fasta"
    fasta.write_text("".join(f">{name}\n{sequence}\n" for name, sequence in records), encoding="utf-8")
    truth = tmp_path / "reference.fasta"
    truth.write_text("".join(f">{name}\n{reference.seq(name)}\n" for name in sorted(reference.sequences())),
                     encoding="utf-8")
    return fasta, truth


def _args(checkpoint, truth, fasta, out, *extra) -> list[str]:
    return ["--checkpoint", str(checkpoint), "--reference", str(truth), "--input", str(fasta), "--out", str(out),
            "--device", "cpu", "--batch-size", "8", *extra]


def _rows(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_rows_are_the_clis(checkpoint, inputs, tmp_path):
    fasta, truth = inputs
    out, cli_out = tmp_path / "api.tsv", tmp_path / "cli.tsv"
    assert main(_args(checkpoint, truth, fasta, out)) == 0
    # The CLI on the same file and reads; it may refuse the job over a random model's assembly failures, but it
    # writes its rows first.
    alignair_main(["predict", "--model", str(checkpoint), "--input", str(fasta), "--out", str(cli_out),
                   "--device", "cpu", "--batch-size", "8", "--quiet"])
    assert out.read_text(encoding="utf-8") == cli_out.read_text(encoding="utf-8")
    rows = _rows(out)
    assert len(rows) == 50 and rows[7]["sequence_id"] == f"{rows[0]['sequence_id']}_dup1"
    assert all(row["v_call"] for row in rows)

    record = json.loads(out.with_name("api.tsv.run.json").read_text())
    cli_record = json.loads(cli_out.with_name("cli.tsv.run.json").read_text())
    assert record["counts"] == {key: cli_record["counts"][key] for key in record["counts"]}
    assert record["counts"]["written_records"] == record["counts"]["input_records"] == 50
    assert (record["device"], record["torch"], record["cuda_device"]) == ("cpu", torch.__version__, None)
    assert (record["batch_size"], record["chunk_size"], record["locus"]) == (8, CHUNK_SIZE, "IGH")
    assert record["api"] == "alignair.predict_sequences" and record["genotype"] is None


def test_a_genotype_constrains_the_calls(checkpoint, inputs, reference, tmp_path):
    fasta, truth = inputs
    allowed = sorted(reference.alleles["v"])[10:14]
    genotype = tmp_path / "genotype.json"
    genotype.write_text(json.dumps({"v": allowed, "d": sorted(reference.alleles["d"]),
                                    "j": sorted(reference.alleles["j"])}))
    out = tmp_path / "genotype.tsv"
    assert main(_args(checkpoint, truth, fasta, out, "--genotype", str(genotype))) == 0
    calls = [set(row["v_call"].split(",")) - {""} for row in _rows(out)]
    assert any(calls) and all(call <= set(allowed) for call in calls)
    record = json.loads(out.with_name("genotype.tsv.run.json").read_text())
    assert record["genotype"] == {"path": str(genotype), "method": "mask", "alleles": {"v": 4, "d": 33, "j": 7}}


@pytest.mark.parametrize("v, error", [
    (["IGHVX9*01"], "NovelAlleleUnsupportedError: unknown alleles in genotype for v: ['IGHVX9*01']"),
    ([], "NovelAlleleUnsupportedError: genotype constraint for gene 'v' allows no allele"),
])
def test_a_failure_on_the_genotype_path_is_told_apart(checkpoint, inputs, reference, tmp_path, capsys, v, error):
    fasta, truth = inputs
    genotype = tmp_path / "genotype.json"
    genotype.write_text(json.dumps({"v": v, "d": sorted(reference.alleles["d"]), "j": sorted(reference.alleles["j"])}))
    out = tmp_path / "refused.tsv"
    assert main(_args(checkpoint, truth, fasta, out, "--genotype", str(genotype))) == GENOTYPE_FAILED == 3
    last = capsys.readouterr().err.strip().splitlines()[-1]
    assert last.startswith(f"{GENOTYPE_NOTE}: {error}") and GENOTYPE_NOTE == "genotype API path failed"
    # Nothing is written: no rows, no temporary file, no record.
    assert list(tmp_path.glob("refused.tsv*")) == []


@pytest.mark.parametrize("raised", [TypeError("unhashable type: 'set'"),
                                    AttributeError("'tuple' object has no attribute 'items'"), KeyError("v")])
def test_a_genotype_of_the_wrong_type_is_a_genotype_failure(checkpoint, inputs, reference, tmp_path, capsys,
                                                            monkeypatch, raised):
    # The errors a genotype of another type raises where it is applied are the genotype path's.
    import alignair_assay.diagnostics.api_predict as api_predict

    def predict(*args, genotype=None, **kwargs):
        raise raised

    fasta, truth = inputs
    genotype = tmp_path / "genotype.json"
    genotype.write_text(json.dumps({"v": sorted(reference.alleles["v"])[:3], "d": sorted(reference.alleles["d"]),
                                    "j": sorted(reference.alleles["j"])}))
    monkeypatch.setattr(api_predict, "predict_file", predict)
    assert main(_args(checkpoint, truth, fasta, tmp_path / "out.tsv", "--genotype", str(genotype))) == GENOTYPE_FAILED
    assert capsys.readouterr().err.strip().splitlines()[-1].startswith(f"{GENOTYPE_NOTE}: {type(raised).__name__}")


def test_any_other_failure_with_a_genotype_is_not_a_genotype_failure(checkpoint, inputs, reference, tmp_path,
                                                                     capsys, monkeypatch):
    # Anything else (a model, device or I/O error) propagates as it would without a genotype: Python's traceback
    # and exit status 1, with no note.
    import alignair_assay.diagnostics.api_predict as api_predict

    def predict(*args, genotype=None, **kwargs):
        raise RuntimeError("CUDA out of memory")

    fasta, truth = inputs
    genotype = tmp_path / "genotype.json"
    genotype.write_text(json.dumps({"v": sorted(reference.alleles["v"])[:3], "d": sorted(reference.alleles["d"]),
                                    "j": sorted(reference.alleles["j"])}))
    monkeypatch.setattr(api_predict, "predict_file", predict)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        main(_args(checkpoint, truth, fasta, tmp_path / "out.tsv", "--genotype", str(genotype)))
    assert GENOTYPE_NOTE not in capsys.readouterr().err


def test_the_traceback_precedes_the_note(checkpoint, inputs, reference, tmp_path, capsys):
    fasta, truth = inputs
    genotype = tmp_path / "genotype.json"
    genotype.write_text(json.dumps({"v": [], "d": sorted(reference.alleles["d"]), "j": sorted(reference.alleles["j"])}))
    main(_args(checkpoint, truth, fasta, tmp_path / "out.tsv", "--genotype", str(genotype)))
    err = capsys.readouterr().err
    assert err.index("Traceback (most recent call last)") < err.index(GENOTYPE_NOTE)


def test_another_reference_is_not_a_genotype_failure(checkpoint, inputs, reference, tmp_path):
    fasta, _ = inputs
    v = sorted(reference.alleles["v"])
    changed = Reference({**reference.alleles, "v": {**reference.alleles["v"], v[3]: reference.seq(v[3]) + "A"}}, {})
    other = tmp_path / "other.fasta"
    other.write_text("".join(f">{name}\n{changed.seq(name)}\n" for name in sorted(changed.sequences())))
    genotype = tmp_path / "genotype.json"
    genotype.write_text(json.dumps({"v": v[:3], "d": sorted(reference.alleles["d"]),
                                    "j": sorted(reference.alleles["j"])}))
    # The model is checked against the verified reference before the genotype is read: a mismatch is the tool's
    # error (Python's exit status 1), never the genotype path's.
    with pytest.raises(ValueError, match="not the truth reference"):
        main(_args(checkpoint, other, fasta, tmp_path / "out.tsv", "--genotype", str(genotype)))
