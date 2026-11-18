"""``predict --genotype``: the genotype file is validated against the model's reference, and the stream predicts with
an aligner bound to the genotype's bank (the reference's subset it names; a gene it does not name keeps all)."""
import json

import pytest

from alignair.cli.predict_run import RunCounts

pytestmark = pytest.mark.integration


def test_predict_streams_with_the_genotypes_bank(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from alignair.cli import predict as predict_cli
    from alignair.cli.main import build_parser
    from alignair.reference import ReferenceSet

    model = tmp_path / "model.alignair"
    model.write_bytes(b"model fixture")
    genotype = tmp_path / "donor.json"
    genotype.write_text(json.dumps({"v": ["V*01"], "j": ["J*01"]}))
    reference = ReferenceSet.from_alleles({"V": {"V*01": "ACGTAC", "V*02": "ACGAAC"}, "D": {"D*01": "GGGCCC"},
                                            "J": {"J*01": "TTTGGA", "J*02": "TTAGGA"}})
    banks, streamed = [], []
    loaded = SimpleNamespace(reference=reference,
                             bound_to=lambda bank: banks.append(bank) or SimpleNamespace(reference=bank))
    monkeypatch.setattr("alignair.registry.resolve_model", lambda *a, **k: model)
    monkeypatch.setattr("alignair.model_file.container.is_alignair_file", lambda path: True)
    monkeypatch.setattr(predict_cli.Aligner, "from_pretrained", lambda *a, **k: loaded)

    def stream(aligner, io, **kwargs):
        streamed.append(aligner)
        return RunCounts(accepted=1, written=1)

    monkeypatch.setattr(predict_cli, "_stream_predict", stream)
    args = build_parser().parse_args([
        "predict", "--model", str(model), "--genotype", str(genotype),
        "--input", "unused.fasta", "--out", str(tmp_path / "out.tsv"),
        "--no-run-metadata", "--quiet",
    ])
    assert args.func(args) == 0
    (bank,) = banks
    assert [aligner.reference for aligner in streamed] == [bank]
    assert (bank.gene("V").names, bank.gene("D").names, bank.gene("J").names) == (["V*01"], ["D*01"], ["J*01"])
