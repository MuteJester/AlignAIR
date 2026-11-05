"""``predict --genotype``: the genotype file is validated against the model's reference and passed to the
prediction as the constraint mapping."""
import json

import pytest

pytestmark = pytest.mark.integration


def test_predict_passes_the_validated_genotype_mapping(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from alignair.cli import predict as predict_cli
    from alignair.cli.main import build_parser

    model = tmp_path / "model.alignair"
    model.write_bytes(b"model fixture")
    genotype = tmp_path / "donor.json"
    genotype.write_text(json.dumps({"v": ["V*01"], "j": ["J*01"]}))
    reference = SimpleNamespace(locus_names=lambda: ("IGH",),
                                gene=lambda gene: SimpleNamespace(names=[f"{gene}*01"]))
    monkeypatch.setattr("alignair.registry.resolve_model", lambda *a, **k: model)
    monkeypatch.setattr("alignair.model_file.container.is_alignair_file", lambda path: True)
    monkeypatch.setattr(predict_cli.Aligner, "from_pretrained",
                        lambda *a, **k: SimpleNamespace(reference=reference))
    captured = {}

    def stream(aligner, **kwargs):
        captured.update(kwargs["overrides"])
        return {"accepted": 1, "rejected": 0, "written": 1, "failed": 0, "partial": 0}, {}, {}

    monkeypatch.setattr(predict_cli, "_stream_predict", stream)
    args = build_parser().parse_args([
        "predict", "--model", str(model), "--genotype", str(genotype),
        "--input", "unused.fasta", "--out", str(tmp_path / "out.tsv"),
        "--no-run-metadata", "--quiet",
    ])
    assert args.func(args) == 0
    assert captured == {"genotype": {"v": {"V*01"}, "j": {"J*01"}}, "genotype_method": "mask"}
