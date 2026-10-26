"""Genotype task 7: `alignair genotype` end-to-end (structure + AIRR validity)."""
import json
import os

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


@pytest.mark.slow
def test_cli_genotype_writes_report_and_valid_airr(tmp_path):
    import GenAIRR.data as gd
    from alignair import model_file as mf
    from alignair.cli.main import build_parser
    from alignair.core import AlignAIR
    from alignair.core.config import AlignAIRConfig
    from alignair.genotype.airr import validate
    from alignair.reference.reference_set import ReferenceSet

    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    model_path = str(tmp_path / "m.alignair")
    mf.save_model(model_path, AlignAIR(cfg).eval(), dataconfigs=["HUMAN_IGH_OGRDB"],
                  training={"steps": 1, "batch_size": 1}, include_trusted_pickle=False)

    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    V, J = ref.gene("V").sequences, ref.gene("J").sequences
    reads = "".join(f">r{i}\n{V[i % 5]}GGGAAA{J[i % 3]}\n" for i in range(24))   # a tiny 'donor'
    fasta = tmp_path / "rep.fasta"
    fasta.write_text(reads)

    stem = str(tmp_path / "out")
    args = build_parser().parse_args(["genotype", str(fasta), "--model", model_path,
                                      "--out", stem, "--device", "cpu"])
    assert args.func(args) == 0
    assert os.path.exists(stem + ".genotype.report.txt")
    report = json.load(open(stem + ".genotype.report.json"))
    genotype_set = json.load(open(stem + ".genotype.airr.json"))
    assert report["genes"]                                   # at least one gene called
    assert validate(genotype_set) == []                      # conforms to the AIRR GenotypeSet schema
    gt = genotype_set["genotype_class_list"][0]
    assert gt["locus"] == "IGH" and gt["inference_process"] == "repertoire_sequencing"
