"""``alignair predict --reference FASTA``: the model is bound to the FASTA's alleles (novel ones included), completed
against the reference it came with."""
from types import SimpleNamespace

import pytest

from alignair.reference.reference_set import GeneReference, ReferenceSet

pytestmark = pytest.mark.component


def _known():
    v = GeneReference(["IGHV1-1*01"], ["ACGTACGTAC"], {"IGHV1-1*01": 0}, anchors={"IGHV1-1*01": 4},
                      gapped={"IGHV1-1*01": "ACGT..ACGTAC"})
    d = GeneReference(["IGHD1-1*01"], ["GGGCCC"], {"IGHD1-1*01": 0})
    j = GeneReference(["IGHJ1*01"], ["TTTGGAA"], {"IGHJ1*01": 0}, anchors={"IGHJ1*01": 3})
    return ReferenceSet({"V": v, "D": d, "J": j}, has_d=True)


def _run(tmp_path, monkeypatch):
    from alignair.cli import predict as predict_cli
    from alignair.cli.main import build_parser
    model = tmp_path / "model.alignair"
    model.write_bytes(b"model fixture")
    fasta = tmp_path / "germline.fasta"
    fasta.write_text(">IGHV1-1*01\nACGTACGTAC\n>IGHV1-1*novel\nACGAACGTAC\n>IGHD1-1*01\nGGGCCC\n>IGHJ1*01\nTTTGGAA\n")
    loads = []

    def from_pretrained(*args, reference=None, **kwargs):
        loads.append(reference)
        return SimpleNamespace(model=SimpleNamespace(), reference=reference or _known())
    monkeypatch.setattr("alignair.registry.resolve_model", lambda *a, **k: model)
    monkeypatch.setattr("alignair.model_file.container.is_alignair_file", lambda path: True)
    monkeypatch.setattr(predict_cli.Aligner, "from_pretrained", from_pretrained)
    streamed = []
    monkeypatch.setattr(predict_cli, "_stream_predict", lambda aligner, **kwargs: streamed.append(aligner) or (
        {"accepted": 1, "rejected": 0, "written": 1, "failed": 0, "partial": 0}, {}, {}))
    args = build_parser().parse_args(["predict", "--model", str(model), "--reference", str(fasta), "--input",
                                      "unused.fasta", "--out", str(tmp_path / "out.tsv"), "--no-run-metadata",
                                      "--quiet"])
    return args.func(args), loads, streamed


def test_the_model_is_bound_to_the_completed_fasta(tmp_path, monkeypatch):
    status, loads, streamed = _run(tmp_path, monkeypatch)
    assert status == 0
    bound = loads[-1]
    assert bound.gene("V").names == ["IGHV1-1*01", "IGHV1-1*novel"]
    assert bound.gene("V").gapped["IGHV1-1*novel"] == "ACGA..ACGTAC" and bound.gene("V").anchors["IGHV1-1*novel"] == 4
    assert streamed and streamed[0].reference is bound
