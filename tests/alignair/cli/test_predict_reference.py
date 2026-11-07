"""``alignair predict --reference FASTA``: the model is bound to the FASTA's alleles (novel ones included), annotated
against the reference it came with when it loads; the alleles left without an anchor are named on stderr."""
from types import SimpleNamespace

import pytest

from alignair.reference.reference_set import GeneReference, ReferenceSet

pytestmark = pytest.mark.component


def _known():
    v = GeneReference(["IGHV1-1*01"], ["ACGTTGTTAC"], {"IGHV1-1*01": 0}, anchors={"IGHV1-1*01": 4},
                      gapped={"IGHV1-1*01": "ACGT..TGTTAC"})
    d = GeneReference(["IGHD1-1*01"], ["GGGCCC"], {"IGHD1-1*01": 0})
    j = GeneReference(["IGHJ1*01"], ["TTTGGAA"], {"IGHJ1*01": 0}, anchors={"IGHJ1*01": 3})
    return ReferenceSet({"V": v, "D": d, "J": j}, has_d=True)


FASTA = ">IGHV1-1*01\nACGTTGTTAC\n>IGHV1-1*novel\nACGATGTTAC\n>IGHD1-1*01\nGGGCCC\n>IGHJ1*01\nTTTGGAA\n"


def _run(tmp_path, monkeypatch, *, bind=lambda reference: reference, patch_load=True, fasta_text=FASTA):
    from alignair.cli import predict as predict_cli
    from alignair.cli.main import build_parser
    model = tmp_path / "model.alignair"
    model.write_bytes(b"model fixture")
    fasta = tmp_path / "germline.fasta"
    if fasta_text is not None:
        fasta.write_text(fasta_text)
    loads = []

    def from_pretrained(*args, reference=None, **kwargs):
        loads.append(reference)
        return SimpleNamespace(model=SimpleNamespace(), reference=_known() if reference is None else bind(reference))
    monkeypatch.setattr("alignair.registry.resolve_model", lambda *a, **k: model)
    monkeypatch.setattr("alignair.model_file.container.is_alignair_file", lambda path: True)
    if patch_load:
        monkeypatch.setattr(predict_cli.Aligner, "from_pretrained", from_pretrained)
    streamed = []
    monkeypatch.setattr(predict_cli, "_stream_predict", lambda aligner, **kwargs: streamed.append(aligner) or (
        {"accepted": 1, "rejected": 0, "written": 1, "failed": 0, "partial": 0}, {}, {}))
    args = build_parser().parse_args(["predict", "--model", str(model), "--reference", str(fasta), "--input",
                                      "unused.fasta", "--out", str(tmp_path / "out.tsv"), "--no-run-metadata",
                                      "--quiet"])
    return args.func(args), loads, streamed


def test_the_model_is_bound_to_the_fastas_alleles_and_warned_of_those_without_an_anchor(tmp_path, monkeypatch,
                                                                                       capsys):
    status, loads, streamed = _run(tmp_path, monkeypatch)
    assert status == 0
    assert loads[-1].gene("V").names == ["IGHV1-1*01", "IGHV1-1*novel"]
    assert streamed and streamed[0].reference.gene("V").names == loads[-1].gene("V").names
    err = capsys.readouterr().err
    assert "warning: --reference" in err and "IGHV1-1*novel (unresolved)" in err


def test_a_fully_anchored_reference_prints_no_warning(tmp_path, monkeypatch, capsys):
    from alignair.reference import annotate
    status, _, streamed = _run(tmp_path, monkeypatch, bind=lambda reference: annotate(reference, known=_known()))
    assert status == 0 and "warning" not in capsys.readouterr().err
    assert streamed[0].reference.gene("V").anchors == {"IGHV1-1*01": 4, "IGHV1-1*novel": 4}


def test_a_broken_model_file_with_a_reference_is_a_model_error(tmp_path, monkeypatch, capsys):
    """Loading the model and binding the FASTA happen in one load: an error of the model file is still the model's."""
    from alignair.errors import ModelError

    def broken(*args, **kwargs):
        raise ModelError("the model file is truncated")
    monkeypatch.setattr("alignair.cli.predict.Aligner.from_pretrained", broken)
    status, _, streamed = _run(tmp_path, monkeypatch, patch_load=False)
    err = capsys.readouterr().err
    assert status != 0 and not streamed
    assert "truncated" in err and "reference_binding" not in err


def test_an_unreadable_reference_is_a_reference_binding_error(tmp_path, monkeypatch, capsys):
    status, loads, streamed = _run(tmp_path, monkeypatch, fasta_text=None)
    assert status != 0 and not streamed and loads == []
    assert "reference_binding" in capsys.readouterr().err
