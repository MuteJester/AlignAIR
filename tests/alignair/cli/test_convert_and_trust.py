"""AlignAIR 3 checkpoints are refused with the version that reads them, by ``Aligner.from_pretrained`` and by
``alignair predict``; a safe model file needs no pickle consent."""

import GenAIRR.data as gd
import pytest
import torch

from alignair import Aligner
from alignair import model_file as mf
from alignair.core import prepare_model
from alignair.errors import ModelError
from alignair.reference import ReferenceSet

pytestmark = pytest.mark.integration


def _model():
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), max_seq_length=128, embed_dim=4,
                          trunk_channels=8)
    return setup.build_model().eval()


def _v3_pt(tmp_path):
    path = str(tmp_path / "legacy.pt")
    torch.save({"config": {"v_allele_count": 2, "j_allele_count": 2}, "model": {}, "step": 7}, path)
    return path


def test_from_pretrained_refuses_a_pt_checkpoint(tmp_path):
    with pytest.raises(ModelError, match=r"AlignAIR 3 checkpoint .*alignair<4"):
        Aligner.from_pretrained(_v3_pt(tmp_path), device="cpu")


def test_from_pretrained_needs_no_trust_for_a_pickle_free_model_file(tmp_path):
    p = str(tmp_path / "m.alignair")
    mf.save_model(p, _model(), ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB),
                  dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1},
                  include_trusted_pickle=False)
    aligner = Aligner.from_pretrained(p, device="cpu")               # no trust, no dataconfig needed
    assert aligner.model is not None and aligner.reference.gene("V").gapped


def test_predict_refuses_a_pt_checkpoint_with_the_version_that_runs_it(tmp_path, capsys):
    from alignair.cli.main import main
    reads = tmp_path / "reads.fasta"
    reads.write_text(">r\nACGT\n")
    rc = main(["predict", "--model", _v3_pt(tmp_path), "--input", str(reads), "--out", str(tmp_path / "out.tsv"),
               "--offline", "--quiet"])
    err = capsys.readouterr().err
    assert rc != 0 and "AlignAIR 3 checkpoint" in err and "alignair<4" in err
    assert not (tmp_path / "out.tsv").exists()


@pytest.mark.parametrize("payload,message", [(None, "model file does not exist"), (b"", "not an AlignAIR model file"),
                                             (b">read\nACGT\n", "not an AlignAIR model file")],
                         ids=["missing", "empty", "fasta"])
def test_a_file_that_is_no_checkpoint_is_not_called_an_alignair_3_one(tmp_path, payload, message):
    path = tmp_path / "model.alignair"
    if payload is not None:
        path.write_bytes(payload)
    with pytest.raises(ModelError, match=message) as caught:
        Aligner.from_pretrained(str(path), device="cpu")
    assert "alignair<4" not in str(caught.value)
