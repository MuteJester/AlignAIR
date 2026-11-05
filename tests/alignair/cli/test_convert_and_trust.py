"""AlignAIR 3 checkpoints are refused with the version that reads them, by ``api.load_model`` and by ``alignair
convert``; a safe model file needs no pickle consent."""
from argparse import Namespace

import GenAIRR.data as gd
import pytest
import torch

from alignair import api
from alignair import model_file as mf
from alignair.cli import convert
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


@pytest.mark.parametrize("trust_pickle", [False, True])
def test_api_load_refuses_a_pt_checkpoint_whatever_the_consent(tmp_path, trust_pickle):
    with pytest.raises(ModelError, match=r"AlignAIR 3 checkpoint .*alignair<4"):
        api.load_model(_v3_pt(tmp_path), dataconfigs=["HUMAN_IGH_OGRDB"], trust_pickle=trust_pickle)


def test_api_load_no_pickle_alignair_needs_no_trust(tmp_path):
    p = str(tmp_path / "m.alignair")
    mf.save_model(p, _model(), dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1},
                  include_trusted_pickle=False)
    m, ref = api.load_model(p)                                       # no trust, no dataconfig needed
    assert m is not None and ref.gene("V").gapped


def test_convert_refuses_a_pt_checkpoint(tmp_path, capsys):
    rc = convert.run(Namespace(src=_v3_pt(tmp_path), dst=str(tmp_path / "out.alignair")))
    assert rc != 0 and "alignair<4" in capsys.readouterr().err
    assert not (tmp_path / "out.alignair").exists()


def test_convert_refuses_a_model_file_that_is_not_a_lab_chip(tmp_path, capsys):
    p = str(tmp_path / "m.alignair")
    mf.save_model(p, _model(), dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1},
                  include_trusted_pickle=False)
    rc = convert.run(Namespace(src=p, dst=str(tmp_path / "out.alignair")))
    assert rc != 0 and "not a lab chip model file" in capsys.readouterr().err
    assert not (tmp_path / "out.alignair").exists()


@pytest.mark.parametrize("dataconfig", [[], ["--dataconfig", "HUMAN_IGH_OGRDB"]])
def test_predict_refuses_a_pt_checkpoint_with_the_version_that_runs_it(tmp_path, capsys, dataconfig):
    from alignair.cli.main import main
    reads = tmp_path / "reads.fasta"
    reads.write_text(">r\nACGT\n")
    rc = main(["predict", "--model", _v3_pt(tmp_path), "--input", str(reads), "--out", str(tmp_path / "out.tsv"),
               "--offline", "--quiet", *dataconfig])
    err = capsys.readouterr().err
    assert rc != 0 and "AlignAIR 3 checkpoint" in err and "alignair<4" in err
    assert not (tmp_path / "out.tsv").exists()
