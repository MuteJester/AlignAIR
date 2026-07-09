import GenAIRR.data as gd
from alignair.core.config import AlignAIRConfig
from alignair.core import AlignAIR
from alignair import model_file as mf
from alignair.cli.main import main


def _save(tmp_path):
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    p = tmp_path / "m.alignair"
    mf.save_model(str(p), AlignAIR(cfg), dataconfigs=["HUMAN_IGH_OGRDB"],
                  training={"steps": 2, "batch_size": 3})
    return str(p)


def test_cli_info(tmp_path, capsys):
    p = _save(tmp_path)
    assert main(["info", p]) == 0
    out = capsys.readouterr().out
    assert "AlignAIR" in out and "HUMAN_IGH_OGRDB" in out and "total_sequences_seen" in out


def test_cli_export_reference(tmp_path, capsys):
    p = _save(tmp_path)
    fasta = tmp_path / "ref.fasta"
    assert main(["export-reference", p, "--fasta", str(fasta)]) == 0
    assert fasta.read_text().startswith(">")
