import pytest
import GenAIRR.data as gd
from alignair.core.config import AlignAIRConfig
from alignair.core import AlignAIR
from alignair import model_file as mf
from alignair.cli.main import main

pytestmark = pytest.mark.integration


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


def test_cli_reference_export_writes_the_fasta(tmp_path, capsys):
    p = _save(tmp_path)
    fasta = tmp_path / "ref.fasta"
    assert main(["reference", "export", p, "--fasta", str(fasta)]) == 0
    assert fasta.read_text().startswith(">")


def test_reference_export_never_unpickles_a_dataconfig_section(tmp_path, capsys):
    """A model file's pickled sections are never loaded by the CLI: the export takes no --dataconfig."""
    p = _save(tmp_path)
    out = tmp_path / "dc.pkl"
    with pytest.raises(SystemExit) as caught:
        main(["reference", "export", p, "--dataconfig", str(out)])
    assert caught.value.code == 2 and not out.exists()


def test_reference_export_needs_somewhere_to_write(tmp_path, capsys):
    with pytest.raises(SystemExit) as caught:
        main(["reference", "export", _save(tmp_path)])
    assert caught.value.code == 2


def test_the_duplicate_export_reference_verb_is_gone(tmp_path, capsys):
    with pytest.raises(SystemExit) as caught:
        main(["export-reference", _save(tmp_path), "--fasta", str(tmp_path / "ref.fasta")])
    assert caught.value.code == 2
