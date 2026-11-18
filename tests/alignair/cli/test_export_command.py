"""``alignair export CHECKPOINT OUT_DIR``: a resumable training checkpoint becomes a pickle-free bundle (model.alignair,
model_card.md, reference_manifest.json; validation_report.json unless --no-validate) whose card carries the given
model id and version, so the catalog's publish_local accepts it as it is. A checkpoint refuses distribution with a
hint that names the command."""
import GenAIRR.data as gd
import pytest

from alignair import Aligner
from alignair import model_file as mf
from alignair.cli.main import build_parser, main
from alignair.core import prepare_model
from alignair.errors import ModelError
from alignair.reference import ReferenceSet
from alignair.registry.publish import publish_local

pytestmark = pytest.mark.integration


@pytest.fixture
def checkpoint(tmp_path):
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), max_seq_length=128, embed_dim=4,
                          trunk_channels=8)
    path = tmp_path / "run" / "model.alignair"
    path.parent.mkdir()
    mf.save_model(str(path), setup.build_model().eval(), setup.reference, dataconfigs=["HUMAN_IGH_OGRDB"],
                  training={"steps": 3, "batch_size": 2})
    assert any(s.startswith("dataconfig/") for s in mf.read_metadata(str(path))["sections"])
    return str(path)


def test_a_checkpoint_refuses_distribution_with_a_hint_naming_alignair_export(checkpoint, tmp_path):
    with pytest.raises(ModelError, match="alignair export"):
        Aligner.from_pretrained(checkpoint, device="cpu").save_pretrained(str(tmp_path / "share"))


def test_export_writes_a_pickle_free_bundle_the_catalog_accepts(checkpoint, tmp_path, capsys):
    out = tmp_path / "bundle"
    assert main(["export", checkpoint, str(out), "--model-id", "alignair4-igh-human", "--model-version", "0.1.0",
                 "--description", "test IGH", "--no-validate"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["model.alignair", "model_card.md", "reference_manifest.json"]
    md = mf.read_metadata(str(out / "model.alignair"))
    assert md["model_id"] == "alignair4-igh-human" and md["model_version"] == "0.1.0"
    assert md["description"] == "test IGH"
    assert not any(s.startswith("dataconfig/") or s == "train_state" for s in md["sections"])
    assert publish_local(str(out / "model.alignair"), "alignair4-igh-human", "0.1.0", str(tmp_path / "reg")) == []
    assert str(out / "model.alignair") in capsys.readouterr().out


def test_export_refuses_an_existing_directory_unless_told_to_overwrite(checkpoint, tmp_path, capsys):
    out = tmp_path / "bundle"
    out.mkdir()
    assert main(["export", checkpoint, str(out), "--no-validate"]) != 0
    assert "already exists" in capsys.readouterr().err
    assert main(["export", checkpoint, str(out), "--no-validate", "--overwrite"]) == 0


def test_export_validates_by_default():
    args = build_parser().parse_args(["export", "ckpt.alignair", "out"])
    assert args.validate is True and args.model_id is None and args.model_version is None
