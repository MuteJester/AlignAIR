import pytest
import GenAIRR.data as gd
from alignair.core import AlignAIR
from alignair import model_file as mf
from alignair import Aligner
from alignair.core import prepare_model
from alignair.reference import ReferenceSet

pytestmark = pytest.mark.integration


def test_from_pretrained_reads_a_model_file_without_its_dataconfig(tmp_path):
    cfg = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), max_seq_length=128, embed_dim=4,
                        trunk_channels=8).config
    m = AlignAIR(cfg)
    p = tmp_path / "m.alignair"
    mf.save_model(str(p), m, ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB),
                  dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1})
    aligner = Aligner.from_pretrained(str(p), device="cpu")     # NO dataconfigs= needed
    model, reference = aligner.model, aligner.reference
    assert reference.gene("V").names[0].startswith("IGH")
    assert model.cfg == cfg
    assert len(reference.gene("V")) == len(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB).gene("V"))
