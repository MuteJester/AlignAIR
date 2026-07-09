import GenAIRR.data as gd
from alignair.core.config import AlignAIRConfig
from alignair.core import AlignAIR
from alignair import model_file as mf
from alignair.api import load_model as api_load


def test_api_load_model_reads_alignair_without_dataconfig(tmp_path):
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    m = AlignAIR(cfg)
    p = tmp_path / "m.alignair"
    mf.save_model(str(p), m, dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1})
    model, reference = api_load(str(p))                 # NO dataconfigs= needed
    assert reference.gene("V").names[0].startswith("IGH")
    assert model.cfg.v_allele_count == cfg.v_allele_count
