import GenAIRR.data as gd
import torch
from alignair.core.config import AlignAIRConfig
from alignair.core import AlignAIR
from alignair.reference.reference_set import ReferenceSet
from alignair import model_file as mf


def _fresh_model():
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    return AlignAIR(cfg), cfg


def test_save_and_read_metadata(tmp_path):
    model, cfg = _fresh_model()
    p = tmp_path / "m.alignair"
    mf.save_model(str(p), model, dataconfigs=["HUMAN_IGH_OGRDB"],
                  training={"steps": 10, "batch_size": 4, "lr": 1e-4}, description="test")
    md = mf.read_metadata(str(p))
    assert md["model_class"] == "AlignAIR"
    assert md["model"]["allele_counts"]["v"] == cfg.v_allele_count
    assert md["training"]["total_sequences_seen"] == 40
    assert md["reference"]["dataconfigs"][0]["name"] == "HUMAN_IGH_OGRDB"
    assert "config" in md["sections"] and md["sections"]["config"]["format"] == "json"
    assert md["sections"]["weights"]["format"] == "safetensors"
    assert md["sections"]["dataconfig/0"]["format"] == "python-pickle"
