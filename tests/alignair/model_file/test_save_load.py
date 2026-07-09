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


def test_load_model_rebuilds_and_matches(tmp_path):
    model, cfg = _fresh_model()
    model.eval()
    p = tmp_path / "m.alignair"
    mf.save_model(str(p), model, dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1})
    lm = mf.load_model(str(p))
    assert lm.config.__dict__ == cfg.__dict__            # full config, no external hints
    ref0 = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    assert lm.reference.gene("V").names == ref0.gene("V").names
    x = {"tokenized_sequence": torch.zeros(1, cfg.max_seq_length, dtype=torch.long)}
    with torch.no_grad():
        a = model(x)["v_start"]
        b = lm.model.eval()(x)["v_start"]
    assert torch.allclose(a, b)


def test_inference_load_still_works_with_optimizer_present(tmp_path):
    model, cfg = _fresh_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    p = tmp_path / "m.alignair"
    mf.save_model(str(p), model, dataconfigs=["HUMAN_IGH_OGRDB"],
                  training={"steps": 1, "batch_size": 1}, optimizer=opt)
    assert "train_state" in mf.read_metadata(str(p))["sections"]
    lm = mf.load_model(str(p))
    assert lm.model is not None
