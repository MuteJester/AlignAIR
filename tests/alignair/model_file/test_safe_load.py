"""Phase 1 / Task 2: pickle-free publish artifacts + safe (no-pickle) inference load."""
import GenAIRR.data as gd
import pytest
import torch

from alignair.core.config import AlignAIRConfig
from alignair.core import AlignAIR
from alignair import model_file as mf
from alignair.model_file import container as C


def _model():
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    return AlignAIR(cfg).eval(), cfg


def test_publish_artifact_has_no_pickle_sections(tmp_path):
    model, _ = _model()
    p = str(tmp_path / "pub.alignair")
    mf.save_model(p, model, dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1},
                  include_trusted_pickle=False, model_id="human-igh", model_version="1.0.0")
    secs = mf.read_metadata(p)["sections"]
    assert "reference_json" in secs and secs["reference_json"]["format"] == "json"
    assert not any(k.startswith("dataconfig/") for k in secs)   # zero pickle sections
    assert "train_state" not in secs


def test_inference_loads_no_pickle_artifact_with_gapped_and_anchors(tmp_path):
    model, cfg = _model()
    p = str(tmp_path / "pub.alignair")
    mf.save_model(p, model, dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1},
                  include_trusted_pickle=False)
    lm = mf.load_model(p)                                        # default trust_pickle=False
    assert lm.reference.gene("V").gapped and lm.reference.gene("V").anchors
    x = {"tokenized_sequence": torch.zeros(1, cfg.max_seq_length, dtype=torch.long)}
    with torch.no_grad():
        assert torch.allclose(model(x)["v_start"], lm.model.eval()(x)["v_start"])


def test_identity_fields_roundtrip(tmp_path):
    model, _ = _model()
    p = str(tmp_path / "pub.alignair")
    mf.save_model(p, model, dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1},
                  model_id="human-igh", model_version="2.1.0",
                  card={"species": "homo_sapiens", "locus": "IGH", "min_alignair": "0.3.0"})
    md = mf.read_metadata(p)
    assert md["model_id"] == "human-igh" and md["model_version"] == "2.1.0"
    assert md["model_format_version"] == C.MAJOR_VERSION and md["created_by_alignair"]
    assert md["species"] == "homo_sapiens" and md["locus"] == "IGH" and md["min_alignair"] == "0.3.0"
    assert md["reference"]["allele_order_sha256"] and md["reference"]["reference_fasta_sha256"]


def test_load_refuses_legacy_without_reference_json(tmp_path):
    # craft a legacy-style container: weights + config + a pickle dataconfig, but NO reference_json
    model, cfg = _model()
    from alignair.model_file import serialize
    header = {"format_version": 1, "model_class": "AlignAIR",
              "reference": {"dataconfigs": [{"index": 0, "section": "dataconfig/0", "name": "HUMAN_IGH_OGRDB"}]},
              "_formats": {"config": "json", "weights": "safetensors", "dataconfig/0": "python-pickle"}}
    p = str(tmp_path / "legacy.alignair")
    C.write_container(p, header, {
        "config": (serialize.config_to_bytes(cfg), "zlib"),
        "weights": (serialize.state_dict_to_bytes(model.state_dict()), "none"),
        "dataconfig/0": (serialize.dataconfig_to_bytes(gd.HUMAN_IGH_OGRDB), "zstd"),
    })
    with pytest.raises(ValueError, match="reference_json|trust"):
        mf.load_model(p)                                        # safe default refuses
    lm = mf.load_model(p, trust_pickle=True)                    # explicit trust -> legacy pickle path
    assert lm.reference.gene("V").names


def test_allele_order_mismatch_is_a_hard_error():
    # the verify guard: a reference whose order != the card hash must be rejected
    from alignair.model_file import serialize
    from alignair.reference.reference_set import ReferenceSet
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    md = {"reference": {"allele_order_sha256": serialize.allele_order_sha256(ref),
                        "reference_fasta_sha256": serialize.reference_fasta_sha256(ref)}}
    mf._verify_reference_integrity(ref, md)                     # matching -> ok
    d = ref.to_serializable()
    d["genes"]["V"]["names"] = list(reversed(d["genes"]["V"]["names"]))
    d["genes"]["V"]["sequences"] = list(reversed(d["genes"]["V"]["sequences"]))
    with pytest.raises(ValueError, match="allele_order"):
        mf._verify_reference_integrity(ReferenceSet.from_serializable(d), md)


def test_resumable_checkpoint_still_has_pickle_and_resumes(tmp_path):
    model, _ = _model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3); opt.step()
    p = str(tmp_path / "ckpt.alignair")
    mf.save_model(p, model, dataconfigs=["HUMAN_IGH_OGRDB"],
                  training={"steps": 5, "batch_size": 2}, optimizer=opt)   # default include_trusted_pickle=True
    secs = mf.read_metadata(p)["sections"]
    assert "dataconfig/0" in secs and "train_state" in secs and "reference_json" in secs
    ts = mf.load_training_state(p)
    assert ts.step == 5 and ts.optimizer_state is not None
