"""Pickle-free publish artifacts + safe (no-pickle) inference load."""
import GenAIRR.data as gd
import pytest

from alignair.core import prepare_model
from alignair.reference.reference_set import ReferenceSet
import torch

from alignair.core import AlignAIR
from alignair import model_file as mf
from alignair.model_file import container as C

pytestmark = pytest.mark.integration


def _model():
    from alignair.reference import ReferenceSet
    cfg = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), max_seq_length=128, embed_dim=4,
                        trunk_channels=8).config
    model = AlignAIR(cfg).eval()
    model.bind_reference(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB))   # the chip reads its bank from it
    return model, cfg


def test_publish_artifact_has_no_pickle_sections(tmp_path):
    model, _ = _model()
    p = str(tmp_path / "pub.alignair")
    mf.save_model(p, model, ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB),
                  dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1},
                  include_trusted_pickle=False, model_id="human-igh", model_version="1.0.0")
    secs = mf.read_metadata(p)["sections"]
    assert "reference_json" in secs and secs["reference_json"]["format"] == "json"
    assert not any(k.startswith("dataconfig/") for k in secs)   # zero pickle sections
    assert "train_state" not in secs



def test_a_publish_artifact_records_no_host_or_user_and_a_checkpoint_does(tmp_path):
    """A model file made to be shared names neither the machine nor the login it was written on; a resumable
    checkpoint, which stays with its run, still does."""
    model, _ = _model()
    published, checkpoint = str(tmp_path / "pub.alignair"), str(tmp_path / "ckpt.alignair")
    for path, trusted in ((published, False), (checkpoint, True)):
        mf.save_model(path, model, ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB),
                      dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1},
                      include_trusted_pickle=trusted)
    assert {"host", "user"}.isdisjoint(mf.read_metadata(published)["provenance"])
    assert {"host", "user"} <= set(mf.read_metadata(checkpoint)["provenance"])

def test_inference_loads_no_pickle_artifact_with_gapped_and_anchors(tmp_path):
    model, cfg = _model()
    p = str(tmp_path / "pub.alignair")
    mf.save_model(p, model, ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB),
                  dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1},
                  include_trusted_pickle=False)
    lm = mf.load_model(p)
    assert lm.reference.gene("V").gapped and lm.reference.gene("V").anchors
    x = {"tokenized_sequence": torch.zeros(1, cfg.max_seq_length, dtype=torch.long)}
    with torch.no_grad():
        assert torch.allclose(model(x)["v_start"], lm.model.eval()(x)["v_start"])


def test_identity_fields_roundtrip(tmp_path):
    model, _ = _model()
    p = str(tmp_path / "pub.alignair")
    mf.save_model(p, model, ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB),
                  dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1},
                  model_id="human-igh", model_version="2.1.0",
                  card={"species": "homo_sapiens", "locus": "IGH", "min_alignair": "0.3.0"})
    md = mf.read_metadata(p)
    assert md["model_id"] == "human-igh" and md["model_version"] == "2.1.0"
    assert md["model_format_version"] == C.MAJOR_VERSION and md["created_by_alignair"]
    assert md["species"] == "homo_sapiens" and md["locus"] == "IGH" and md["min_alignair"] == "0.3.0"
    assert md["reference"]["allele_order_sha256"] and md["reference"]["reference_fasta_sha256"]


def test_update_card_patches_metadata_and_preserves_model(tmp_path):
    """update_card sets descriptive fields (model_id/version/locus/species) on an existing pickle-free
    artifact while keeping every section byte-for-byte — the loaded model + reference hashes are
    identical, so a re-carded artifact is safe to publish."""
    model, cfg = _model()
    src = str(tmp_path / "src.alignair")
    mf.save_model(src, model, ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB),
                  dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1},
                  include_trusted_pickle=False)                    # no model_id/version yet
    assert mf.read_metadata(src).get("model_id") is None
    dst = str(tmp_path / "carded.alignair")
    md = mf.update_card(src, dst, {"model_id": "alignair-igh-human", "model_version": "1.0.0",
                                   "locus": "IGH", "species": "Homo sapiens", "receptor": "IG"})
    assert md["model_id"] == "alignair-igh-human" and md["model_version"] == "1.0.0"
    assert md["locus"] == "IGH" and md["species"] == "Homo sapiens"
    # integrity: still pickle-free, hashes unchanged, weights bit-identical on reload
    src_md, dst_md = mf.read_metadata(src), mf.read_metadata(dst)
    assert dst_md["reference"]["allele_order_sha256"] == src_md["reference"]["allele_order_sha256"]
    assert "reference_json" in dst_md["sections"]
    assert not any(k.startswith("dataconfig/") or k == "train_state" for k in dst_md["sections"])
    lm = mf.load_model(dst)                                        # loads safe (no trust needed)
    x = {"tokenized_sequence": torch.zeros(1, cfg.max_seq_length, dtype=torch.long)}
    with torch.no_grad():
        assert torch.allclose(model(x)["v_start"], lm.model.eval()(x)["v_start"])


def test_update_card_rejects_structural_fields(tmp_path):
    model, _ = _model()
    src = str(tmp_path / "src.alignair")
    mf.save_model(src, model, ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB),
                  dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1},
                  include_trusted_pickle=False)
    with pytest.raises(ValueError, match="structural/integrity"):
        mf.update_card(src, str(tmp_path / "x.alignair"), {"reference": {"allele_order_sha256": "deadbeef"}})


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
    with pytest.raises(ValueError, match="no reference_json: it was written before AlignAIR 4"):
        mf.load_model(p)                                        # no pickled reference is ever read


def test_allele_order_mismatch_is_a_hard_error():
    # the verify guard: a reference whose order != the card hash must be rejected
    from alignair.model_file import io as model_file_io, serialize
    from alignair.reference.reference_set import ReferenceSet
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    md = {"reference": {"allele_order_sha256": serialize.allele_order_sha256(ref),
                        "reference_fasta_sha256": serialize.reference_fasta_sha256(ref)}}
    model_file_io._verify_reference_integrity(ref, md)                     # matching -> ok
    d = ref.to_serializable()
    d["genes"]["V"]["names"] = list(reversed(d["genes"]["V"]["names"]))
    d["genes"]["V"]["sequences"] = list(reversed(d["genes"]["V"]["sequences"]))
    with pytest.raises(ValueError, match="allele_order"):
        model_file_io._verify_reference_integrity(ReferenceSet.from_serializable(d), md)


def test_resumable_checkpoint_still_has_pickle_and_resumes(tmp_path):
    model, _ = _model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3); opt.step()
    p = str(tmp_path / "ckpt.alignair")
    mf.save_model(p, model, ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), dataconfigs=["HUMAN_IGH_OGRDB"],
                  training={"steps": 5, "batch_size": 2}, optimizer=opt)   # default include_trusted_pickle=True
    secs = mf.read_metadata(p)["sections"]
    assert "dataconfig/0" in secs and "train_state" in secs and "reference_json" in secs
    ts = mf.load_training_state(p)
    assert ts.step == 5 and ts.optimizer_state is not None


def test_a_model_files_reference_it_cannot_read_is_an_integrity_error(tmp_path, monkeypatch):
    """A fault in the file's own reference is the file's, never the reference a user binds."""
    from alignair.errors import DataContractError, ModelIntegrityError
    model, _ = _model()
    p = str(tmp_path / "pub.alignair")
    mf.save_model(p, model, ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB),
                  dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1},
                  include_trusted_pickle=False)

    def unreadable(data):
        raise DataContractError("reference info names an allele its gene does not hold", context={"gene": "V"})
    monkeypatch.setattr(mf.serialize, "reference_from_json", unreadable)
    with pytest.raises(ModelIntegrityError):
        mf.load_model(p)
