"""Phase 1 / Task 3: .pt pickle-gating in api.load_model + convert upgrade paths."""
from argparse import Namespace

import GenAIRR.data as gd
import pytest
import torch

from alignair import api
from alignair import model_file as mf
from alignair.cli import convert
from alignair.core import AlignAIR
from alignair.core.config import AlignAIRConfig
from alignair.model_file import container as C, serialize


def _model():
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    return AlignAIR(cfg).eval(), cfg


def _legacy_pt(tmp_path):
    model, cfg = _model()
    p = str(tmp_path / "legacy.pt")
    torch.save({"config": cfg.__dict__, "model": model.state_dict(), "step": 7}, p)
    return p


def test_api_load_pt_requires_trust_pickle(tmp_path):
    p = _legacy_pt(tmp_path)
    with pytest.raises(ValueError, match="trust"):
        api.load_model(p, dataconfigs=["HUMAN_IGH_OGRDB"])            # default refuses pickle
    m, ref = api.load_model(p, dataconfigs=["HUMAN_IGH_OGRDB"], trust_pickle=True)
    assert m is not None and ref.gene("V").names


def test_api_load_no_pickle_alignair_needs_no_trust(tmp_path):
    model, _ = _model()
    p = str(tmp_path / "m.alignair")
    mf.save_model(p, model, dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1},
                  include_trusted_pickle=False)
    m, ref = api.load_model(p)                                       # no trust, no dataconfig needed
    assert m is not None and ref.gene("V").gapped


def test_convert_pt_to_no_pickle_alignair(tmp_path):
    src = _legacy_pt(tmp_path)
    dst = str(tmp_path / "out.alignair")
    assert convert.run(Namespace(src=src, dst=dst, dataconfig=["HUMAN_IGH_OGRDB"],
                                 trust_pickle=False)) == 1           # refuse without trust
    assert convert.run(Namespace(src=src, dst=dst, dataconfig=["HUMAN_IGH_OGRDB"],
                                 trust_pickle=True)) == 0
    secs = mf.read_metadata(dst)["sections"]
    assert "reference_json" in secs and not any(k.startswith("dataconfig/") for k in secs)
    m, ref = api.load_model(dst)                                     # safe-loadable, no trust
    assert m is not None


def test_convert_legacy_alignair_upgrades_to_safe(tmp_path):
    # craft a legacy .alignair WITH pickle dataconfig but NO reference_json, then upgrade it
    model, cfg = _model()
    header = {"format_version": 1, "model_class": "AlignAIR",
              "training": {"steps": 3, "batch_size": 2},
              "reference": {"dataconfigs": [{"index": 0, "section": "dataconfig/0", "name": "HUMAN_IGH_OGRDB"}]},
              "_formats": {"config": "json", "weights": "safetensors", "dataconfig/0": "python-pickle"}}
    src = str(tmp_path / "legacy.alignair")
    C.write_container(src, header, {
        "config": (serialize.config_to_bytes(cfg), "zlib"),
        "weights": (serialize.state_dict_to_bytes(model.state_dict()), "none"),
        "dataconfig/0": (serialize.dataconfig_to_bytes(gd.HUMAN_IGH_OGRDB), "zstd")})
    dst = str(tmp_path / "upgraded.alignair")
    assert convert.run(Namespace(src=src, dst=dst, dataconfig=None, trust_pickle=False)) == 1
    assert convert.run(Namespace(src=src, dst=dst, dataconfig=None, trust_pickle=True)) == 0
    secs = mf.read_metadata(dst)["sections"]
    assert "reference_json" in secs and not any(k.startswith("dataconfig/") for k in secs)
    api.load_model(dst)                                              # now safe-loadable
