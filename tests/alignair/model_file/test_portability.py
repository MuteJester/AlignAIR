import json

import GenAIRR.data as gd
from safetensors.torch import load as st_load

from alignair.core.config import AlignAIRConfig
from alignair.core import AlignAIR
from alignair import model_file as mf
from alignair.model_file import container as C


def _save(tmp_path):
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    p = tmp_path / "m.alignair"
    mf.save_model(str(p), AlignAIR(cfg), dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1})
    return str(p)


def test_portable_sections_need_no_pickle(tmp_path):
    p = _save(tmp_path)
    json.loads(C.read_section(p, "config").decode())               # json only
    st_load(C.read_section(p, "weights"))                          # safetensors only
    assert C.read_section(p, "reference").decode().startswith(">")  # fasta only
    md = mf.read_metadata(p)
    assert md["sections"]["config"]["format"] == "json"
    assert md["sections"]["weights"]["format"] == "safetensors"
    assert md["sections"]["dataconfig/0"]["format"] == "python-pickle"


def test_read_metadata_reads_only_the_header(tmp_path, monkeypatch):
    p = str(tmp_path / "big.alignair")
    C.write_container(p, {"format_version": 1}, {"weights": (b"X" * 5_000_000, "none")})
    base = C.read_header(p)["_sections_base"]                       # 16 + header_len
    counted = {"n": 0}
    real_open = open

    def counting_open(file, *a, **k):
        f = real_open(file, *a, **k)
        if str(file) == p:
            orig = f.read
            def read(n=-1):
                b = orig(n); counted["n"] += len(b); return b
            f.read = read
        return f

    monkeypatch.setattr("builtins.open", counting_open)
    md = mf.read_metadata(p)
    assert md["sections"]["weights"]["payload_length"] == 5_000_000
    assert counted["n"] == base                                    # header bytes only, never the 5MB payload


def test_codec_matrix_roundtrips():
    for codec in ("none", "zlib", C.available_codec("zstd")):
        data = b"AIRR" * 5000
        assert C.decompress(C.compress(data, codec), codec) == data
