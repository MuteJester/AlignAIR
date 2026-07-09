import pytest
from alignair.model_file import container as C


@pytest.mark.parametrize("codec", ["none", "zlib"])
def test_codec_roundtrip(codec):
    data = b"AlignAIR" * 1000
    blob = C.compress(data, codec)
    assert C.decompress(blob, codec) == data


def test_sha256_hex_stable():
    assert C.sha256_hex(b"abc") == C.sha256_hex(b"abc")
    assert len(C.sha256_hex(b"abc")) == 64


def test_available_codec_falls_back_when_zstd_missing():
    assert C.available_codec("zstd") in ("zstd", "zlib")


def _sections():
    return {"a": (b"hello" * 100, "zlib"), "b": (b"\x00\x01\x02", "none")}


def test_write_read_roundtrip(tmp_path):
    p = tmp_path / "m.alignair"
    C.write_container(str(p), {"format_version": 1, "note": "hi"}, _sections())
    h = C.read_header(str(p))
    assert h["note"] == "hi"
    assert set(h["sections"]) == {"a", "b"}
    assert h["sections"]["a"]["codec"] == "zlib"
    assert h["sections"]["a"]["payload_length"] == 500
    assert C.read_section(str(p), "a") == b"hello" * 100
    assert C.read_section(str(p), "b") == b"\x00\x01\x02"


def test_bad_magic_is_not_alignair(tmp_path):
    p = tmp_path / "x.pt"
    p.write_bytes(b"PK\x03\x04not-ours")
    assert C.is_alignair_file(str(p)) is False
    with pytest.raises(ValueError):
        C.read_header(str(p))


def test_is_alignair_true_for_written_file(tmp_path):
    p = tmp_path / "m.alignair"
    C.write_container(str(p), {"format_version": 1}, {"a": (b"x", "none")})
    assert C.is_alignair_file(str(p)) is True


def test_corrupt_section_fails_checksum(tmp_path):
    p = tmp_path / "m.alignair"
    C.write_container(str(p), {"format_version": 1}, {"a": (b"data" * 100, "zlib")})
    raw = bytearray(p.read_bytes())
    raw[-1] ^= 0xFF
    p.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum|corrupt"):
        C.read_section(str(p), "a")
