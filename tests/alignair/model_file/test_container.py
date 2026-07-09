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
