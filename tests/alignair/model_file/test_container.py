import pytest
from alignair.model_file import container as C

pytestmark = pytest.mark.integration


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


@pytest.mark.parametrize("payload, message", [(None, "no such model file"), (b"", "not an AlignAIR model file"),
                                              (b"PK\x03\x04rest", "AlignAIR 3 checkpoint"),
                                              (b">read\nACGT\n", "not an AlignAIR model file")])
def test_the_header_reader_names_what_a_file_that_is_no_model_file_is(tmp_path, payload, message):
    from alignair.errors import ModelError
    path = tmp_path / "model.alignair"
    if payload is not None:
        path.write_bytes(payload)
    with pytest.raises(ModelError, match=message):
        C.read_header(str(path))


def test_a_missing_model_file_is_model_not_found_on_every_path_that_reads_one(tmp_path):
    """As ``Aligner.from_pretrained`` and ``alignair info`` say: the container reader is what ``model_file.load_model``,
    ``read_metadata``, ``alignair export``, ``reference export`` and ``init_from`` reach."""
    from alignair import model_file
    from alignair.errors import ModelError, ModelNotFoundError
    missing = tmp_path / "missing.alignair"
    for read in (C.read_header, model_file.read_metadata, model_file.load_model):
        with pytest.raises(ModelNotFoundError) as caught:
            read(str(missing))
        assert caught.value.code == "ALIGNAIR_MODEL_NOT_FOUND" and "by its id" not in (caught.value.hint or "")
    with pytest.raises(ModelError) as caught:                     # a directory is there: not "not found"
        C.read_header(str(tmp_path))
    assert not isinstance(caught.value, ModelNotFoundError)
