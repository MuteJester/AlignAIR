"""Low-level .alignair binary container: magic + JSON header + independently-compressed sections."""
from __future__ import annotations

import hashlib
import json
import struct

MAGIC = b"ALGNAIR\x01"          # 7-char tag + 1-byte MAJOR format version
MAJOR_VERSION = 1
_HEADER_LEN = struct.Struct("<Q")   # u64 little-endian


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def available_codec(preferred: str) -> str:
    """Resolve a preferred codec to one actually usable now (zstd only if zstandard is importable)."""
    if preferred == "zstd":
        try:
            import zstandard  # noqa: F401
        except Exception:
            return "zlib"
    return preferred


def compress(data: bytes, codec: str) -> bytes:
    if codec == "none":
        return data
    if codec == "zlib":
        import zlib
        return zlib.compress(data, 6)
    if codec == "zstd":
        import zstandard
        return zstandard.ZstdCompressor(level=10).compress(data)
    raise ValueError(f"unknown codec {codec!r}")


def decompress(blob: bytes, codec: str) -> bytes:
    if codec == "none":
        return blob
    if codec == "zlib":
        import zlib
        return zlib.decompress(blob)
    if codec == "zstd":
        import zstandard
        return zstandard.ZstdDecompressor().decompress(blob)
    raise ValueError(f"unknown codec {codec!r}")


def is_alignair_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(7) == MAGIC[:7]
    except OSError:
        return False


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic[:7] != MAGIC[:7]:
            raise ValueError("not an AlignAIR model file (bad magic)")
        if magic[7] > MAJOR_VERSION:
            raise ValueError(f"model written by a newer AlignAIR (format v{magic[7]} > {MAJOR_VERSION})")
        (header_len,) = _HEADER_LEN.unpack(f.read(8))
        header = json.loads(f.read(header_len).decode("utf-8"))
    header["_sections_base"] = 16 + header_len          # absolute start of the sections region
    return header


def write_container(path: str, header: dict, sections: dict) -> None:
    formats = header.get("_formats", {})
    index, blobs, offset = {}, [], 0
    for name, (payload, codec) in sections.items():
        codec = available_codec(codec)
        blob = compress(payload, codec)
        index[name] = {"offset": offset, "compressed_length": len(blob), "payload_length": len(payload),
                       "codec": codec, "compressed_sha256": sha256_hex(blob),
                       "payload_sha256": sha256_hex(payload), "format": formats.get(name, "bytes")}
        blobs.append(blob)
        offset += len(blob)
    header = {k: v for k, v in header.items() if not k.startswith("_")}
    header["sections"] = index
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(_HEADER_LEN.pack(len(header_bytes)))
        f.write(header_bytes)
        for blob in blobs:
            f.write(blob)


def read_section(path: str, name: str) -> bytes:
    header = read_header(path)
    if name not in header["sections"]:
        raise KeyError(f"section {name!r} not in model file")
    s = header["sections"][name]
    with open(path, "rb") as f:
        f.seek(header["_sections_base"] + s["offset"])
        blob = f.read(s["compressed_length"])
    if sha256_hex(blob) != s["compressed_sha256"]:
        raise ValueError(f"section {name!r} failed compressed checksum (corrupt/modified file)")
    payload = decompress(blob, s["codec"])
    if sha256_hex(payload) != s["payload_sha256"]:
        raise ValueError(f"section {name!r} failed payload checksum (corrupt/modified file)")
    return payload
