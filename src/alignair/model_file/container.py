"""Low-level .alignair binary container: magic + JSON header + independently-compressed sections."""
from __future__ import annotations

import hashlib
import json
import struct

from ..errors import (MissingDependencyError, ModelFormatError, ModelIntegrityError,
                      ModelSectionError, error_context)

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
        except ImportError:
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
    raise ModelFormatError(f"unknown codec {codec!r}", context={"codec": codec})


def decompress(blob: bytes, codec: str) -> bytes:
    if codec == "none":
        return blob
    if codec == "zlib":
        import zlib
        try:
            return zlib.decompress(blob)
        except zlib.error as exc:
            raise ModelFormatError("invalid zlib model section", context={"codec": codec}) from exc
    if codec == "zstd":
        try:
            import zstandard
        except ImportError as exc:
            raise MissingDependencyError("Reading this model section requires zstandard.",
                                         context={"dependency": "zstandard", "codec": codec},
                                         hint='Install with: pip install "AlignAIR[zstd]"') from exc
        try:
            return zstandard.ZstdDecompressor().decompress(blob)
        except zstandard.ZstdError as exc:
            raise ModelFormatError("invalid zstd model section", context={"codec": codec}) from exc
    raise ModelFormatError(f"unknown codec {codec!r}", context={"codec": codec})


def is_alignair_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(7) == MAGIC[:7]
    except OSError:
        return False


def read_header(path: str) -> dict:
    import os

    with error_context(path=path, stage="model_header"), open(path, "rb") as f:
        magic = f.read(8)
        if len(magic) != 8:
            raise ModelFormatError("truncated AlignAIR model header")
        if magic[:7] != MAGIC[:7]:
            raise ModelFormatError("not an AlignAIR model file (bad magic)")
        if magic[7] > MAJOR_VERSION:
            raise ModelFormatError(f"model written by a newer AlignAIR (format v{magic[7]} > {MAJOR_VERSION})",
                                   hint="Upgrade AlignAIR to a version supporting this model format.")
        size = f.read(8)
        if len(size) != 8:
            raise ModelFormatError("truncated model header length")
        (header_len,) = _HEADER_LEN.unpack(size)
        if header_len > os.fstat(f.fileno()).st_size - 16:
            raise ModelFormatError("truncated model JSON header", context={"header_length": header_len})
        try:
            header = json.loads(f.read(header_len).decode("utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise ModelFormatError("invalid model JSON header") from exc
        if not isinstance(header, dict) or not isinstance(header.get("sections"), dict):
            raise ModelFormatError("model header must contain a sections mapping")
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
    with error_context(path=path, section=name, stage="model_section"):
        return _read_section(path, name)


def _read_section(path: str, name: str) -> bytes:
    header = read_header(path)
    if name not in header["sections"]:
        raise ModelSectionError(f"section {name!r} not in model file")
    s = header["sections"][name]
    if (not isinstance(s, dict)
            or any(not isinstance(s.get(k), int) or isinstance(s[k], bool) or s[k] < 0
                   for k in ("offset", "compressed_length", "payload_length"))
            or any(not isinstance(s.get(k), str)
                   for k in ("codec", "compressed_sha256", "payload_sha256"))):
        raise ModelFormatError(f"invalid metadata for model section {name!r}")
    with open(path, "rb") as f:
        f.seek(0, 2)
        available = f.tell() - header["_sections_base"]
        if s["offset"] > available or s["compressed_length"] > available - s["offset"]:
            raise ModelFormatError(f"section {name!r} extends beyond the model file")
        f.seek(header["_sections_base"] + s["offset"])
        blob = f.read(s["compressed_length"])
    if sha256_hex(blob) != s["compressed_sha256"]:
        raise ModelIntegrityError(f"section {name!r} failed compressed checksum (corrupt/modified file)",
                                  hint="Re-download the model or restore a verified copy.")
    payload = decompress(blob, s["codec"])
    if sha256_hex(payload) != s["payload_sha256"]:
        raise ModelIntegrityError(f"section {name!r} failed payload checksum (corrupt/modified file)",
                                  hint="Re-download the model or restore a verified copy.")
    if len(payload) != s["payload_length"]:
        raise ModelIntegrityError(f"section {name!r} payload length does not match its metadata")
    return payload
