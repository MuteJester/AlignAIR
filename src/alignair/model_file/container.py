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
