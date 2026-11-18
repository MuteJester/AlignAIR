"""The ``.alignair`` binary container: an 8-byte magic, a JSON header, then the sections, each compressed on its own.

Byte layout (integers little-endian):

    offset   size   content
    0        7      b"ALGNAIR", the format tag
    7        1      the major format version (``MAJOR_VERSION``, 1); a reader refuses a newer one
    8        8      u64: H, the JSON header's length in bytes
    16       H      the JSON header, UTF-8
    16 + H   ...    the sections' blobs, back to back

The header is one JSON object. Its ``sections`` maps each section's name to its blob and how to read it:

- ``offset``: the blob's start, counted from the end of the header (byte 16 + H);
- ``compressed_length``: the blob's length in the file;
- ``payload_length``: the section's length once decompressed;
- ``codec``: ``none``, ``zlib`` or ``zstd`` (reading ``zstd`` needs the ``zstandard`` package);
- ``compressed_sha256`` and ``payload_sha256``: the SHA-256 hex digests of the blob and of the decompressed payload,
  both checked on every read;
- ``format``: how the payload is encoded: ``json``, ``safetensors``, ``fasta``, ``python-pickle`` or ``bytes``.

The header's other keys describe the model (``model_file.io.save_model``): the format versions, the model's settings,
its training, its reference's digests, its provenance and the card's descriptive fields. A model file's sections are
``config`` (JSON), ``weights`` (safetensors), ``reference`` (FASTA) and ``reference_json`` (the full reference, JSON),
and two kinds of Python pickle: ``dataconfig/<i>``, each training dataconfig, unless the file was saved with
``include_trusted_pickle=False`` (a publish artifact has none), and ``train_state`` (the optimizer, scheduler and
random state) when it was saved with any of them, as a training checkpoint is.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct

from ..errors import (
    MissingDependencyError,
    ModelError,
    ModelFormatError,
    ModelIntegrityError,
    ModelNotFoundError,
    ModelSectionError,
    error_context,
)

MAGIC = b"ALGNAIR\x01"          # 7-char tag + 1-byte MAJOR format version
MAJOR_VERSION = 1
_HEADER_LEN = struct.Struct("<Q")   # u64 little-endian


def sha256_hex(data: bytes) -> str:
    """The SHA-256 hex digest of ``data``."""
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
    """``data`` compressed with ``codec`` (``none``, ``zlib`` or ``zstd``)."""
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
    """``blob`` decompressed with ``codec``; a ModelFormatError for a damaged blob or an unknown codec."""
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


def is_alignair_file(path: str | os.PathLike[str]) -> bool:
    """Whether ``path`` starts with the ``.alignair`` format tag."""
    try:
        with open(path, "rb") as f:
            return f.read(7) == MAGIC[:7]
    except OSError:
        return False


# The first bytes of a torch checkpoint (an AlignAIR 3 ``.pt``): a zip archive, or a pickle stream (protocol >= 2).
_TORCH_MAGIC = (b"PK\x03\x04", b"\x80")


def _not_a_model_file(path, head: bytes) -> ModelError:
    """The refusal of a file whose first bytes ``head`` are not a model file's: an AlignAIR 3 checkpoint (a torch
    ``.pt``, refused with the version that runs it), or a file that is no model at all."""
    if head.startswith(_TORCH_MAGIC):
        return ModelFormatError(f"{path} is an AlignAIR 3 checkpoint (not an .alignair model file); AlignAIR 4 reads "
                                "chip models only — use alignair<4 to run it", context={"path": str(path)},
                                hint="pip install 'alignair<4' runs it.")
    return ModelFormatError(f"{path} is not an AlignAIR model file (.alignair)", context={"path": str(path)},
                            hint="An empty or truncated file is an interrupted download: fetch the model again.")


def read_header(path: str | os.PathLike[str]) -> dict:
    """The model file's JSON header. A ModelNotFoundError for a missing file, a ModelError for a path that is not a
    file; a ModelFormatError for a file that is not a model file (an AlignAIR 3 checkpoint named as such), a damaged
    one or a newer one."""

    if not os.path.exists(path):
        raise ModelNotFoundError(f"no such model file: {path}", context={"path": str(path)}, hint="Check the path.")
    if not os.path.isfile(path):
        raise ModelError(f"model path must be a file: {path}", context={"path": str(path)})
    with error_context(path=path, stage="model_header"), open(path, "rb") as f:
        magic = f.read(8)
        if magic[:7] != MAGIC[:7]:
            raise _not_a_model_file(path, magic)
        if len(magic) != 8:
            raise ModelFormatError("truncated AlignAIR model header")
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


def write_container(path: str | os.PathLike[str], header: dict, sections: dict) -> None:
    """Write ``sections`` (name -> ``(payload, codec)``) under ``header`` at ``path``; the header's ``_formats`` gives
    each section's ``format`` and is not written (no key starting with ``_`` is)."""
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


def read_section(path: str | os.PathLike[str], name: str) -> bytes:
    """Section ``name``'s payload, both checksums and its length checked."""
    with error_context(path=path, section=name, stage="model_section"):
        return _read_section(path, name)


def _read_section(path: str | os.PathLike[str], name: str) -> bytes:
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
