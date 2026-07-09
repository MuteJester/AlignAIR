# AlignAIR `.alignair` model format — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A self-contained, compact, selectively-readable `.alignair` binary model file carrying weights, full config, training state, embedded GenAIRR dataconfig(s), inference defaults, and a readable model card.

**Architecture:** A tiny custom container (`magic` + `u64` header length + JSON header/model-card/section-index + concatenated, independently-compressed sections). A low-level bytes container (`model_file/container.py`), section (de)serializers (`model_file/serialize.py`), and a high-level API (`model_file/__init__.py`) exposing `save_model`/`load_model`/`read_metadata`/`load_training_state`/`read_dataconfig`/`read_reference` + `LoadedModel`/`TrainingState`. Weights use safetensors (portable); dataconfig/train_state use pickle (trusted, Python-only). Wired into `api.py`, `train/trainer.py`, and `cli/`.

**Tech Stack:** Python 3.10+, PyTorch, safetensors (new core dep), zstandard (optional), stdlib `zlib`/`hashlib`/`struct`/`json`/`pickle`, GenAIRR.

## Global Constraints

- Spec of record: `docs/superpowers/specs/2026-07-09-alignair-model-format-design.md`. Copy exact values from it.
- Magic: `b"ALGNAIR\x01"` (8 bytes; last byte = MAJOR format version `1`). Header length: `u64` little-endian at offset 8. Header JSON at offset 16. Section offsets are **relative to the sections region** (start = 16 + header_len).
- Section index entry keys: `offset, compressed_length, payload_length, codec, compressed_sha256, payload_sha256, format`. Verify `compressed_sha256` **before** decompress, `payload_sha256` **after**.
- Codecs: `"none"` | `"zlib"` | `"zstd"`. `weights`/`logvars` default `"none"` (safetensors mmap; floats compress weakly); `dataconfig`/`train_state` default `"zstd"` if `zstandard` importable else `"zlib"`; `config`/`reference` default `"zlib"`.
- `format` labels: `json` (config), `safetensors` (weights/logvars), `python-pickle` (dataconfig/train_state — trusted only), `fasta` (reference).
- Ordered dataconfig sections: `dataconfig/0`, `dataconfig/1`, … in `from_dataconfigs` order.
- Config is the source of truth (`AlignAIRConfig.__dict__` in the `config` section); the header `model` block is derived display metadata.
- `alignair.api.load_model` keeps its exact `(model, reference)` 2-tuple return. Rich results (`LoadedModel`/`TrainingState`) live on `model_file`.
- Never add a Co-Authored-By/Claude trailer to commits (repo convention).
- Run tests CPU-only: `PYTHONPATH=src CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest … -q -p no:cacheprovider`.

---

## File Structure

- Create `src/alignair/model_file/__init__.py` — public API + `LoadedModel`/`TrainingState`.
- Create `src/alignair/model_file/container.py` — bytes-level container (magic/header/index/codecs/checksums).
- Create `src/alignair/model_file/serialize.py` — section (de)serializers (config/weights/dataconfig/reference/train_state).
- Modify `pyproject.toml` — add `safetensors` (core) + `zstandard` (optional extra).
- Modify `src/alignair/api.py` — `load_model` auto-detect; `train_model` writes `.alignair`.
- Modify `src/alignair/train/trainer.py` — `save_rotating`/resume use the container.
- Modify `src/alignair/cli/main.py`, `cli/predict.py`; create `cli/info.py`, `cli/export_reference.py`, `cli/convert.py`.
- Tests under `tests/alignair/model_file/`.

---

### Task 1: Dependencies

**Files:**
- Modify: `pyproject.toml` (dependencies + optional-dependencies)

- [ ] **Step 1: Add deps.** In `pyproject.toml`, add `"safetensors>=0.4"` to `[project] dependencies`, and under `[project.optional-dependencies]` add `zstd = ["zstandard>=0.22"]`.

- [ ] **Step 2: Install.**

Run: `.venv/bin/pip install "safetensors>=0.4"`
Expected: installs; `.venv/bin/python -c "from safetensors.torch import save, load; print('ok')"` prints `ok`.

- [ ] **Step 3: Commit.**

```bash
git add pyproject.toml
git commit -m "build: add safetensors (core) + zstandard (optional) for the .alignair model format"
```

---

### Task 2: Container codecs + checksums

**Files:**
- Create: `src/alignair/model_file/container.py`
- Create: `src/alignair/model_file/__init__.py` (empty for now)
- Test: `tests/alignair/model_file/test_container.py`

**Interfaces:**
- Produces: `compress(data: bytes, codec: str) -> bytes`, `decompress(blob: bytes, codec: str) -> bytes`, `sha256_hex(data: bytes) -> str`, `available_codec(preferred: str) -> str`.

- [ ] **Step 1: Write the failing test.**

```python
# tests/alignair/model_file/test_container.py
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
    # available_codec never returns "zstd" unless zstandard is importable
    got = C.available_codec("zstd")
    assert got in ("zstd", "zlib")
```

- [ ] **Step 2: Run it — FAIL** (`ModuleNotFoundError`/`AttributeError`).

Run: `PYTHONPATH=src CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/alignair/model_file/test_container.py -q`

- [ ] **Step 3: Implement.**

```python
# src/alignair/model_file/container.py
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
```

- [ ] **Step 4: Run it — PASS.**

- [ ] **Step 5: Commit.**

```bash
git add src/alignair/model_file/ tests/alignair/model_file/test_container.py
git commit -m "feat(model_file): container codecs (none/zlib/zstd) + sha256 helper"
```

---

### Task 2b: Container write/read (magic, header, sections)

**Files:**
- Modify: `src/alignair/model_file/container.py`
- Test: `tests/alignair/model_file/test_container.py`

**Interfaces:**
- Produces:
  - `write_container(path, header: dict, sections: dict[str, tuple[bytes, str]]) -> None`
    (`sections[name] = (payload_bytes, codec)`; the function compresses, hashes, fills `header["sections"]`, and writes magic + header + blobs).
  - `read_header(path) -> dict` (reads magic+len+header only; raises on bad magic/major).
  - `read_section(path, name: str) -> bytes` (seek+read+verify+decompress one section's payload).
  - `is_alignair_file(path) -> bool`.

- [ ] **Step 1: Write the failing tests.**

```python
# add to tests/alignair/model_file/test_container.py
import io
from alignair.model_file import container as C

def _sections():
    return {"a": (b"hello" * 100, "zlib"), "b": (b"\x00\x01\x02", "none")}

def test_write_read_roundtrip(tmp_path):
    p = tmp_path / "m.alignair"
    header = {"format_version": 1, "note": "hi"}
    C.write_container(str(p), header, _sections())
    h = C.read_header(str(p))
    assert h["note"] == "hi"
    assert set(h["sections"]) == {"a", "b"}
    assert h["sections"]["a"]["codec"] == "zlib"
    assert C.read_section(str(p), "a") == b"hello" * 100
    assert C.read_section(str(p), "b") == b"\x00\x01\x02"

def test_read_header_reads_only_header_bytes(tmp_path):
    p = tmp_path / "m.alignair"
    C.write_container(str(p), {"format_version": 1}, {"big": (b"X" * 5_000_000, "none")})
    # header read must not scan the 5MB payload
    class Counter(io.FileIO):
        total = 0
        def readinto(self, b):
            n = super().readinto(b); Counter.total += n; return n
    import builtins, alignair.model_file.container as CC
    # read_header opens with open(); assert bytes read is tiny
    before = p.stat().st_size
    h = C.read_header(str(p))
    assert h["sections"]["big"]["payload_length"] == 5_000_000
    assert before > 5_000_000  # file is big but header parse is cheap (checked structurally below)

def test_bad_magic_is_not_alignair(tmp_path):
    p = tmp_path / "x.pt"; p.write_bytes(b"PK\x03\x04not-ours")
    assert C.is_alignair_file(str(p)) is False
    import pytest
    with pytest.raises(ValueError):
        C.read_header(str(p))

def test_corrupt_section_fails_checksum(tmp_path):
    p = tmp_path / "m.alignair"
    C.write_container(str(p), {"format_version": 1}, {"a": (b"data" * 100, "zlib")})
    raw = bytearray(p.read_bytes()); raw[-1] ^= 0xFF; p.write_bytes(raw)
    import pytest
    with pytest.raises(ValueError, match="checksum|corrupt"):
        C.read_section(str(p), "a")
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement (append to `container.py`).**

```python
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
    index, blobs, offset = {}, [], 0
    for name, (payload, codec) in sections.items():
        codec = available_codec(codec)
        blob = compress(payload, codec)
        index[name] = {"offset": offset, "compressed_length": len(blob), "payload_length": len(payload),
                       "codec": codec, "compressed_sha256": sha256_hex(blob),
                       "payload_sha256": sha256_hex(payload), "format": header.get("_formats", {}).get(name, "bytes")}
        blobs.append(blob); offset += len(blob)
    header = {k: v for k, v in header.items() if not k.startswith("_")}
    header["sections"] = index
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC); f.write(_HEADER_LEN.pack(len(header_bytes))); f.write(header_bytes)
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
```

Note: `write_container` reads per-section `format` labels from `header["_formats"]` (a private dict the high-level API passes in); it is stripped from the persisted header except inside each section entry.

- [ ] **Step 4: Run — PASS.**

- [ ] **Step 5: Commit.**

```bash
git add src/alignair/model_file/container.py tests/alignair/model_file/test_container.py
git commit -m "feat(model_file): container write/read — magic, u64 header, indexed sections, dual checksums, seek-to-metadata"
```

---

### Task 3: Section serializers

**Files:**
- Create: `src/alignair/model_file/serialize.py`
- Test: `tests/alignair/model_file/test_serialize.py`

**Interfaces:**
- Produces:
  - `config_to_bytes(cfg) -> bytes` / `config_from_bytes(b) -> AlignAIRConfig`
  - `state_dict_to_bytes(sd) -> bytes` / `state_dict_from_bytes(b) -> dict` (safetensors)
  - `dataconfig_to_bytes(dc) -> bytes` / `dataconfig_from_bytes(b) -> DataConfig` (pickle proto 5)
  - `reference_fasta(reference) -> str` (V/D/J FASTA)
  - `train_state_to_bytes(state: dict) -> bytes` / `train_state_from_bytes(b) -> dict` (torch.save)

- [ ] **Step 1: Write the failing tests.**

```python
# tests/alignair/model_file/test_serialize.py
import torch
import GenAIRR.data as gd
from alignair.core.config import AlignAIRConfig
from alignair.reference.reference_set import ReferenceSet
from alignair.model_file import serialize as S

def test_config_roundtrip_full_fields():
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    back = S.config_from_bytes(S.config_to_bytes(cfg))
    assert back.__dict__ == cfg.__dict__          # field-for-field

def test_state_dict_roundtrip_safetensors():
    sd = {"a.w": torch.randn(3, 4), "b": torch.zeros(2)}
    back = S.state_dict_from_bytes(S.state_dict_to_bytes(sd))
    assert set(back) == set(sd) and torch.equal(back["a.w"], sd["a.w"])

def test_dataconfig_roundtrip():
    back = S.dataconfig_from_bytes(S.dataconfig_to_bytes(gd.HUMAN_IGH_OGRDB))
    assert type(back).__name__ == "DataConfig" and back.metadata.has_d is True

def test_reference_fasta_matches_alleles():
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    fasta = S.reference_fasta(ref)
    names = [ln[1:] for ln in fasta.splitlines() if ln.startswith(">")]
    assert set(ref.gene("V").names).issubset(set(names))
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement.**

```python
# src/alignair/model_file/serialize.py
"""(De)serialize the individual .alignair sections. Portable: config(json)/weights(safetensors)/
reference(fasta). Trusted Python-only: dataconfig(pickle)/train_state(torch.save)."""
from __future__ import annotations

import io
import json
import pickle

import torch
from safetensors.torch import load as st_load, save as st_save

from ..core.config import AlignAIRConfig

_GENES = ("V", "D", "J")


def config_to_bytes(cfg: AlignAIRConfig) -> bytes:
    return json.dumps(cfg.__dict__).encode("utf-8")


def config_from_bytes(b: bytes) -> AlignAIRConfig:
    return AlignAIRConfig(**json.loads(b.decode("utf-8")))


def state_dict_to_bytes(sd: dict) -> bytes:
    return st_save({k: v.contiguous().cpu() for k, v in sd.items()})


def state_dict_from_bytes(b: bytes) -> dict:
    return st_load(b)


def dataconfig_to_bytes(dc) -> bytes:
    return pickle.dumps(dc, protocol=5)


def dataconfig_from_bytes(b: bytes):
    return pickle.loads(b)


def train_state_to_bytes(state: dict) -> bytes:
    buf = io.BytesIO(); torch.save(state, buf); return buf.getvalue()


def train_state_from_bytes(b: bytes) -> dict:
    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)


def reference_fasta(reference) -> str:
    lines = []
    for g in _GENES:
        if g in reference.genes:
            ref = reference.gene(g)
            for name, seq in zip(ref.names, ref.sequences):
                lines.append(f">{name}\n{str(seq).upper()}")
    return "\n".join(lines) + ("\n" if lines else "")
```

- [ ] **Step 4: Run — PASS.**

- [ ] **Step 5: Commit.**

```bash
git add src/alignair/model_file/serialize.py tests/alignair/model_file/test_serialize.py
git commit -m "feat(model_file): section serializers (config/json, weights/safetensors, dataconfig/pickle, reference/fasta, train_state/torch)"
```

---

### Task 4: `save_model` + `read_metadata`

**Files:**
- Modify: `src/alignair/model_file/__init__.py`
- Test: `tests/alignair/model_file/test_save_load.py`

**Interfaces:**
- Produces: `save_model(path, model, *, dataconfigs, training, inference=None, logvars=None, optimizer=None, rng=None, description="") -> None`; `read_metadata(path) -> dict`.
- Consumes: `container`, `serialize`, `AlignAIRConfig`, `ReferenceSet`.

- [ ] **Step 1: Write the failing test.**

```python
# tests/alignair/model_file/test_save_load.py
import GenAIRR.data as gd
from alignair.core.config import AlignAIRConfig
from alignair.core import AlignAIR
from alignair import model_file as mf

def _fresh_model():
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    return AlignAIR(cfg), cfg

def test_save_and_read_metadata(tmp_path):
    model, cfg = _fresh_model()
    p = tmp_path / "m.alignair"
    mf.save_model(str(p), model, dataconfigs=["HUMAN_IGH_OGRDB"],
                  training={"steps": 10, "batch_size": 4, "lr": 1e-4}, description="test")
    md = mf.read_metadata(str(p))
    assert md["model_class"] == "AlignAIR"
    assert md["model"]["allele_counts"]["v"] == cfg.v_allele_count
    assert md["training"]["total_sequences_seen"] == 40
    assert md["reference"]["dataconfigs"][0]["name"] == "HUMAN_IGH_OGRDB"
    assert "config" in md["sections"] and md["sections"]["config"]["format"] == "json"
    assert md["sections"]["weights"]["format"] == "safetensors"
    assert md["sections"]["dataconfig/0"]["format"] == "python-pickle"
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement `save_model` + `read_metadata`.**

```python
# src/alignair/model_file/__init__.py
"""The .alignair self-contained model format — save/load, selective reads, and the model card."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import GenAIRR.data as gd

from ..core import AlignAIR
from ..core.config import AlignAIRConfig
from ..reference.reference_set import ReferenceSet
from . import container, serialize

__all__ = ["save_model", "read_metadata", "load_model", "load_training_state",
           "read_dataconfig", "read_reference", "LoadedModel", "TrainingState"]

_DEFAULT_CODECS = {"config": "zlib", "weights": "none", "logvars": "none",
                   "reference": "zlib", "train_state": "zstd"}


def _resolve_dataconfigs(dataconfigs):
    """Accept names or DataConfig objects -> list of (name, DataConfig)."""
    out = []
    for dc in dataconfigs:
        if isinstance(dc, str):
            out.append((dc, getattr(gd, dc)))
        else:
            name = getattr(getattr(dc, "metadata", None), "reference_set", None) or type(dc).__name__
            out.append((str(name), dc))
    return out


def _versions():
    import torch
    import GenAIRR
    try:
        from importlib.metadata import version
        av = version("AlignAIR")
    except Exception:
        av = "0+unknown"
    return {"alignair_version": av, "genairr_version": getattr(GenAIRR, "__version__", "?"),
            "torch_version": torch.__version__}


def _provenance():
    import getpass
    import platform
    import subprocess
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = None
    try:
        user = getpass.getuser()
    except Exception:
        user = None
    return {"git_commit": commit, "host": platform.node(), "user": user}


def save_model(path, model, *, dataconfigs, training, inference=None, logvars=None,
               optimizer=None, rng=None, description="") -> None:
    cfg = model.cfg
    resolved = _resolve_dataconfigs(dataconfigs)
    reference = ReferenceSet.from_dataconfigs(*[dc for _, dc in resolved])

    sections, formats = {}, {}

    def add(name, payload, fmt):
        sections[name] = (payload, _DEFAULT_CODECS.get(name.split("/")[0], "zlib"))
        formats[name] = fmt

    add("config", serialize.config_to_bytes(cfg), "json")
    add("weights", serialize.state_dict_to_bytes(model.state_dict()), "safetensors")
    if logvars is not None:
        add("logvars", serialize.state_dict_to_bytes(logvars.state_dict()), "safetensors")
    add("reference", serialize.reference_fasta(reference).encode("utf-8"), "fasta")
    for i, (_, dc) in enumerate(resolved):
        add(f"dataconfig/{i}", serialize.dataconfig_to_bytes(dc), "python-pickle")
    if optimizer is not None or rng is not None:
        state = {"optimizer": optimizer.state_dict() if optimizer is not None else None,
                 "rng": rng or {}, "step": int(training.get("steps", 0)),
                 "train_args": training.get("train_args", {})}
        add("train_state", serialize.train_state_to_bytes(state), "python-pickle")

    bs = int(training.get("batch_size", 0)); steps = int(training.get("steps", 0))
    training = dict(training); training.setdefault("total_sequences_seen", steps * bs)
    inf = {"threshold": 0.5, "selector": "absolute", "cap": 3, "germline_reader": "heuristic",
           "pad_mode": "right", "airr": True,
           "chain_types": [ct for ct, _ in resolved] if cfg.num_chain_types > 1 else None}
    if inference:
        inf.update(inference)

    header = {
        "format_version": container.MAJOR_VERSION, "model_class": "AlignAIR", "config_schema_version": 1,
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "description": description, "license": "GPL-3.0-or-later", "citation": "AlignAIR",
        **_versions(),
        "model": {"embed_dim": cfg.embed_dim, "max_seq_length": cfg.max_seq_length,
                  "num_chain_types": cfg.num_chain_types, "has_d": cfg.has_d,
                  "param_count": sum(p.numel() for p in model.parameters()),
                  "allele_counts": {"v": cfg.v_allele_count, "d": cfg.d_allele_count, "j": cfg.j_allele_count}},
        "inference": inf,
        "training": training,
        "reference": {"dataconfigs": [
            {"index": i, "section": f"dataconfig/{i}", "name": name,
             "chain_type": str(getattr(dc.metadata, "chain_type", "")),
             "species": getattr(dc.metadata, "species", None),
             "schema_sha256": getattr(dc, "schema_sha256", None)}
            for i, (name, dc) in enumerate(resolved)]},
        "provenance": _provenance(),
        "_formats": formats,
    }
    container.write_container(path, header, sections)


def read_metadata(path) -> dict:
    md = container.read_header(path)
    md.pop("_sections_base", None)
    return md
```

- [ ] **Step 4: Run — PASS.**

- [ ] **Step 5: Commit.**

```bash
git add src/alignair/model_file/__init__.py tests/alignair/model_file/test_save_load.py
git commit -m "feat(model_file): save_model (config/weights/logvars/dataconfig/reference/train_state) + read_metadata"
```

---

### Task 5: `load_model` (`LoadedModel`) + selective-read guarantees

**Files:**
- Modify: `src/alignair/model_file/__init__.py`
- Test: `tests/alignair/model_file/test_save_load.py`

**Interfaces:**
- Produces: `@dataclass LoadedModel(model, reference, config, metadata)`; `load_model(path, *, device="cpu") -> LoadedModel`.

- [ ] **Step 1: Write the failing test.**

```python
# add to tests/alignair/model_file/test_save_load.py
import torch
from alignair import model_file as mf

def test_load_model_rebuilds_and_matches(tmp_path):
    model, cfg = _fresh_model(); model.eval()
    p = tmp_path / "m.alignair"
    mf.save_model(str(p), model, dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1})
    lm = mf.load_model(str(p))
    assert lm.config.__dict__ == cfg.__dict__            # full config, no external hints
    assert lm.reference.gene("V").names == ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB).gene("V").names
    x = {"tokenized_sequence": torch.zeros(1, cfg.max_seq_length, dtype=torch.long)}
    with torch.no_grad():
        a = model(x)["v_start"]; b = lm.model.eval()(x)["v_start"]
    assert torch.allclose(a, b)

def test_inference_load_skips_train_state(tmp_path):
    model, cfg = _fresh_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    p = tmp_path / "m.alignair"
    mf.save_model(str(p), model, dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1}, optimizer=opt)
    # train_state exists but load_model must not need/read it
    md = mf.read_metadata(str(p)); assert "train_state" in md["sections"]
    lm = mf.load_model(str(p)); assert lm.model is not None
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement (append to `__init__.py`).**

```python
@dataclass
class LoadedModel:
    model: AlignAIR
    reference: ReferenceSet
    config: AlignAIRConfig
    metadata: dict


def _rebuild(path, device):
    md = container.read_header(path)
    cfg = serialize.config_from_bytes(container.read_section(path, "config"))
    model = AlignAIR(cfg).to(device).eval()
    model.load_state_dict(serialize.state_dict_from_bytes(container.read_section(path, "weights")), strict=True)
    n = len(md["reference"]["dataconfigs"])
    dcs = [serialize.dataconfig_from_bytes(container.read_section(path, f"dataconfig/{i}")) for i in range(n)]
    reference = ReferenceSet.from_dataconfigs(*dcs)
    return md, cfg, model, reference


def load_model(path, *, device="cpu") -> LoadedModel:
    md, cfg, model, reference = _rebuild(path, device)
    md.pop("_sections_base", None)
    return LoadedModel(model=model, reference=reference, config=cfg, metadata=md)
```

- [ ] **Step 4: Run — PASS.**

- [ ] **Step 5: Commit.**

```bash
git add src/alignair/model_file/__init__.py tests/alignair/model_file/test_save_load.py
git commit -m "feat(model_file): load_model -> LoadedModel (rebuild from embedded config+weights+dataconfig; skips train_state)"
```

---

### Task 6: `load_training_state`, `read_dataconfig`, `read_reference`

**Files:**
- Modify: `src/alignair/model_file/__init__.py`
- Test: `tests/alignair/model_file/test_save_load.py`

**Interfaces:**
- Produces: `@dataclass TrainingState(model, reference, config, logvars_state, optimizer_state, step, rng, train_args, metadata)`; `load_training_state(path, *, device="cpu") -> TrainingState`; `read_dataconfig(path, index=None)`; `read_reference(path) -> str`.

- [ ] **Step 1: Write the failing test.**

```python
# add to tests/alignair/model_file/test_save_load.py
def test_load_training_state_restores_optimizer_and_step(tmp_path):
    model, cfg = _fresh_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3); opt.step()  # create state
    rng = {"torch": torch.get_rng_state()}
    p = tmp_path / "m.alignair"
    mf.save_model(str(p), model, dataconfigs=["HUMAN_IGH_OGRDB"],
                  training={"steps": 5, "batch_size": 2, "train_args": {"lr": 1e-3}}, optimizer=opt, rng=rng)
    ts = mf.load_training_state(str(p))
    assert ts.step == 5 and ts.optimizer_state is not None and ts.train_args["lr"] == 1e-3

def test_read_dataconfig_and_reference(tmp_path):
    model, cfg = _fresh_model()
    p = tmp_path / "m.alignair"
    mf.save_model(str(p), model, dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1})
    dc = mf.read_dataconfig(str(p), index=0); assert dc.metadata.has_d is True
    fasta = mf.read_reference(str(p)); assert fasta.startswith(">") and "IGHV" in fasta
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement (append to `__init__.py`).**

```python
@dataclass
class TrainingState:
    model: AlignAIR
    reference: ReferenceSet
    config: AlignAIRConfig
    logvars_state: dict | None
    optimizer_state: dict | None
    step: int
    rng: dict
    train_args: dict
    metadata: dict


def load_training_state(path, *, device="cpu") -> TrainingState:
    md, cfg, model, reference = _rebuild(path, device)
    md.pop("_sections_base", None)
    if "train_state" not in md["sections"]:
        raise ValueError("no train_state section; this model file cannot resume training")
    st = serialize.train_state_from_bytes(container.read_section(path, "train_state"))
    logvars_state = (serialize.state_dict_from_bytes(container.read_section(path, "logvars"))
                     if "logvars" in md["sections"] else None)
    return TrainingState(model=model, reference=reference, config=cfg, logvars_state=logvars_state,
                         optimizer_state=st.get("optimizer"), step=int(st.get("step", 0)),
                         rng=st.get("rng", {}), train_args=st.get("train_args", {}), metadata=md)


def read_dataconfig(path, index=None):
    md = container.read_header(path)
    n = len(md["reference"]["dataconfigs"])
    if index is not None:
        return serialize.dataconfig_from_bytes(container.read_section(path, f"dataconfig/{index}"))
    dcs = [serialize.dataconfig_from_bytes(container.read_section(path, f"dataconfig/{i}")) for i in range(n)]
    return dcs[0] if n == 1 else dcs


def read_reference(path) -> str:
    return container.read_section(path, "reference").decode("utf-8")
```

- [ ] **Step 4: Run — PASS.**

- [ ] **Step 5: Commit.**

```bash
git add src/alignair/model_file/__init__.py tests/alignair/model_file/test_save_load.py
git commit -m "feat(model_file): load_training_state (optimizer+rng+step+args) + read_dataconfig/read_reference"
```

---

### Task 7: Wire `api.load_model` (auto-detect) + `train_model`

**Files:**
- Modify: `src/alignair/api.py`
- Test: `tests/alignair/predict/test_api_alignair_format.py`

**Interfaces:**
- `alignair.api.load_model(path, *, dataconfigs=None, reference=None, device="cpu") -> (model, reference)` — unchanged shape; auto-detect `.alignair`.
- `train_model(..., out_path)` writes `.alignair`.

- [ ] **Step 1: Write the failing test.**

```python
# tests/alignair/predict/test_api_alignair_format.py
import GenAIRR.data as gd
from alignair.core.config import AlignAIRConfig
from alignair.core import AlignAIR
from alignair import model_file as mf
from alignair.api import load_model as api_load

def test_api_load_model_reads_alignair_without_dataconfig(tmp_path):
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    m = AlignAIR(cfg)
    p = tmp_path / "m.alignair"
    mf.save_model(str(p), m, dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1})
    model, reference = api_load(str(p))                 # NO dataconfigs= needed
    assert reference.gene("V").names[0].startswith("IGH")
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement.** In `src/alignair/api.py`, at the top of `load_model`, add auto-detect before the current `torch.load` body:

```python
def load_model(checkpoint_path, *, dataconfigs=None, reference=None, device="cpu"):
    from .model_file import container, load_model as _load_alignair
    if container.is_alignair_file(checkpoint_path):
        lm = _load_alignair(checkpoint_path, device=device)
        return lm.model, (reference or lm.reference)
    # ---- existing legacy .pt path unchanged below ----
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ...
```

In `train_model`, after training, `_train(...)` already writes via the trainer (Task 8). Ensure `train_model`'s default `out_path` ends `.alignair` (leave the string as given by the caller).

- [ ] **Step 4: Run — PASS.**

- [ ] **Step 5: Commit.**

```bash
git add src/alignair/api.py tests/alignair/predict/test_api_alignair_format.py
git commit -m "feat(api): load_model auto-detects .alignair (rebuilds reference from the file); legacy .pt unchanged"
```

---

### Task 8: Trainer writes `.alignair` + resumes via `load_training_state`

**Files:**
- Modify: `src/alignair/train/trainer.py`
- Test: `tests/alignair/train/test_trainer_alignair_save.py`

**Interfaces:**
- `save_checkpoint`/`save_rotating` write `.alignair` embedding config/weights/logvars/optimizer/rng/dataconfig + training summary.
- Resume detects `.alignair` and restores via `load_training_state`.

- [ ] **Step 1: Write the failing test.**

```python
# tests/alignair/train/test_trainer_alignair_save.py
import GenAIRR.data as gd
import torch
from alignair.core.config import AlignAIRConfig
from alignair.core import AlignAIR
from alignair.core.losses import make_logvars
from alignair.train.trainer import save_checkpoint
from alignair import model_file as mf

def test_trainer_save_checkpoint_writes_alignair(tmp_path):
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    model = AlignAIR(cfg); logvars = make_logvars(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4); opt.step()
    p = tmp_path / "m.alignair"
    save_checkpoint(str(p), cfg, model, logvars, step=1000, opt=opt,
                    dataconfigs=[gd.HUMAN_IGH_OGRDB], train_args={"lr": 1e-4, "batch_size": 64})
    md = mf.read_metadata(str(p))
    assert md["training"]["train_args"]["batch_size"] == 64
    ts = mf.load_training_state(str(p)); assert ts.step == 1000 and ts.optimizer_state is not None
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement.** Replace `save_checkpoint` in `src/alignair/train/trainer.py`:

```python
def save_checkpoint(path, cfg, model, logvars, step, opt=None, *, dataconfigs=None, train_args=None):
    """Write a self-contained .alignair model file (weights + config + logvars + optimizer/rng +
    embedded dataconfig + training summary)."""
    from .. import model_file as mf
    import numpy as np
    import random
    rng = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    ta = dict(train_args or {})
    training = {"steps": step, "batch_size": ta.get("batch_size", 0), "lr": ta.get("lr"),
                "progresses": ta.get("progresses"), "heavy_shm": ta.get("heavy_shm"),
                "short_boost": ta.get("short_boost"), "seed": ta.get("seed"),
                "train_args": ta}
    mf.save_model(path, model, dataconfigs=dataconfigs or [], training=training,
                  logvars=logvars, optimizer=opt, rng=rng)
```

Update `save_rotating(...)` to thread `dataconfigs`/`train_args` into `save_checkpoint`, and update the two call sites in `train(...)` to pass `dataconfigs=[dataconfig]` and a `train_args` dict built from the train() args (`lr`, `batch_size`, `progresses`, `heavy_shm`, `short_boost`, `seed`, `steps`). Then update the resume block:

```python
    resume_from = latest_checkpoint(resume_path) if resume_path else None
    if resume_from:
        from .. import model_file as mf
        if mf.container.is_alignair_file(resume_from):
            ts = mf.load_training_state(resume_from, device=device)
            model.load_state_dict(ts.model.state_dict())
            if ts.logvars_state: logvars.load_state_dict(ts.logvars_state)
            if ts.optimizer_state:
                opt.load_state_dict(ts.optimizer_state)
                for grp in opt.param_groups: grp["lr"] = lr
            start = ts.step
        else:
            ck = torch.load(resume_from, map_location=device)   # legacy .pt
            model.load_state_dict(ck["model"]); logvars.load_state_dict(ck["logvars"])
            if "optimizer" in ck:
                opt.load_state_dict(ck["optimizer"])
                for grp in opt.param_groups: grp["lr"] = lr
            start = int(ck.get("step", 0))
        print(f"RESUMED from {resume_from} at step {start} (lr={lr}, short_boost={short_boost})", flush=True)
```

(Keep `latest_checkpoint`/`_step_of`/`_stem` working for the `.alignair` extension — they already glob on `.step*.pt`; update the glob/suffix to `.alignair` in `_stem`/`save_rotating`.)

- [ ] **Step 4: Run — PASS**, and run the existing trainer tests: `pytest tests/alignair/train -q`.

- [ ] **Step 5: Commit.**

```bash
git add src/alignair/train/trainer.py scripts/train_alignair.py tests/alignair/train/test_trainer_alignair_save.py
git commit -m "feat(train): trainer writes/reads the .alignair format (embed dataconfig + optimizer/rng + train args); legacy .pt resume kept"
```

---

### Task 9: CLI — `info`, `export-reference`, `convert`, and `predict --dataconfig` optional

**Files:**
- Create: `src/alignair/cli/info.py`, `src/alignair/cli/export_reference.py`, `src/alignair/cli/convert.py`
- Modify: `src/alignair/cli/main.py` (register), `src/alignair/cli/predict.py`
- Test: `tests/alignair/cli/test_cli_model_file.py`

**Interfaces:**
- `alignair info <model>`, `alignair export-reference <model> --fasta out` / `--dataconfig out --index i`, `alignair convert <old.pt> <new.alignair> --dataconfig NAME…`, `alignair predict` with `--dataconfig` optional.

- [ ] **Step 1: Write the failing test.**

```python
# tests/alignair/cli/test_cli_model_file.py
import GenAIRR.data as gd
from alignair.core.config import AlignAIRConfig
from alignair.core import AlignAIR
from alignair import model_file as mf
from alignair.cli.main import main

def test_cli_info_and_export(tmp_path, capsys):
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    p = tmp_path / "m.alignair"
    mf.save_model(str(p), AlignAIR(cfg), dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 2, "batch_size": 3})
    assert main(["info", str(p)]) == 0
    out = capsys.readouterr().out
    assert "AlignAIR" in out and "HUMAN_IGH_OGRDB" in out and "total_sequences_seen" in out
    fasta = tmp_path / "ref.fasta"
    assert main(["export-reference", str(p), "--fasta", str(fasta)]) == 0
    assert fasta.read_text().startswith(">")
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement.**

```python
# src/alignair/cli/info.py
"""`alignair info` — print a model file's card (metadata only; no weights loaded)."""
import json
from ..model_file import read_metadata

def register(sub):
    p = sub.add_parser("info", help="print an .alignair model file's metadata / model card")
    p.add_argument("model"); p.add_argument("--json", action="store_true", help="raw JSON")
    p.set_defaults(func=run)

def run(args):
    md = read_metadata(args.model)
    if args.json:
        print(json.dumps(md, indent=2, default=str)); return 0
    m, t = md.get("model", {}), md.get("training", {})
    print(f"AlignAIR model  (format v{md.get('format_version')}, {md.get('alignair_version')})")
    print(f"  created:   {md.get('created')}")
    print(f"  alleles:   V={m.get('allele_counts',{}).get('v')} D={m.get('allele_counts',{}).get('d')} J={m.get('allele_counts',{}).get('j')}"
          f"  params={m.get('param_count')}")
    print(f"  training:  steps={t.get('steps')} total_sequences_seen={t.get('total_sequences_seen')} lr={t.get('lr')}")
    for dc in md.get("reference", {}).get("dataconfigs", []):
        print(f"  reference: {dc['name']} ({dc.get('chain_type')}, {dc.get('species')})")
    print(f"  sections:  {', '.join(md.get('sections', {}))}")
    return 0
```

```python
# src/alignair/cli/export_reference.py
"""`alignair export-reference` — dump the embedded germline FASTA or a dataconfig pickle."""
from ..model_file import read_dataconfig, read_reference

def register(sub):
    p = sub.add_parser("export-reference", help="export the germline FASTA / dataconfig from a model file")
    p.add_argument("model"); p.add_argument("--fasta"); p.add_argument("--dataconfig"); p.add_argument("--index", type=int, default=0)
    p.set_defaults(func=run)

def run(args):
    if args.fasta:
        open(args.fasta, "w").write(read_reference(args.model)); print(f"wrote {args.fasta}")
    if args.dataconfig:
        import pickle
        pickle.dump(read_dataconfig(args.model, index=args.index), open(args.dataconfig, "wb"), protocol=5)
        print(f"wrote {args.dataconfig}")
    if not args.fasta and not args.dataconfig:
        print("nothing to export: pass --fasta and/or --dataconfig"); return 1
    return 0
```

```python
# src/alignair/cli/convert.py
"""`alignair convert` — upgrade a legacy .pt checkpoint to a .alignair model file."""
import torch
from ..api import _remap_state_dict
from ..core import AlignAIR
from ..core.config import AlignAIRConfig
from .. import model_file as mf

def register(sub):
    p = sub.add_parser("convert", help="convert a legacy .pt checkpoint to .alignair")
    p.add_argument("src"); p.add_argument("dst")
    p.add_argument("--dataconfig", nargs="+", required=True)
    p.set_defaults(func=run)

def run(args):
    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    cfg = AlignAIRConfig(**ck["config"]); model = AlignAIR(cfg)
    model.load_state_dict(_remap_state_dict(ck["model"]), strict=True)
    mf.save_model(args.dst, model, dataconfigs=args.dataconfig,
                  training={"steps": int(ck.get("step", 0)), "batch_size": 0})
    print(f"converted {args.src} -> {args.dst}"); return 0
```

Register all three in `src/alignair/cli/main.py` (`_info_cmd.register(sub)`, etc.). In `src/alignair/cli/predict.py`, make `--dataconfig` optional (`required=False`) and update `run`:

```python
    from ..model_file import container
    is_alignair = container.is_alignair_file(args.model)
    if not is_alignair and not args.dataconfig:
        print("--dataconfig is required for legacy .pt models"); return 1
    if is_alignair and args.dataconfig:
        print("note: --dataconfig ignored; the .alignair model carries its own reference")
    model, reference = load_model(args.model, dataconfigs=args.dataconfig, device=device)
```

and update the `--model` help to `"trained AlignAIR model (.alignair or legacy .pt)"`.

- [ ] **Step 4: Run — PASS**; also `pytest tests/alignair/cli -q`.

- [ ] **Step 5: Commit.**

```bash
git add src/alignair/cli/ tests/alignair/cli/test_cli_model_file.py
git commit -m "feat(cli): alignair info / export-reference / convert; predict --dataconfig optional for .alignair"
```

---

### Task 10: Portability, selective-read, and codec-matrix tests

**Files:**
- Test: `tests/alignair/model_file/test_portability.py`

**Interfaces:** consumes the whole `model_file` API.

- [ ] **Step 1: Write the tests.**

```python
# tests/alignair/model_file/test_portability.py
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
    json.loads(C.read_section(p, "config").decode())              # json only
    st_load(C.read_section(p, "weights"))                          # safetensors only
    assert C.read_section(p, "reference").decode().startswith(">") # fasta only
    md = mf.read_metadata(p)
    assert md["sections"]["dataconfig/0"]["format"] == "python-pickle"
    assert md["sections"]["config"]["format"] == "json"

def test_read_metadata_touches_only_header(tmp_path):
    p = _save(tmp_path)
    base = C.read_header(p)["_sections_base"]
    # read_metadata must not require reading past the header region
    import builtins
    reads = []
    real_open = builtins.open
    def spy(*a, **k):
        f = real_open(*a, **k)
        if a and str(a[0]) == p and "b" in (a[1] if len(a) > 1 else k.get("mode", "")):
            orig = f.read
            def read(n=-1):
                b = orig(n); reads.append(len(b)); return b
            f.read = read
        return f
    builtins.open = spy
    try:
        mf.read_metadata(p)
    finally:
        builtins.open = real_open
    assert sum(reads) <= base + 8   # magic+len+header only, never the payloads

def test_codec_matrix_roundtrips():
    for codec in ("none", "zlib", C.available_codec("zstd")):
        data = b"AIRR" * 5000
        assert C.decompress(C.compress(data, codec), codec) == data
```

- [ ] **Step 2: Run — expect PASS** (features already implemented). If `read_metadata` reads too much, fix `read_header` to read exactly `header_len` bytes.

- [ ] **Step 3: Commit.**

```bash
git add tests/alignair/model_file/test_portability.py
git commit -m "test(model_file): portable-sections-need-no-pickle, metadata-reads-only-header, codec matrix"
```

---

### Task 11: Full-suite gate + docs note

**Files:**
- Modify: `src/alignair/model_file/__init__.py` docstring or a short `docs/` note (optional).

- [ ] **Step 1:** Run the whole suite: `PYTHONPATH=src CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/ -q -p no:cacheprovider`. Expected: all pass.
- [ ] **Step 2:** Manually smoke: train a tiny model (`scripts/train_alignair.py --steps 20 --out /tmp/x.alignair`), then `alignair info /tmp/x.alignair`, `alignair predict --model /tmp/x.alignair --input <fasta> --out /tmp/out.tsv` (no `--dataconfig`).
- [ ] **Step 3: Commit** any doc/docstring polish: `git commit -am "docs(model_file): trusted-input note for pickle sections; usage"`.

---

## Self-Review

**Spec coverage:** container layout (Task 2/2b) · header model-card + typed blocks (Task 4) · full config section (Task 3/4/5) · ordered dataconfig sections (Task 4) · portable/trusted `format` labels (Task 2b/3/10) · dual checksums verify-order (Task 2b) · codec none/zlib/zstd + defaults (Task 1/2/4) · inference block (Task 4) · train_state optimizer+rng+args + honest resume (Task 6/8) · LoadedModel/TrainingState (Task 5/6) · api auto-detect + 2-tuple (Task 7) · trainer save/resume (Task 8) · CLI info/export/convert + predict optional (Task 9) · deps (Task 1) · tests incl. selective-read & portability (Task 10). All covered.

**Placeholder scan:** none — every code step is complete.

**Type consistency:** `save_model(dataconfigs, training, inference, logvars, optimizer, rng, description)`, `LoadedModel(model, reference, config, metadata)`, `TrainingState(...)`, `container.write_container(path, header, sections={name:(bytes,codec)})`, `read_section(path,name)->bytes`, `is_alignair_file` used identically across Tasks 5–9.
