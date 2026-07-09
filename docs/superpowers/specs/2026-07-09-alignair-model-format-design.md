# AlignAIR self-contained model format (`.alignair`) — Design

**Status:** approved design (2026-07-09), ready for implementation plan.

## Goal

Replace the bare `torch.save` checkpoint with a **self-contained, inspectable, compact binary
model file** that carries everything needed to run, resume, or understand an AlignAIR model:

- the model **weights**,
- **full training metadata** (steps, batch size, total sequences seen, all hyperparameters,
  curriculum, versions, provenance),
- the **GenAIRR `DataConfig`(s)** so the germline reference and alleles travel *inside* the model
  (no external `dataconfigs=` needed to load or inspect it),
- **AIRR-community-useful metadata** (a readable model card + a derived germline FASTA) to ease
  adoption and interoperation.

The file must be **compressed on save** and **selectively decompressed on read** — a reader takes
only the sections it needs (just the metadata, just the model for inference, the model+optimizer for
training, or just the dataconfig), never paying to decompress the rest.

## Non-goals

- No C/C++ extension. The reader/writer is **pure Python** over a **documented, language-agnostic
  binary layout** (so a C/C++/Rust/JS reader can be added later with no file change). Reading
  metadata is a `seek`+small-read — there is no compute to accelerate.
- Not a general model zoo / registry. One file = one model.

## Why a custom container (vs alternatives)

- **safetensors** is tensor-only; the 6 MB pickled `DataConfig` and the reference FASTA are not
  tensors, and stuffing them into safetensors' string metadata bloats the always-read header,
  defeating "metadata without reading the file." We *use safetensors for the `weights` section*
  (standard, safe, multi-language) but wrap it in our container.
- **HDF5** meets the seek/partial-read goal but is a heavy dependency and overkill here.
- **zip/npz** has a central directory but no first-class "header model card"; less clean for a
  seek-to-metadata peek and per-section codecs.

A tiny custom container (magic + versioned JSON header + indexed, independently-compressed sections)
gives exactly the required properties with ~one file of Python.

## On-disk layout

```
offset 0   magic          8 bytes    b"ALGNAIR\x01"   (7-char tag + 1-byte format version)
offset 8   header_len     u64 LE     length in bytes of the JSON header
offset 16  header         header_len bytes  UTF-8 JSON: the model card + section index
offset ..  sections       concatenated payload blobs (order matches the index; opaque bytes)
EOF
```

- **Read metadata:** open → read 16 bytes → read `header_len` → `json.loads`. Payloads untouched.
- **Read a section:** from the index get `{offset, length, codec, sha256}` → `seek(offset)` →
  `read(length)` → decompress(codec) → verify sha256 → deserialize.
- Magic's last byte is the format version; a reader rejects unknown major versions with a clear
  error. Legacy `.pt` files (no magic) are detected and loaded via `torch.load` (back-compat).

### Header (JSON model card + index)

```jsonc
{
  "format_version": 1,
  "alignair_version": "0.x", "genairr_version": "2.2.0", "torch_version": "2.x",
  "created": "2026-07-09T12:34:56Z",
  "description": "human IGH, GenAIRR end-loss curriculum",   // free-text model card
  "license": "GPL-3.0-or-later", "citation": "AlignAIR ...",
  "model": {
    "embed_dim": 512, "max_seq_length": 576, "num_chain_types": 1, "has_d": true,
    "param_count": 14312345,
    "allele_counts": {"v": 198, "d": 33, "j": 7},
    "chain_types": ["IGH"]
  },
  "training": {
    "steps": 500000, "batch_size": 64, "total_sequences_seen": 32000000,
    "lr": 0.0001, "progresses": [0.3, 0.6, 0.9], "heavy_shm": 0.25, "short_boost": 1,
    "seed": 0, "curriculum": "genairr end-loss amplicon mix",
    "final_losses": {"segmentation": 6.0, "...": 0.0}, "resumed_from": "step435000"
  },
  "reference": {
    "chains": [{"dataconfig_name": "HUMAN_IGH_OGRDB", "schema_sha256": "abc...",
                "allele_counts": {"v": 198, "d": 33, "j": 7}}]
  },
  "provenance": {"git_commit": "…", "host": "…", "user": "…"},
  "sections": {
    "weights":          {"offset": 0,        "length": 40000000, "codec": "zlib", "sha256": "…"},
    "logvars":          {"offset": 40000000, "length": 2048,     "codec": "zlib", "sha256": "…"},
    "optimizer":        {"offset": 40002048, "length": 80000000, "codec": "zlib", "sha256": "…"},
    "dataconfig/IGH":   {"offset": ...,      "length": 6300000,  "codec": "zlib", "sha256": "…"},
    "reference":        {"offset": ...,      "length": 120000,   "codec": "zlib", "sha256": "…"}
  }
}
```

`offset` in the index is **relative to the start of the sections region** (i.e., after the header),
so the header can be written last without patching absolute offsets.

### Sections (payloads)

| name | content | serialization | notes |
|---|---|---|---|
| `weights` | model `state_dict` | **safetensors** bytes | standard, safe, cross-language |
| `logvars` | Kendall log-var `state_dict` | safetensors bytes | small |
| `optimizer` | AdamW `state_dict` | `torch.save` bytes | optional; present ⇒ resumable |
| `dataconfig/<chain>` | full GenAIRR `DataConfig` | `pickle` (protocol 5) | ~6 MB each; reference+alleles+sim params travel with the model |
| `reference` | derived V/D/J germline | **FASTA** text | for IgBLAST / other AIRR tools; regenerated from the dataconfig |

Each section is compressed independently at save (`zlib` from stdlib by default; `zstd` used if
`zstandard` is importable, recorded in `codec`). Decompression is lazy — only for read sections.

## Public API

New module `alignair/model_file.py` (container read/write) + a thin `alignair/save.py` façade.

```python
# save (used by the trainer and by api.save_model)
save_model(path, model, *, logvars=None, optimizer=None, dataconfigs, training: dict,
           description="", codec="zlib") -> None

# read — each reads ONLY the needed sections
read_metadata(path) -> dict                      # header only; no torch needed
load_model(path, *, device="cpu", for_training=False) -> (model, reference[, train_state])
read_dataconfig(path) -> DataConfig | list        # dataconfig section only
read_reference(path) -> str                       # the germline FASTA
load_all(path) -> dict                            # everything
```

- `load_model(path)` (default, inference): reads `weights` + `dataconfig` → builds `AlignAIR` +
  `ReferenceSet`; **skips** optimizer/logvars. Rebuilds the reference from the embedded dataconfig,
  so the caller passes **no** `dataconfigs=`.
- `load_model(path, for_training=True)`: also reads `logvars` + `optimizer` + `training.step`;
  returns the extra `train_state` for exact resume.
- Back-compat: `alignair.api.load_model` auto-detects — magic ⇒ new reader; otherwise the existing
  `torch.load` path (still accepts `dataconfigs=`/`reference=` for legacy `.pt`).

## CLI

- `alignair info <model.alignair>` — pretty-print the model card (versions, allele counts,
  training summary, provenance) using `read_metadata` (instant, no weights loaded).
- `alignair export-reference <model.alignair> --fasta out.fasta` — dump the germline FASTA
  (via `read_reference`), or `--dataconfig out.pkl` to dump the pickled `DataConfig`.

## Integration & migration

- `train/trainer.save_rotating`/`save_checkpoint` write `.alignair`, capturing the training dict
  live (steps, batch_size, lr, progresses, short_boost, seed, total_sequences, versions, git).
  Resume reads with `for_training=True`.
- `api.save_model`/`train_model` produce `.alignair`; `api.load_model` reads it.
- Existing `.pt` checkpoints remain loadable (auto-detect) — no forced migration; an optional
  `alignair convert <old.pt> <new.alignair> --dataconfig …` helper can upgrade them.

## Testing

- **Round-trip:** save → `read_metadata` matches the input card; `load_model` reproduces
  predictions bit-for-bit; `for_training=True` resumes (optimizer state restored).
- **Selective read:** `read_metadata` opens without importing torch and reads < header_len+16 bytes
  (assert via a byte-counting file wrapper); `read_dataconfig` does not touch the `weights` bytes.
- **Integrity:** corrupting a section's bytes fails its sha256 check with a clear error; unknown
  magic/version raises a clear error; legacy `.pt` still loads.
- **Interop:** the exported `reference` FASTA parses and matches the dataconfig's allele set.
- **Compactness:** compressed file is meaningfully smaller than the naive `torch.save`.

## Error handling

- Wrong magic → try legacy `torch.load`; if that also fails, a clear "not an AlignAIR model" error.
- Unknown format major version → explicit "written by a newer AlignAIR" error.
- Section sha256 mismatch → "corrupt/modified model file" error naming the section.
- Missing `optimizer` section with `for_training=True` → clear "no optimizer state; cannot resume".
- GenAIRR `DataConfig` schema-version mismatch on unpickle → surface GenAIRR's own error verbatim.
```
