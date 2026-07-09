# AlignAIR self-contained model format (`.alignair`) — Design

**Status:** approved design (2026-07-09, revised after review), ready for implementation plan.

## Goal

Replace the bare `torch.save` checkpoint with a **self-contained, inspectable, compact binary
model file** that carries everything needed to run, resume, or understand an AlignAIR model:

- the model **weights** and the **full model config** needed to reconstruct the network,
- **full training metadata** (steps, batch size, total sequences seen, every hyperparameter,
  curriculum, versions, provenance) plus the **binary training state** for resume,
- the **GenAIRR `DataConfig`(s)** so the germline reference and alleles travel *inside* the model,
- **inference defaults** (thresholds, reader, chain order) so the model "just runs",
- **AIRR-community-useful metadata** (a readable model card + a derived germline FASTA).

The file is **compressed per section on save** and **selectively decompressed on read** — a reader
takes only the sections it needs (metadata only, model-for-inference, model-for-training, or just
the dataconfig), never paying to decompress the rest.

## Non-goals

- No C/C++ extension. The **container + reader/writer are pure Python** over a **documented layout**.
- Not a general model zoo / registry. One file = one model.

## The four contracts (revision drivers)

1. **Full config, not a summary.** Persist the entire `AlignAIRConfig` dataclass so `AlignAIR` can be
   rebuilt without guessing (`vocab_size`, `embed_dim`, `filters`, `block_out`, latent sizes,
   `has_d`, `num_chain_types`, counts, future fields). The header's `model` block is *derived* display
   metadata only.
2. **Ordered dataconfig sections.** Sections are `dataconfig/0`, `dataconfig/1`, … in
   `AlignAIRConfig.from_dataconfigs` order (that order defines chain-type indices), each with metadata
   `{index, name, chain_type, species, source, schema_sha256, allele_counts}`. Never key by locus.
3. **Portable vs trusted sections, stated explicitly.** JSON/safetensors/FASTA sections are
   language-agnostic and safe. `dataconfig`/`train_state` are **Python pickle** — trusted-input only,
   not readable from Rust/JS. The header marks each section's `format`.
4. **Honest resume semantics.** "Resume" restores weights + logvars + optimizer + step **+ RNG states
   (python/numpy/torch/cuda) + the full train-arg set**. Data-stream position is reconstructed from
   `(seed, step)` for the seeded curriculum streams; per-batch augmentation RNG is covered by the
   stored torch RNG state. We call this **resumable**, and document that bit-exactness holds only on
   the same library/hardware stack.

## On-disk layout

```
offset 0   magic          8 bytes    b"ALGNAIR\x01"   (7-char tag + 1-byte MAJOR format version)
offset 8   header_len     u64 LE     length in bytes of the JSON header
offset 16  header         header_len bytes  UTF-8 JSON: model card + section index
offset ..  sections       concatenated payload blobs (opaque bytes; described by the index)
EOF
```

- **Read metadata:** open → read 16 bytes → read `header_len` → `json.loads`. Payloads untouched.
- **Read a section:** index gives `{offset, compressed_length, payload_length, codec,
  compressed_sha256, payload_sha256, format}`. `seek(offset)` → read `compressed_length` →
  **verify `compressed_sha256`** → decompress(codec) → **verify `payload_sha256`** → deserialize.
- Offsets are relative to the start of the sections region (header can be written last).
- Magic's last byte is the MAJOR version; a reader rejects unknown majors with a clear error.
  Legacy `.pt` files (no magic) are detected and loaded via `torch.load`.

### Header (JSON) — readable model card + typed blocks + section index

```jsonc
{
  "format_version": 1, "model_class": "AlignAIR", "config_schema_version": 1,
  "alignair_version": "0.x", "genairr_version": "2.2.0", "torch_version": "2.x",
  "created": "2026-07-09T12:34:56Z",
  "description": "human IGH, GenAIRR end-loss curriculum",
  "license": "GPL-3.0-or-later", "citation": "AlignAIR …",

  "model": {                          // DERIVED summary for display/discovery (source of truth = the `config` section)
    "embed_dim": 512, "max_seq_length": 576, "num_chain_types": 1, "has_d": true,
    "param_count": 14312345, "allele_counts": {"v": 198, "d": 33, "j": 7}
  },
  "inference": {                      // PredictConfig defaults so the model "just runs"
    "threshold": 0.5, "selector": "absolute", "cap": 3,
    "germline_reader": "heuristic", "pad_mode": "right", "airr": true,
    "chain_types": ["IGH"]
  },
  "training": {                       // human-readable summary (binary resume state lives in the `train_state` section)
    "steps": 500000, "batch_size": 64, "total_sequences_seen": 32000000,
    "lr": 0.0001, "progresses": [0.3, 0.6, 0.9], "heavy_shm": 0.25, "short_boost": 1,
    "seed": 0, "curriculum": "genairr end-loss amplicon mix",
    "final_losses": {"segmentation": 6.0}, "resumed_from": "step435000",
    "train_args": { /* full arg dict actually used */ }
  },
  "reference": {
    "dataconfigs": [
      {"index": 0, "section": "dataconfig/0", "name": "HUMAN_IGH_OGRDB",
       "chain_type": "BCR_HEAVY", "species": "human", "source": "OGRDB",
       "schema_sha256": "…", "allele_counts": {"v": 198, "d": 33, "j": 7}}
    ]
  },
  "provenance": {"git_commit": "…", "host": "…", "user": "…"},

  "sections": {
    "config":        {"offset": …, "compressed_length": …, "payload_length": …, "codec": "zlib",
                      "compressed_sha256": "…", "payload_sha256": "…", "format": "json"},
    "weights":       {"…": "…", "codec": "none",   "format": "safetensors"},
    "logvars":       {"…": "…", "codec": "none",   "format": "safetensors"},
    "train_state":   {"…": "…", "codec": "zstd",   "format": "python-pickle"},
    "dataconfig/0":  {"…": "…", "codec": "zstd",   "format": "python-pickle"},
    "reference":     {"…": "…", "codec": "zlib",   "format": "fasta"}
  }
}
```

### Sections (payloads)

| name | content | serialization | `format` | portable? |
|---|---|---|---|---|
| `config` | full `AlignAIRConfig.__dict__` | JSON | `json` | ✅ yes |
| `weights` | model `state_dict` | **safetensors** | `safetensors` | ✅ yes |
| `logvars` | Kendall log-var `state_dict` | safetensors | `safetensors` | ✅ yes |
| `train_state` | `{optimizer, rng:{python,numpy,torch,cuda}, step, train_args}` | `torch.save` | `python-pickle` | ⚠️ trusted, Python-only |
| `dataconfig/<i>` | full GenAIRR `DataConfig` (ordered) | `pickle` (proto 5) | `python-pickle` | ⚠️ trusted, Python-only + GenAIRR |
| `reference` | derived V/D/J germline | **FASTA** text | `fasta` | ✅ yes |

**Codecs are per section:** `"none"` (raw — keeps safetensors mmap/zero-copy for `weights`),
`"zlib"` (stdlib default), or `"zstd"` (only if `zstandard` is importable). Float weights compress
weakly, so `weights` defaults to `none`; pickled `dataconfig`/`train_state` compress well
(`zstd`→`zlib` fallback). **The plan will benchmark `none`/`zlib`/`zstd` per section type** and set
the shipped defaults from measured size/speed.

## Dependencies

- `safetensors` becomes a **core dependency** (weights + logvars serialization for the default save
  format). `zstandard` stays **optional** (`codec:"zstd"` used only when installed; `zlib` otherwise).
  Both added to `pyproject.toml` by the plan.

## Public API

New module `alignair/model_file.py` (container read/write) + a `LoadedModel` result type; wired into
`alignair/api.py`.

```python
@dataclass
class LoadedModel:            # what load_model returns (stable, extensible — no variable tuples)
    model: AlignAIR
    reference: ReferenceSet
    config: AlignAIRConfig
    metadata: dict            # the header

@dataclass
class TrainingState:
    model: AlignAIR; reference: ReferenceSet; config: AlignAIRConfig
    logvars_state: dict; optimizer_state: dict | None
    step: int; rng: dict; train_args: dict; metadata: dict

save_model(path, model, *, config, dataconfigs, training: dict, inference: dict | None = None,
           logvars=None, optimizer=None, rng=None, description="", codec="auto") -> None

read_metadata(path) -> dict                        # header only; no torch/safetensors load
load_model(path, *, device="cpu") -> LoadedModel   # config+weights+dataconfig(+inference); skips train_state
load_training_state(path, *, device="cpu") -> TrainingState   # + logvars+optimizer+rng+step
read_dataconfig(path, index=None) -> DataConfig | list[DataConfig]
read_reference(path) -> str                        # the germline FASTA
```

**`alignair/api.py` back-compat:** the existing `load_model(path, *, dataconfigs=None,
reference=None, device) -> (model, reference)` keeps its **exact 2-tuple shape**. Internally it
auto-detects: magic ⇒ read the container and return `(lm.model, lm.reference)` (ignoring
`dataconfigs=`, or erroring if the caller passes one that disagrees with the embedded config);
otherwise the current `torch.load` path (still honoring `dataconfigs=`/`reference=` for legacy `.pt`).
`train_model` writes `.alignair`. The richer `LoadedModel`/`TrainingState` live on `model_file`.

## CLI

- `alignair info <model.alignair>` — pretty-print the card via `read_metadata` (instant; no weights).
- `alignair export-reference <model> --fasta out.fasta` (or `--dataconfig out.pkl --index 0`).
- `alignair predict`: `--dataconfig` becomes **optional** — supplied by a `.alignair` file, still
  **required for legacy `.pt`**; if given alongside a `.alignair` whose embedded config disagrees,
  error. Help text updated to accept `.alignair`/`.pt`.
- `alignair convert <old.pt> <new.alignair> --dataconfig …` — optional upgrade of legacy checkpoints.

## Integration & migration

- `train/trainer` writes `.alignair`, capturing the `training` summary + `train_state` (optimizer,
  RNG states, step, full train args) live. Resume reads with `load_training_state`.
- Existing `.pt` checkpoints remain loadable via auto-detect — no forced migration.

## Testing

- **Round-trip:** save → `read_metadata` matches the input card; `load_model` rebuilds `AlignAIR`
  from the `config` section and reproduces predictions bit-for-bit; `load_training_state` restores
  optimizer + RNG + step and a resumed step is deterministic on the same stack.
- **Config completeness:** a model saved and reloaded with **no external config** matches the
  original `AlignAIRConfig` field-for-field.
- **Ordered dataconfigs:** a 2-chain save preserves order; `chain_types`/chain-type indices match.
- **Selective read:** `read_metadata` reads `< 16 + header_len` bytes and imports neither torch nor
  safetensors (asserted via a byte-counting file wrapper + import guard); `read_dataconfig` never
  reads the `weights` bytes; inference `load_model` never reads `train_state`.
- **Integrity:** flipping a byte in a section fails `compressed_sha256`; a bad codec/truncation fails
  `payload_sha256`; unknown magic/major version → clear error; legacy `.pt` still loads.
- **Portability:** `config`/`reference`/`weights` sections parse with only JSON/FASTA/safetensors
  (no pickle); `format` labels are correct.
- **Codec matrix:** `none`/`zlib`/`zstd` all round-trip for each section; sizes recorded to pick
  defaults.
- **Interop:** exported `reference` FASTA parses and matches the dataconfig allele set.

## Error handling

- Wrong magic → try legacy `torch.load`; if that fails too, clear "not an AlignAIR model" error.
- Unknown MAJOR version → "written by a newer AlignAIR" error.
- `compressed_sha256`/`payload_sha256` mismatch → "corrupt/modified model file", naming the section.
- `load_training_state` with no `train_state` section → "no training state; cannot resume".
- Loading a `python-pickle` section is gated behind a "trusted input" note in the docs (pickle
  executes code); portable sections never require it.
- GenAIRR `DataConfig` schema mismatch on unpickle → surface GenAIRR's own error verbatim.
