# Python API

AlignAIR exposes a small, stable Python API. The CLI is a thin client of these
same functions, so anything you can do on the command line you can do from a
notebook or a pipeline — and you get the same results.

```python
from alignair import load_model, predict, read_sequences

model = load_model("runs/my_model/bundle")        # bundle dir, .pt, catalog id, or HF repo id
ids, reads, _ = read_sequences("reads.fastq")
result = predict(model, reads)                     # uses the bundle's embedded reference
result.to_airr("out.tsv", ids)                     # AIRR-C rearrangement TSV
```

Everything else (`alignair.inference`, `alignair.io`, `alignair.serialization`,
`alignair.core`, `alignair.nn`, …) is implementation detail and may change
without notice. Import only from the top-level `alignair` namespace.

## Public surface

| Name | What it is |
|------|------------|
| `load_model(spec, device=None) -> LoadedModel` | Load a model from a bundle directory, a raw `.pt` checkpoint, a catalog id, or an `org/name` Hugging Face repo id (auto-downloaded). |
| `predict(model, reads, *, reference=None, genotype=None, …) -> PredictionBatch` | Align a list of nucleotide strings. |
| `read_sequences(path) -> (ids, sequences, metadata)` | Eager reader for FASTA/FASTQ/CSV/TSV (`-` for stdin). |
| `iter_sequences(path, chunk_size=20000)` | Streaming generator for bounded-memory runs over large files. |
| `write_airr(path, ids, sequences, predictions, locus)` / `AirrWriter` | Write an AIRR-C TSV (one-shot or incremental). |
| `compare_airr(a, b)` | Per-gene agreement report between two AIRR TSVs. |
| `ReferenceSet` | A runtime (dynamic) germline reference — `from_fasta`, `from_yaml`, `from_dataconfigs`, `from_genotype`, `subset`. |
| `LoadedModel`, `PredictionBatch` | Typed result objects (see below). |

## Dynamic reference

The reference is a **runtime input**, not baked into the weights. Supply a donor
genotype (a subset of the trained alleles, or one with novel alleles) and the
model uses it directly:

```python
result = predict(model, reads, genotype="donor.yaml")   # YAML or FASTA path,
result = predict(model, reads, genotype={"V": {...}})    # a dict,
from alignair import ReferenceSet                          # or a ReferenceSet
result = predict(model, reads, reference=ReferenceSet.from_fasta("donor.fasta"))
```

Reference precedence: explicit `reference=` > `genotype=` > the model's
embedded/default reference.

## Result objects

`load_model` returns a `LoadedModel` (`.model`, `.reference_set`, `.locus`,
`.calibration`, `.device`). `predict` returns a `PredictionBatch`:

```python
result = predict(model, reads)
len(result)              # number of reads
result.sequences         # canonical (forward) sequences the coordinates refer to
result.predictions       # list of AIRR-ready dicts (v/d/j calls, coords, junction, …)
result.locus             # e.g. "IGH"
result.to_airr("out.tsv", ids)
```

`predict` reorients reverse-complemented inputs, so `result.sequences` are the
sequences that the coordinates in `result.predictions` actually index into.
