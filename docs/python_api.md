# Python API

AlignAIR exposes a small, stable Python API. The CLI is a thin client of these
same functions, so anything you can do on the command line you can do from a
notebook or a pipeline — and you get the same results.

```python
from alignair import Aligner, read_sequences

# a .alignair model file, a catalog id, or an org/name Hugging Face repo id
aligner = Aligner.from_pretrained("runs/my_model/bundle/model.alignair")
ids, reads, _ = read_sequences("reads.fastq")      # (ids, sequences, info)
result = aligner.predict(reads)                    # uses the model's embedded reference
result.ids = ids                                   # keep the FASTA ids (else seq0, seq1, …)
result.write_airr("out.tsv")                       # AIRR-C rearrangement TSV
```

`Aligner` is the high-level entry point. Everything under `alignair.core`,
`alignair.nn`, `alignair.io`, `alignair.predict`, … is implementation detail and
may change without notice — import only from the top-level `alignair` namespace.

## Public surface

| Name | What it is |
|------|------------|
| `Aligner.from_pretrained(model, *, device="auto", revision=None, offline=False, token=None) -> Aligner` | Load a model from a `.alignair` file, a catalog id, or an `org/name` Hugging Face repo id (auto-downloaded). |
| `Aligner.predict(sequences, *, batch_size="auto", genotype=None, genotype_method="mask", airr=True) -> PredictionResult` | Align a list of nucleotide strings. |
| `Aligner.predict_iter(sequences, *, chunk_size=20000, …)` | Streaming generator over an iterable of sequences (bounded memory). |
| `load_model(path, *, dataconfigs=None, reference=None, device="cpu", trust_pickle=False) -> (AlignAIR, ReferenceSet)` | Lower-level load: returns the raw model + reference tuple. |
| `predict_sequences(model, reference, sequences, *, device=None, batch_size=64, **overrides) -> list[dict]` | Lower-level functional predict (what `Aligner.predict` calls). |
| `train_model(dataconfigs, *, out_path, steps=100000, device="cpu", **overrides) -> str` | Train a model; returns the path to the written `model.alignair`. |
| `read_sequences(path) -> (ids, sequences, info)` | Eager reader for FASTA/FASTQ/CSV/TSV (`-` for stdin). |
| `iter_sequences(path, chunk_size=20000)` | Streaming generator for bounded-memory runs over large files. |
| `write_airr(path, ids, sequences, predictions, locus="IGH", columns=None)` / `AirrWriter` | Write an AIRR-C TSV (one-shot or incremental). |
| `compare_airr(a, b)` | Per-gene agreement report between two AIRR TSVs. |
| `ReferenceSet` | A germline reference (embedded in the model file) — `from_dataconfigs`, `from_fasta`, `from_yaml`, `subset`. |
| `PredictionResult` | Typed result object (see below). |

## Donor genotype constraint

Each model is tied to the germline reference it was trained on (embedded in the model file). At
inference you can constrain calls to a **donor genotype — a subset of that reference** — with no
retraining. Alleles the model was not trained on are not callable (train a new model to add them).

Pass a `{gene: set(allele_names)}` dict, or load one from a YAML/FASTA file against the model's
reference:

```python
from alignair.genotype.constraint import load_genotype

genotype = load_genotype("donor.yaml", reference=aligner.reference)   # YAML or FASTA path -> dict
result = aligner.predict(reads, genotype=genotype)

# or pass the dict directly
result = aligner.predict(reads, genotype={"V": {"IGHV1-2*02", "IGHV3-23*01"}})
```

Every call in the result is then guaranteed to be within that genotype. `genotype_method` selects how
the constraint is applied (`mask` default / `renormalize` / `redistribute`).

## Result objects

`aligner.predict(...)` returns a `PredictionResult` — iterable and `len()`-able over the per-read
AIRR-convention records:

```python
result = aligner.predict(reads)
len(result)              # number of reads
result.locus             # e.g. "IGH"
records = result.to_dicts()          # list of AIRR-ready dicts (v/d/j calls, coords, junction, …)
result.ids = ids                     # optionally attach the FASTA ids before writing
result.write_airr("out.tsv", columns="airr")   # columns: a preset or explicit field list
```

The coordinates in each record refer to that record's canonical `sequence` (reverse-complemented
inputs are reoriented, and `rev_comp` follows the AIRR convention).

`load_model(path)` is the lower-level door: it returns a raw `(model, reference)` tuple you can feed
to `predict_sequences(model, reference, reads)`.
