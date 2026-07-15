# Design & internals

AlignAIR is a single neural model that takes an immunoglobulin / TCR **DNA** read and produces the full
V(D)J alignment — allele calls, segment coordinates, orientation, and per-read quality signals — which a
light post-processing stage turns into a schema-valid AIRR record. This page describes how it works.

## End-to-end pipeline

```
read (nucleotides)
  │  tokenize (A/C/G/T/N + pad, fixed window)
  ▼
in-model orientation head ──► detect the read's orientation, re-orient to the forward frame
  ▼
convolutional feature encoder (residual conv tower, TF-faithful)
  ├─► per-gene branches (V, D, J):
  │      ├─ segmentation heads ─► start / end position of the segment
  │      └─ classification head ─► per-allele scores (over the embedded catalog)
  └─► meta heads ─► mutation rate · indel count · productivity · locus (multi-chain)
  ▼
post-processing (alignair.predict)
  ├─ allele selection ─► top call + an equivalence set of indistinguishable alleles
  ├─ germline alignment ─► exact CIGAR, coordinates, and % identity (parasail)
  └─ AIRR assembly ─► junction / CDR3, np1/np2, gapped alignments, productivity flags
  ▼
AIRR rearrangement record
```

The neural network localises each segment and scores alleles; a classical **gapped alignment** against
the called germline (via `parasail`) then produces the exact CIGAR, coordinates, and identity, and the
AIRR assembler derives the junction and the remaining fields. The germline reference is **embedded in
the model file**, so calls are always drawn from a known, fingerprinted catalog (see the
[model contract](model_contract.md)).

## Key design choices

- **In-model orientation.** The read's orientation (forward / reverse-complement / complement / reverse)
  is predicted from the initial embeddings and corrected inside the model, so every downstream head and
  the coordinates operate on one canonical frame. This is why AlignAIR handles arbitrary-orientation
  reads that classical seed-and-extend tools miss.
- **Segmentation + classification, jointly.** Each gene has heads that regress the segment's boundaries
  *and* score alleles, from a shared convolutional representation — one forward pass yields calls,
  coordinates, and quality signals together.
- **Honest ambiguity.** Allele selection emits an equivalence set (`*_call_set`), not a forced single
  call, when a read genuinely cannot distinguish alleles (e.g. short fragments).
- **Fixed-reference classifier.** The classification heads are tied to the embedded catalog; a donor
  genotype can *subset* that catalog at inference, but adding alleles requires training. See the
  [model contract](model_contract.md).

## Results

On a frozen 4,400-case benchmark AlignAIR beats IgBLAST on 23 of 24 metrics — largest on short
fragments, arbitrary orientation, and D/J calling. Full methodology and per-stratum numbers are on the
[Benchmarks](benchmarks.md) page.

## Module map

| Concern | Location |
| --- | --- |
| Model assembly + forward | `alignair/core/model.py` |
| Config (heads, conv schedule, allele sizes) | `alignair/core/config.py` |
| Per-gene branch (segmentation + classification) | `alignair/core/gene_branch.py` |
| Orientation / state heads | `alignair/nn/heads/` |
| Prediction pipeline (orchestrator) | `alignair/predict/pipeline.py` |
| Allele selection (equivalence set) | `alignair/predict/threshold.py` |
| Germline alignment / reader | `alignair/predict/germline.py` |
| AIRR assembly (junction, regions, alignments) | `alignair/predict/airr/` |
| Training (curriculum, GenAIRR simulator) | `alignair/train/` |
| Self-contained `.alignair` model format | `alignair/model_file/` |

## Training

Models train on data streamed from the [GenAIRR](https://github.com/MuteJester/GenAIRR) simulator — no
static dataset — over a curriculum that mixes clean reads with fragments, SHM, indels, and arbitrary
orientation. A run embeds the germline reference into the resulting `.alignair` file, so the model is
self-contained. See [Train your own](getting_started.md#2-get-a-model) and the
[CLI reference](cli.md#alignair-train).
