# Speed & throughput

Indicative resource numbers for AlignAIR on one machine. Speed depends on the model architecture, read
length, and reference size (not on the trained weights), so treat these as planning figures, not
accuracy figures (see [Benchmarks](benchmarks.md) for accuracy).

Measured on: **Intel Core i7-10700F (16 threads)** + **NVIDIA GeForce RTX 3090 Ti**.
Workload: 400 human-IGH reads (avg 373 nt), full 198-allele V reference, batch 64.

## Prediction throughput

| device | output (`--columns`) | reads/s | peak mem |
| --- | --- | --- | --- |
| CPU | `full` (with gapped alignment) | ~36 | ~0.8 GB |
| CPU | `core` (no gapped alignment) | ~44 | ~0.8 GB |
| CUDA | `full` (with gapped alignment) | ~190 | ~0.8 GB |
| CUDA | `core` (no gapped alignment) | ~215 | ~0.8 GB |

The main throughput knobs:

- **`--device cuda`** — the largest lever; the neural stage is GPU-friendly.
- **`--columns core` / `minimal`** — skip the gapped-alignment assembly (`sequence_alignment` /
  `germline_alignment` / exact CIGARs / identity) when you only need calls + coordinates. The default
  `full` produces every field.
- **`--batch-size`** — larger batches improve GPU utilisation up to memory limits.
- **`--chunk-size`** — controls streaming granularity for large inputs; it bounds memory, not speed.

Memory stays flat (~0.8 GB here) regardless of input size, because prediction streams the input in
chunks rather than loading it all at once.

## Training step time

| preset | device | s / step | est. wall (preset steps) | peak mem |
| --- | --- | --- | --- | --- |
| desktop | CPU | ~1.1 | ~50 min | ~2.2 GB |
| desktop | CUDA | ~1.1 | ~50 min | ~1.5 GB |
| full | CUDA | ~1.2 | a few hours | ~3.2 GB |

For the small `desktop` model each step is **data-generation bound** (the GenAIRR simulation + target
building dominate), so CPU and GPU step times are close; the GPU pulls ahead on the larger `full`
preset and bigger batches. Estimate a specific run before committing GPU time with:

```bash
alignair train --dataconfig HUMAN_IGH_OGRDB --out my_model --preset desktop --plan
```
