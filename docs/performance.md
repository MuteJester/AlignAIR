# Performance

Indicative resource numbers for AlignAIR on one machine. Speed depends on the model architecture, read length, and reference size (not on the trained weights), so these are measured with **untrained** representative models — use them for planning, not as accuracy figures (see [Benchmarks](benchmarks.md) for accuracy).

Measured on: **Intel Core i7-10700F (16 threads)** + **NVIDIA GeForce RTX 3090 Ti**.  
Workload: 400 human-IGH reads (avg 373 nt), full 198-allele V reference, batch 64.  
Reproduce: `PYTHONPATH=src .venv/bin/python scripts/perf_table.py`.

## Prediction throughput (desktop model)

| device | gapped alignment | V reader | reads/s | peak mem |
| --- | --- | --- | --- | --- |
| CPU | yes | learned | 36 | 0.80 GB |
| CPU | no | learned | 36 | 0.80 GB |
| CPU | no | parasail | 44 | 0.80 GB |
| CUDA | yes | learned | 191 | 0.81 GB |
| CUDA | no | learned | 211 | 0.81 GB |
| CUDA | no | parasail | 216 | 0.81 GB |

`--no-full-alignment` skips the parasail gapped alignment (exact cigars / germline_alignment / identity) for the faster coordinate approximation; `--v-reader parasail` swaps the learned V reader for the classical one. The two together are the fastest configuration.

## Training step time

| preset | device | s / step | est. wall (preset steps) | peak mem |
| --- | --- | --- | --- | --- |
| desktop | CPU | 1.06 | ~53m11s | 2.17 GB |
| desktop | CUDA | 1.06 | ~53m02s | 1.45 GB |
| standard | CUDA | 1.15 | ~2h32m | 3.20 GB |

For the small `desktop` model, each step is **data-generation bound** (the GenAIRR simulation + target building dominate), so CPU and GPU step times are close; the GPU pulls ahead on the larger `standard` model and bigger batches. The `standard` preset on CPU is impractically slow and is omitted. Add calibration time (a few minutes) unless `--no-calibrate`. Estimate your own run first with `alignair train --plan`.
