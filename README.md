<p align="center">
  <img src="https://alignair.ai/_next/static/media/logo_alignair11bw.17e5d8d6.svg" width="220" alt="AlignAIR logo" />
</p>

<h1 align="center">AlignAIR</h1>
<p align="center">
  A neural aligner for immunoglobulin (IG) and T-cell receptor (TCR) repertoires. A single end-to-end
  model predicts V/D/J allele calls, segment coordinates, and the junction, and writes standard AIRR
  output. The germline reference is an input, not weights: restrict calls to a donor genotype, or bind
  another reference with novel alleles, without retraining. Train a model for a new locus or species.<br>
  <a href="https://hub.docker.com/r/alignair/alignair"><img alt="Docker Hub: alignair/alignair" src="https://img.shields.io/badge/docker-alignair%2Falignair-2496ED?logo=docker&logoColor=white"></a>
  <a href="https://doi.org/10.1093/nar/gkaf651"><img src="https://img.shields.io/badge/DOI-10.1093%2Fnar%2Fgkaf651-blue" alt="DOI"></a>
  <a href="LICENSE"><img alt="GPLv3" src="https://img.shields.io/badge/license-GPLv3-blue.svg"></a>
  <a href="https://mutejester.github.io/AlignAIR/"><img alt="Documentation" src="https://img.shields.io/badge/docs-guides%20%26%20lessons-6d5cf5"></a>
</p>

<p align="center">
  <b>📖 New to AlignAIR? Start on the docs site — reference, guides, and interactive lessons:<br>
  <a href="https://mutejester.github.io/AlignAIR/">mutejester.github.io/AlignAIR</a></b>
</p>

---

## Overview

- **End-to-end neural model.** One network corrects the read's orientation, segments it into V, D and J, finds the
  junction anchors, and scores every allele of the bound germline reference, in a single forward pass. Deterministic
  steps then refine the calls and boundaries against the germline and assemble a standard AIRR record (junction,
  productivity, gapped alignments).
- **Self-contained models.** Each model file embeds a fingerprinted germline reference and loads without executing any
  pickle. Load one from a local path, a Hugging Face repo (`hf://org/repo`), or a catalog id. AlignAIR 4 pretrained
  models come one per locus: IGH first, then IGK, IGL, TRA, TRB, TRD and TRG.
- **Reference without retraining.** The model scores the alleles of whichever reference it is bound to. At predict time
  you can restrict calls to a donor's genotype, or bind another germline set: novel alleles, new genes, a curated
  reference. A reference missing one of the model's segments is refused; for a new locus or species, train a model.
- **Donor genotyping.** `--infer-genotype` infers a donor's V genotype from the reads themselves (TIgGER's frequency
  rule) and re-calls the reads it settles.
- **Uncertainty-aware.** When a read cannot tell alleles apart (e.g. a short fragment), `v_call` lists, after the call,
  the V alleles that match the read as well as it does, as IgBLAST lists equally scoring alleles. Each gene also gets a
  candidate-set column (`*_call_set`).
- **AIRR output.** Standard AIRR rearrangement TSV (V/D/J calls, coordinates, junction, productivity) that reads directly
  into Change-O / Scirpy / Immcantation.
- **Model evaluation.** `alignair benchmark` checks a model on simulated reads with known truth. The source benchmark
  harness supports paired tool comparisons, scoring V/D/J allele accuracy, segment coordinates, the junction,
  productivity, and orientation. See [benchmarks](https://mutejester.github.io/AlignAIR/#/docs/benchmarks) for the
  evaluation workflows.

## Install

```bash
pip install "AlignAIR[cli]"            # core + CLI (recommended)
alignair doctor                        # verify Python / PyTorch+CUDA / GenAIRR
```

Install extras: `[cli]` or `[hub]` (`huggingface_hub`, needed to load a model straight from a Hugging Face repo; the
catalog downloads with the standard library), `[test]` (pytest, official AIRR validation, recipe checks), `[all]`.
CSV/TSV reading uses Python's standard library; pandas is not
required for inference or training. PyTorch is auto-detected for GPU; for a CPU-only install, `pip install torch
--index-url https://download.pytorch.org/whl/cpu` first.

Or Docker (no local install needed):

```bash
docker pull alignair/alignair:latest
docker run --rm alignair/alignair:latest doctor

# align reads: mount an input dir + an output dir, and persist the model cache across runs
docker run --rm \
  -v "$PWD:/data" -v alignair-cache:/home/appuser/.cache/alignair \
  alignair/alignair:latest \
  predict --input /data/reads.fasta --out /data/out.tsv --model <v4-igh-model>
```

The same image is mirrored at `ghcr.io/mutejester/alignair` (identical tags) if Docker Hub's anonymous
pull limit is a problem, e.g. on a shared cluster. The default image is CPU-only; pin a version tag
(`alignair/alignair:3.0.0`) for reproducibility. The
container runs as a non-root user, so mount a writable output dir (add `--user $(id -u):$(id -g)` if
your host uid differs). Models are not baked in - the `alignair-cache` volume above keeps a downloaded catalog
`--model <id>` from being re-fetched on every run (a model pulled straight from a Hugging Face repo goes to the
Hugging Face cache, `/home/appuser/.cache/huggingface`, instead). GPU is auto-detected when you run in a CUDA base
image.

## Quick start

See it work end-to-end in one command - offline, no model download needed (it trains a tiny demo
model, aligns simulated reads, and aligns them again under a donor genotype):

```bash
alignair demo
```

Or use a **pretrained model**. `--model` takes a `.alignair` path, a Hugging Face repo (`hf://org/repo`), or an id from
the catalog at the public [model hub](https://huggingface.co/AlignAIR/AlignAIR-pretrained); a remote model is
downloaded on first use and cached.

<!-- PENDING-V4-MODEL-ID: the AlignAIR 4 IGH model's id/repo is not decided. -->
> `<v4-igh-model>` below is a placeholder for an AlignAIR 4 human IGH model; the id is not published yet. Any
> AlignAIR 4 `.alignair` path works in its place. The hub catalog's current entries (`alignair-igh-human@1.0.0` and
> the others) are AlignAIR 3 models, which AlignAIR 4 refuses ("use alignair<4 to run it").

```bash
alignair models list                          # the catalog (fetched live from the hub)
alignair predict --input reads.fasta --out out.tsv --model <v4-igh-model>

# from a Hugging Face repo, pinned to a branch, tag or commit
alignair predict --input reads.fasta --out out.tsv --model hf://<org>/<repo> --revision <commit>

# restrict calls to a donor's genotype (YAML or JSON: V/D/J -> allele names of the model's reference)
alignair predict --input reads.fasta --out out.tsv --model <v4-igh-model> --genotype donor.yaml

# or infer the donor's V genotype from the reads themselves and re-call the reads it settles
alignair predict --input reads.fasta --out out.tsv --model <v4-igh-model> --infer-genotype
```

A catalog id downloads, hash-verifies and caches the model on first use (pin a version with
`--model <id>@<version>`); a Hugging Face repo takes `--revision` and, for a private repo, `--hf-token` (or
`$HF_TOKEN`). `alignair info model.alignair` prints a model file's card: its reference, allele counts, training steps
and sections.

From Python:

```python
from alignair import Aligner

aligner = Aligner.from_pretrained("model.alignair", device="auto")   # or "hf://org/repo" (revision=...), or a catalog id
result = aligner.predict(["CAGGTGCAGCTGGTGCAGTCTGGG..."])
result.write_airr("out.tsv")                                         # an AIRR rearrangement TSV
```

Prefer your own reference or species? Train a model, then align with it:

```bash
alignair train --dataconfig HUMAN_IGH_OGRDB --out my_model --preset desktop
alignair predict --input reads.fasta --out out.tsv --model my_model/bundle/model.alignair
```

<!-- PENDING-V4-NUMBERS: training time per preset (was "~minutes on a GPU"); re-time on a quiet machine. -->

See [`examples/`](examples/) for runnable data.

## Reference: an input, not weights

Each model file embeds the germline reference it was trained on, fingerprinted; that is the default set of callable
alleles. The model scores the alleles of whichever reference it is bound to, so you can change that set at predict time
without retraining.

- **Restrict to a donor's genotype** with `--genotype donor.yaml` (YAML or JSON). The keys are the segments `V`, `D`
  and `J`; each lists allele names of the bound reference (a name-to-sequence mapping also works; the sequences are
  ignored). A segment you leave out keeps all its alleles. A read whose V gene is not in the genotype gets no V call,
  and its row is tagged `partial` (reason `no_v`).

  ```yaml
  # donor.yaml
  V: [IGHVF1-G3*02, IGHVF6-G21*07, IGHVF6-G21*12]
  J: [IGHJ4*02, IGHJ6*02]
  ```

- **Bind another reference** with `--reference germline.fasta`: one FASTA holding the model's segments (V, D and J;
  V and J for a model without D), each allele's segment read from its name. For example, the model's reference plus
  novel alleles, or a curated set. An allele that is one of the model's keeps its IMGT gaps and anchor. A V allele
  that differs from one of the model's alleles of the same length at no more than 10% of its bases takes that
  allele's gaps and anchor. Give other V alleles IMGT-gapped (`.` gaps), so their anchor can be found from the gaps.
  Alleles left without an anchor are named in a warning on stderr. With `--reference`, `--genotype` may name the
  reference's novel alleles.
- **Export the model's own reference** as a starting point: `alignair reference export model.alignair --fasta ref.fasta`.

From Python, `Aligner.bound_to(reference)` returns a copy bound to another reference (the original is unchanged), and
`Aligner.from_pretrained(..., reference=...)` binds one at load:

```python
from alignair.reference import ReferenceSet

donor = aligner.bound_to(ReferenceSet.from_fasta("germline.fasta"))
result = donor.predict(reads)
result = aligner.predict(reads, genotype={"V": ["IGHVF1-G3*02", "IGHVF6-G21*07"]})   # a subset, for one call
```

What a swap refuses, with its error:

- A reference without one of the model's segments, e.g. an IGK, IGL, TRA or TRG reference (no D) on an IGH model:
  `[ALIGNAIR_CONFIG_INVALID] the bound reference has no D alleles for the chip's genes`.
- On a model of several loci, an allele that belongs to none of them (by its sequence, else its name):
  `[ALIGNAIR_INPUT_INVALID] the reference's allele IGHV1-2*02 belongs to none of the model's loci (IGK, IGL)`.
- A genotype naming an allele the bound reference does not hold:
  `[ALIGNAIR_GENOTYPE_UNKNOWN_ALLELE] unknown alleles in genotype for v: ['IGHV1-2*99'] (not in the reference)`. Add
  its sequence with `--reference`.

What a swap does not check: the species, and, on a single-locus model, the locus of a reference that has the model's
segments. A TRB reference binds to an IGH model, and its records still say `IGH`. How well a model calls the alleles of
another species or locus has not been measured. For those, **train a model**:

```bash
# train for any of GenAIRR's built-in references (human, mouse, rat, rabbit, dog, ...: alignair reference list)
alignair train --dataconfig MOUSE_IGH_IMGT --out runs/mouse_igh --preset desktop

# or train from your OWN germline FASTAs (custom reference or species)
alignair train --v-fasta v.fasta --d-fasta d.fasta --j-fasta j.fasta \
  --chain-type BCR_HEAVY --out runs/my_ref --preset desktop
```

Presets: `quick` (smoke), `desktop`, `full` (paper-grade). Preview the reference, config and model size without
training with `--plan`.

### Sharing a trained model

`alignair train` writes two kinds of file:

- `runs/.../model.alignair` (and `model.step<N>.alignair`): resumable training checkpoints. They hold pickled sections
  (`dataconfig/0`, `train_state`) for resuming, and are not for sharing. `alignair info` lists a file's sections.
- `runs/.../bundle/`: a self-contained, pickle-free `model.alignair` (the reference is **embedded**), a `model_card.md`,
  a `reference_manifest.json`, and a `validation_report.json`. This is the file to share and to predict with:
  `alignair predict ... --model runs/.../bundle/model.alignair`.

`Aligner.save_pretrained(directory)` copies a pickle-free model file, and `Aligner.push_to_hub(repo_id)` uploads it
(`model.alignair` only) to a Hugging Face repo (needs `huggingface_hub` and a write token); both refuse a checkpoint.
After training from Python (`run_training`), export the bundle yourself:

```python
from alignair.genairr import resolve_dataconfigs
from alignair.train.build import export_bundle

export_bundle("runs/igh/model.alignair", resolve_dataconfigs(["HUMAN_IGH_OGRDB"]), "runs/igh/bundle")
```

There is no `alignair export` command. `alignair convert` converts only lab chip model files (`lab-chip-v1`,
`lab-chipw64-v1`), not training checkpoints.

## Donor genotype inference

`--infer-genotype` reads the whole input file twice, so all of its reads should come from one donor:

1. It aligns every read, and counts, for each V gene, the reads whose `v_call` is a single allele.
2. For each gene, it keeps the most frequent alleles until they explain 87.5% of those reads (`--genotype-explain`,
   TIgGER's default). A gene with fewer than 10 such reads (`--genotype-min-reads`) has no evidence and keeps all its
   alleles.
3. It reads the file again and re-calls only the reads whose call the genotype excludes while their `v_call` lists an
   allele it allows, with the genotype's V alleles as the bank. Their rows replace the first pass's.

It writes the genotype as an AIRR GenotypeSet (`out.tsv.genotype.json`, or `--genotype-out`), and records it in
`out.tsv.run.json`. It infers V only: D and J are not genotyped. It does not find novel alleles, mark deleted genes,
or separate donors, and it does not change a read whose V call is a single allele. It cannot be combined with
`--genotype`, and needs files, not `-`. `alignair genotype --airr table.tsv --out genotype.json` applies the same rule
to any tool's AIRR table.

## Output

`alignair predict` writes a **schema-valid AIRR rearrangement TSV** (validates against the official
`airr` library; reads back with Change-O / Immcantation): `sequence_id`, `sequence`, `rev_comp`, `locus`,
`productive`, `v_call`/`d_call`/`j_call`, `junction`/`junction_aa`, gapped `sequence_alignment` /
`germline_alignment`, per-gene `*_cigar`, `*_identity`, and sequence/germline coordinates, plus a
per-gene candidate-set column (`*_call_set`). The gapped
alignment fields are produced by AlignAIR's own IMGT-gap reconstruction (no external aligner required).

**Reproducible rows.** On the CPU, the same file with the same settings always gives the same rows, and
`--threads N` gives the rows of one single-threaded process at the same `--batch-size`. On a GPU, some of the
model's sums run in an order that can vary between runs, so the last bits of a row's floats can differ from run to
run. Reads are batched with reads of their 32-position length bucket, so reads in other buckets never change a
read's row. Which and how many reads of its own bucket share its batch, the batch size, the chunk size, the CPU
thread count and the device can still move the last bits of the floats and, rarely (in one measurement, 1 read in
2,000), a call between two near-tied alleles. The default batch size depends on the device and on `--threads`, so
pass `--batch-size`, `--chunk-size` and `--threads` to reproduce a run exactly.

Every run also writes a **`<output>.run.json` provenance sidecar** (model + fingerprint, reference,
command, device, and record counts). Validate any TSV explicitly:

```bash
alignair validate-airr out.tsv      # structural check: columns, coordinate/CIGAR bounds, productivity
```

## Commands

| Command | Purpose |
| --- | --- |
| `alignair demo` | offline end-to-end trial (tiny train → predict → predict under a donor genotype) |
| `alignair predict` | align reads → AIRR rearrangement TSV |
| `alignair train` | train a model for your own reference / species (built-in dataconfig or custom FASTA) |
| `alignair info` | print a model file's card (reference, allele counts, training, sections) |
| `alignair models` | list / download / manage catalog models (alias `model`) |
| `alignair reference` | list built-in references, or export a model's reference |
| `alignair export-reference` | export a model's germline FASTA (same as `reference export`) |
| `alignair genotype` | a donor's V genotype from an AIRR table (the `--infer-genotype` rule) |
| `alignair compare` | agreement report between two AIRR TSVs (e.g. AlignAIR vs IgBLAST) on your data |
| `alignair analyze` | summarize an AIRR TSV (repertoire composition, prediction QC) |
| `alignair benchmark` | evaluate a model on simulated GenAIRR reads, per stratum |
| `alignair validate-airr` | structural check of a rearrangement TSV (columns, coordinate/CIGAR bounds, productivity); not the official AIRR-C schema validator |
| `alignair doctor` | check the environment (Python, PyTorch+CUDA, GenAIRR) |
| `alignair convert` | convert a lab chip model file (`lab-chip-v1`, `lab-chipw64-v1`) into a pickle-free chip-v1 `.alignair` |

Run `alignair <command> --help` for options.

Runtime errors support `--debug` for chained tracebacks and `--error-format json` for structured
diagnostics on stderr. Python callers can catch `alignair.errors.AlignAIRError`; see
[Errors and diagnostics API](https://mutejester.github.io/AlignAIR/#/docs/errors) for exception types and reporting helpers.

## Documentation

Full docs, reference, and interactive lessons: **<https://mutejester.github.io/AlignAIR/>**

- [Getting started](https://mutejester.github.io/AlignAIR/#/docs/getting-started)
- [Pretrained models](https://mutejester.github.io/AlignAIR/#/docs/models)
- [Command-line reference](https://mutejester.github.io/AlignAIR/#/docs/cli)
- [Benchmarks](https://mutejester.github.io/AlignAIR/#/docs/benchmarks)
- [Design & internals](https://mutejester.github.io/AlignAIR/#/docs/design)
- [Troubleshooting](https://mutejester.github.io/AlignAIR/#/docs/troubleshooting)

## Development

```bash
git clone https://github.com/MuteJester/AlignAIR && cd AlignAIR
pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

## Citation

If you use AlignAIR, please cite:

> Thomas Konstantinovsky, Ayelet Peres, Ran Eisenberg, Pazit Polak, Ofir Lindenbaum, Gur Yaari.
> Enhancing sequence alignment of adaptive immune receptors through multi-task deep learning.
> *Nucleic Acids Research*, Volume 53, Issue 13, 22 July 2025, gkaf651.
> <https://doi.org/10.1093/nar/gkaf651>

## License

GPL-3.0-or-later (see [`LICENSE`](LICENSE)).

## Contact

Issues: [GitHub issues](https://github.com/MuteJester/AlignAIR/issues) · Email: thomaskon90@gmail.com · Site: https://alignair.ai
