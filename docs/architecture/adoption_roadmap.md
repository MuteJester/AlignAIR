# AlignAIR Adoption Roadmap

Audit date: 2026-06-22. Scope: this repository branch plus current AIRR and incumbent-tool distribution norms.

## Executive Summary

AlignAIR has a strong technical adoption hook: the PyTorch `alignair` package can condition inference on a runtime reference, including YAML/FASTA genotypes and novel alleles, and it already has a versioned bundle format, AIRR-style TSV writer, common sequence readers, internal training loop, and a serious GenAIRR-backed benchmark module. The main adoption risk is not the model idea. It is that the public product surface is split between the current PyTorch CLI and older TensorFlow/app.py documentation and Docker assumptions.

The highest-leverage launch path is:

1. Make one documented public entrypoint work everywhere: `alignair predict`, `alignair bundle`, `alignair train`, and `alignair model`.
2. Publish at least one real pretrained bundle with a model card, checksums, license, validation report, and a one-command download path.
3. Turn the internal training script into a first-class `alignair train` workflow for custom references/species.
4. Validate AIRR output against AIRR-C tooling and make Immcantation/Change-O compatibility explicit.
5. Replace stale docs with two runnable journeys: "use pretrained model" and "train/fine-tune on my own reference".

## Evidence Base

### Repository Audit Anchors

- Packaging metadata declares `AlignAIR` version `2.0.2`, Python `>=3.9,<3.13`, GPL-3.0-or-later, core dependencies including `torch` and `GenAIRR`, optional extras for `cli`, `reader`, `train`, `viz`, `server`, and a console script `alignair = alignair.cli:main` (`pyproject.toml:7-12`, `pyproject.toml:31-88`).
- The current CLI exposes only `predict` and `bundle`; there is no user-facing `train` or model-download command (`src/alignair/cli.py:98-127`).
- `predict` accepts a bundle or raw checkpoint, optional YAML/FASTA genotype, optional calibration JSON, `--batch`, and `--device`, then writes AIRR-style TSV (`src/alignair/cli.py:37-76`, `src/alignair/cli.py:102-116`).
- The sequence reader supports FASTA, FASTQ, CSV/TSV, TXT, and `.gz`, normalizes IUPAC ambiguity to `N`, and drops reads with too much invalid content (`src/alignair/io/sequence_reader.py:1-5`, `src/alignair/io/sequence_reader.py:46-113`).
- The AIRR writer emits core fields plus custom uncertainty fields; it converts starts to 1-based and emits canonical forward sequence with `rev_comp` (`src/alignair/io/airr.py:1-18`, `src/alignair/io/airr.py:21-52`).
- The current PyTorch bundle format is `model.pt`, `config.json`, `reference.json`, optional `calibration.json`, `meta.json`, `VERSION`, and `fingerprint.txt`, with SHA-256 fingerprint verification on load (`src/alignair/serialization/dnalignair_bundle.py:1-13`, `src/alignair/serialization/dnalignair_bundle.py:26-82`).
- Runtime reference handling exists for GenAIRR dataconfigs, YAML genotypes, FASTA genotypes, subsets, and novel alleles (`src/alignair/reference/reference_set.py:31-57`, `src/alignair/reference/reference_set.py:59-150`).
- Training exists as an internal script using `AlignAIRGym`, `GymTrainer`, seeds, periodic evaluation, and CSV logging, but it is not a package CLI (`scripts/train_dnalignair.py:1-7`, `scripts/train_dnalignair.py:24-111`).
- The benchmark module can generate, export, evaluate, normalize AIRR/IgBLAST TSVs, add bootstrap CIs, and compare outputs (`src/alignair/benchmark/README.md:1-31`, `src/alignair/benchmark/README.md:33-220`).
- Top-level README and MkDocs pages are stale for this branch: they document `python app.py run`, `list-pretrained`, TensorFlow `SavedModel`, and `from_pretrained` classes that are not the current CLI surface (`README.md:41-68`, `README.md:87-95`, `README.md:121-156`, `README.md:202-220`; `docs/getting_started.md:23-104`; `docs/index.md:39-80`).
- Dockerfile is not launch-ready for this branch: it copies a missing `checkpoints` directory, healthchecks TensorFlow although TensorFlow is not a dependency, and uses `ENTRYPOINT ["python", "app.py"]` even though the repo root contains only a trivial `main.py` and no `app.py` (`Dockerfile:19-26`, `Dockerfile:37-42`; `main.py:1-3`).
- Governance assets are incomplete: generic issue templates exist, but no `CONTRIBUTING.md`, `CHANGELOG.md`, `CITATION.cff`, conda recipe, Galaxy wrapper, or nf-core module was found in the audit.

### External Norms Used

- AIRR Software WG compliance expects public source/versioning, standard AIRR file formats, example data with automated checks, run parameters in output, remotely built containers, and clear support statements ([AIRR Software WG guidance](https://docs.airr-community.org/en/stable/swtools/airr_swtools_standard.html)).
- AIRR Rearrangement schema covers input identifiers, V(D)J calls, productivity, junction, alignments, and coordinate fields; custom columns are allowed if unique and snake_case-like ([AIRR Rearrangement schema](https://docs.airr-community.org/en/latest/datarep/rearrangements.html)).
- OGRDB is the AIRR Community germline reference source with curated IG/TR germline sets, downloads, APIs/tools, and frequent updates ([OGRDB](https://ogrdb.airr-community.org/), [OGRDB help](https://ogrdb.airr-community.org/help)).
- Change-O/Immcantation use AIRR-C TSV as the default standard and map AIRR fields into downstream repertoire workflows ([Change-O data standards](https://changeo.readthedocs.io/en/stable/standard.html), [Immcantation portal](https://immcantation.readthedocs.io/en/latest/)).
- IgBLAST has precompiled binaries, configurable custom germline databases, OGRDB/AIRR-C germline support, and AIRR tabular output; Change-O wrappers expect AIRR outfmt 19 or compatible tabular output ([NCBI IgBLAST setup](https://ncbi.github.io/igblast/cook/How-to-set-up.html), [NCBI IgBLAST web docs](https://www.ncbi.nlm.nih.gov/igblast/), [Change-O IgBLAST example](https://changeo.readthedocs.io/en/stable/examples/igblast.html)).
- MiXCR lowers adoption with downloadable binaries, conda/Homebrew installs, and `exportAirr` for AIRR output ([MiXCR installation](https://mixcr.com/mixcr/getting-started/installation/), [MiXCR exportAirr](https://mixcr.com/mixcr/reference/mixcr-exportAirr/)).
- IgDiscover is the closest UX analog for personalized references: it has a Conda-based install, test datasets, user guide, configuration docs, run docs, and changelog ([IgDiscover docs](https://gkhlab.gitlab.io/igdiscover22/), [IgDiscover installation](https://gkhlab.gitlab.io/igdiscover22/installation.html)).
- Immcantation publishes versioned DockerHub containers that include dependencies, IgBLAST, references, and example scripts; nf-core/airrflow packages AIRR workflows with containers and CI-tested releases ([Immcantation Docker](https://immcantation.readthedocs.io/en/stable/docker/intro.html), [nf-core/airrflow](https://nf-co.re/airrflow/3.3.0/)).
- Bioconda recipes automatically unlock BioContainers for many users, while Galaxy and nf-core integrations rely on conda/containerized command-line tools ([BioContainers/Bioconda integration](https://biocontainers-edu.readthedocs.io/en/latest/conda_integration.html), [Galaxy conda dependencies](https://docs.galaxyproject.org/en/latest/admin/conda_faq.html), [nf-core module docs](https://nf-co.re/docs/developing/pipelines/adding-modules)).
- Hugging Face model cards make models discoverable through metadata such as license, datasets, task tags, versions, intended use, and evaluation results ([HF model cards](https://huggingface.co/docs/hub/en/model-cards), [HF model release checklist](https://huggingface.co/docs/hub/en/model-release-checklist)).

## 1. Install And Packaging

| Item | Current State | Gap Vs Expectations | Recommendation | Priority | Effort |
| --- | --- | --- | --- | --- | --- |
| Public Python install | `pyproject.toml` is modern and declares `alignair` console script, but docs still use `python app.py` and README says Python `<3.12` while package allows `<3.13` (`pyproject.toml:7-12`, `README.md:87-95`). | Users cannot tell whether PyPI, editable install, or Docker is canonical. AIRR compliance also expects versioned source and easy future reproducibility. | Make `pip install AlignAIR[cli]` and `alignair --help` the canonical path. Verify the PyPI project exists before marketing it. Add a smoke test in CI that installs the built wheel and runs `alignair predict` on bundled example data. | P0 | M |
| Extras and dependency story | Extras exist for `cli`, `reader`, `train`, etc. `parasail` is optional under `reader`; `huggingface-hub` is in `cli` but no model-download UX exists (`pyproject.toml:43-77`). | Users will not know when to install `[reader]`, `[train]`, or GPU-specific PyTorch. Broad `torch>=2.2` and `GenAIRR>=2.2.0` ranges reduce reproducibility. | Document `AlignAIR[cli]`, `AlignAIR[train]`, `AlignAIR[reader]`, and `AlignAIR[all]`. Add tested version ranges or lockfiles for release builds. Keep `parasail` optional unless benchmarked as a production default; expose it as an expert `--v-reader parasail` option if retained. | P1 | S |
| Docker | Dockerfile uses `python:3.11-slim`, installs `.[cli]`, copies missing `checkpoints`, healthchecks TensorFlow, and runs missing `app.py` (`Dockerfile:1-42`). | Current container is likely broken on this branch. AIRR Software WG expects a remotely built container that runs example data as part of build/test. | Replace entrypoint with `alignair`, remove TensorFlow healthcheck, do not require a local `checkpoints/` directory, add a container smoke test, and publish tagged CPU images to GHCR and Docker Hub. Add a separate CUDA image only if GPU inference/training is documented and tested. | P0 | M |
| Conda/Bioconda/BioContainers | No conda recipe found. | AIRR/bioinformatics users frequently install through conda/Bioconda; Bioconda also unlocks BioContainers automatically. | Add a Bioconda recipe after wheel install is stable. Use it to enable Galaxy/nf-core packaging. Include `run_exports`/pins for PyTorch, GenAIRR, and Python versions. | P1 | M |
| Release artifacts | No changelog or release workflow found. README has Docker and DOI badges but not install verification (`README.md:7-10`). | Users need versioned, reproducible releases, not only a moving `latest` image. | Add GitHub release workflow that builds wheels, Docker images, smoke-test artifacts, and publishes release notes. Tag Docker images as `2.0.2`, `2.0`, and `latest` only after tests pass. | P1 | M |

## 2. Pretrained Models

| Item | Current State | Gap Vs Expectations | Recommendation | Priority | Effort |
| --- | --- | --- | --- | --- | --- |
| Model bundle format | PyTorch bundle format is coherent and integrity-checked (`src/alignair/serialization/dnalignair_bundle.py:1-13`, `src/alignair/serialization/dnalignair_bundle.py:60-82`). | It is not yet discoverable as a model zoo. README advertises old bundle paths and TensorFlow `SavedModel` layout (`README.md:121-156`, `README.md:202-220`). | Make the PyTorch bundle spec public and stable. Include `model.pt`, `config.json`, `reference.json`, calibration, provenance, license, validation report, and checksums. Remove TensorFlow bundle docs from the current PyTorch branch. | P0 | S |
| Model zoo | README lists `IGH_S5F_576`, `IGH_S5F_576_Extended`, `IGL_S5F_576`, `TCRB_UNIFORM_576`, but repo audit did not find a `checkpoints/` directory and CLI has no `list-pretrained` (`README.md:41-57`, `README.md:99-117`; `src/alignair/cli.py:98-127`). | Users cannot run the 5-minute journey without a model. | Publish a minimal model zoo with at least `human-igh-ogrdb-v1` first. Each model should state species, locus, reference source/version/DOI, training simulator config, read-length/chemistry assumptions, calibration, expected hardware, validation metrics, license, and known limitations. | P0 | M |
| Download/from-pretrained UX | `huggingface-hub` is listed in extras but no `alignair model download` or Python `from_pretrained` path exists (`pyproject.toml:45-49`; no matches in current `src/alignair` audit). | ML users expect `from_pretrained("org/model")`; bioinformatics users expect `alignair model list/download`. | Add `alignair model list`, `alignair model download`, `alignair model inspect`, and `DNAlignAIRBundle.from_pretrained(...)` backed by Hugging Face Hub or release assets. Support pinned revisions and offline cache. | P0 | M |
| Model versioning and integrity | Bundle fingerprint exists, but model provenance metadata is sparse (`src/alignair/serialization/dnalignair_bundle.py:30-56`). | HF and reproducible bioinformatics norms expect metadata, model cards, checksums, license, and version lineage. | Store SHA-256 manifest, training commit, AlignAIR/GenAIRR/PyTorch versions, seed, dataconfig/reference version, calibration data version, benchmark report hash, and `base_model`/`new_version` relationships in model cards. | P1 | M |

## 3. Inference UX And CLI

| Item | Current State | Gap Vs Expectations | Recommendation | Priority | Effort |
| --- | --- | --- | --- | --- | --- |
| Core command | `alignair predict` exists and accepts FASTA/FASTQ/CSV/TSV/TXT `.gz`, genotype YAML/FASTA, bundle/raw checkpoint, calibration, batch, and device (`src/alignair/cli.py:37-76`, `src/alignair/io/sequence_reader.py:1-5`). | Good base, but docs do not use it, and errors/logging are print-based. | Make `alignair predict reads.fq.gz -o rearrangements.tsv --model alignair/human-igh-ogrdb-v1` the first README example. Add `--quiet`, `--verbose`, progress, structured log option, deterministic exit codes, and input summary manifest. | P0 | S |
| Output contract | `write_airr` emits key AIRR-like fields and custom uncertainty fields (`src/alignair/io/airr.py:13-18`). Tests cover coordinates and custom set fields (`tests/alignair/test_cli_io.py:72-109`). | Need formal AIRR-C validation, run-parameter provenance, and compatibility with Change-O/Immcantation readers. AIRR custom fields should be clearly reserved or prefixed. | Add `alignair validate-airr` or CI validation using AIRR library. Include a sidecar `run.json` with model hash, reference hash, command args, device, versions, and calibration. Consider `alignair_v_call_set`/reserved field proposal for uncertainty columns. | P0 | M |
| Input formats | Common single-read formats are supported (`src/alignair/io/sequence_reader.py:46-113`). | No explicit paired-end/UMI story, no direct AIRR input mode beyond a table with `sequence`, and no stdin/stdout guidance. | Document supported input columns. Add `--input-format`, `--id-column`, `--sequence-column`, stdin/stdout support, and later paired/assembled read layouts. | P1 | M |
| Multi-locus/multi-chain | Code can build references from multiple GenAIRR dataconfigs and has `locus` labeling, but CLI only accepts one `--dataconfig` default unless repeated only in `bundle`; README's multi-chain docs are stale (`src/alignair/reference/reference_set.py:31-57`, `src/alignair/cli.py:112-116`, `README.md:188-198`). | Users with IGK/IGL/TCR data need explicit model/reference compatibility and no false D calls on no-D loci. | Add model metadata enforcement: supported loci, has-D, max length, reference kind. Add `--locus auto|IGH|IGK|IGL|TRA|TRB|TRD|TRG` only where the model supports it. | P1 | M |
| Throughput and hardware | `--batch` and auto CUDA/CPU exist (`src/alignair/cli.py:45`, `src/alignair/cli.py:115-116`). | No published throughput/memory table; CPU vs GPU install story unclear. | Publish benchmarked reads/sec and memory for CPU, single GPU, and common read lengths. Add `alignair doctor` for dependency, CUDA, model, and reference checks. | P1 | M |

## 4. Train-Your-Own-Model UX

This is the differentiator and the biggest adoption gap. Dynamic genotype inference means users can often bring a custom genotype at prediction time without retraining. But the broader promise, "bring your own reference/species and get your own aligner," needs a first-class training workflow.

| Item | Current State | Gap Vs Expectations | Recommendation | Priority | Effort |
| --- | --- | --- | --- | --- | --- |
| Training entrypoint | Training exists only as `scripts/train_dnalignair.py`, invoked with `PYTHONPATH=src`, raw hyperparameters, and CSV logging (`scripts/train_dnalignair.py:1-7`, `scripts/train_dnalignair.py:24-111`). | No `alignair train`; no project layout; no reference validation; no checkpoint/resume; no final bundle; no model card; no guided evaluation. | Add `alignair train` as P0. It should take a reference, output directory, preset, seed, device, training budget, and optional base model, then write a loadable bundle plus validation report. | P0 | L |
| Minimum viable train UX | Internal pieces exist: `ReferenceSet`, `AlignAIRGym`, `GymTrainer`, evaluation, bundle save, calibration script, benchmark CLI. | Pieces are not connected into a user journey. | MVP command: `alignair train --reference my_ogrdb_or_fasta --locus IGH --out runs/my_species_igh --preset standard --steps N --device cuda --seed 1`. It should: validate reference, infer V/D/J presence, start GenAIRR simulation, train with resume checkpoints, evaluate smoke metrics every interval, calibrate set calls, save final bundle, and emit `model_card.md` plus `validation_report.json`. | P0 | L |
| Fine-tune vs scratch | Docs mention fine-tuning in old TensorFlow terms (`README.md:202-220`) and internal training starts from random model (`scripts/train_dnalignair.py:50-57`). | Users need guidance: when can they fine-tune a human IGH model vs train from scratch for a new species/locus/chemistry? | Define policy: fine-tune when locus architecture, max length, read chemistry, and segment structure match; train from scratch for new locus topology, very divergent species, or new assay distributions. Add `alignair train --base-model <bundle>` and report which layers/reference embeddings are transferred. | P0 | M |
| Data and compute expectations | Training script defaults are smoke-scale, not launch-scale (`scripts/train_dnalignair.py:26-41`). | New users need wall time, GPU memory, steps, expected metrics, and failure modes. | Publish presets: `smoke`, `desktop-cpu`, `single-gpu`, `paper-grade`. Each preset should list reads simulated, batch, approximate wall time, GPU memory, stopping criteria, and expected benchmark floor. | P1 | M |
| Built-in evaluation | Benchmark is powerful but separate. Scripts mention head-to-head, calibration, held-out alleles, real-data validation (`scripts/README.md:13-39`). | Users training custom references need one command to know whether the model works and how it compares to IgBLAST/MiXCR if installed. | Add `alignair train --evaluate` and `alignair evaluate` wrappers around benchmark build/export/run/evaluate. If IgBLAST is installed, run a baseline with the same reference; otherwise produce AlignAIR-only report and instructions. | P1 | L |
| Ideal train UX | None. | IgDiscover-like personalized workflows are successful because they provide config, test data, run directory conventions, and output interpretation. | Ideal flow: `alignair init my-project`, `alignair reference fetch ogrdb --species ... --locus ...`, `alignair train`, `alignair benchmark --against igblast`, `alignair model push`. Include notebook/Colab path for small runs and HPC/SLURM examples for serious training. | P1 | L |

## 5. Docs And Onboarding

| Item | Current State | Gap Vs Expectations | Recommendation | Priority | Effort |
| --- | --- | --- | --- | --- | --- |
| README | README has a strong short tagline and badges, but most commands are stale for this branch (`README.md:5-10`, `README.md:41-95`, `README.md:121-220`). | First-time users will fail before seeing the model's strengths. | Rewrite README around current PyTorch CLI. First viewport: what AlignAIR does, why dynamic reference matters, install, one pretrained command, one custom genotype command, one train command. | P0 | M |
| Getting started docs | MkDocs pages use old `python app.py` commands and placeholder links (`docs/getting_started.md:23-104`, `docs/index.md:75-92`). | Docs conflict with current package and Docker. | Replace with a 5-minute quickstart using real example data and real model. Add "pretrained inference" and "train own model" tutorials as separate pages. | P0 | M |
| Example data | Tests have small FASTA/CSV data; benchmark can export examples (`tests/alignair/test_cli_io.py:16-43`, `src/alignair/benchmark/README.md:110-130`). | No polished, documented example dataset for users. | Ship `examples/` with tiny FASTQ, custom genotype YAML/FASTA, expected AIRR TSV, and smoke command. Use it in Docker, wheel, and docs tests. | P0 | S |
| Notebooks/Colab | One Jupyter notebook exists, but currentness is unknown (`docs/tutorials/AlignAIR_On_Jupyter_Notebooks.ipynb`). | AIRR users often evaluate in notebooks; training needs guided interpretation. | Add two maintained notebooks: pretrained inference and custom-reference training smoke. Add Colab badges only if model/download sizes fit. | P1 | M |
| Benchmark story | `docs/dnalignair.md` includes benchmark numbers and dynamic genotype claims (`docs/dnalignair.md:53-74`, `docs/dnalignair.md:89-118`). | The requested 4400-case/22-stratum/23-of-24 claim is not present as a frozen public report in audited docs. | Publish a "Why AlignAIR vs IgBLAST/MiXCR" page with frozen benchmark artifacts, exact commit/model/reference versions, confidence intervals, and known losses. | P1 | M |
| Troubleshooting | No focused troubleshooting guide found. | Bioinformatics install failures often involve CUDA, conda, BLAST paths, reference names, and coordinate conventions. | Add troubleshooting for install, CUDA, GenAIRR, parasail, reference parsing, missing anchors/junctions, AIRR validation, and Docker volumes. | P1 | S |

## 6. Interoperability And Standards

| Item | Current State | Gap Vs Expectations | Recommendation | Priority | Effort |
| --- | --- | --- | --- | --- | --- |
| AIRR schema | Output includes common AIRR fields and coordinate conversion; benchmark can normalize AIRR/IgBLAST TSVs (`src/alignair/io/airr.py:13-52`, `src/alignair/benchmark/README.md:147-187`). | AIRR compliance requires more than "AIRR-style": formal field validation, examples, run params, and standard-format read/write support. | Make AIRR-C TSV the default output, validate examples with the AIRR reference library, include field dictionary, and document any custom fields. | P0 | M |
| OGRDB ingestion | GenAIRR dataconfigs and FASTA/YAML work (`src/alignair/reference/reference_set.py:31-150`). | OGRDB is the community reference source and supports downloads/API/tools; AlignAIR lacks a direct fetch/version/provenance path. | Add `alignair reference fetch ogrdb`, `alignair reference validate`, and `alignair reference convert` for OGRDB FASTA/JSON to AlignAIR YAML/bundle reference. Store OGRDB set/version/DOI in run and model metadata. | P1 | M |
| Immcantation/Change-O | Output field names overlap AIRR/Change-O expectations; benchmark accepts AIRR/IgBLAST TSV. | Need actual compatibility tests with `AIRRReader`, Change-O, and downstream germline/clonal workflows. | Add an integration test that feeds AlignAIR TSV into Change-O/airr readers and documents any missing optional fields needed by Immcantation. | P1 | M |
| Galaxy/nf-core | No wrappers found. | Many AIRR users run through Galaxy/Nextflow rather than Python. | After CLI and Bioconda stabilize, create a Galaxy tool wrapper and an nf-core module for `alignair predict`; later add `alignair train` as a separate heavier module. | P1 | L |
| Multi-tool baselines | Benchmark can compare to AIRR/IgBLAST outputs; scripts mention IgBLAST drivers (`src/alignair/benchmark/README.md:169-199`, `scripts/README.md:13-21`). | MiXCR, IgDiscover, partis, and Immcantation pipelines are adoption comparators, not just scientific baselines. | Provide converters/adapters and example benchmark recipes for IgBLAST, MiXCR `exportAirr`, and partis where legally/practically feasible. | P2 | M |

## 7. Trust And Reproducibility

| Item | Current State | Gap Vs Expectations | Recommendation | Priority | Effort |
| --- | --- | --- | --- | --- | --- |
| Benchmarks | Benchmark module is strong and documents generation, readiness, export, evaluation, bootstrap, performance, and comparison (`src/alignair/benchmark/README.md:1-31`, `src/alignair/benchmark/README.md:66-108`, `src/alignair/benchmark/README.md:200-220`). | Results are not packaged as public, frozen, citable release artifacts. | Publish benchmark case manifests, prediction outputs, reports, notebooks, and model bundle hashes for each release. Add a "reproduce the paper benchmark" command. | P1 | L |
| Citation | README has a Zenodo DOI badge (`README.md:7-10`), but no `CITATION.cff` found. | GitHub, Zenodo, and papers should agree on citation metadata. | Add `CITATION.cff`, paper/preprint link, Zenodo release metadata, and model-specific citation guidance. | P0 | S |
| Provenance | Bundle has fingerprint and meta notes; training has seed; benchmark has deterministic generation notes (`src/alignair/serialization/dnalignair_bundle.py:30-68`, `scripts/train_dnalignair.py:41-45`, `src/alignair/benchmark/README.md:69-75`). | Bundle metadata lacks complete software/reference/training provenance. | Add structured provenance to bundle and every run output: commit SHA, package versions, reference hash, OGRDB version, GenAIRR config, seed, hardware, command, calibration version. | P0 | M |
| Determinism | Seeds appear in training and benchmark docs, but not surfaced in CLI output (`scripts/train_dnalignair.py:41-45`, `src/alignair/benchmark/README.md:69-75`). | Users need reproducible predictions/training reports. | Add `--seed` where stochasticity exists, emit it in run manifests, and document deterministic limits for CUDA/PyTorch. | P1 | S |
| Validation reports | Tests cover CLI IO and bundle equivalence; benchmarks can produce reports (`tests/alignair/test_cli_io.py:112-149`, `src/alignair/benchmark/README.md:156-220`). | Public users need plain-language validation attached to models, not just tests. | Every model release should ship `validation_report.json` plus human-readable summary over loci, length bins, mutation bins, fragments, orientation, genotype subset, and novel-allele stress tests. | P1 | M |

## 8. Community And Governance

| Item | Current State | Gap Vs Expectations | Recommendation | Priority | Effort |
| --- | --- | --- | --- | --- | --- |
| Positioning | README tagline is clear, but it does not explain dynamic genotype or train-your-own in the first 30 seconds (`README.md:5-10`). | Adoption hook is buried. | Lead with: "A neural AIRR aligner that conditions on your reference at runtime; use pretrained human IG/TCR models or train your own model for any species/reference." | P0 | S |
| Contribution process | Docs link to `CONTRIBUTING.md`, but file was not found (`docs/index.md:82-88`). Issue templates are generic web/mobile templates. | Open-source contributors need project-specific bug reports, reference/model issue templates, and contribution rules. | Add `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, PR template, issue templates for bug, model request, reference parsing, training failure, and AIRR compatibility. | P1 | S |
| License clarity | `pyproject.toml` and README say GPL-3.0-or-later/GPLv3, but docs index says AGPL-3 (`pyproject.toml:11`, `README.md:7-10`, `docs/index.md:3`). | License inconsistency creates legal friction. GPLv3 is acceptable for many research users but can deter closed-source commercial pipeline integration. | Fix license inconsistency. Clarify code license, model license, data/reference licenses, and whether commercial users can request alternative licensing. | P0 | S |
| Releases/changelog | No changelog found. | Users need to know which model/code/reference combinations are compatible. | Add `CHANGELOG.md` with package, bundle format, model zoo, and reference compatibility changes. | P1 | S |
| Support | README lists GitHub issues and email; AIRR compliance expects clear support expectations (`README.md:343-347`; AIRR guidance). | No response-time or scope expectations. | Add support policy: GitHub issues for bugs, discussions for usage, security contact, what is supported for pretrained vs custom-trained models. | P1 | S |
| Community examples | No examples gallery found. | Adoption grows when users see species/reference workflows like their own. | Add examples gallery: human IGH OGRDB, human IGK/IGL, TCRB, custom FASTA, novel allele, OGRDB non-human set, benchmark vs IgBLAST. | P2 | M |

## Top-10 Highest-ROI Items

1. P0 / M: Fix the public command surface and docs so `pip install AlignAIR[cli]`, `alignair predict`, and Docker all run the same current PyTorch workflow.
2. P0 / M: Publish one working pretrained `human-igh-ogrdb-v1` bundle with model card, checksum, calibration, validation report, and download command.
3. P0 / L: Add MVP `alignair train` that turns a YAML/FASTA/OGRDB reference into a trained, resumable, bundled model with validation output.
4. P0 / M: Make AIRR-C output validation and run provenance mandatory for examples, CI, and released models.
5. P0 / M: Replace README/MkDocs quickstarts with two runnable journeys: pretrained inference and train-your-own reference.
6. P0 / S: Fix license/citation basics: GPL vs AGPL inconsistency, `CITATION.cff`, model/data license statements.
7. P1 / M: Add `alignair model list/download/inspect` and Python `from_pretrained` backed by Hugging Face Hub or release assets.
8. P1 / M: Add OGRDB reference fetch/validate/convert commands with versioned reference provenance.
9. P1 / M: Publish tagged CPU Docker/GHCR images and a Bioconda recipe to unlock BioContainers, Galaxy, and nf-core.
10. P1 / L: Publish frozen benchmark artifacts and an IgBLAST/MiXCR/partis comparison workflow with confidence intervals and known limitations.

## Smallest P0 Set For A Credible Public Launch

1. One install path works from zero: wheel install plus Docker image, both smoke-tested on example data.
2. One pretrained model is discoverable and runnable by a documented command.
3. `alignair train` MVP exists for custom references and writes a loadable bundle plus validation report.
4. AIRR TSV output passes validation, and every run emits model/reference/parameter provenance.
5. README and docs stop referencing stale `app.py`, TensorFlow `SavedModel`, missing checkpoints, and placeholder links.
6. License, citation, support, and release versioning are unambiguous.

With those six items, AlignAIR can credibly launch as both a pretrained AIRR aligner and the first practical "bring your own reference, train your own neural aligner" workflow for the AIRR community.

## Round 2 Review - 2026-06-23

This section supersedes the priority order above for pre-release adoption work. A large part of the
first roadmap has been implemented. Under the current constraint, do not optimize for publishing
to PyPI, pushing Docker images, or uploading hub models yet. Optimize for what a real AIRR user can
try locally, trust, compare, and drop into an existing workflow before public release.

### Current Assessment

What is genuinely adoption-ready:

- The public CLI surface is now real: `predict`, `train`, `model`, `reference`, `validate-airr`,
  `doctor`, `bundle`, and `--version` are wired in one entrypoint (`src/alignair/cli.py:532-644`).
- `predict` is much more pipeline-friendly than the first audit: it resolves local/catalog/HF model
  specs, accepts genotype YAML/FASTA, supports stdin/stdout, column overrides, quiet/progress modes,
  a seed, `--v-reader learned|parasail`, AIRR validation, and a provenance sidecar
  (`src/alignair/cli.py:46-140`, `src/alignair/cli.py:151-187`, `src/alignair/cli.py:537-565`).
- AIRR output moved from "AIRR-style" to a much stronger contract: core AIRR columns, per-gene
  CIGARs, identities, sequence/germline alignments, and uncertainty columns are emitted
  (`src/alignair/io/airr.py:15-22`, `src/alignair/io/airr.py:80-103`). The gapped alignment path is
  real parasail alignment, not just coordinate decoration (`src/alignair/io/alignment.py:1-7`,
  `src/alignair/io/alignment.py:67-118`).
- Training is no longer just a research script. `alignair train` supports built-in GenAIRR
  references, custom V/D/J FASTAs, presets, fine-tuning, resume, calibration, validation reports,
  model cards, and self-contained bundles (`src/alignair/cli.py:322-357`,
  `src/alignair/cli.py:387-529`, `src/alignair/serialization/dnalignair_bundle.py:53-109`).
- Governance and engineering scaffolding are credible: CI tests Python 3.10-3.12 and Docker smoke
  (`.github/workflows/ci.yml:8-41`), the Dockerfile is now an `alignair` CPU entrypoint with a
  `doctor` healthcheck (`Dockerfile:1-38`), and docs/README now lead with dynamic reference,
  benchmarks, CLI, and training (`README.md:17-22`, `README.md:44-89`).

What is half-done or lower-quality than it looks:

- The "first run" still is not self-contained. The README tells users to run `--model
  human-igh-ogrdb` (`README.md:44-58`), the catalog advertises `human-igh-ogrdb` and will try to
  download `AlignAIR/human-igh-ogrdb` (`src/alignair/hub.py:13-20`, `src/alignair/hub.py:46-50`),
  but examples still say a model is "coming soon" and require a bundle/checkpoint
  (`examples/README.md:10-24`). Until a local demo model or tiny train-then-predict path exists,
  a new user can still bounce before seeing AlignAIR work.
- The docs are improved but inconsistent. `docs/getting_started.md` says `alignair model download`
  is on the roadmap (`docs/getting_started.md:21-37`), while the CLI and README now document the
  command as implemented (`src/alignair/cli.py:190-218`, `README.md:107-117`).
- Input handling is convenient, but not yet repertoire-scale. FASTQ and stdin are read fully into
  memory (`src/alignair/io/sequence_reader.py:23-36`, `src/alignair/io/sequence_reader.py:100-105`),
  and `predict` materializes all predictions before writing (`src/alignair/cli.py:114-128`). This
  is fine for demos, risky for multi-million-read AIRR-seq runs.
- "Schema-valid" is necessary but not the same as workflow-compatible. Immcantation/Change-O,
  10x Cell Ranger, nf-core/airrflow, and MiXCR users think in pipeline inputs, sample metadata,
  cell barcodes, UMI/duplicate counts, clone/QC summaries, and tool-to-tool comparison, not just a
  valid TSV. Immcantation explicitly supports IgBLAST-style annotation workflows, Change-O has a
  10x MakeDb path, nf-core/airrflow handles targeted bulk, 10x, assembled reads, and TCR/BCR
  inputs, Cell Ranger has `filtered_contig_annotations.csv`/`all_contig_annotations.csv`, and
  MiXCR users can export AIRR directly
  ([Immcantation intro](https://immcantation.readthedocs.io/en/stable/getting_started/intro-lab.html),
  [Change-O 10x example](https://changeo.readthedocs.io/en/latest/examples/10x.html),
  [nf-core/airrflow](https://github.com/nf-core/airrflow),
  [10x VDJ annotations](https://www.10xgenomics.com/support/software/cell-ranger/latest/analysis/outputs/cr-5p-outputs-annotations-vdj),
  [MiXCR exportAirr](https://mixcr.com/mixcr/reference/mixcr-exportAirr/)).
- The 4,400-case benchmark is strong, but it is still mostly a simulated-data trust story
  (`docs/benchmarks.md:1-43`). The caveat that AlignAIR is slower than IgBLAST is honest
  (`docs/benchmarks.md:38-43`), but a routine production user will still ask: "what happens on my
  10x/bulk/UMI/primer-biased real data, and how much RAM/GPU time will it cost?"
- The train-your-own path is powerful, but still expert-shaped. Users must know GenAIRR
  dataconfig names or chain-type strings, and the custom FASTA path can require
  `--allow-curatable` (`src/alignair/cli.py:330-357`, `src/alignair/cli.py:594-624`). There is no
  `alignair train plan`/dry-run explaining expected wall time, anchor problems, or target metrics
  before a long run.

### New Or Refined Recommendations

| Priority | Effort | Recommendation | Why It Drives Adoption |
| --- | --- | --- | --- |
| P0 | M | Build a no-network, no-published-model "first success" path. Either ship a deliberately tiny toy bundle in `examples/` or add a documented `alignair demo train-and-predict` recipe that trains a tiny smoke model, predicts on `examples/reads.fasta`, validates AIRR, and shows expected output. Fix docs so README, getting-started, examples, and hub catalog all tell the same truth until real models are published. | A user who cannot run one command successfully will not read the benchmark or try dynamic genotype. This is the biggest current bounce risk. |
| P0 | M | Add a "prove it on my data" comparison workflow: `alignair compare` or documented scripts that take AlignAIR AIRR TSV plus IgBLAST/MiXCR AIRR TSV, align rows by `sequence_id`, and emit an HTML/Markdown report: call agreement, D/J improvements, V disagreements, fragment/orientation cases, junction mismatches, uncertainty set rescue, runtime, and examples to inspect. | Users switching from IgBLAST or MiXCR need side-by-side evidence on their own repertoire, not only simulated aggregate claims. This turns skepticism into an evaluation path. |
| P0 | L | Make prediction repertoire-scale: chunked FASTA/FASTQ/table reading, chunked AIRR writing, bounded-memory stdin, multi-sample manifest input, resumable output, per-sample run summaries, and an optional `--max-reads`/`--sample` dry run. | AIRR users run large files and cohorts. Current buffered IO is good UX for small data but can fail silently as a production story. |
| P0 | M | Add real-data input adapters before adding more model-zoo polish: 10x `filtered_contig_annotations.csv`/`all_contig_annotations.csv`, AIRR input TSV with sample metadata, and a "sequence column cookbook" for pRESTO/Change-O/Immcantation exports. Preserve `cell_id`, barcode, UMI/duplicate/consensus counts, sample ID, and chain columns into output where present. | AlignAIR wins adoption by fitting existing workflows. 10x and Immcantation users should not have to reshape common files by hand. |
| P0 | M | Turn dynamic genotype into a killer guided workflow: `alignair genotype-template`, `alignair reference validate --explain`, a donor-genotype example with one novel allele, and a before/after report comparing full-reference vs donor-reference calls and uncertainty set sizes. | Dynamic reference is the unique reason to choose AlignAIR over classical tools. It needs to be visible as a workflow, not just a flag. |
| P1 | M | Add a training preflight: `alignair train plan --reference ...` and `alignair train plan --v-fasta ...` should list inferred locus, V/D/J counts, anchor coverage, expected preset wall time/GPU memory, output files, calibration cost, and likely failure modes before training starts. | Train-your-own is the differentiator, but without a dry-run plan it feels risky and expert-only. |
| P1 | M | Build a real-data validation matrix now, even if private: public OAS/10x/bulk datasets where licenses allow local recipes, plus internal reports for 10x BCR, 10x TCR, bulk IGH, short amplicons, noisy/primer-heavy reads, non-human references, and custom FASTA training. Track agreement with IgBLAST/MiXCR and where uncertainty is useful. | Wet-lab users distrust simulation-only claims. A matrix of known-good and known-bad real regimes is more persuasive than another aggregate benchmark. |
| P1 | M | Add an "actionable uncertainty" layer: docs and optional summary columns/report that translate `*_call_set`, `*_call_level`, and `*_set_confidence` into recommendations: accept allele call, collapse to gene, abstain, inspect read, or use donor genotype. Include thresholds used in benchmarks. | Calibrated uncertainty is unfamiliar to many AIRR users. If they do not know how to act on it, they may treat it as noise. |
| P1 | L | Improve throughput enough that the speed caveat stops dominating: batch auto-tuning, profiler output in `run.json`, separate fast presets (`--fast`, `--no-full-alignment`, `--v-reader parasail`), CPU multiprocessing for parasail-heavy steps, `torch.compile`/ONNX experiments, and a published internal speed table by read length/reference size. | The 2x deficit vs IgBLAST is not a launch blocker for accuracy/dynamic-genotype use cases, but it is a blocker for default replacement in routine high-throughput pipelines. |
| P1 | M | Strengthen model/reference compatibility guardrails. Turn locus mismatch into a hard error by default with `--force-locus-mismatch`, and add bundle metadata for supported locus, reference species/source, max length, expected assay/read regime, training preset, and calibration regime. | A plausible-looking wrong run destroys trust faster than a clear refusal. The current warning is useful but too easy to miss in pipelines (`src/alignair/cli.py:90-99`). |
| P1 | M | Add downstream field coverage consciously. Create a matrix of AIRR/Change-O/Alakazam/Scirpy/common QC fields and decide what AlignAIR can emit now: `c_call` when available, `vj_in_frame`, `stop_codon`, `np1`/`np2`, CDR/FWR regions, `duplicate_count`, `consensus_count`, `cell_id`, `sample_id`, and per-record QC flags. | "Reads back with Change-O" is a baseline. Adoption comes when downstream clone/QC workflows do not need custom glue code. |
| P1 | M | Add local workflow templates: a small Nextflow/Snakemake example and a Galaxy wrapper draft that run local AlignAIR over a manifest and validate output. Do not publish them yet; use them to discover CLI and output gaps. | Wrapper work forces workflow-fit discipline and prepares for nf-core/Galaxy later without making release publishing the objective. |
| P2 | S | Clean up copy and command consistency. Remove "coming soon" for implemented commands, mark model catalog entries as "placeholder/unavailable" unless a local path is used, and make `examples/README.md` match `README.md` and `docs/getting_started.md`. | This is small but high-trust: inconsistent docs make users wonder whether the benchmark claims are also stale. |
| P2 | M | Add a "known failure modes" page tied to examples: exact junction-nt deficit, custom FASTA missing anchors causing empty junctions, short-read V ambiguity, no-D loci, locus mismatches, contaminants, and when to prefer IgBLAST/MiXCR. | Honest limitations increase trust and help users choose AlignAIR for the cases where it is strongest. |

### Top 5 Pre-Release Adoption Moves

1. Build a local runnable demo that does not depend on a published model: train tiny, predict,
   validate AIRR, and show the dynamic-genotype path.
2. Add an IgBLAST/MiXCR comparison report for users' own data, because that is how most switchers
   will decide.
3. Make `predict` scale to real repertoires: chunked IO, bounded memory, multi-sample manifests,
   resumable output, and per-sample summaries.
4. Add 10x/Cell Ranger and Immcantation/Change-O workflow adapters plus downstream metadata
   preservation.
5. Turn dynamic genotype and train-your-own into guided workflows with templates, preflight plans,
   and concrete before/after reports.
