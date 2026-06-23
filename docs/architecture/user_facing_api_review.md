# AlignAIR User-Facing API and Adoption Review

Date: 2026-06-23

Scope: CLI, Python package API, model loading, prediction IO, AIRR output, training UX,
Docker/packaging, docs, and workflow fit for AIRR community users. This review combines a
local audit with an independent sub-agent review. No feature implementation is included here.

## Executive Judgment

AlignAIR is now a credible pre-release command-line product, not just a research codebase.
The current CLI covers the important user journeys: offline demo, prediction, AIRR validation,
tool-to-tool comparison, reference validation/conversion/template generation, model listing,
training, bundling, and environment diagnosis.

It is not yet industry-grade as a public Python library or as a drop-in AIRR workflow component.
The biggest remaining risks are:

1. The public Python API is effectively undefined.
2. Public docs are split between current CLI docs and stale TensorFlow/notebook pages that still
   live under `docs/`.
3. Model/reference mismatch can still produce plausible but wrong output because it is only a
   warning.
4. AIRR TSV validity is good, but downstream workflow compatibility is not yet proven across
   10x, Change-O/Immcantation, Scirpy, MiXCR, and nf-core-style pipelines.
5. There is no published production model, so the first-run story depends on demo/training.

## What Is Strong Now

| Surface | Current state | Evidence |
| --- | --- | --- |
| CLI coverage | Strong. Top-level commands now cover `demo`, `predict`, `train`, `model`, `reference`, `compare`, `validate-airr`, `doctor`, and `bundle`. | `src/alignair/cli.py` |
| First run | Much improved. `alignair demo` runs offline, trains a tiny model, predicts, validates AIRR, and exercises genotype mode. | `src/alignair/cli.py:639` |
| Prediction IO | Strong baseline. FASTA/FASTQ/CSV/TSV/TXT, gzip, stdin/stdout, chunked streaming, quiet mode, stderr progress, provenance sidecar, and metadata join are present. | `src/alignair/cli.py:46`, `src/alignair/io/sequence_reader.py:83`, `src/alignair/io/sequence_reader.py:169` |
| AIRR output | Strong baseline. Core AIRR fields, canonical orientation, CIGARs, identities, coordinates, uncertainty extensions, and incremental writer are implemented. | `src/alignair/io/airr.py:15`, `src/alignair/io/airr.py:97` |
| Dynamic reference | Strong differentiator. Runtime YAML/FASTA genotype, dataconfig references, subset references, and custom FASTA training are supported. | `src/alignair/reference/reference_set.py`, `src/alignair/cli.py:64` |
| Train-your-own | Strong pre-release UX. Built-in reference, custom FASTA, presets, resume, base model, dry-run plan, calibration, validation report, model card, and bundle output exist. | `src/alignair/cli.py:459` |
| Tool comparison | Useful adoption path. `alignair compare` lets users compare AlignAIR against IgBLAST/MiXCR AIRR TSVs by `sequence_id` with set-rescue reporting. | `src/alignair/compare.py`, `src/alignair/cli.py:205` |
| Docker | Credible CPU image. It uses `alignair` as entrypoint, non-root user, CPU PyTorch, examples, and `doctor` healthcheck. | `Dockerfile` |
| Governance | Credible baseline. CI, release workflow, changelog, citation, contributing, security policy, and Docker smoke exist. | `.github/workflows/ci.yml`, `CITATION.cff` |

## Verified Locally

Commands run from `/home/thomas/Desktop/AlignAIR`:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.cli --help
PYTHONPATH=src .venv/bin/python -m alignair.cli doctor
PYTHONPATH=src .venv/bin/python -m alignair.cli model list
PYTHONPATH=src .venv/bin/python -m alignair.cli train --reference HUMAN_IGH_OGRDB \
  -o /tmp/alignair_plan_review --preset smoke --steps 1 --plan --device cpu
PYTHONPATH=src .venv/bin/python -m alignair.cli demo --steps 1 \
  -o /tmp/alignair_demo_review --device cpu
PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/test_cli_io.py \
  tests/alignair/test_compare.py -q
PYTHONPATH=src .venv/bin/python -m mkdocs build --strict
```

Results:

- CLI help and `doctor` worked. `doctor` reported Python 3.12, Torch 2.11 CUDA available, GenAIRR
  2.2.0, and parasail present.
- `model list` truthfully reports `human-igh-ogrdb` as `not published`.
- `train --plan` gives a real dry run with reference counts, anchor coverage, model size, timing,
  outputs, and warnings.
- `demo --steps 1 --device cpu` completed in about 18 seconds locally and validated
  `/tmp/alignair_demo_review/demo.tsv` as AIRR-C rearrangement output.
- Focused CLI/IO tests passed: `30 passed in 1.31s`.
- Strict MkDocs build initially failed on missing nav targets and two outside-docs links; this pass
  fixed those blockers and `mkdocs build --strict` now passes. MkDocs still discovers legacy
  notebook/pipeline pages outside the nav, so stale-public-doc cleanup remains necessary.

## Adoption Gaps

### P0 - Fix Before Broad External Adoption

| Priority | Effort | Recommendation | Why it matters |
| --- | --- | --- | --- |
| P0 | M | Define a stable public Python API. Add exports such as `alignair.load_model`, `alignair.predict`, `alignair.read_sequences`, `alignair.write_airr`, `alignair.ReferenceSet`, and typed result objects. Keep CLI orchestration as a client of that API. | `src/alignair/__init__.py` currently exports nothing. Notebook users, workflow authors, and downstream tools should not have to import private CLI helpers. |
| P0 | M | Finish the public docs cleanup. Remove or quarantine stale TensorFlow/legacy notebook and old pipeline pages from the published docs output, keep the current CLI pages in nav, and keep `mkdocs build --strict` passing. | The README and nav are current now, but legacy docs still live under `docs/` and are discovered by MkDocs. Stale docs reduce trust faster than missing features. |
| P0 | S | Harden model/reference compatibility. Turn bundle locus/reference mismatches into an error by default and add `--force-locus-mismatch` for expert override. Add bundle metadata for supported locus, has-D, max length, training reference, assay/read regime, and calibration regime. | A warning is too easy to miss in batch pipelines; wrong-locus output can look valid while being biologically meaningless. |
| P0 | M | Add real workflow fixtures and tests: tiny 10x `filtered_contig_annotations.csv`, AIRR input TSV with sample metadata, expected output TSV, and downstream readback checks. | Metadata preservation exists, but AIRR users need proof that Cell Ranger/10x and Immcantation-style data survive end to end. |
| P0 | S | Make unavailable model IDs impossible to misuse. `alignair model list` should separate "available now" from "planned" and avoid showing a "use directly" hint for unpublished IDs. | The current text is honest but still implies a direct-use path immediately after listing an unavailable catalog entry. |
| P0 | M | Add a release smoke that runs the same first-run path for wheel and Docker: `doctor`, `demo --steps 1`, `validate-airr`, and a small `compare` report. | AIRR Software WG guidance expects example data, expected-output checks, run parameters, and containerized execution as part of tool quality. |

### P1 - Needed For Industry-Grade Workflow Fit

| Priority | Effort | Recommendation | Why it matters |
| --- | --- | --- | --- |
| P1 | M | Add a downstream field coverage matrix and choose what AlignAIR will emit now: `c_call`, `vj_in_frame`, `stop_codon`, `np1`/`np2`, CDR/FWR regions, duplicate/consensus counts, sample/cell IDs, and QC flags. | AIRR-valid output is necessary, but Change-O, Alakazam, Scirpy, and nf-core users also care about common downstream fields. |
| P1 | M | Add local workflow-wrapper drafts for Nextflow, Snakemake, and Galaxy around `alignair predict`; keep them internal until stable. | Wrappers expose rough edges around manifests, paths, exit codes, logs, containers, and multi-sample output. |
| P1 | M | Add multi-sample manifest support and per-sample run summaries. | AIRR users usually process cohorts, not one file. A manifest mode is easier to wrap and audit than repeated shell loops. |
| P1 | S | Align the CLI implementation with its dependency story. Either use Typer/Rich for polished help/tables/errors or remove those extras. Add `--json`, `--verbose`, and shell completion where useful. | The CLI works, but it is plain `argparse` while `pyproject.toml` advertises modern CLI dependencies. |
| P1 | M | Expand provenance in both run JSON and bundle metadata: commit SHA, package versions, reference hash/source/version, calibration hash, training seed/config, device/CUDA details, and benchmark/report IDs. | Current provenance is useful, but not enough for reproducible scientific release artifacts. |
| P1 | M | Make conda/BioContainers readiness real. Replace the placeholder SHA in `conda/meta.yaml`, add a local recipe build/test job, and smoke `alignair doctor` plus a tiny demo. | Bioconda is the path to BioContainers and many Galaxy/nf-core deployments. |
| P1 | M | Improve `train --plan`: list valid GenAIRR dataconfigs/chain types, explain anchor problems, estimate GPU memory, and show expected validation thresholds for each preset. | Train-your-own is the differentiator; planning needs to feel predictable before users spend GPU hours. |
| P1 | M | Publish a hardware/performance table for prediction and training: CPU, one GPU, common read lengths, reference sizes, `--no-full-alignment`, and `--v-reader parasail`. | Docs honestly say AlignAIR is slower than IgBLAST in the benchmark. Users need concrete resource expectations. |

### P2 - Adoption Polish And Ecosystem Expansion

| Priority | Effort | Recommendation | Why it matters |
| --- | --- | --- | --- |
| P2 | M | Add `alignair reference fetch ogrdb` or an equivalent OGRDB/AIRR GermlineSet import path with source/version provenance. | OGRDB and AIRR GermlineSet/GenotypeSet formats are natural reference sources for the community. |
| P2 | M | Maintain two notebooks: pretrained inference and custom-reference training smoke. Add Colab only if model sizes and install time are acceptable. | Notebooks are valuable for evaluation, but stale notebooks are worse than no notebooks. |
| P2 | S | Add a "known failure modes" page: junction exactness, missing anchors, short-read ambiguity, no-D loci, contaminants, model/reference mismatch, and when to prefer IgBLAST/MiXCR. | Honest limitations improve trust and support quality. |
| P2 | M | Add an examples gallery: human IGH OGRDB, IGK/IGL, TCRB, 10x BCR, 10x TCR, custom FASTA, novel allele, and tool comparison. | AIRR users adopt faster when they see a workflow close to their own experiment. |

## Suggested Public API Shape

The CLI should remain the primary beginner path, but the library needs a stable import surface.
One possible minimal API:

```python
from alignair import load_model, predict, read_sequences, write_airr, ReferenceSet

bundle = load_model("runs/my_model/bundle", device="cuda")
reference = ReferenceSet.from_yaml("donor.yaml")
ids, reads = read_sequences("reads.fastq")
result = predict(bundle, reads, reference=reference, batch_size=128)
write_airr("out.tsv", ids, result.sequences, result.predictions, locus=result.locus)
```

Design constraints:

- Typed result objects, not loose dicts only.
- Same behavior as CLI for genotype handling, canonical orientation, calibration, and full alignment.
- Explicit stability policy: public API lives under `alignair.*`; implementation details stay under
  `alignair.inference`, `alignair.io`, `alignair.serialization`, etc.
- CLI tests should assert parity with Python API tests on the same tiny fixture.

## Ecosystem Expectations Used For This Review

- AIRR Software WG guidance asks tools to support standard AIRR formats, include example data and
  automated checks, provide run parameters as output, and ship a remotely buildable container:
  https://docs.airr-community.org/en/stable/swtools/airr_swtools_standard.html
- AIRR Rearrangement is the standard annotated sequence schema AlignAIR is targeting:
  https://docs.airr-community.org/en/latest/datarep/rearrangements.html
- AIRR Python reference library provides read/write/validation functions:
  https://docs.airr-community.org/en/latest/packages/airr-python/overview.html
- Change-O documents a 10x path where `filtered_contig_annotations.csv` is incorporated into AIRR
  Rearrangement output:
  https://changeo.readthedocs.io/en/latest/examples/10x.html
- nf-core/airrflow is the workflow bar for AIRR adoption: Nextflow, containers, bulk and single-cell
  AIRR-seq, AIRR rearrangement input, and test profiles:
  https://nf-co.re/airrflow/dev/ and
  https://nf-co.re/airrflow/dev/docs/usage/single_cell_tutorial
- MiXCR exports AIRR-formatted TSV via `mixcr exportAirr`, making it a natural comparison input:
  https://mixcr.com/mixcr/reference/mixcr-exportAirr/

## Recommended Next Sprint

1. Stabilize the Python API and make the CLI call it.
2. Quarantine stale notebook/legacy pages from the published docs output while keeping
   `mkdocs build --strict` passing.
3. Add 10x/AIRR metadata fixtures with downstream readback tests.
4. Make model/reference mismatch a default error.
5. Add wheel and Docker smoke tests that run `demo --steps 1`, validate output, and compare two tiny
   AIRR TSVs.

This would move AlignAIR from "promising and locally usable" to "credible for outside AIRR users to
try without handholding."
