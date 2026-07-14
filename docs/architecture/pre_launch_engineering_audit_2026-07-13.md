# AlignAIR Modernization: Pre-Launch Engineering Audit and Action Plan

**Audit date:** 2026-07-13  
**Review posture:** senior deep-learning, scientific-software, packaging, and public-API review  
**Audited state:** the current working tree, including uncommitted modernization work  
**Primary objective:** make the code and product surface production-ready before the documentation phase

## Executive verdict

**Current release recommendation: NO-GO for a public production launch.**

AlignAIR contains a promising PyTorch implementation, a strong safe model-container foundation,
useful AIRR post-processing, a sizeable test suite, and the beginnings of a verified model registry.
The repository is substantially beyond a research prototype. However, several issues can produce
scientifically incorrect or misleading output, and the implementation currently does not match the
product described by the README, workflows, CI, Docker health check, or previous architecture
reviews.

The most important issue is architectural, not cosmetic: the active `AlignAIR` model uses
fixed-size V/D/J classification heads whose output indices are tied to the ordered training
reference. It has no runtime germline-reference encoder. It can mask calls to a subset of the
training alleles, but it cannot add a novel allele as a callable class at inference time. The CLI
explicitly drops unknown genotype alleles. Therefore the current claims that the reference is not
baked into the weights and that unseen alleles are embedded and called at runtime are not true for
this code path.

This must be resolved before launch in one of two ways:

1. **Recommended product direction:** finish and make the genuinely reference-conditioned model the
   production architecture. Allele scoring must be a read/reference compatibility operation over a
   runtime allele set, with tests proving zero-shot novel-allele behavior.
2. **Smaller interim release:** explicitly define this version as a fixed-reference classifier.
   Support donor subsets safely, require training/fine-tuning to add alleles/species, and remove all
   runtime-novel-reference claims from code comments, metadata, workflows, and product material.

Do not try to hide this distinction behind documentation. It controls model semantics, artifact
compatibility, API design, benchmark interpretation, and the core adoption promise.

## What is already worth keeping

- The `.alignair` container separates safe JSON/safetensors/reference sections from explicitly
  trusted pickle sections, checks section hashes, and supports pickle-free distribution artifacts.
- Embedded ordered reference metadata and allele-order/reference hashes are the right safeguards
  for fixed-head models.
- Registry downloads are streamed, hash-checked, and atomically installed.
- Legacy `.pt` loading is denied by default in the main loading API and requires explicit trust.
- The inference pipeline has clear stages for forward inference, cleanup, segment correction,
  allele selection, germline alignment, and AIRR assembly.
- AIRR output already contains a broad potential field set, canonical-orientation machinery,
  exact/derived CIGAR support, junction logic, and uncertainty-extension slots.
- GenAIRR-backed online generation, curricula, model diagnostics, calibration work, and the
  benchmark package provide strong ingredients for a serious validation program.
- `pytest` currently passes **514 tests** on the audited Linux/Python 3.12 environment.
- A clean sdist and native Linux CPython 3.12 wheel build successfully and pass `twine check`.

These are good foundations. The launch work should consolidate them behind one truthful product
contract rather than replace them wholesale.

## Launch scorecard

| Area | Current judgment | Launch status |
| --- | --- | --- |
| Scientific model contract | Fixed allele heads conflict with runtime-reference and novel-allele claims | **P0 blocker** |
| Prediction correctness | Canonical sequence/output mismatch, possible out-of-bounds segments, silent AIRR fallbacks | **P0 blocker** |
| Multi-locus correctness | Locus mapping is not propagated; global `has_d` drives assembly; cross-locus calls are unconstrained | **P0 blocker** |
| Genotype constraints | Unknown alleles are dropped; empty/all-zero masks can still call a disallowed allele | **P0 blocker** |
| CLI/product contract | README, CI, Docker, workflows, and actual commands disagree | **P0 blocker** |
| Public Python API | Thin tuple/list API; no registry-aware load, typed results, stable config, or custom-reference workflow | **P0 blocker** |
| Model format safety | Good base; needs compatibility policy, resource limits, privacy cleanup, and stronger publish gate | P1 |
| Hugging Face distribution | Custom public-file registry exists; no direct repo API/private auth/resume/remote publish flow | **P0 for model launch** |
| Linux packaging | Source and local wheel build work | P1 hardening |
| macOS/Windows support | Binary extension smoke only; no full runtime/training/inference validation | **P0 for cross-platform claim** |
| AIRR interoperability | Broad fields, but semantics/completeness and downstream compatibility are not release-gated | **P0 blocker** |
| Tests | Large suite, but missing key product invariants and release-contract tests | **P0 blocker** |
| Release automation | Current CI/release smoke invokes commands that do not exist | **P0 blocker** |
| Performance/robustness | No production resource envelopes, adaptive batching, or failure budgets | P1 |

## P0 release gates

Every item in this section should be closed before publishing production models or calling the
package production-ready.

### P0-1 — Freeze a truthful model and reference contract

**Observed implementation**

- `AlignAIRConfig` stores fixed `v_allele_count`, `d_allele_count`, and `j_allele_count` values.
- `GeneBranch` constructs fixed output heads from those counts.
- `predict.pipeline` maps output column indices to `ReferenceSet` allele names.
- The runtime model forward pass receives sequence tokens only. No germline allele tensor or
  reference embedding is supplied.
- `load_genotype(..., drop_unknown=True)` removes alleles absent from the embedded training
  reference.
- `api.load_model` accepts a caller-supplied `reference` for a safe `.alignair` model and returns it
  instead of the verified embedded reference without checking allele counts, order, or hashes. For a
  fixed-head architecture this can mislabel output indices or fail late in prediction, bypassing the
  otherwise strong embedded-reference integrity check.

**Risk**

A user can reasonably believe that a novel allele in YAML/FASTA becomes callable without training,
when it cannot. This is a scientific-validity problem and the largest adoption risk in the package.

**Required action** — 🟡 PARTIAL (2026-07-13): decision-independent correctness fixes landed; the
architecture DECISION (fixed-reference vs runtime-reference product) still needs the user.

DONE (safe under either eventual direction, correct for the model that exists today):
- `load_model` now validates any caller-supplied `reference` against the embedded fixed head via
  `_assert_reference_matches` (rejects reordered/extra/missing/mismatched-gene-set references) so
  output columns can no longer be mislabeled — the verified embedded reference is the default.
- Typed `NovelAlleleUnsupportedError` (a `ValueError` subclass) raised from the genotype path when a
  file names alleles outside the trained catalog; the CLI rejects rather than drops them (with P0-5).
- Corrected the misleading "novel alleles are rows the encoder embeds at predict time" comments in
  `reference_set.from_genotype`/`from_fasta` — they now state the fixed-head order contract.
- Tests: `test_api_reference_validation.py`, `test_constraint.py`.

CONTRACT FROZEN (2026-07-13): **fixed-reference classifier (v1)** — `docs/architecture/model_contract.md`
(architecture contract versioned separately from the container format). Black-box acceptance test
`tests/alignair/test_fixed_reference_contract.py` proves a novel allele fails with
`NovelAlleleUnsupportedError` and a donor subset of the trained catalog is accepted. *Remaining (P1):*
stamp an `architecture_version` into model metadata + refuse to load a mismatched contract.

### P0-2 — Make CLI, package, workflows, container, and CI one product — ✅ MOSTLY DONE (2026-07-13)

Implemented the 6 missing commands the README/CI/Docker/release-smoke depend on, unifying the surface:
`--version` (top-level), `doctor` (env diagnostics — Docker health check + smoke exit 0), `demo` (tiny
train→predict→bundle→donor, no network), `validate-airr` (AIRR TSV column/coordinate/CIGAR validation),
`compare` (wraps `compare_airr`), `reference list`/`export` (+ `model` alias of `models`). The full
`scripts/release_smoke.sh` now runs green end-to-end from the source CLI. Parser-contract test
(`test_cli_contract.py`) asserts the exact command surface + `--help`/`--version` parse; command
behavior tests in `test_new_commands.py`. `validate-airr` caught a real empty-segment coordinate bug
(fixed — see P0-3). README `bundle` (phantom) row corrected to `convert`.

*Remaining:* run the contract suite against the **installed wheel** (not `PYTHONPATH=src`) on CI; a CI
step that extracts+dry-runs shell commands from README; quarantine the legacy `src/AlignAIR` CLI;
reconcile remaining README prose (e.g. the `reference` description) in the docs phase.

### P0-3 — Correct canonical orientation and AIRR sequence/coordinate consistency — ✅ DONE (2026-07-13; orientation contract corrected 2026-07-14)

**Orientation correction (AIRR-community review):** the earlier fix emitted `sequence`=canonical with
`rev_comp=T`, which is inconsistent — AIRR defines `rev_comp=T` to mean coordinates are on
`RC(sequence)`. Now, for a reverse-complement read, `sequence` is the **original query** and coords are
on `RC(sequence)`==the aligned frame (IgBLAST/AIRR convention); complement/reverse-only emit the
canonical frame with `rev_comp=F` and the true transform in `orientation`. Round-trip test proves
`RC(sequence)`==coordinate frame (`test_airr_writer.py`).


`_build_row` now emits the record's own canonical `sequence` (`p["sequence"]`) and takes every slice
from it — never the parallel input list, which is now treated purely as the original read and preserved
in an `input_sequence` extension. AIRR `rev_comp` is set **only** for a true reverse-complement
(orientation id 1); complement-only/reverse-only keep `rev_comp=F` with the full transform in a new
`orientation` field. `segmentation_low_quality` surfaced as a T/F extension. Absent/zero-length segments (e.g. Short-D) now
emit **blank** coords/CIGAR instead of an invalid AIRR `start>end` span (bug found by `validate-airr`).
Tests cover the reoriented-read invariant + all four transforms in `test_airr_writer.py`; per-CIGAR
query-consumption ≤ len(sequence) is enforced by `validate-airr` (P0-2). *Remaining:* official AIRR
readback is folded into P0-14.

### P0-4 — Guarantee bounded, ordered segment coordinates — ✅ DONE (2026-07-13)

`correct_segments` rewritten as a constrained projection (non-decreasing boundary chain capped at
`seq_len`): `0 <= start <= end <= seq_len` and V/D/J ordering hold jointly by construction; squeezed
segments collapse to absent (zero-length) instead of manufacturing one-base spans past the read. V-anchor
collapse is reported via `Segments.low_quality` and surfaced as `segmentation_low_quality` on records.
Property test over random logits/lengths + edge cases (len 1/2, no-D, overflow reproduction) in
`tests/alignair/predict/test_segment.py`. *Remaining:* AIRR-row rejection gate is folded into P0-7/P0-14.

### P0-5 — Make genotype constraints mathematically safe — ✅ DONE (2026-07-13)

`genotype_allowed_mask()` validates every constrained gene keeps ≥1 supported allele and raises before
inference otherwise (all-novel file → actionable error). `select_alleles(..., allowed=mask)` now selects
over the explicit allowed-index set, so a constrained call is **always** a member of the allowed set —
never a disallowed argmax/index-0 fallback from zeroed probabilities. Empty-set softmax NaN eliminated;
`_assert_finite` guards run after calibration and after every constraint transform. CLI rejects novel
alleles (`drop_unknown=False`) instead of silently dropping them. Tests in `test_threshold.py` +
`test_constraint.py` (allowed-set restriction, all-zero-probs, empty-set raise, finiteness for every
method). *Remaining:* explicit `no_call` when the best allowed allele is below a quality floor (deferred;
selection currently keeps top-1-within-allowed).

### P0-6 — Fix multi-locus and no-D semantics before advertising unified models — ✅ DONE (2026-07-13)

`ReferenceSet` now carries a versioned per-locus schema (`loci`: ordered AIRR locus, chain type, has-D,
and per-gene `[start,end)` allele-index ranges), derived at build and persisted in `reference_json`; for
models saved before the schema existed it is **inferred from allele-name prefixes** (`IGKV…→IGK`), so
existing artifacts get it with no re-save. The public API propagates the ordered loci into
`PredictConfig.chain_types`; the pipeline builds a **per-read** locus-allowed mask (`_locus_allowed`) and
`select_alleles` selects only within it (2-D per-read masks; empty allowed set → explicit no-call), so a
cross-locus call — an `IGKV` allele on an `IGL` read, or any D on a light-chain read — is **impossible by
construction**. AIRR assembly uses each record's locus-specific has-D. Single-chain records get their one
locus label (no silent `IGH`); the CLI validates `--locus` against the model's loci and records the locus
mapping in `<out>.run.json`. **Verified end-to-end on the real IGK+IGL model: 24/24 correct locus, 0
cross-locus V calls.** Tests: `test_locus_schema.py`, `test_multichain_locus.py`, `test_threshold.py`.

### P0-7 — Stop swallowing AIRR assembly failures — ✅ DONE (2026-07-13)

`build_airr` now catches only expected data-edge exceptions (`ValueError/IndexError/KeyError`), tagging
those rows with `airr_assembly_status="failed"` + an `airr_assembly_error` reason while preserving the
light record; any other exception re-raises as `AirrAssemblyError` with the record identifier (never
silent — surfaces programming defects). `strict=True` gates release fixtures. Successful rows are tagged
`"ok"`. The CLI accounts failures, records counts+reasons in `<out>.run.json`, and **fails the job** when
the failure rate exceeds `--max-assembly-failures` (default 1%) unless `--permissive`. Tests in
`test_airr.py`. *Remaining:* official `airr`-package validation of release fixtures (P0-14).

### P0-8 — Define sequence length, ambiguity, and malformed-input policies — ✅ MOSTLY DONE (2026-07-13)

Single input gate `apply_input_policy()` in `predict()` (reached by both CLI and Python API): uppercases,
**rejects empty** reads loudly, and crops over-length reads to the model window *consistently* — the
cropped string is what the tokenizer, coords, germline reader and AIRR all see, so no silent truncation
with mismatched coordinates; cropped rows are flagged `length_cropped`. Content validator unified as
`validate_sequence(seq, max_len) -> (cleaned, reason)`. Strict FASTQ parser with line-numbered errors
(`@`/`+` structure, seq==qual length, truncation). Duplicate IDs disambiguated deterministically with
preserved input order. AIRR output written **atomically** (temp + rename; discarded on error) — tests in
`test_sequence_reader.py` + `test_airr_writer.py`. **Streaming (2026-07-14): the predict CLI now streams
reader-chunk → predict → AIRR assembly → metadata join → writer → counters in bounded memory
(`_stream_predict`, `--chunk-size` default 20000); order + cross-chunk dup-id preserved, rejects streamed
incrementally + atomically, and full run.json accounting (input/accepted/rejected/cropped/complete/
partial/failed/written). Order/chunk + bounded-memory tests in `test_streaming_predict.py`.** *Remaining:*
a live 1M-read CI memory test (the unit test proves chunked processing + bounded peak locally).

### P0-9 — Establish a stable, registry-aware Python API — 🟡 MOSTLY DONE (2026-07-13)

New `aligner.py` is the stable object API (exported from `alignair`): **`Aligner.from_pretrained`** unifies
local path / catalog id / `id@revision` loading (via the registry resolver) + **`device="auto"`** (CUDA→
MPS→CPU with fallback via `resolve_device`); **`aligner.predict(...)`** returns a typed
**`PredictionResult`** (iterable/`len`/`to_dicts`/`write_airr`, no duplicate sequence arg) and
**`predict_iter`** streams in bounded memory. Typed **`TrainingConfig.from_genairr(..., preset=…)`** +
**`train(config, output_dir)`** → **`TrainingRun.best_aligner()`** replaces unrestricted `**overrides`.
The **predict CLI is now a thin client of `Aligner`**; a CLI/API parity test asserts identical output on
the same fixture. Public surface is snapshot-tested; the functional façade stays for back-compat. Tests:
`test_aligner.py`, `test_cli_api_parity.py`.

*Remaining:* direct HF `org/repo` loading (P0-11) behind `from_pretrained`; static type-checking gate
(mypy) over the public surface; move the `train` CLI onto `TrainingConfig` too; `save_pretrained`.

The current public API returns `(model, reference)` tuples and lists of loose dictionaries. Its
`load_model` accepts local paths only, while the CLI separately resolves registry IDs. Training is a
single function with unrestricted `**train_overrides`, defaults to 100,000 CPU steps, and does not
provide the custom FASTA/preset/validation/bundle workflow described elsewhere.

**Required action**

Adopt a small object API and make the CLI a thin client of it. A suitable target is:

```python
from alignair import Aligner, TrainingConfig, run_training  # `train` would collide with the train pkg

aligner = Aligner.from_pretrained(
    "AlignAIR/human-igh-ogrdb",
    revision="v1.0.0",
    device="auto",
)

result = aligner.predict(
    ["CAGGTGCAGCTG..."],
    reference="donor.yaml",       # only if supported by the model contract
    batch_size="auto",
)
result.write_airr("predictions.tsv")

run = run_training(
    TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", preset="desktop"),
    output_dir="runs/human_igh",
)
aligner = run.best_model
```

Required API properties:

- typed immutable configuration and result models;
- structured warnings/errors rather than `print` plus integer codes;
- `device="auto"` with CPU, CUDA, and Apple MPS support and explicit fallback reporting;
- local path, catalog alias, and direct HF repo/revision loading through the same method;
- sync streaming/iterator prediction and eager convenience prediction;
- no duplicate sequence/reference arguments;
- explicit public namespace and semantic-version/deprecation policy;
- CLI/Python parity tests using identical fixtures.

**Acceptance criteria**

- The primary use, custom-train, resume, offline-load, and custom-reference journeys take no private
  imports.
- Static type checking covers the public surface.
- Public signatures are snapshot-tested and documented as stable for the release series.

### P0-10 — Make training resumable, reproducible, validated, and difficult to misuse — 🟡 MOSTLY DONE (2026-07-13)

New `train/guards.py`: `validate_training_request` runs a **preflight** check (steps/batch/lr/max_len/
progresses/heavy_shm/short_boost/grad_clip + non-empty V/J reference) in `api.train_model` **before the
model is allocated** and at the top of `train()`; `check_finite_loss` **aborts on NaN/Inf** (`train_step`,
before backward corrupts grads) with a per-task diagnostic. `train_step` measures the **gradient norm**
(logged) and clips to `--grad-clip` when set. Resume now **restores Python/NumPy/Torch/CUDA RNG**
(`_restore_rng`); documented as *statistically* (not bitwise) reproducible since the GenAIRR stream is
re-seeded by `seed+start`. A fixed-seed **validation loop** (`validate`) runs every `--val-every` steps
and writes a **best checkpoint** (`<out>.best.alignair`) by mean V/D/J top-1. Legacy `.pt` resume now
requires `--resume-trust-pickle` (arbitrary-code pickle gate). CLI help fixed to `.alignair`. Tests:
`test_guards.py` + `test_trainer_robustness.py` (incl. slow end-to-end val/best/resume).

*Remaining:* OOM auto-recovery; a frozen scientific acceptance suite as a release gate; resource-estimate
presets; per-model published model-card completeness (calibration/validation/provenance) — ties P0-16.

### P0-11 — Ship a real Hugging Face model experience — 🟡 MOSTLY DONE (2026-07-13)

New `registry/hf.py` adds the **one-repo-per-model** path via `huggingface_hub`:
`Aligner.from_pretrained("hf://org/repo")` / `"org/repo"` pulls the `.alignair` with `hf_hub_download`
— revision (branch/tag/commit) pinning, tokens (arg or `$HF_TOKEN`) for private/gated repos, offline
(`local_files_only`), the standard HF cache, retries/resumable transfers, a user-agent. Both loaders
funnel through `resolve_model` (catalog aliases stay a convenience). The resolved HF **commit SHA** is
captured on the aligner (`source_commit`) for provenance. `Aligner.save_pretrained` (copy artifact) and
maintainer-only `push_to_hub` added; the CLI gained `--revision`/`--hf-token`. **Publishing is now
transactional** — `publish_local` stages + validates before committing, so a failed validation leaves
neither an updated catalog nor a copied artifact (was: wrote before validating). Tests: `test_hf.py`
(spec parse/route/offline/commit/save, no network) + transactional-publish test.

*Remaining (maintainer/CI-verified):* real private-token + pinned-revision + interrupted-download live
tests; min/max AlignAIR/format/architecture version enforcement on download; publishing the full
card/manifest/validation+benchmark-report bundle.

**Required action**

- Use `huggingface_hub` for the supported HF path: `hf_hub_download`/`snapshot_download`, tokens,
  revisions, local-files-only mode, standard cache, retries, resumable transfers, and user agent.
- Prefer one HF repository per production model or define a rigorously versioned central catalog.
- Support immutable commit SHA/release tag pinning; store resolved commit in run provenance.
- Implement `Aligner.from_pretrained`, `save_pretrained`, and a maintainer-only `push_to_hub` or
  release command.
- Publish a pickle-free `.alignair`, model card, reference manifest, validation report, benchmark
  report, license/use constraints, checksums, and minimal example input/expected output.
- Enforce minimum/maximum compatible AlignAIR/model-format/architecture versions on download.
- Make catalog aliases conveniences, not a second incompatible loading system.

**Acceptance criteria**

- A new Linux/macOS/Windows user can install, download, predict, go offline, and repeat the run.
- Public, private-token, pinned-revision, corrupt-cache, interrupted-download, and offline tests pass.
- Publishing is transactional: failed validation leaves neither an updated catalog nor a copied
  invalid artifact. The current local publisher writes before validation and must be changed.

### P0-12 — Prove full runtime support on Linux, macOS, and Windows — 🟡 PARTIAL (2026-07-13)

Fixed the two portability bugs that were verifiable/fixable here:
- **Cache locking is now portable.** The Windows `fcntl` no-op (which let concurrent downloads corrupt
  the cache) is replaced by an atomic `O_EXCL` lockfile protocol (`_excl_lock`) with stale-lock
  reclamation + timeout on non-POSIX; POSIX keeps robust `flock`. Tested (mutual exclusion, stale
  reclaim, timeout) in `test_lock.py`.
- **Per-OS paths via `platformdirs`** (core dep) for cache *and* config, honoring `ALIGNAIR_CACHE_DIR`/
  `ALIGNAIR_CONFIG_DIR`/XDG; Linux paths unchanged, macOS/Windows now correct.
- **Apple MPS** is selected by `device="auto"` (`resolve_device`, P0-9); `doctor` reports the resolved
  device + cache/config dirs.

*Remaining (needs CI on macOS/Windows runners):* run the installed-wheel golden suite on Win/macOS
Intel+arm64 for Py 3.10–3.12; CUDA in a GPU workflow; per-backend numerical tolerances; Unicode/CRLF/
long-path/AV audit.

**Observed gap**

The main test/CLI/wheel/release jobs run on Ubuntu. The separate wheel workflow builds on Windows and
macOS but only loads the optional Cython kernel; it does not install runtime dependencies, load a
model, predict, train, write AIRR, or exercise cache paths. Apple MPS is not selected by device auto
detection. Windows model-cache locking has no implementation because the `fcntl` fallback is a
no-op.

**Required action**

- Run the installed-wheel contract suite on Ubuntu, Windows, macOS Intel (while supported), and
  macOS arm64 for Python 3.10-3.12.
- Test CPU everywhere, MPS on Apple Silicon, and CUDA in a separate GPU workflow.
- Replace OS-specific cache locking with a portable lock library or an atomic lock protocol.
- Use `platformdirs` for cache/config locations while honoring existing environment overrides.
- Audit path quoting, Unicode paths/IDs, CRLF, spawn multiprocessing, antivirus file locking, and
  long Windows paths.
- Decide whether the C extension is truly optional. If yes, publish a universal pure-Python wheel
  plus optional accelerators or ensure source installs without a compiler produce a usable wheel.

**Acceptance criteria**

- The same golden prediction and one-step train/save/load/resume workflow passes on all supported
  operating systems from the exact artifacts intended for release.
- Numerical tolerances are defined per backend; scientific calls/coordinates remain invariant.
- Concurrent downloads cannot corrupt the Windows cache.

### P0-13 — Repair wheel, conda, Docker, and release publication flow

**Observed issues**

- The tag-triggered PyPI job builds and uploads only the native Linux wheel plus sdist.
- The cross-platform wheel workflow is triggered manually or after a GitHub release is published,
  uploads CI artifacts only, and is not connected to PyPI publication.
- Release smoke is guaranteed to fail because it invokes missing commands.
- The conda recipe claims Python 3.9 while the project requires Python 3.10, contains a placeholder
  source hash, depends on unavailable GenAIRR packaging, and tests missing commands.
- Docker can build a package from copied `pyproject.toml` and `src`, but omits `setup.py`, so the
  advertised optional compiled kernel is not part of that build. Its health check invokes a missing
  command.

**Required action**

- Build all wheels once, test each, aggregate them with the sdist, and publish that exact artifact
  set under a protected release environment.
- Use trusted publishing (OIDC) for PyPI where possible.
- Generate versions from one source and verify tag/package/model compatibility.
- Make conda/Bioconda support honest: either finish the dependency chain and recipe or remove it
  from the launch promise.
- Build multi-architecture container images, pin base/dependency versions, emit an SBOM, scan them,
  and run as non-root with writable cache/output behavior tested.
- Test the installed artifact, not the source checkout, at every release gate.

**Acceptance criteria**

- TestPyPI rehearsal installs on all OS targets before production publication.
- Wheel, sdist, conda (if claimed), and container produce the same version and golden result.
- Release automation cannot upload if scientific, API, AIRR, or artifact-integrity gates fail.

### P0-14 — Make AIRR output semantically correct, not only parseable — 🟡 MOSTLY DONE (2026-07-13)

- **`productive` is now a DERIVED fact** (in-frame AND no stop codon via `quality.airr_productive`); the
  model's neural call is kept separately as `productive_prediction`. Verified on the real IGH model: the
  two differ on 6/30 reads.
- **Advertised fields populated:** `d_identity`/`j_identity` are now computed (generic
  `quality.segment_identity`), not empty placeholders (was V-only).
- **Machine-readable field map** (`io/airr_field_map.py`) documents every emitted column's source
  (neural/derived/germline/read/extension) + null policy; a completeness test guarantees no undocumented
  column ships.
- **Cross-field invariant:** `validate-airr` now rejects `productive=T` with `vj_in_frame=F` or
  `stop_codon=T`, on top of the coordinate/CIGAR-consumption checks (P0-2). Silent assembly failure,
  orientation mismatch, and global locus/has-D are already fixed (P0-7/P0-3/P0-6).
- Tests: `test_semantics.py`, `test_field_map.py`.

*Remaining:* run the official `airr` Python validator on golden fixtures in CI; a formal JSON schema for
the extension fields; per-assay field-completeness/assembly-failure budgets (ties into P0-15/P0-16).

### P0-15 — Build an interoperability contract for the AIRR ecosystem

Do not attempt to clone every proprietary output format by default. Make AIRR Rearrangement the
canonical contract, then provide tested compatibility profiles/adapters where widely adopted tools
require additional conventions.

**Required compatibility matrix**

| Consumer/tool | Minimum launch proof | Likely adapter work |
| --- | --- | --- |
| AIRR Python | Schema validation and lossless readback | Standard field typing/nulls |
| Change-O / Immcantation | Import and a minimal downstream operation | Coordinate, CIGAR, junction, identity fields; metadata preservation |
| IgBLAST workflows | Compare against outfmt 19/AIRR on identical reads | Import comparator; optional IgBLAST-like report profile only if demanded |
| MiXCR | Compare/import `exportAirr` outputs | Normalized calls, locus, productivity, cell metadata |
| partis | Document/import the supported handoff | Germline/reference provenance and field conversion |
| Scirpy / single-cell Python | Load predicted AIRR with cell/contig metadata | `cell_id`, locus/chain, productive, junction AA, duplicate/UMI fields |
| nf-core/airrflow | Tiny pipeline fixture and container execution | Samplesheet, resources, provenance, failure codes |
| 10x Cell Ranger inputs | FASTA + annotations join | Stable ID join, barcode/UMI/constant-region preservation |
| Galaxy | Planemo-tested tool with pinned container/model | Datatypes, collections, model cache, test data |

**Required action**

- Create `alignair interop check --profile <consumer>` or equivalent Python validators.
- Preserve unknown input metadata columns only through an explicit safe policy; avoid collisions
  with standard/generated fields.
- Produce versioned tiny fixtures for each supported consumer.
- Publish a feature matrix that distinguishes validated, best-effort, and unsupported integrations.

**Acceptance criteria**

- At least AIRR Python, Change-O/Immcantation, Scirpy/10x, and one workflow engine are executable
  release gates.
- IgBLAST and MiXCR comparisons use the same reference, orientation, and scoring definitions.
- Unsupported fields/tools fail clearly rather than implying compatibility from a `.tsv` suffix.

### P0-16 — Add tests for product invariants and scientific claims — 🟡 MOSTLY DONE (2026-07-13)

Added the missing test layers on top of the existing unit/contract coverage:
- **Property/invariant** (`property/test_invariants.py`): orientation-transform involution, deterministic
  prob-sorted selection, CIGAR query-consumption ≤ read length over randomized inputs.
- **Golden AIRR + official validation** (`golden/test_golden_airr.py`): fixed inputs → exact normalized
  rows that **pass the official `airr` library's schema validation** (also closes P0-14's remaining
  item; logical fields normalized to `T`/`F`).
- **Robustness/fuzz** (`fuzz/test_robustness.py`): malformed genotype/container/FASTQ, empty file,
  NaN/Inf injection, oversized read → clean typed errors, never a crash or silent garbage.
- **Version-controlled release gates** (`alignair/validation/gates.py`): `SCIENTIFIC_THRESHOLDS`
  (per-task floors) + a `CLAIM_TESTS` map from each model-card claim to the named test that proves it.
  Tests assert every claim points at a real test, and — on a machine with the model — the shipped IGH
  model **clears the scientific gates**. Editing a threshold is a reviewed code change (no silent bypass).

*Remaining:* differential IgBLAST/MiXCR gate (P0-15, needs those tools); artifact/system + performance-
regression layers (need CI); wire the gates into the release workflow.

The 514+ passing tests are valuable but miss several failures above. Test count is not a
release metric; coverage of invariants is.

**Required test layers**

1. **Pure unit tests:** tensor shapes, losses, transforms, serialization, parsers.
2. **Property tests:** finite probabilities, coordinate bounds, CIGAR consumption, allowed genotype
   calls, orientation invariance, deterministic ordering.
3. **API/CLI contract tests:** identical behavior and errors through public Python and installed CLI.
4. **Golden AIRR tests:** fixed inputs/model artifact and exact/normalized outputs.
5. **Scientific validation:** per-locus/allele/gene accuracy, boundary errors, junction exactness,
   productivity, calibration/coverage, corruption strata, novel-allele tests if claimed.
6. **Differential tests:** AlignAIR versions and IgBLAST/MiXCR baselines under a frozen benchmark
   manifest.
7. **Robustness/fuzz tests:** malformed FASTA/FASTQ/TSV/YAML/container headers, long reads, empty
   references, large allele sets, corrupted artifacts, NaN/Inf injection.
8. **Artifact/system tests:** installed wheels on every OS, container, cache concurrency, offline HF.
9. **Performance regressions:** runtime, peak RAM/VRAM, model-load time, throughput, and output size.

**Acceptance criteria**

- Every scientific/product claim in the model card maps to a named automated test/report metric.
- Release thresholds are version controlled and cannot be bypassed silently.
- Failures produce artifacts sufficient to reproduce the exact case.

## P1 production-hardening work

These items should be completed before describing the system as industry-grade, even if a limited
technical preview is allowed after all P0 gates pass.

### P1-1 — Centralize configuration and validation

- Replace unrestricted `**overrides` at public boundaries with typed config models.
- Validate counts, maximum length, kernel schedules, batch size, thresholds, temperatures, device,
  reference non-emptiness, anchors, and locus schemas.
- Preserve unknown future config fields in metadata without silently applying incompatible defaults.
- Emit a fully resolved run config to provenance.

### P1-2 — Numerical reliability controls

- Assert finite tensors at model outputs, calibrated probabilities, losses, gradients, and metrics.
- Add gradient norm logging/clipping and task-weight monitoring.
- Define mixed-precision policy per CPU/CUDA/MPS backend; test AMP underflow/overflow.
- Seed Python, NumPy, Torch CPU/CUDA and data workers consistently.
- Add deterministic mode for reproducibility and a faster nondeterministic mode with provenance.
- Calibrate uncertainty on held-out, assay-representative data. Report ECE/Brier score, set coverage,
  set size, selective risk, and calibration under shift—not only top-1 accuracy.
- Treat the statement that a smoothed BCE sigmoid is a calibrated posterior as a hypothesis to test,
  not an implementation guarantee.

### P1-3 — Resource-aware inference and training

- Auto-tune batch size conservatively from device memory and reference size.
- Handle OOM by reducing batch size once with a clear report, not an infinite retry.
- Record peak RAM/VRAM and per-stage timing.
- Benchmark the pure-Python and compiled CIGAR paths for equivalence and speed.
- Set launch envelopes for common CPU, Apple Silicon, and NVIDIA hardware.
- Add cancellation-safe checkpoint/output cleanup.

### P1-4 — Harden the model container

- Bound header length, section count, compressed/uncompressed section sizes, and config dimensions
  before allocation/decompression to prevent resource-exhaustion artifacts.
- Validate all required metadata and sections before model construction.
- Add explicit architecture/config/model-format compatibility ranges.
- Preserve a migration tool and golden artifacts for every supported format version.
- Remove user/host provenance from public artifacts or make it opt-in; it currently risks leaking
  workstation identity.
- Capture model weights license, training/reference data licenses, and reference redistribution
  constraints separately from package GPL licensing.

### P1-5 — Transactional publishing and supply-chain controls

- Validate into a temporary staging directory before updating a registry.
- Sign releases/checksums or publish attestations; checksum data fetched from the same mutable
  registry protects corruption but not registry compromise.
- Pin GitHub Actions to reviewed commit SHAs for high-assurance release jobs.
- Generate SBOM/provenance attestations and dependency/license scans.
- Add a vulnerability response process for model artifacts as well as code.

### P1-6 — Improve error and logging design

- Define a public exception hierarchy: input, reference, compatibility, download, artifact,
  inference, assembly, training, and validation errors.
- Send logs/progress to stderr and data to stdout when streaming.
- Add `--quiet`, `--verbose`, and `--json` consistently.
- Use stable exit codes for usage, data, model, resource, and internal failures.
- Never catch broad exceptions without preserving sequence ID, stage, exception, and counts.

### P1-7 — Repository organization and ownership

Recommended package boundaries:

```text
alignair/
  api/              # only stable public objects and protocols
  model/            # architectures and forward contracts
  training/         # configs, engine, checkpoints, validation
  inference/        # predictor, batching, reference scoring
  references/       # parsing, schemas, provenance, locus partitions
  airr/             # canonical record model, assembly, validation
  interop/          # consumer profiles and adapters
  artifacts/        # .alignair container and migrations
  hub/              # HF/catalog/cache behavior
  cli/              # thin API adapter only
```

Additional actions:

- Decide whether `alignair_benchmark` is a separate distributable/dev project; it is currently
  excluded from wheels but occupies a large second architecture under `src`.
- Keep experimental genotype inference and X-ray tooling behind explicit experimental namespaces
  until their contracts stabilize.
- Remove stale duplicate data/training implementations after migration.
- Add `CODEOWNERS` for model, AIRR semantics, packaging/release, and security-sensitive artifact
  loading.
- Add linting, formatting, type checking, dependency checks, and dead-code/import-boundary checks.

### P1-8 — Provenance and reproducibility

Every prediction run should record:

- AlignAIR version and commit/build identity;
- model repo, revision/commit, artifact SHA-256, architecture and format version;
- reference source/version/hashes and genotype hash;
- complete resolved inference config and calibration identity;
- input hash or manifest hashes, accepted/rejected counts, and output hash;
- device/backend, OS, Python, Torch, CUDA/MPS and relevant library versions;
- timestamps, random seed/determinism mode, stage timings, resource peaks, and warning/error counts.

Avoid absolute local model paths and usernames in portable public provenance unless explicitly
requested.

## P2 adoption and expansion work

- Add an OGRDB/AIRR GermlineSet and GenotypeSet import/export path with source/version provenance.
- Add a service-safe inference interface only after the local API is stable; define concurrency,
  model residency, request limits, and observability rather than shipping a thin Flask wrapper.
- Add ONNX/other runtime export only if scientific parity can be proven for dynamic reference and
  post-processing behavior.
- Add model quantization only behind per-locus accuracy/calibration gates.
- Create community benchmark submissions and reproducible comparison containers.
- Establish a model lifecycle: candidate, validated, production, deprecated, withdrawn.
- Add telemetry only as explicit opt-in; scientific tools should be offline and private by default.

## Recommended implementation sequence

### Workstream 0 — Contract freeze (2-4 engineering days)

1. Decide fixed-head versus runtime-reference architecture for the launch.
2. Freeze supported loci, assays, sequence lengths, platforms, output profile, and model IDs.
3. Write executable acceptance tests for the chosen claims.
4. Stop feature work that assumes the other architecture.

**Exit:** one approved product/model/API contract and a P0 issue for every failing acceptance test.

### Workstream 1 — Correctness containment (approximately 1-2 weeks)

1. Fix canonical sequence ownership.
2. Replace segment repair with constrained bounded projection/no-call behavior.
3. Fix genotype allowed-set selection and empty-set errors.
4. Implement per-record locus/has-D semantics and cross-locus masking.
5. Replace broad AIRR exception swallowing with typed status/failure budgets.
6. Enforce long-read/input policies and finite-value checks.

**Exit:** property/invariant suite passes on CPU; no silent scientific failure path remains.

### Workstream 2 — One public API and CLI (approximately 1-2 weeks)

1. Implement `Aligner`, typed results/configs, registry-aware loading, and device resolution.
2. Move CLI orchestration onto the public API.
3. Implement the agreed training/custom-reference surface and structured errors.
4. Update workflows/tests mechanically to that contract; documentation prose remains phase two.

**Exit:** all supported journeys pass API/CLI parity tests from an installed wheel.

### Workstream 3 — Model production and validation (duration depends on architecture/training)

1. Train candidate models from frozen reference/data manifests.
2. Calibrate on independent validation sets.
3. Run scientific, robustness, and differential benchmark gates.
4. Produce pickle-free inference artifacts and complete model cards/reports.
5. Independently reproduce one candidate from its training manifest.

**Exit:** each model meets declared per-stratum accuracy, calibration, failure, and resource bars.

### Workstream 4 — Distribution and platform matrix (approximately 1 week after API freeze)

1. Implement direct HF revision-aware load/publish and transactional catalog validation.
2. Aggregate/test/publish wheels for Linux/macOS/Windows.
3. Add CPU/MPS/CUDA backend parity tests.
4. Repair container, conda claim, release smoke, and protected publication flow.

**Exit:** clean-machine install/download/offline-predict succeeds on every supported platform.

### Workstream 5 — AIRR ecosystem certification (approximately 1-2 weeks, parallelizable)

1. Freeze the AIRR field map and extension schema.
2. Validate official AIRR round trips and cross-field invariants.
3. Add executable Change-O/Immcantation, Scirpy/10x, workflow, IgBLAST, and MiXCR fixtures.
4. Mark every ecosystem surface as validated, best-effort, or unsupported.

**Exit:** compatibility matrix is evidence-backed and release-gated.

### Workstream 6 — Production rehearsal

1. Publish a release candidate to TestPyPI and staging HF repositories.
2. Run fresh Windows/macOS/Linux installations with empty caches.
3. Run a representative cohort at production scale and inject interruption/corruption/OOM cases.
4. Verify rollback/withdrawal of a model and package release.
5. Freeze artifacts, checksums, reports, and release notes.

**Exit:** release checklist signed by model/science, software, AIRR interoperability, and release
owners. Only then start the final documentation polish phase.

## Proposed release-blocking metric set

Exact numerical thresholds must be chosen from the intended use and benchmark baselines, but every
production model should publish and gate at least:

- V/D/J allele, gene, and family accuracy with confidence intervals;
- top-k/set coverage, mean set size, and selective accuracy;
- boundary MAE and exact boundary rate by segment;
- junction nucleotide/amino-acid exact rate;
- productivity, frame, stop-codon, and orientation performance;
- calibration ECE/Brier and reliability curves per gene/locus;
- results by read length, SHM, indels, sequencing error, truncation, orientation, allele frequency,
  locus, and reference size;
- novel-allele/reference-shift results only if the architecture supports that claim;
- AIRR assembly success and official validation rate;
- throughput and peak RAM/VRAM on supported hardware;
- comparison against frozen IgBLAST and MiXCR configurations using identical references/cases.

Do not use one aggregate accuracy number as the launch bar. AIRR adopters will encounter exactly the
short-read, high-SHM, ambiguous-allele, no-D, multi-locus, and reference-shift corners that averages
hide.

## Definition of production-ready

The modernization can be considered code-complete—and ready for the separate documentation
improvement phase—when all of the following are true:

- The implemented model architecture matches every public scientific claim.
- No P0 item in this report remains open.
- Prediction invariants are property-tested and no stage silently degrades a row.
- The public Python API and CLI are stable, typed, parity-tested, and used by all wrappers.
- Custom GenAIRR DataConfig training, validation, checkpoint/resume, export, and inference work from
  a clean installation without private imports.
- Published models are safe, immutable, validated, calibrated, revision-pinned, and easy to load
  directly from HF online or offline.
- Linux, macOS, and Windows execute the same installed-artifact golden workflows; Apple MPS/CUDA
  support is either proven or explicitly unsupported.
- AIRR output is semantically validated and executable compatibility tests pass for the declared
  downstream ecosystem.
- The release workflow publishes the exact cross-platform artifacts it tested and supports rollback.
- Performance/resource envelopes, failure budgets, security boundaries, and model lifecycle policy
  are explicit and enforced.

At that point, better documentation really will be the remaining major product task. In the current
state, documentation polish would conceal unresolved contract and correctness problems rather than
finish the product.

## Verification performed during this audit

The following checks were run against the audited working tree:

| Check | Result |
| --- | --- |
| `pytest -q` | **514 passed**, 6 warnings, approximately 104 seconds |
| `compileall` over `src` and `tests` | Passed |
| Clean sdist + wheel build | Passed |
| `twine check` on both artifacts | Passed |
| Built wheel type | `alignair-2.0.2-cp312-cp312-linux_x86_64.whl` |
| Official AIRR Python read/validation of a sample writer row | Passed basic parse/schema validation |
| Active CLI help inspection | Confirmed command/product divergence described above |
| Segment-bound reproduction | 10-base read produced D/J ends at 11/12 after ordering repair |
| Strict MkDocs attempt in current venv | Not run: MkDocs executable is absent from the existing venv |
| Docker build | Passed locally in approximately 198 seconds |
| Docker `--help` | Passed and confirmed the active nine-command surface |
| Docker `doctor` / configured health check | Failed with parser exit 2 because `doctor` does not exist |
| Compiled inference extension inside Docker | Absent (`_derive_cy.c`/`.pyx` present, no `.so`); `setup.py` is not copied into the image build context |

Passing unit tests and artifact construction are meaningful positives, but they do not override the
reproduced scientific-output and product-contract blockers.
