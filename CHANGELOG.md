# Changelog

All notable changes to AlignAIR are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow semantic versioning.

## [4.0.0] - 2026-11-18

AlignAIR 4 is a new network and a new model contract: AlignAIR 3 model files do not run on it. Its pretrained models
are `alignair4-<locus>-human` 1.0.0, one per locus (IGH, IGK, IGL, TRA, TRB, TRD, TRG).

### Changed
- **One network, the chip (`chip-v1`).** A dilated-convolution trunk gives every base its features once; a region
  parser, V/J anchor heads, a mutation reader and per-segment scans of the bound reference's alleles read them, and
  deterministic steps refine the calls and boundaries against the germline. The `conv-v1` network is removed.
- **AlignAIR 3 model files are refused** with the version that runs them: a `.pt`, a `conv-v1` `.alignair`, or a
  container without `reference_json` ("this is an AlignAIR 3 model ...; use alignair<4 to run it"). The hub catalog keeps
  the AlignAIR 3 entries (`alignair-igh-human`, `alignair-igkl-human`, `alignair-tcrb-human` at 1.0.0) for
  AlignAIR 3; AlignAIR 4's are the `alignair4-<locus>-human` models.
- **The reference is an input, not weights.** A model scores the alleles of the reference it is bound to: its
  embedded reference by default, or another one (see Added). The fixed-reference contract and its reference-identity
  check are gone; novel alleles are callable without retraining.
- **`--genotype` binds the genotype's subset of the reference as the bank.** Genotype files are YAML or JSON keyed by
  segment (`V`, `D`, `J`); a name the bound reference does not hold is refused, as is a segment left with no allele.
  `--genotype-method` and the post-hoc genotype masks are removed. A read whose V gene is outside the genotype is
  called to the closest allele the genotype allows, or, when none is close, gets no V call (`airr_assembly_status`
  `partial`, reason `no_v`). `--genotype` also reads the AIRR GenotypeSet that `alignair genotype` and
  `--genotype-out` write (each allele's segment from its name); a GenotypeSet restricts only the genes it names, so
  the genes its rule left open keep all their alleles, while a YAML/JSON genotype restricts each segment it names.
- **Calls.** `v_call` lists, after the call, the V alleles that match the read as well (IgBLAST's convention for
  equally scoring alleles); an allele the read matches at more bases than the model's V call becomes the call; on a
  multi-locus model each read keeps to one locus and the record carries its V call's locus; a D-presence head gates
  the D call, and D is aligned without gaps.
- **Junction and productivity follow by rule from the V and J anchors** the model finds in the read;
  `productive_prediction` repeats the derived call. The learned productivity and chain-type heads and allele
  temperature calibration are removed.
- **Reads are batched with reads of their 32-position width bucket**, so other reads' lengths never change a read's
  row; the default batch size follows the device (256 on a GPU, 64 on the CPU, 16 per worker with `--threads`).
- **`alignair genotype`** now infers a donor's V genotype from an AIRR table: `alignair genotype TSV --out JSON
  [--locus] [--explain] [--min-reads]`.
- Model loading takes no dataconfigs or `trust_pickle`: `alignair predict` loses `--dataconfig` and `--trust-pickle`,
  `alignair benchmark` loses `--trust-pickle`.
- A model file written for sharing records neither the host nor the user it was written by.
- **A reference of another locus** (every locus-showing allele name of one other locus, e.g. TRB on an IGH model) binds
  with one `LocusMismatchWarning` ("the bound reference is TRB, but the model was trained on IGH") and keeps its own
  locus, so the records' `locus` comes from the alleles called.
- **`<out>.run.json`** records `--reference` and `--genotype` (paths and SHA-256), the fingerprints of the reference the
  calls came from (the bound one, with the embedded one's beside them) and the genotype bank's allele order.
- **Genotype files** that are not YAML/JSON keyed by V, D, J (a FASTA, a broken file, a list, a segment given one
  string) or that are keyed by gene names are refused with a ConfigurationError whose hint gives the format, instead
  of an internal error or a misleading "alleles the reference does not hold".
- `alignair reference list --chain` matches a chain type, a locus or a whole token of either or of the name (`IGH`
  no longer lists light chains).
- `--device` defaults to `auto` (CUDA, then Apple MPS, then CPU) on every command that takes it (`predict`, `demo`,
  `benchmark`, `train`), as the Python API does, with one help text.
- The missing-segment refusal has a hint; the refusal to share a checkpoint names `alignair export`.
- Training reads get library flanks by default (a 5' leader, a primer masked to N, a 3' constant region), in
  `alignair train` and `run_training`.
- **Error classes end in `Error`, and every one is in `alignair.errors`.** `DuplicateMetadataId` is
  `DuplicateMetadataIdError`, `PredictionCountMismatch` is `PredictionCountMismatchError`,
  `alignair.registry.IntegrityError` is `DownloadIntegrityError`, and `NonFiniteLossError` is
  `NonFiniteTrainingError`; the old names are gone. `alignair.errors` defines and exports every error class
  (`OfflineError` and `TrainingConfigError` included), so `alignair.DuplicateMetadataIdError`,
  `alignair.PredictionCountMismatchError` and `alignair.registry.OfflineError` are `alignair.errors.*` now.
  `OperationIOError` is an `OSError`.
- **Modules moved.** `alignair.compare` is `alignair.analyze.compare`; the germline reader
  (`alignair.predict.heuristic_matcher`, `alignair.predict.germline`) is the `alignair.predict.germline` package
  (`HeuristicGermlineMatcher` is `locate.GermlineLocator`, `match_one` is `locate`); `alignair.train.gym` is
  `alignair.train.records`; `alignair.train.build` is `alignair.train.bundle` (its dataconfig builder is
  `alignair.genairr.custom`); `build_experiment` is `alignair.genairr.experiment.build_experiment`.
- **Python API signatures.** `build_airr(reads, reference, *, has_d=True, strict=False)` replaces
  `chain="heavy"/"light"` (its options, and `correct_segments`' `min_span`, are keyword-only);
  `PredictConfig(loci=...)` replaces `chain_types=`; `select_alleles(probs, names, threshold, cap)` is the one
  allele-set rule; `load_genotype(path)` returns the genotype dict and `to_genotype_set(genotype, locus)` builds the
  AIRR GenotypeSet; `save_model(path, model, reference, *, dataconfigs=(), ...)` embeds the reference it is given, and
  `save_checkpoint`/`save_rotating` take the reference too; `build_experiment` refuses a parameter it does not read
  (a typo no longer simulates the default) and needs `invert_d_prob`; `NucleotideBatchAdapter(augment_orientation=)`
  takes True or False and refuses any other value.
- **Model files** no longer store allele counts, input/label spaces or a label schema; the reference digests identify
  the model. An earlier AlignAIR 4 development build cannot read a file this one writes. A stored configuration that
  is invalid is a `ModelIntegrityError` (the file's fault). A training checkpoint resumes only onto the reference it
  was trained on, its sequences included (`init_from` warm-starts from it otherwise).
- **Metadata joins** (`build_metadata_index`) rename a column that collides with an AIRR field to `meta_<column>` by
  default, as `alignair predict --metadata` does.
- **`alignair info`** prints one model card: the id and version (or "(local)"), species and locus when set, the
  model's allele counts, allele order and sections.
- **`alignair analyze`** reads CSV as well as TSV; its QC counts the `orientation` column's four transforms
  (`reverse_complement`, `complement`, `reverse`, `forward`; was `reoriented`); the always-empty V confidence
  statistics are gone.
- **`alignair benchmark --dataconfig`** takes one name; unknown `--strata` are refused, naming the strata.
- **`alignair doctor`** lists the core dependencies (torch, numpy, GenAIRR, safetensors, PyYAML, platformdirs) and the
  optional ones (zstandard, huggingface-hub), reports each compiled kernel, and its hint for a missing dependency is
  `pip install alignair`.
- `--locus` is spelled as the model spells it; a bare model spec naming a `.alignair` file, or a path below
  `org/repo`, is a path ("model file does not exist"), not a Hugging Face repo; a failed `alignair predict` publishes
  no rejects file; `<out>.run.json` records the predicting AlignAIR as `alignair_version` and the model's as
  `model_created_by_alignair`; `ReferenceSet.from_fasta` refuses, naming them, alleles whose name does not tell their
  segment; a model without a D never gets a D in its AIRR rows, whatever the bound reference holds.
- The CLI options that had no help text have it (`benchmark`, `models prune --keep`, `info`, `train --lr`/`--seed`).
- **One way to train: `run_training(TrainingConfig(...))`.** `TrainingConfig` checks every value when it is built, not
  when the run starts: `lr`, `grad_clip`, `seed`/`init_seed`, `val_every`, the device, the window, `has_d`, the
  model options (their names and values) and the `resume_path`/`init_from` paths. Training draws novel alleles by
  default (`novel_alleles=NovelAlleles()`, the shipped models' recipe); for a reference GenAIRR cannot draw them for
  (a custom reference, which names no species, among them), the run says so, draws none and records why in
  `train_args["novel_alleles_unavailable"]`, an explicit spec included (it was refused). `alignair demo` trains through
  `run_training`, seeded, so the same command trains the same model.
- **One way to predict: `Aligner`.** `Aligner.predict(sequences, *, ids=None, batch_size="auto", threshold=None,
  cap=None, airr=True)`, and `predict_iter` the same: `ids=` fills each record's `sequence_id` and
  `PredictionResult.ids`; `threshold=` and `cap=` are named options; the other keyword options are gone (`has_d`,
  `max_seq_length` and `loci` are the model's). `threshold`, `cap`, `batch_size` and `predict_iter`'s `chunk_size` are
  refused unless a finite number or a positive integer (`batch_size` may also be `"auto"` or None); a `chunk_size` of 0
  or a negative one, which AlignAIR 3 read as one read per chunk, is refused. `predict_raw` defaults to
  `batch_size="auto"` (was 64), and `Aligner(model, reference)` to `device="auto"`.
- **A reference holding alleles of a locus the model does not call is refused when bound**
  (`from_pretrained(reference=)`, `bound_to`), with a ConfigurationError naming it; a reference wholly of one other
  locus still binds, with its warning.
- `ReferenceSet.from_genotype` is `ReferenceSet.from_alleles` (it builds a reference from `{gene: {allele: sequence}}`;
  "genotype" now means a donor's allele names only). Gene letters are accepted in either case (`"V"` or `"v"`) by
  `with_genotype`, genotype files, `ReferenceSet.gene`, `from_alleles` and `subset`.
- **The top-level package exports the supported API**: `Aligner`, `PredictionResult`, `TrainingConfig`,
  `TrainingRun`, `run_training`, `ReferenceSet`, `read_sequences`, `iter_sequences`, `write_airr`, `AirrWriter`,
  `compare_airr`, `AlignAIRError` and `__version__`. `AlignAIR`, `AlignAIRConfig`, `PredictConfig` and
  `resolve_device` are imported from their modules (`alignair.core`, `alignair.core.config`, `alignair.predict`,
  `alignair.aligner`). `alignair.io` exports the batch reader and the metadata index; `alignair.registry` exports the
  calls a user makes (the publishing tools are in `alignair.registry.publish`). The public API is type-hinted.
- **A registry list is `registry=` everywhere**, as on `Aligner.from_pretrained`: `resolve_model(registry=)` (was
  `sources=`), `maybe_notify_updates(registry=)` (was `sources_list=`), `resolve_sources(registry)` (was `cli`), and
  `find_model`'s third argument. One registry may be given as a string. `validate_registry`'s parsed catalog
  argument is `catalog`.
- **Registry failures say what failed**: an id no registry lists, or a model file that does not exist (wherever a
  model file is read), is a `ModelNotFoundError`; a registry or the Hugging Face hub that cannot be reached is a
  `RegistryUnavailableError` (retryable, unless the server answered that the request is wrong), not a raw
  `URLError`/`OSError`; a registry whose catalog or version entry cannot be used is a `RegistryInvalidError`; a
  malformed `hf://` spec or repo id is an `InputError`, and a gated repo says a token is needed. A disk error while
  caching a model, from a registry or the hub, is `ALIGNAIR_IO_FAILED`. An unknown device (`--device gpu`) is a
  `ConfigurationError`.
- **`alignair info MODEL`** takes a model file, a registry id (`id@version`) or `hf://org/repo`, with `--registry`,
  `--offline`, `--revision` and `--hf-token`; `alignair benchmark` and `alignair models get|path` take `--revision`
  and `--hf-token` too.
- `alignair predict --help` lists its options by purpose (model, input and output, output columns, metadata join,
  genotype and reference, performance, quality gates, messages). The CLI describes itself as V(D)J allele assignment
  for IG and TCR repertoires, and its refusals and help texts use plain words.
- **The AIRR table follows the AIRR schema's positions and CIGARs, and Change-O reads it as written.** Every position
  is 1-based and closed: `*_alignment_start` and the FWR/CDR starts and ends (now positions in `sequence`, from a
  region's first to its last read base, blank where the read does not reach) were 0-based or counted in
  `sequence_alignment`. Each CIGAR spans the whole read (`4N291M73S`). `v_germline_start`/`v_germline_end` count in
  the IMGT-gapped V germline from its first position, as Change-O's MakeDb writes them (empty for a row without the
  gapped alignment). `germline_alignment` holds N in the untemplated bases. A row without a D call, a too-short D
  included, holds every base between V and J in `np1`. FWR4 is IMGT's FR4, from the J's Trp or Phe codon (the
  junction's last codon) to the end of the J, as IgBLAST and MakeDb cut it; it used to start after that codon.
  `CreateGermlines.py` builds the germlines from the table without a preparation step (2,855 of 2,872 complete rows of
  3,000 real IGH reads; the rest have an indel in their J alignment). Python records keep Python's terms (0-based,
  end excluded).
- **A read cut inside its J is `partial` (`j_cropped`).** A read longer than 575 nt is aligned on its first 575 nt;
  when that cut fell inside the J (a forward read with a long leader, as on some 10x contigs), the row said
  `complete` while its J call, junction and FWR4 rested on part of the J. A row whose J alignment runs to the cut (the
  read's end for a forward or complemented read, its start for a reverse-complemented or reversed one) is now
  `partial`, reason `j_cropped`.
- `--reference` reads IMGT/GENE-DB headers (`>X00369|IGHV3-11*04|Homo sapiens|F|...`, as in Immcantation's germline
  files): the allele is the field with a `*`, as Change-O reads it, not the header's first word.
- `examples/reads.fasta` is 20 clean, productive human IGH reads, written by `examples/make_example_reads.py`.
- The README and the documentation website are rewritten for repertoire researchers: install and a first alignment,
  pages for IgBLAST, MiXCR and Immcantation users, how-tos for real data, the results and how far to trust them, a
  reference generated from the code (every CLI option, output column, error code and model), training and
  fine-tuning, and, apart, the developer pages. Addresses of the old pages redirect to their replacements.
- The `test` extra adds ruff (`>=0.17.1,<0.18`): `tests/quality/test_lint.py` fails on any finding in `src/alignair`
  under `pyproject.toml`'s `[tool.ruff]` settings.

### Added
- `alignair predict --reference FASTA` and, in Python, `Aligner.from_pretrained(..., reference=...)` and
  `Aligner.bound_to(reference)`: bind another germline reference, novel alleles and new genes included. Missing IMGT
  gaps and anchors are taken from the model's own alleles (the same sequence, or a same-length variant differing at
  no more than 10% of bases) or GenAIRR's resolver (`alignair.reference.annotate`); alleles left without an anchor
  are named on stderr (`alignair.reference.anchor_report`). A reference missing one of the model's segments is
  refused.
- `alignair predict --infer-genotype` (with `--genotype-out`, `--genotype-explain`, `--genotype-min-reads`): two
  passes over the input infer the donor's V genotype by TIgGER's frequency rule, re-call the reads it settles, and
  write it as an AIRR GenotypeSet.
- `alignair predict --threads N`: on the CPU, N single-threaded worker processes.
- `alignair reference export MODEL --fasta OUT --gapped`: the model's V alleles IMGT-gapped, as Change-O's
  `CreateGermlines.py` reads them; a V allele without a gapped form is refused by name.
- `alignair train --init-from MODEL` (a model file, registry id or `hf://` repo) and `TrainingConfig(init_from=)`:
  start a run from a model's weights (those whose shapes match; the rest start fresh).
- `alignair export CHECKPOINT OUT_DIR [--model-id --model-version --description --no-validate --overwrite]`: a training
  checkpoint becomes a pickle-free bundle (model, model card, reference manifest, validation report) whose card the
  catalog accepts as it is. The model card names the model, its reference and version (e.g. OGRDB V8), allele counts,
  training steps, intended use and the genotype and reference-swap usage.
- Publishing to the shared catalog: `alignair.registry.publish.publish_bundle` lays a bundle out beside its card and
  manifest (`<id>/<version>.alignair`, `.model_card.md`, `.reference_manifest.json`, recorded with their SHA-256 and
  checked by the validator), and `upload_version` sends one version's files and `registry.json` in one commit.
  `push_to_hub` uploads the card and manifest beside a model file with it.
- Training: `--data-workers` generates curriculum records in worker processes with batches identical to a serial
  run; an init seed separates the initial weights from the data seed; novel-allele training banks
  (`TrainingConfig.novel_alleles`, see Changed); `train_args` record the recipe.
- A V allele listed by two loci belongs to both (TRAV/DV alleles in TRA and TRD).
- `Aligner.with_genotype(genotype)`: an Aligner bound once to a donor's genotype (`{V|D|J: allele names}` of the
  reference; a gene it does not name keeps all its alleles); this Aligner is unchanged.
- **Classified errors** (`alignair.errors`): every expected failure is an `AlignAIRError` with a stable code
  `ALIGNAIR_<THING>_<STATE>` (`..._NOT_FOUND`, `..._UNAVAILABLE`, `..._INVALID`, ...), a category, the CLI's exit
  code, `retryable` and a context; `alignair --error-format json` prints it as one JSON object, and `--debug` adds the
  traceback.

### Fixed
- On Windows, saving a training checkpoint (`alignair train`, `alignair demo`, `run_training`) no longer fails with
  `OSError: [Errno 9] Bad file descriptor`: the checkpoint is flushed to disk through a handle open for writing, which
  Windows requires.
- A registry given as a Windows folder (`--registry C:\models`, `ALIGNAIR_REGISTRY` or the config file's
  `registries`) opens its models; the drive letter was read as a URL scheme and the folder dropped. A
  `file:///C:/models` URL names that folder too, and the relative `file://catalog` form is unchanged.
- Alleles with exactly the same score (a confident model gives several alleles one score) keep the germline set's
  order in a candidate set, so the same read gets the same candidates, and the same call, on every numpy version and
  CPU; numpy's default sort ordered such ties differently between versions.
- The AIRR table, the `--rejects-out` file and the table `--infer-genotype` re-calls rows into end their lines with
  `\n`, as other AIRR tools write them; they ended them with `\r\n`, which broke awk and other Unix tools.
- `alignair benchmark` no longer fails at its defaults on a short locus such as TRB: a stratum whose 5' loss leaves a
  simulated read no bases (`short_janchor`) draws another read, and a stratum that leaves almost none is refused with
  a hint to choose other strata.

### Removed
- The amino-acid (protein) track: `predict-aa`, `predict_aa`, `load_aa_model`, protein references and class labels,
  conformal and allele calibration, the probability and feature-extraction APIs, augmentation, class sampling and
  partitioning, and the cohort, coverage and report evaluation (kept at tag `aa-track-v3`).
- The WFA, parasail and Biopython germline readers (`alignair.align`) and those dependencies; the heuristic matcher
  is the one germline reader.
- The `alignair_benchmark` package and other research code; the head-to-head evaluation is the source-only
  `alignair_assay` package.
- `PredictConfig.selector` and `pad_mode` (with `SELECTORS`, `largest_gap`, `max_likelihood_percentage`), and
  `correct_segments(max_len=, pad_mode=)`.
- `AlignAIR.from_dataconfigs`, `AlignAIRConfig.from_dataconfigs`/`from_reference` and the allele-count and
  label-space settings: build a model for a reference with `alignair.core.prepare_model`
  (`ReferenceSet.from_dataconfigs` stays).
- `ReferenceSet.from_yaml`, `to_yaml` and `infer_locus`; a saved reference reloads with the locus schema it was saved
  with.
- `alignair.data.tokenizer` and `alignair.data.encoders` (the tokens are `alignair.core.tokens`; `tokenize_batch`).
- The ten AIRR extension columns that were never filled: `*_resolved_call`, `*_call_level`, `*_set_confidence` and
  `is_contaminant`.
- `alignair reference export --dataconfig/--index` and the `export-reference` verb: `alignair reference export MODEL
  --fasta OUT` writes a model file's germline FASTA.
- `alignair convert`: AlignAIR 3 `.pt` checkpoints are refused (run them with `alignair<4`).
- The functional API: `alignair.load_model` (use `Aligner.from_pretrained(path)`), `alignair.predict_sequences`
  (use `Aligner.from_pretrained(path).predict(seqs).records`, or `Aligner.from_model(model, reference)`) and
  `alignair.train_model` (use `run_training(TrainingConfig.from_genairr(...))`, which seeds the run); the module
  `alignair.api`. `train_model`'s `progresses=`, `heavy_shm=` and `mutation_cap=` have no replacement: the
  curriculum's record mix is the recipe's (`TrainingConfig` sets `short_boost`, `flanks`, `novel_alleles` and
  `inverted_d`). `Aligner.predict(genotype=, genotype_method=)`: use `aligner.with_genotype(genotype).predict(...)`.
- `alignair analyze --format` and `alignair benchmark --format` (use `--json`); `alignair models info` (use
  `alignair info <id>`) and the `model` alias of `alignair models`.
- `load_metadata` (use `alignair.io.build_metadata_index`; `lookup_metadata_rows` takes its index).

## [3.0.0] - 2026-07-18

A complete PyTorch rewrite of AlignAIR (`alignair` package) with a first-class CLI, a self-contained
model format, pretrained models, and a modern documentation site. This is a major, breaking release —
the legacy TensorFlow lineage is gone.

### Added
- **`alignair` CLI**: `predict`, `train`, `demo`, `models`, `reference`, `convert`, `validate-airr`,
  `compare`, `analyze`, `benchmark`, `genotype`, `doctor`, plus `--version`.
- **Pretrained models** published to a public hub (`AlignAIR/AlignAIR-pretrained`): human IGH,
  IGK+IGL, and TRB. `--model <id>` downloads, hash-verifies, and caches automatically — no login.
  `alignair models list/get/info/...` manages the catalog, which updates live from the registry.
- **Self-contained `.alignair` model format**: weights + the embedded, fingerprinted germline
  reference + model card in one file that loads **without executing any pickle**.
- **Donor-genotype constraint**: `predict --genotype` restricts calls to a subset of the model's
  reference (YAML or FASTA) with no retraining. A model is a **fixed-reference classifier** — novel
  alleles require training (see the model contract).
- **`alignair train`**: train a model for your own reference/species from a built-in GenAIRR
  dataconfig or custom V/D/J FASTAs; exports a self-contained, pickle-free bundle + `model_card.md`,
  `reference_manifest.json`, and `validation_report.json`.
- **AIRR-C output**: schema-valid rearrangement TSV (`sequence_alignment`, `*_cigar`, `*_identity`),
  a per-gene candidate-set column (`*_call_set`), and a `<output>.run.json` provenance sidecar.
- **Bounded-memory streaming** prediction for repertoire-scale inputs; per-read metadata join
  (`--metadata`, e.g. 10x annotations).
- **In-model orientation** detection/correction (forward / reverse-complement / complement / reverse).
- CPU **Docker** image, cross-platform CI, and a compiled (optional) Cython germline-CIGAR kernel.

### Changed
- Output is a schema-valid AIRR rearrangement TSV; ambiguity is surfaced as a candidate set.
- Documentation rebuilt as a React/TypeScript site (reference, guides, and interactive lessons); license clarified as GPL-3.0-or-later.

### Removed
- The entire legacy TensorFlow lineage (`app.py` entrypoint, SavedModel workflow, `src/AlignAIR`).
