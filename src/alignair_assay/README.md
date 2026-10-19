# alignair_assay

## Purpose

`alignair_assay` scores immune-receptor aligners (AlignAIR, IgBLAST, ...) against **observable truth**: what a
perfect aligner could recover from each simulated read. It is a development package, excluded from the wheel, and
the instrument for improving AlignAIR's architecture, wiring and loss. GenAIRR is the only simulator.

The flow is: suites (declarative GenAIRR recipes) → simulated cases → observable truth → each tool's AIRR
rearrangement TSV → per-case scores → coverage-aware summaries and paired, Holm-corrected comparisons.

Units everywhere: coordinates are 0-based and half-open, rates run from 0 to 1, and wall times are in seconds.

## Layers and dependency rule

Each subpackage is a layer and may import only the layers in its row.

| Layer | Holds | May import from `alignair_assay.` |
|---|---|---|
| `core` | shared primitives: `reference` (the germline reference from a dataconfig or a FASTA, `DEFAULT_REFERENCE`, `dataconfig`), `cigar`, `seqops`, `prediction` (normalized prediction records), `jsonl`, `cases` (case schema, case and FASTA files), `provenance`, `layout` (`RunLayout`, the run-directory path schema), `genotype` (`NovelAllele`, `DonorGenotype`: donor genotypes as plain data) | core |
| `truth` | observable truth (pure): `model`, `segment`, `boundaries`, `alleles`, `genairr` (record decoding), `build`, `bench`, `case` (truth from a case), `io`, `oracle`, `summary` (what a truth set could not score) | core, truth |
| `airr` | AIRR TSV `reader` and `writer` | core, airr |
| `metrics` | per-case scorers (pure) and the metric registries | core, truth, metrics |
| `stats` | aggregation with coverage and the paired comparison (pure) | core, metrics, stats |
| `report` | JSON reports and markdown summaries | core, stats, report |
| `scoring` | `pipeline` (score one tool against a truth file) and `slices` | core, truth, airr, metrics, stats, scoring |
| `suites` | the suite registry (`__init__`), suite types (`base`), simulated donors (`donors`), the deterministic generator (`engine`), shared repertoire recipes (`recipes`) and the catalog (`catalog/`, one module per suite) | core, suites |
| `runners` | one runner per tool (`igblast`, `alignair`: the AlignAIR image and a local checkpoint, `oracle`) and the registry (`__init__`: `RUNNERS`, `make_runner`), the runner interface, run records and timed commands (`base`), the shared reference (`reference`), the genotype arm (`arms`) | core, airr, truth, suites, runners |
| `workflow` | build → run → score → report orchestration over a run directory: `build` (one suite) | everything except cli |
| `cli` | the `python -m alignair_assay` commands | everything |
| `testing.py` | test-only helpers: `assert_oracle_perfect` (returns the oracle's per-case scores) | everything |

Further rules:

- `alignair` is imported only by `core/reference.py`, and `alignair_benchmark` is never imported.
- `truth`, `metrics` and `stats` are pure: no file IO, no subprocess, no CLI, no network. `truth/io.py` reads and
  writes truth files through `core.jsonl`.
- GenAIRR is imported only from `core`, `truth` (record decoding), `suites` and `workflow`.

`tests/alignair_assay/test_architecture.py` enforces the table and the first two rules by reading every module's
imports, function-level ones included. A new subpackage fails that test until it has a row in the test's `ALLOWED`
map. Tests live in `tests/alignair_assay/<layer>/`, mirroring the package; `tests/alignair_assay/conftest.py`
provides the `genairr` guard, the session `reference` (`HUMAN_IGH_OGRDB`) and `load_fixture(name)` for the
committed bench cases in `tests/alignair_assay/fixtures/`.

## Truth conventions

**Cases.** A case (`core.cases.make_case`) is one JSON object per line: `case_id`, `suite`, `stratum`, `orientation`
(0 forward, 1 reverse complement), `sequence` (the presented read, uppercase; the reverse complement of the record
sequence when the orientation is 1), `record` (the forward GenAIRR record, verbatim), `labels` (`{str: str}` slice
dimensions) and `groups` (`{str: str}` grouping keys that are not sliced: `subject`, `clone`, `pair`).
`truth.case.truth_from_case` builds the truth and copies `suite`, `labels` and `groups` onto it; it raises when the
sequence does not match the record and orientation. `scoring.slices` slices by the fixed dimensions plus every label
key; a label named like a fixed dimension is an error.

**Observable truth.** Each simulated record is projected onto what the read shows (`truth.build`):

- A segment is observed when at least one of its germline bases is in the read. An unobserved segment has no call
  or coordinate truth; a tool that calls it scores `{g}.spurious_call`.
- The observed alignment drops edge gaps: a clipped or edge-deleted germline base cannot be told apart from extra
  trimming.
- The allele truth is a set: every reference allele whose sequence contains the observed germline, plus the
  simulator's own calls that are in the reference. Disagreements with the simulator's tie list are counted.
- Each segment edge is an inclusive interval of equally defensible positions. An edge is exact when the prediction
  lies inside its interval, and its error is the distance to the interval.
- The junction exists only when both conserved anchors (V Cys, J Trp/Phe) are aligned read bases. Productivity
  flags exist only with a junction. The junction translation and the stop-codon flag are scored only on
  frame-unambiguous reads: an in-frame junction and no V or J frameshift.
- The clone key is (V genes, J genes, junction length).
- Truth is in the canonical forward frame. bench_v2 orientation ids 2 and 3 (complement only, reverse only) are
  excluded (`non_biological_orientation`) and never scored, as are genes whose truth could not be built
  (`ObservableTruth.unscorable`).
- `truth.oracle.oracle_prediction` is the prediction a perfect aligner makes. It must score 1.0 on every metric that
  is neither lower-is-better nor informational, 0.0 on every lower-is-better metric, and no `nan` anywhere.

**`CaseScores` NaN rules.** `CaseScores` maps a metric name (`<scope>.<metric>`, e.g. `v.top1_in_set`) to a float.

- An absent key means the metric does not apply to the case.
- `nan` means the metric applies but the tool did not cover it.
- Accuracy metrics score 0.0 when the prediction is missing. The exception is germline-frame edges, which stay
  `nan` without a top-1 call in the truth set, because the frames cannot be compared.
- Summaries report coverage next to every mean.
- `metrics.lower_is_better` and `metrics.INFORMATIONAL` are the registries. A comparison runs Holm over every
  non-informational metric either tool produced.

**Minimum observed lengths.** `metrics.case.MIN_OBSERVED = {"v": 20, "d": 8, "j": 10}` observed germline bases.
Below its minimum a segment gets only `{g}_short.top1_in_set` and `{g}_short.abstain_ok`, for which abstaining is
as correct as a right call. A missing row is a failure, not an abstention.

## Run directory layout

Every path inside a run directory comes from `core.layout.RunLayout`; no other module joins run-directory paths.
`RunLayout` only builds paths (no IO), and suite, tool and arm names must each be one path component.

```
RUN/
  manifest.json                       config + config_hash + split + host + git + genairr provenance
  reference/germline.fasta            shared reference given to every tool (+ reference.json: digest, counts, source)
  suites/<S>/cases.jsonl.gz           one case per line
  suites/<S>/truth.jsonl.gz           header + observable truth per case
  suites/<S>/reads.fasta              presented reads, id = case_id
  suites/<S>/donors.json              suites with donors only
  suites/<S>/build.json               counts, warnings, timing, config hash; written last
  runs/<tool>/<S>/<arm>/predictions.tsv + run.json + logs/
  runs/<tool>/<S>/genotype/donors/    per donor: reads.<subject>.fasta, predictions.<subject>.tsv (+ logs/<subject>/)
  runs/<tool>/_work/                  the tool's one-time setup (IgBLAST: the germline databases; AlignAIR: the model
                                      cache and the model's exported reference)
  scores/<S>/<arm>/<tool>.* and comparison.{json,md}
  speed/speed.json
  report.json, report.md
```

| Path | `RunLayout` |
|---|---|
| `manifest.json` | `manifest` |
| `reference/`, `germline.fasta`, `reference.json` | `reference_dir`, `reference_fasta`, `reference_json` |
| `suites/<S>/` and its files | `suite_dir(s)`, `cases(s)`, `truth(s)`, `reads(s)`, `donors(s)`, `build_json(s)` |
| `runs/<tool>/<S>/<arm>/` and its files | `run_dir(tool, s, arm)`, `predictions(...)`, `run_json(...)`, `logs_dir(...)` |
| `runs/<tool>/<S>/genotype/donors/` files, `logs/<subject>/` | `donor_reads(tool, s, arm, subject)`, `donor_predictions(...)`, `donor_logs_dir(...)` |
| `runs/<tool>/_work/` | `work_dir(tool)` |
| `scores/<S>/<arm>/` | `scores_dir(s, arm)` |
| `speed/`, `speed.json` | `speed_dir`, `speed_json` |
| `report.json`, `report.md` | `report_json`, `report_md` |

Suites are `S0` (toy: tests and smoke runs), `S0_EXHAUST` and `S0_COHORT` (tests only) and `S1`–`S12`; arms are
`shared` and `genotype`; splits are `dev` (the default, for model iteration) and `test` (locked, used only for
claims).

## Commands

Run from the repository root with `PYTHONPATH=src`:

```
python -m alignair_assay build --suites S0,S1 --run-dir RUN [--split dev] [--reference HUMAN_IGH_OGRDB] \
    [--seed 20261019] [--scale 1.0] [--workers 8]
python -m alignair_assay run --run-dir RUN --tool igblast|alignair|alignair-local|oracle --suites S0,S1 \
    [--arm shared|genotype] [--threads 8] [--cpus 0-7] [--igblast-dir DIR] \
    [--image IMAGE] [--model alignair-igh-human@1.0.0] [--cache-dir DIR] \
    [--checkpoint MODEL.alignair] [--device cpu] [--batch-size 64]
python -m alignair_assay truth --bench-cases CASES.jsonl[.gz] [--reference HUMAN_IGH_OGRDB] --out TRUTH.jsonl.gz
python -m alignair_assay score --truth TRUTH --predictions AIRR.tsv --tool NAME --out-dir DIR
python -m alignair_assay compare --truth TRUTH --a A.tsv --a-name A --b B.tsv --b-name B --out-dir DIR [--alpha 0.05]
```

- `build` generates each named suite into `RUN` (`workflow.build.build_suite`): `cases.jsonl.gz`, `truth.jsonl.gz`,
  `reads.fasta`, `donors.json` (suites with donors only) and `build.json`. Every name is checked before anything is
  built; an unknown suite or split, or a scale that is not positive, exits with status 2. The truth header records
  `kind`, `version`, `reference`, `reference_digest`, the GenAIRR provenance, `suite`, `split`, `seed`, `scale` and
  `n_cases`. `build.json` records the same configuration, its `config_hash` (SHA-256 of the canonical JSON of
  `{suite, split, reference_digest, seed, scale}`), the per-stratum generation stats, the truth summary
  (`truth.summary.truth_summary`: excluded cases, warnings, unscorable genes, simulator disagreements), the worker
  count and the wall time in `seconds`; a suite with donors adds `donors_summary` (see Donors). It is removed first
  and written last, so it marks a complete build. The command prints one line per suite.
- `run` runs one registered tool on built suites of `RUN`. It first checks every input: the tool and its options,
  that each suite is built (`build.json`) against one truth reference, whose dataconfig must still have the
  digest the truth headers record, and, for `--arm genotype`, that each suite has donors (`donors.json`). The run
  directory and the tool's paths (`--igblast-dir`, `--cache-dir`, `--checkpoint`) are made absolute, so every
  recorded command can be re-run from anywhere. It then makes `reference/germline.fasta` the shared reference
  (`runners.reference.ensure_shared_reference`): an existing file, such as a tool's exported reference, is verified
  against the truth reference, and any difference stops the command before anything runs, naming both
  references' allele counts and the first allele that is missing, extra or has a different sequence; otherwise
  the file is written from the truth reference. `reference/reference.json` records its `digest`, `counts`,
  `source` (`truth` or `existing`) and `sha256`. The tool is prepared once, then run per suite: on
  `suites/<S>/reads.fasta` for the `shared` arm (`runners.base.run_suite`), or donor by donor for the `genotype`
  arm (`runners.arms.run_genotype_arm`, see The genotype arm). Either way the run writes `predictions.tsv`, `logs/`
  and `run.json`. The command prints one line per run (return code, reads, wall time, peak RSS) and exits with
  status 2 when any run's return code is not 0. Failed runs are recorded too, and the error message names the
  failed donor and the tool's error line.
- `truth` writes the truth file. Its header records `kind`, `version`, `reference`, `reference_digest` and the
  GenAIRR provenance: `genairr_version`, `genairr_source`, `genairr_git_commit` and `genairr_git_dirty`. The
  command then prints the excluded cases, warnings, unscorable genes and simulator disagreements.
- `score` writes `<tool>.scores.jsonl.gz`, `<tool>.report.json` and `<tool>.report.md`.
- `compare` writes `comparison.json` and `comparison.md`, and prints the headline table. Its 95% and simultaneous
  CIs of 0/1 metrics are clipped to [-1, 1].
- Predictions are matched to truth by AIRR `sequence_id` == truth `case_id`. A truth case without a prediction is
  scored as missing, never dropped.
- A missing file, an unknown reference, a truth file built against another reference, a prediction file
  matching no case, `compare` given the same name for both tools, an unknown tool, an unbuilt suite, a shared
  reference that is not the truth reference, or a failed tool run exits with status 2.

To add a command, write one module in `cli/` with `register(subparsers)` and `run(args) -> int`, and add it to
`cli.COMMANDS`. The module owns its own input checks: `run` raises ValueError (or FileNotFoundError) for invalid
input, which `cli.main` prints as `error: ...` with exit status 2.

## Adding a metric

1. Write the scorer in the `metrics/` module of its kind (`alleles`, `boundaries`, `junction`, `shm`, `flags`). It
   returns `CaseScores` named `<scope>.<metric>` and follows the NaN rules above.
2. Call it from `metrics.case.score_case`. `scoring.pipeline.score_predictions` scores every case through it, so the
   pipeline and the oracle check see the same metrics.
3. Declare its status in `metrics/__init__.py`. A name ending in one of the `_LOWER_IS_BETTER` suffixes is
   lower-is-better. A metric that is reported but not compared goes into `INFORMATIONAL`.
4. Extend the oracle tests in `tests/alignair_assay/metrics/` so the oracle scores perfectly on it, and add a test
   for a wrong prediction.
5. For a headline metric, add it to `report.markdown.HEADLINE`.

## Adding a suite

A suite is a declarative GenAIRR recipe per stratum; `suites.engine` turns it into cases deterministically.

1. Write one module `suites/catalog/sNN_<slug>.py` defining a `suites.base.Suite`: `name` (`"S13"`), `slug`,
   `title`, `purpose` (one sentence: the mechanism it isolates), `primary` (the metrics its verdict rests on),
   `strata`, and optionally `donors` (a `suites.donors.DonorSpec`, see Donors) and `arms`. Names may not
   contain `:`.
2. Give each `Stratum` a `name`, `n` (cases at scale 1.0; clones for a clonal stratum) and a `recipe`: a
   module-level function (or `functools.partial` of one) `recipe(ctx) -> Experiment` that returns a fresh
   `Experiment.on(ctx.config)` chain. `ctx.config` is the GenAIRR DataConfig the truth reference
   (`ctx.reference`) is built from; never build from a short alias. `ctx.n` is the stratum's requested count at
   the current scale (a clonal recipe passes it as `n_clones`), and `ctx.donors` the suite's donors. A
   repertoire-style stratum uses `functools.partial(suites.recipes.repertoire, ...)` (see Suite catalog).
3. Choose the stratum options:
   - `labels` (slice dimensions on every case) and `case_labels(record, ctx)` (per-record labels, merged over
     them);
   - `rc_fraction`, the probability a case is presented reverse-complemented, or `both_orientations`, which
     presents every record as `:fwd` and `:rc` cases grouped by `groups["pair"]`;
   - `accept(record)` for rejection sampling (plain strata only);
   - `cohort` (run over the suite's donors with `run_cohort`, which needs `Suite.donors`; cases get
     `groups["subject"]`) or `clonal` (the recipe ends in `clonal_lineage`; every record is kept; cases get
     `groups["clone"]`).
4. Register the suite in `_CATALOG` in `suites/__init__.py`.
5. Test it in `tests/alignair_assay/suites/` at a small scale: counts, labels and groups, and the oracle check
   `testing.assert_oracle_perfect` on its truth. A suite that fails the oracle check has a truth defect.

Engine rules (`suites.engine.generate_suite`):

- Every seed derives from `(seed, split, suite, stratum, ...)` with `derive_seed` (SHA-256): round `r` uses
  `(seed, split, suite, stratum, r)` and orientation `(seed, split, suite, stratum, "orientation")`, so the `dev`
  and `test` splits share no generation seed.
- A stratum requests `max(1, round(n * scale))` cases. Without `accept` each round asks for the remainder; with
  it the first round asks for twice the remainder and later rounds for the remainder over the acceptance so far
  (at most 20 times). The first accepted records are kept in generation order. A stratum still short after 50
  rounds raises `RuntimeError("<suite>/<stratum>: accepted k of n after 50 rounds")`; a clonal run with fewer
  clones, or a cohort run with fewer records, than requested raises too. Nothing is short-filled.
- A suite's donors are drawn once with `derive_seed(seed, split, suite, "donors", tag)`; their count does not
  scale. A cohort stratum runs `recipe(ctx).run_cohort([d.genotype(ctx.config) for d in donors],
  n_per_subject=ceil(n / len(donors)), seed=<round-0 seed>)` and keeps the first `n` records in record order
  (subject order), so the last donors may get fewer cases, or none; each case gets `groups["subject"]` =
  the record's `subject_id`. Only the first `ceil(n / n_per_subject)` donors, whose records are kept, are
  simulated: GenAIRR draws each subject's seed from the cohort seed in subject order, so the kept records are
  those of a run over every donor, without compiling a simulator for the others.
- Case ids are `<suite>:<stratum>:<index:06d>` (`:fwd`/`:rc` appended for pairs). Orientations are drawn per case,
  in order, from the stratum's orientation seed.
- One spawned process per stratum (`--workers` caps them); a worker receives only names and numbers, resolves
  the suite from the registry and re-draws its donors from the same seed. The output does not depend on the
  worker count.

## Suite catalog

| Suite | Slug | Strata (cases at scale 1.0) | Presentation | Donors | Labels | Primary |
|---|---|---|---|---|---|---|
| `S0` | `toy` | `plain` 30, `with_d` 20 (accepts records with a D call) | rc 0.5 / 0 | none | — | `v.top1_in_set`, `j.top1_in_set`, `junction.nt_exact` |
| `S0_EXHAUST` | `exhaust` | `never` 5 (accepts nothing; raises) | — | none | — | — |
| `S0_COHORT` | `cohort` | `cohort` 10, `few` 2; cohort | rc 0.5 / 0 | `DonorSpec(3, "S0")` | — | `v.top1_in_set`, `j.top1_in_set` |
| `S1` | `core` | `{design}_{shm}` for the 4 read designs × `igm` 500, `igg` 500, `high` 250 = 5000; cohort | rc 0.5 | `DonorSpec(20, "S1")` | `design`, `shm` | `HEADLINE` |
| `S10` | `heldout` | `uniform` (uniform SHM model), `hkl_s5f` (S5F kernel `hkl_s5f`), `np_shift` (NP lengths +4), `trim_shift` (trims +3); each 600, full, rate 0.075; cohort | rc 0.5 | `DonorSpec(20, "S10")` | `heldout` | `HEADLINE` |
| `S11` | `orientation` | `full_igg` 500, `fr3_igg` 500 | both orientations (pairs) | none | `design` | `orientation.correct` + `HEADLINE` |
| `S12` | `noise` | `seq_err_0.005`, `seq_err_0.01` (`sequencing_errors` rate), `pcr_0.002` (`pcr_amplify` rate), `n_bases_3`, `n_bases_8` (`ambiguous_base_calls` count), `pol_indel_1`, `pol_indel_2` (`polymerase_indels` count); each 300, full, rate 0.075 | rc 0.5 | none | `noise` (the stratum name) | `HEADLINE` |

`HEADLINE` is `report.markdown.HEADLINE` (clone key, junction, V, J, V mutation count, D, productive). Suites may
not import `report`, so `suites.recipes.HEADLINE` repeats it and the suite tests pin the two equal.

The repertoire suites share `suites.recipes.repertoire(ctx, *, design, rate, model, s5f_model, np_shift,
trim_shift, noise)`, which chains, in this order:

1. `recombine()`, with every NP1 and NP2 length of `ctx.config.NP_lengths` shifted by `np_shift` when it is set;
2. `trim(...)` when `trim_shift` is set (GenAIRR requires it immediately after `recombine`): every length of the
   pooled trim marginals shifted by `trim_shift` and clamped to GenAIRR's cap for its side;
3. `mutate(model, rate, s5f_model)`, skipped at rate 0;
4. the `noise` step, a `(step name, keyword arguments)` pair applied verbatim;
5. `end_loss_5prime` and `end_loss_3prime` with the read design's ranges.

Read designs (`READ_DESIGNS`, end loss in nucleotides, inclusive; `(0, 0)` skips the step) are realistic
FRx → J products that keep the junction: `full` none; `fr1` 5′ 0–25; `fr2` 5′ 105–135; `fr3` 5′ 225–255
(J-anchored); every amplicon design also loses 0–5 at the 3′ end. SHM classes (`SHM_CLASSES`, Poisson per-base
rates of the S5F `hh_s5f` kernel) are `igm` 0.03, `igg` 0.075 and `high` 0.20.

`pooled_trim_marginals(config)` gives, per `trim()` keyword (`v_3`, `d_5`, `d_3`, `j_5`), the distribution
GenAIRR's `recombine()` samples by default: the mean over genes of `config.trim_dicts`, restricted to GenAIRR's
cap (the segment's shortest allele minus a 3-base anchor margin, halved for the two D sides) and normalised, so
`trim(**pooled_trim_marginals(config))` reproduces the default records exactly. GenAIRR aborts the whole run
when a sampled D trim pair exceeds the D allele, and the shortest human IGH D allele is 11 nt, so the cap is 4
per D side: S10's `trim_shift` moves V 3′ and J 5′ trims by a full +3 but D trims only to 3 or 4.
GenAIRR's record `np1_length` / `np2_length` can be a base shorter than the sampled NP length.

## Donors

Cohort strata simulate reads from donors (`suites.donors`), each a diploid genotype over the reference genes.
`sample_donors(reference, n, seed, params)` draws donors `donor00`, `donor01`, ... independently (each from its
own stream seeded by the seed and its id, so a smaller cohort is a prefix of a larger one). With the default
`DonorParams`:

- A Poisson(`mean_deleted_genes` = 6.7) number of genes is homozygously deleted, drawn uniformly without
  replacement from the `deletable` segments (V and D). J is never deleted, and every deletable segment keeps at
  least one gene.
- Every remaining gene with two or more reference alleles is heterozygous with probability `het_fraction` = 0.40;
  the others are homozygous. Alleles are uniform within a gene.

A `Donor` holds `alleles` (gene -> its one or two alleles, sorted; deleted genes absent), `deleted` (gene names)
and `novel` (name -> `core.genotype.NovelAllele`, for novel alleles carried under their base's gene).
`donor.view()` is the plain `core.genotype.DonorGenotype` (carried allele names, deleted genes) that truth,
metrics and scoring read. `donor.genotype(config)` is the strict GenAIRR `Genotype` the cohort runs (built with
`from_dataconfig`, then `homozygous` / `heterozygous` with the first sorted allele on haplotype 0, `delete_gene`
on both haplotypes, and `add_novel_allele` for novel alleles; tagged with the subject id).
`donor.tool_genotype(reference)` is what a tool is given in the genotype arm: per gene letter, the sorted
reference alleles carried, each novel allele replaced by its base.

`suites/<S>/donors.json` (`donors_to_json` / `donors_from_json`) holds `kind` (`alignair_assay.donors`),
`version` and per donor `subject_id`, `alleles`, `deleted` and `novel`. `build.json`'s `donors_summary`
(`donors_summary`) records the realised `donors` count, `mean_alleles_per_gene` (per gene letter and `all`),
`mean_deleted_genes` and `heterozygous_share` (of carried genes with two or more reference alleles). Over 300
human IGH donors, across seeds, the defaults realise 6.5 to 6.9 deleted genes per donor, a heterozygous share of
0.39 to 0.40 and 1.22 to 1.24 alleles per gene (V about 1.32).

## Adding a runner

A runner turns a FASTA of reads into an AIRR rearrangement TSV. It implements `runners.base.Runner`:

- `name`: the tool name, used as `runs/<tool>/` (one path component).
- `prepare(layout, reference) -> dict`: the tool's one-time setup for run directory `layout`, against the shared
  reference (already verified equal to the truth reference). Put setup files in `layout.work_dir(name)`. Return
  the setup's provenance, and raise (FileNotFoundError, ValueError, RuntimeError) when the tool cannot be set up.
- `run(fasta, out_tsv, *, suite, arm, log_dir, alleles=None) -> RunResult`: annotate every read of `fasta` into
  `out_tsv`, keeping logs and scratch files in `log_dir`. `alleles`, when given, restricts the tool to those
  reference alleles (the genotype arm): IgBLAST builds a database from them in `log_dir/db`, AlignAIR gets them as
  a `--genotype` JSON, and the oracle ignores them. A tool that fails must return a non-zero return code in the
  `RunResult`, not raise, so the failure is recorded and never scored. This also covers a restriction the tool
  cannot take, such as an allele outside its reference, a gene letter left with no allele or a database that
  does not build. Put the reason in the provenance's `error`: `runners.base.error_line(log)` picks the failed
  command's error line, and `runners.base.failed_run` records a run that failed before its command ran. Run
  external commands through `runners.base.timed_command`, which measures the wall time and the peak RSS (GNU
  `/usr/bin/time`), pins the command to `cpus` with `taskset` and logs its output.

`RunResult` holds `tool`, `suite`, `arm`, `predictions` (the TSV path), `n_reads` (reads given), `wall_seconds`,
`peak_rss_kb` (None when not measured), `returncode`, `commands` (every tool command executed, as argument lists
that can be re-run; environment variables as an `env` prefix; without the `time`/`taskset` wrappers, whose CPUs
the provenance records) and `provenance`. Provenance must identify what
ran: the tool's version text and the SHA-256 of its binaries or image, of its model or database files, and the
threads and CPUs. `run_suite` adds the reads' path and SHA-256 (`reads`) and writes the record as `run.json`,
removing the old one first, so a run that raises leaves none. Anything that scores a run first calls
`require_successful_run(run_json)`, which refuses a missing record or a non-zero return code, naming the tool,
suite, arm and return code.

To register a runner, write one module in `runners/` and add a factory `Callable[[RunnerOptions], Runner]` to
`RUNNERS` in `runners/__init__.py` (by convention a `from_options` classmethod). `RunnerOptions` holds every
tool's settings (`igblast_dir`, `threads`, `cpus`, the AlignAIR image, model, checkpoint, device, batch size and
cache directory); a factory reads the fields it needs and raises ValueError naming a missing one. Expose new
options as `run` flags. Test the runner in `tests/alignair_assay/runners/`; a test that needs the tool installed
skips unless its environment variable is set (`ALIGNAIR_ASSAY_IGBLAST_DIR`, `ALIGNAIR_ASSAY_DOCKER`,
`ALIGNAIR_ASSAY_MODEL`).

Registered runners:

| Tool | Runner | Setup (`prepare`) | Run |
|---|---|---|---|
| `oracle` | `runners.oracle.OracleRunner` | binds the run directory | writes `truth.oracle.oracle_prediction` rows for the reads of the FASTA, in its order, from the suite's truth (in-process: no command, no RSS) |
| `igblast` | `runners.igblast.IgBlastRunner` (`--igblast-dir`, `--threads`, `--cpus`) | checks the release; writes `germline_{v,d,j}.fasta` and builds the BLAST databases (`makeblastdb -parse_seqids -dbtype nucl`) in `runs/igblast/_work/` | `IGDATA=<dir> igblastn -germline_db_V/D/J <db>/V,D,J -auxiliary_data optional_file/human_gl.aux -organism human -ig_seqtype Ig -num_threads <threads> -outfmt 19 -query <fasta> -out <tsv>`; provenance: version text, SHA-256 of `igblastn`, `makeblastdb`, the auxiliary file and every database file, threads and CPUs |
| `alignair` | `runners.alignair.AlignAIRImage` (`--image`, `--model`, `--cache-dir`, `--threads`, `--cpus`) | inspects the image (pulling it when it is not local); installs the model into the cache (see AlignAIR model cache) with `models get`, then `models verify` and `models path`; records `models info` and `doctor --json`; exports the model's reference with `reference export <model file> --fasta` (3.0.0 crashes on a registry id there) into `runs/alignair/_work/` and refuses it unless it is the truth reference | `docker run --rm -u <uid>:<gid> -v <cache>:/home/appuser/.cache/alignair -v <fasta dir>:/in:ro -v <tsv dir>:/out [--cpuset-cpus <cpus>] [-e OMP_NUM_THREADS=<t> -e MKL_NUM_THREADS=<t>] --entrypoint sh <image> -c 'alignair "$@" 2>&1; rc=$?; cat /sys/fs/cgroup/memory.peak; exit $rc' sh predict --model <model> --input /in/<fasta> --out /out/<tsv> --device cpu --quiet [--genotype /in/genotype.json]`; the wall time includes the container's start; the peak RSS is the container's cgroup v2 `memory.peak` (resident memory plus page cache; None with an `rss_note` when unreadable); provenance: image reference, ID and RepoDigests, the cache record, model spec, path and SHA-256, the model card, the doctor output, the exported reference, the moved `predict` sidecar (`logs/alignair.run.json`), the genotype file, threads and CPUs |
| `alignair-local` | `runners.alignair.AlignAIRLocal` (`--checkpoint`, `--device`, `--batch-size`, `--threads`, `--cpus`) | checks the checkpoint; probes the torch version and CUDA device the subprocess sees; exports the checkpoint's reference into `runs/alignair-local/_work/` with this checkout's CLI and refuses it unless it is the truth reference | `PYTHONPATH=<checkout>/src [OMP_NUM_THREADS=<t> MKL_NUM_THREADS=<t>] <python> -c "import sys; from alignair.cli.main import main; sys.exit(main())" predict --model <checkpoint> --input <fasta> --out <tsv> --device <device> --batch-size <n> --quiet [--genotype <log_dir>/genotype.json]` under `timed_command` (peak RSS of the CLI process); provenance: checkpoint path and SHA-256, the checkout's path and commit state, interpreter, torch version, CUDA device name, device, batch size, the exported reference, the moved sidecar, the genotype file, threads and CPUs |

### AlignAIR model cache

The image keeps its models in `/home/appuser/.cache/alignair`, where the runner mounts its cache directory:
`--cache-dir`, or by default `runs/alignair/_work/cache` of the run directory, so a run neither depends on nor
changes the user's cache. When the model is pinned (`id@version`) and the cache directory lacks it, `prepare`
copies `models/<id>/<version>.alignair` and its `.sha256` from the user's AlignAIR cache (`ALIGNAIR_CACHE_DIR`,
else `$XDG_CACHE_HOME/alignair`, else `~/.cache/alignair`; only read). `models get` then downloads the model
only when it is still missing. The run records which happened as `cache.model_source`: `cache_dir` (already
there), `host_cache` (copied from `cache.seeded_from`) or `registry` (downloaded). `models verify` re-hashes it
against the registry and stops the run on a mismatch; offline it reports `UNKNOWN` and the run goes on.

### The genotype arm

`run --arm genotype` (`runners.arms.run_genotype_arm`) runs a tool on a suite with donors one donor at a time, each
time restricted to what that donor carries. The suite's reads are split by their cases' `subject` group (a read
without a case, a case without a subject or a subject that is not a donor is an error). Each donor with reads gets
`runs/<tool>/<S>/genotype/donors/reads.<subject>.fasta`, in suite order, and one `Runner.run` with `alleles` =
the flattened `donor.tool_genotype(reference)`, where a novel allele is given as its base. That run writes
`predictions.<subject>.tsv` and keeps its logs in `logs/<subject>/`. IgBLAST builds the donor's database in
`logs/<subject>/db`. AlignAIR gets `{"v": [...], "d": [...], "j": [...]}` as `--genotype` with the default
`mask` method. The image reads it with a copy of the donor's FASTA from `logs/<subject>/input/`, the local runner
from `logs/<subject>/genotype.json`. Donors without cases are skipped and listed as `skipped_donors`.

The donors' TSVs are then concatenated into the arm's `predictions.tsv` under one header. A donor whose header
differs from the first donor's, or whose TSV has a row for a read that is not its own, stops the arm with a
ValueError naming the donor, and no run record is written. The arm's `run.json` sums the donors' wall times,
takes their largest peak RSS and lists their commands in order. Its provenance holds per donor the reads, wall
time, peak RSS, return code, allele counts and the run's provenance, plus `skipped_donors`, `donors_json`
(path, SHA-256) and `reads`. The donors run in `donors.json` order and stop at the first failed run. The arm is
then recorded with that return code, the `failed_donor` and its `error` line, and is not scored.

AlignAIR 3.0.0's `predict --genotype` crashes (`AttributeError: 'tuple' object has no attribute 'items'`):
`alignair/cli/predict.py` passes `load_genotype`'s `(genotype, unknown)` tuple on as the genotype. Until a fixed
AlignAIR is released, every AlignAIR genotype-arm run is recorded as failed with that error line.
