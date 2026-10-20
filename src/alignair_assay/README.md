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
| `core` | shared primitives: `reference` (the germline reference from a dataconfig or a FASTA, `DEFAULT_REFERENCE`, `dataconfig`), `cigar`, `seqops`, `prediction` (normalized prediction records), `jsonl`, `cases` (case schema, case and FASTA files), `provenance`, `layout` (`RunLayout`, the run-directory path schema), `genotype` (`NovelAllele`, `DonorGenotype`: donor genotypes as plain data), `headline` (`HEADLINE`, the headline metrics shared by suites and reports) | core |
| `truth` | observable truth (pure): `model`, `segment`, `boundaries`, `alleles`, `genairr` (record decoding), `build`, `bench`, `case` (truth from a case), `io`, `oracle`, `summary` (what a truth set could not score) | core, truth |
| `airr` | AIRR TSV `reader` and `writer` | core, airr |
| `metrics` | per-case scorers (pure), the group scorers that read a case's whole clone (`groups`), the scorers against a read's donor genotype (`genotype`), and the metric registries | core, truth, metrics |
| `stats` | aggregation with coverage and the paired comparison (pure) | core, metrics, stats |
| `report` | JSON reports and markdown summaries (`build`, `markdown`), and the run report rendered from a run summary (`run`) | core, stats, report |
| `scoring` | `pipeline` (score one tool against a truth file), `slices` and `oracle` (the oracle rule on per-case scores) | core, truth, airr, metrics, stats, scoring |
| `suites` | the suite registry (`__init__`), suite types (`base`), simulated donors (`donors`), novel alleles (`novel`: invariant positions and validated SNP placement), the deterministic generator (`engine`), shared repertoire recipes (`recipes`) and the catalog (`catalog/`, one module per suite) | core, suites |
| `runners` | one runner per tool (`igblast`, `alignair`: the AlignAIR image and a local checkpoint, `oracle`) and the registry (`__init__`: `RUNNERS`, `make_runner`), the runner interface, run records and timed commands (`base`), the shared reference (`reference`), the genotype arm (`arms`) | core, airr, truth, suites, runners |
| `workflow` | build → run → score → report orchestration over a run directory: `build` (one suite), `scoring` (`score_file`, `read_genotypes`, score and comparison reports and their files, shared by `score`, `compare` and `all`), `summary` (the run summary, read back from the run directory) and `orchestrate` (`RunConfig`, `run_all`, `score_and_report`) | everything except cli |
| `cli` | the `python -m alignair_assay` commands | everything |
| `testing.py` | test-only helpers: `assert_oracle_perfect` (the oracle rule of `scoring.oracle`, with the donor genotypes when given; returns the oracle's per-case scores) | everything |

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
key; a label named like a fixed dimension is an error. The fixed dimensions are `stratum`, `orientation`,
`v_len_bin` (observed V germline bases: `absent`, `<20`, `20-49`, `50-99`, `100-149`, `150-199`, `200-249`,
`>=250`), `d_len_bin` (observed D germline bases: `absent`, `<5`, `5-7`, `8-9`, `10-11`, `12-14`, `>=15`) and
`d_inverted` (`yes`, `no`, `absent`).

**Observable truth.** Each simulated record is projected onto what the read shows (`truth.build`):

- A segment is observed when at least one of its germline bases is in the read. An unobserved segment has no call
  or coordinate truth; a tool that calls it scores `{g}.spurious_call`.
- The observed alignment drops edge gaps: a clipped or edge-deleted germline base cannot be told apart from extra
  trimming.
- The primary allele is the generative one, the allele GenAIRR sampled (`truth_{g}_call`; the suite engine
  simulates with `expose_provenance=True`). GenAIRR's `{g}_call` is a best-match re-call made after SHM, which
  differs from the generative allele in a few percent of mutated reads. A record without `truth_{g}_call` (bench
  cases, records simulated without provenance) falls back to the first entry of `{g}_call`, and the truth summary
  counts those fallbacks per gene (`primary_fallbacks`).
- Coordinates, CIGAR and SHM truth are observed against the generative allele. When it is not in the call list,
  GenAIRR projects the CIGAR edges onto the re-called allele (a deletion fill to that allele's end, NP bases it
  matches claimed as germline) while the core alignment stays generative. `truth.genairr` undoes this: a re-called
  forward segment is clipped to the generative allele's trimmed germline `[trim_5, len - trim_3)`, and a re-called
  inverted D has its fills decoded with the re-called allele's length (or is dropped as unscorable when they do not
  determine its span). That both reproduce GenAIRR's own generative alignment is measured, not proven: on 3000
  records at SHM rate 0.15 every re-called segment matched it (V 187, D 182, J 8; inverted D 190 with every D
  inverted). Known limitation: GenAIRR's fills stop at the projected allele's length, so when the re-called allele
  is shorter than the generative one, a generative deletion at or past the re-call's length is not in the CIGAR
  and cannot be recovered (only records with indel noise can have one; SHM makes no indels).
- The allele truth is a set. `AlleleTruth.generative` holds every reference allele whose sequence contains the
  observed germline of the generative allele. The scored set `AlleleTruth.alleles` adds the simulator's calls that
  are in the reference, since a sequence-only aligner can defend either. The truth summary
  (`truth.summary.simulator_calls`) counts two things per gene. `call_list_differs` counts the cases whose
  simulator call list is not exactly the generative set (`AlleleTruth.disagrees_with_simulator`); most of these
  are tie lists that include the generative allele. `recalled` counts the cases whose first simulator call is
  another allele (`AlleleTruth.recalled`): the reads GenAIRR re-called after SHM. GenAIRR lists the generative
  allele first whenever it lists it, so a re-call is a non-empty call list without it; a fallback primary is never
  re-called. Truth files written before the generative primary (version 1) load with `generative` equal to
  `alleles` and `primary_fallback` True.
- The scored set of a forward segment (V, J and forward D) also holds its as-near alleles
  (`truth.build.as_near_alleles`): the other alleles of the primary's gene that cover the observed germline span
  and, compared position by position through the truth alignment's aligned columns, mismatch the read at no more
  columns than the primary (a read `N` never mismatches). A sequence-only aligner cannot tell them from the
  primary on that read. GenAIRR's re-call list does not always show them: it leaves out some ties, and in a cohort
  it is restricted to the donor's genotype. Scoring only what it lists would reward GenAIRR's conventions, which
  the assay must not do. They join `alleles` only: `generative` (so `top1_generative`), the primary and
  `simulator_ties` are unchanged. At the default seed (dev, scale 1.0) the rule grows the scored set of V on 55
  of 5000 S1 reads and D on 2 (J none), and V on 128 of 3000 S7 reads (107 of them novel). An inverted D keeps
  GenAIRR's list only. Remaining limitation: alleles of another gene that come as near are scored only when
  GenAIRR lists them, and in cohort suites that list is genotype-restricted. Truth version 3 marks this rule
  (no field changed), so older builds are redone rather than reused.
- A primary allele that is not in the reference makes its segment unscorable, except a novel V allele of the
  read's donor (`truth_from_case(case, reference, novel=donor.novel)`; `workflow.build` passes each case its own
  donor's, since novel names are per donor). Its truth is what a perfect aligner restricted to the reference
  reports. The segment is observed against the novel sequence and then re-based onto the novel allele's base:
  the same coordinates and CIGAR, the base's germline, and mismatches counted against the base, so every SNP the
  read shows is one. Boundaries, the allele set (the base's observable set) and the junction anchors come from
  the base. In GenAIRR's call list the novel name (GenAIRR calls the novel allele) becomes the base. The as-near
  alleles are the base's, and they matter most here: GenAIRR's re-call compares the read with the novel allele,
  so it can never show that SHM brought the read as near to a sibling as to the base. On S7 at the default seed
  that is 107 of 1568 novel reads (22 strictly nearer); no allele of another gene comes as near.
  `ObservableTruth.v_novel_snps` holds the read positions (canonical frame, ascending) aligned to the
  novel allele's SNPs inside the observed germline span `[germ_start, germ_end)`; it is None for every other
  read. A novel allele whose base is not in the reference, or whose sequence does not differ from the base's
  exactly at its SNPs, raises ValueError naming it. SNPs sit only where every allele of the base's gene agrees
  with the base (`suites.novel`), so over any germline span the base is strictly the nearest allele of its gene
  to the novel allele; only SHM can bring a read nearer to a sibling, which the as-near alleles cover.
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

**Allele metrics** (`metrics.alleles`, per observed segment `g`):

- `{g}.top1_in_set`: the top-1 call is in the scored set (generative set plus the simulator's re-call and
  the as-near alleles).
- `{g}.top1_generative`: the top-1 call is in the generative set, i.e. true-allele accuracy.
- `{g}.gene_top1`: the top-1 call's gene is a gene of the scored set.
- `{g}.any_overlap` and `{g}.set_precision`: the overlap of the tool's call set with the scored set.
- `{g}.set_size` (informational): the size of the tool's call set.
- `{g}.no_call` (lower is better): the tool made no top-1 call.

**Minimum observed lengths.** `metrics.case.MIN_OBSERVED = {"v": 20, "d": 8, "j": 10}` observed germline bases.
Below its minimum a segment gets only `{g}_short.top1_in_set` and `{g}_short.abstain_ok`, for which abstaining is
as correct as a right call. A missing row is a failure, not an abstention.

**Group metrics** (`metrics.groups`, merged into each case's scores by `scoring.pipeline.score_predictions`):

- `clone.agree_modal`: the tool gives the case the same clone assignment as the rest of its clone, by the "first
  call" grouping convention. A case's predicted key is (gene of its top-1 V call, gene of its top-1 J call,
  junction length); it is None when any part is missing, including a missing row. The clone's modal key is its
  most common non-None key, ties going to the key with the smallest `repr`. The case scores 1.0 when its key is
  the modal key and 0.0 otherwise (a None key included). It applies to cases with `groups["clone"]` and a truth
  clone key, in clones with at least two such cases. Set-overlap clone keys are scored per read by
  `clone_key.exact`. Members of a clone share their recombination, so the oracle's key (the generative alleles'
  genes and the truth junction length) is the same across a clone and the oracle always agrees.

**Novel-allele metric** (`metrics.novel`, informational):

- `v_novel.snp_mutations`: on a read whose V is a novel allele with SNPs in the observed V
  (`truth.v_novel_snps`), how many of those read positions the tool's V alignment aligns to a germline base
  that differs from the read base (a read `N` never does; a position the alignment does not cover does not
  count). `nan` without a usable V alignment. A fixed-reference tool aligning the base reports every SNP the read
  shows, and so does the oracle; a tool that inferred the novel allele reports none. It is reported but not
  compared.

**Genotype metrics** (`metrics.genotype`, lower is better). With donor genotypes
(`scoring.pipeline.score_predictions(..., genotypes=)`: subject id -> `Donor.view()`), every scored case with
`groups["subject"]` is also scored against its donor's genotype; a subject without a genotype is a ValueError
naming the case. They apply to V and J when the segment is observed with at least `MIN_OBSERVED` bases and is
scorable, and are `nan` without a top-1 call or without a row. A call in the scored truth set is defensible
whatever the genotype (a simulator re-call, an as-near allele, a novel allele's base), so only calls outside it
count:

- `{g}.outside_genotype` (V, J): the top-1 call is neither carried by the donor nor in the scored truth set.
- `v.deleted_gene_call`: the top-1 V call's gene is one of the donor's deleted genes, and the call is not in the
  scored truth set. J genes are never deleted.

The oracle calls each read's generative allele, which its donor carries (a novel allele's base is in the truth
set), so it scores 0.0 on both. Every suite with donors (S1, S7, S8, S10) gets them under `all`, and `score` and
`compare` with `--donors`.

**Excess alleles** (`metrics.genotype.excess_alleles(truths, preds, genotypes, min_reads=3)`, donor level, not a
per-case metric). Per gene letter (V, J), the reads that count are those the genotype metrics apply to with a
top-1 call, grouped into (donor, gene of the top-1 call) pairs. A pair with at least `min_reads` reads is
considered, and it is an excess when its distinct top-1 alleles called on at least `min_reads` reads each
outnumber the alleles of that gene the donor carries (none for a deleted gene). Counting all calls, a called
allele that is in the read's truth set but not carried (one identical to a carried allele over the read, say)
counts towards an excess, so a perfect sequence-only tool can have one where the oracle has none. The avoidable
count therefore leaves those defensible calls out, the same calls the per-read metrics excuse; calls of carried
alleles and calls outside the truth set count as before, so an avoidable excess is always an excess, and a
perfect sequence-only tool never has one. The table holds `min_reads` and per letter `n` (pairs considered, the
same for both counts), `excess` and `fraction` (all calls), and `avoidable_excess` and `avoidable_fraction`
(fractions None when `n` is 0). A tool's score report holds it as `excess_alleles` when it was scored with
donors, and both reports tabulate both fractions with n next to each. The oracle never has an excess.

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
  scores/<S>/<arm>/                   <tool>.scores.jsonl.gz, <tool>.report.{json,md}, comparison.{json,md}
  speed/speed.json
  report.json, report.md              the run summary and the run report (`all`, `report`)
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
| `scores/<S>/<arm>/<tool>.scores.jsonl.gz`, `.report.json`, `.report.md` | `tool_scores(s, arm, tool)`, `tool_report_json(...)`, `tool_report_md(...)` |
| `scores/<S>/<arm>/comparison.json`, `comparison.md` | `comparison_json(s, arm)`, `comparison_md(s, arm)` |
| `speed/`, `speed.json` | `speed_dir`, `speed_json` |
| `report.json`, `report.md` | `report_json`, `report_md` |

Suites are `S0` (toy: tests and smoke runs), `S0_EXHAUST` and `S0_COHORT` (tests only) and `S1`–`S12`; arms are
`shared` and `genotype`; splits are `dev` (the default, for model iteration) and `test` (locked, used only for
claims).

## Commands

Run from the repository root with `PYTHONPATH=src`:

```
python -m alignair_assay all --run-dir RUN --suites S1,S10,S11,S12 --tools igblast,alignair [--quick] \
    [--split dev] [--reference HUMAN_IGH_OGRDB] [--seed 20261019] [--scale 1.0] [--workers 8] [--threads 8] \
    [--igblast-dir DIR] [--alignair-image IMAGE] [--alignair-model alignair-igh-human@1.0.0] \
    [--alignair-checkpoint MODEL.alignair] [--device cpu] [--speed] [--force]
python -m alignair_assay report --run-dir RUN [--a igblast --b alignair]
python -m alignair_assay build --suites S0,S1 --run-dir RUN [--split dev] [--reference HUMAN_IGH_OGRDB] \
    [--seed 20261019] [--scale 1.0] [--workers 8]
python -m alignair_assay run --run-dir RUN --tool igblast|alignair|alignair-local|oracle --suites S0,S1 \
    [--arm shared|genotype] [--threads 8] [--cpus 0-7] [--igblast-dir DIR] \
    [--image IMAGE] [--model alignair-igh-human@1.0.0] [--cache-dir DIR] \
    [--checkpoint MODEL.alignair] [--device cpu] [--batch-size 64]
python -m alignair_assay truth --bench-cases CASES.jsonl[.gz] [--reference HUMAN_IGH_OGRDB] --out TRUTH.jsonl.gz
python -m alignair_assay score --truth TRUTH --predictions AIRR.tsv --tool NAME --out-dir DIR [--donors DONORS.json]
python -m alignair_assay compare --truth TRUTH --a A.tsv --a-name A --b B.tsv --b-name B --out-dir DIR [--alpha 0.05] \
    [--donors DONORS.json]
```

- `all` (`workflow.orchestrate.run_all`) does everything over one run directory, and can be rerun to resume. See
  The all command below.
- `report` (`workflow.orchestrate.score_and_report`) re-scores, re-compares and rewrites the run report of a run
  directory made by `all`, from its existing runs and with the configuration in its `manifest.json`; nothing is
  built or run. `--a`/`--b` may swap the run's two tools; any other tool is refused, since the manifest records no
  settings to check its runs against. A run is scored only when it matches the manifest (see step 5 below):
  one made with other tool settings (a manual `run`, or an `all --force` that stopped before redoing it), on
  changed files, or on a suite whose build no longer matches is listed as `stale` and not scored. It exits with
  status 2 under the same conditions as `all`, and when the directory has no manifest.
- `build` generates each named suite into `RUN` (`workflow.build.build_suite`): `cases.jsonl.gz`, `truth.jsonl.gz`,
  `reads.fasta`, `donors.json` (suites with donors only) and `build.json`. Every name is checked before anything is
  built; an unknown suite or split, or a scale that is not positive, exits with status 2. The truth header records
  `kind`, `version`, `reference`, `reference_digest`, the GenAIRR provenance, `suite`, `split`, `seed`, `scale` and
  `n_cases`. `build.json` records the same configuration, its `config_hash` (SHA-256 of the canonical JSON of
  `{suite, split, reference_digest, seed, scale}`), the per-stratum generation stats (`requested` in
  `requested_unit`: `clones` for a clonal stratum, `records` otherwise; `accepted`, `generated` and `cases` count
  records; `rounds`; `clones` for a clonal stratum), the truth summary
  (`truth.summary.truth_summary`: excluded cases, warnings, unscorable genes, `simulator_calls`, primary
  fallbacks), the worker count and the wall time in `seconds`; a suite with donors adds `donors_summary`
  (see Donors). It is removed first and written last, so it marks a complete build. The command prints one line per
  suite.
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
  command then prints the excluded cases, warnings, unscorable genes, the two simulator-call counts ("simulator
  call list differs from the generative set" and "simulator first call is not the generative allele", see Truth
  conventions) and primary fallbacks.
- `score` writes `<tool>.scores.jsonl.gz`, `<tool>.report.json` and `<tool>.report.md`.
- `compare` writes `comparison.json` and `comparison.md`, and prints the headline table. Its 95% and simultaneous
  CIs of 0/1 metrics are clipped to [-1, 1].
- `--donors` (`score`, `compare`) takes a suite's `donors.json` and scores each read with a subject against its
  donor's genotype (`workflow.scoring.score_file`): the genotype metrics join the scores and the comparison family,
  the score report holds the excess-alleles table, and the provenance records the donors file as an input. A read
  whose subject the file does not have exits with status 2. Without it, nothing changes.
- `score`, `compare` and `all` build these reports with the same functions (`workflow.scoring`). Their provenance
  records the `command`, each input's path and SHA-256, the checkout's git state, the matching counts and the
  truth's `simulator_calls`. The markdown leads with the headline metrics (`core.headline.HEADLINE`; under `all`,
  the suite's primary metrics), which are also the rows of its slice tables.
- Predictions are matched to truth by AIRR `sequence_id` == truth `case_id`. A truth case without a prediction is
  scored as missing, never dropped.
- A missing file, an unknown reference, a truth file built against another reference, a prediction file
  matching no case, `compare` given the same name for both tools, an unknown tool, an unbuilt suite, a shared
  reference that is not the truth reference, or a failed tool run exits with status 2.

### The all command

`all` runs these steps in order (`workflow.orchestrate.run_all`). It prints one line per step.

1. **Manifest.** The configuration is checked before anything is written (`check_config`). It needs known suites,
   a known split, exactly two different registered tools, a positive scale, at least one worker and thread, and
   the options each tool needs (`--igblast-dir` for igblast, `--alignair-checkpoint` for alignair-local), checked
   by building its runner, which runs, pulls and downloads nothing; `--speed` is not available yet. Each tool's
   runner is then built and fingerprinted (`Runner.fingerprint`, see Adding a runner), which
   identifies the tool by content: IgBLAST's directory and the SHA-256 of `igblastn`, `makeblastdb` and the
   auxiliary file; the AlignAIR image reference and the image ID it resolves to (pulling it when it is not local),
   and the model spec and the SHA-256 of the model file it resolves to (installing it into the run's model cache);
   the local checkpoint's path, SHA-256 and device; nothing for the oracle. Then `manifest.json` is written,
   holding the `config`, its `identity`, the `config_hash` (SHA-256 of the canonical JSON of the identity), the
   split, the host, the checkout's git state, the GenAIRR provenance and `warnings` (an AlignAIR image not pinned
   by digest, or a model not pinned to a version). The identity is what the outputs depend on: suites, tools,
   split, reference, seed, scale and the tools' fingerprints (`settings`). Workers, threads and `--speed` are not
   part of it, so an upgrade in place, a moved image tag or a retrained checkpoint is a new configuration even
   when every path and name is the same.
2. **Build.** Each suite is built, unless its `build.json` already has the suite's config hash and its truth file
   the current truth version.
3. **Shared reference.** With `alignair` among the tools and no `reference/germline.fasta` yet, the image exports
   its model's reference there. `ensure_shared_reference` then verifies the file against the truth reference, or
   writes it from the truth. A model with another reference stops the run before any tool runs, and its export is
   removed.
4. **Runs.** The oracle runs on every suite's shared arm. Then each tool runs on each suite and each of the
   suite's arms, the shared arm through `run_suite` and the genotype arm through `run_genotype_arm`. Every tool
   with a run to make is prepared before the first run. Each run records its tool's fingerprint as the
   provenance's `settings`.
5. **Scoring.** Every suite and arm is scored afresh (`score_and_report`), after its old `scores/<S>/<arm>/` is
   removed. A run is scored only when its `workflow.summary.run_statuses` entry is `ok`: `require_successful_run`
   passes, its predictions exist, its recorded `settings` equal the manifest's fingerprint of its tool, the reads
   (and the truth and donors, when the run recorded them) still have their recorded SHA-256, and its suite's build
   matches the configuration (`workflow.build.build_is_current`). Anything else is `missing`, `failed` or
   `stale`, and is listed, never scored. Each scored run gets the `score` command's three files, the oracle on the
   shared arm. A suite with a `donors.json` is scored as `score --donors` scores it: against each read's donor
   genotype, with the donors file recorded as an input; when its `donors.json` cannot be read, every run of that
   suite is listed as `not scored`, with the reason, and the other suites are scored as usual. When both tools are
   scored, the arm gets the `compare`
   command's `comparison.json` and `comparison.md` (a against b, Holm at 0.05).
6. **Report.** `report.json` holds the run summary (`workflow.summary.collect_run_summary`) and `report.md` its
   rendering (`report.run.run_report_markdown`).

`all` exits with status 0, or with status 2 when any run failed, is missing, stale or could not be scored, or the
oracle is not perfect on a suite. The report names which.

**Resume.** Rerunning `all` with the same configuration skips each suite whose build is current and each run whose
`run.json` shows return code 0 with predictions, the same `settings` (the tool's current fingerprint) and
unchanged input files. A failed run is
redone. Scores, comparisons and the report are always recomputed from the current runs, so a change to the scoring
code takes effect on the next `all`. It never mixes in stale outputs.

**Another configuration.** In a run directory whose manifest has another config hash, `all` stops before changing
anything. The error names every field that differs, as `field old -> new`, for example `seed 20261019 -> 7`.
`--force` replaces the manifest; its `replaced` records the old hash and the changed fields. The steps then redo
what the change affects. A new seed, split, scale or reference rebuilds the suites, which changes their reads, so
every run is redone. A changed tool fingerprint (`<tool>.<key>` in the error) redoes that tool's runs only. Outputs of suites or tools no longer in the
configuration stay on disk but are not reported. `--quick` is scale 0.02 (S1 then has 100 cases); an explicit
`--scale` wins over it.

**The run report** (`report.md`) has these sections; a section without data is left out.

1. **Failed runs.** Every run that failed, is missing, stale or could not be scored: tool, suite, arm, status,
   return code, failed donor, the tool's error line and the run record. These runs are not scored, and the
   comparisons that need them are left out.
2. **Oracle check failures.** Suites where the oracle breaks the oracle rule, with the first offending cases. Such
   a suite has a truth defect.
3. **Summary.** Per suite and arm: cases, excluded cases, the oracle check, and how many of the suite's primary
   metrics each tool wins and how many are tied or inconclusive.
4. **Suites.** Per suite: title, purpose, and per arm the primary-metric rows of its comparison, linked to its
   `comparison.md`.
5. **S10 vs S1 (held-out settings).** Per tool, S10's primary-metric means against those of S1's `full_igg`
   stratum, which has S10's read design and SHM rate under the default distributions. The table shows the
   difference and the mean per held-out setting, each mean with its n and coverage. These are unpaired and
   informational: no test is made.
6. **S11 orientation parity.** Per tool, the forward reads against their reverse-complemented twins, paired by
   `groups["pair"]` and compared with `stats.compare` (Holm at 0.05 across the tool's metrics). The section lists
   the metrics that differ and the primary-metric rows.
7. **Genotype arms.** For suites with a genotype arm, each tool's primary-metric means in each arm, with n and
   coverage.
8. **Excess alleles.** For suites with donors, the excess-alleles table (see Genotype metrics) of each run: the
   oracle's shared arm (always 0%), then each tool in each arm: for V and J, the share counting all calls and the
   avoidable share, each with its n.
9. **Speed.** Present when `speed/speed.json` exists.

To add a command, write one module in `cli/` with `register(subparsers)` and `run(args) -> int`, and add it to
`cli.COMMANDS`. The module owns its own input checks: `run` raises ValueError (or FileNotFoundError) for invalid
input, which `cli.main` prints as `error: ...` with exit status 2.

## Adding a metric

1. Write the scorer in the `metrics/` module of its kind (`alleles`, `boundaries`, `junction`, `shm`, `flags`,
   `novel`, `groups` for a metric that reads the predictions for a case's whole group, or `genotype` for one that
   reads the case's donor genotype). It returns `CaseScores` named `<scope>.<metric>` and follows the NaN rules
   above.
2. Call it from `metrics.case.score_case`; a group or genotype scorer is merged instead by
   `scoring.pipeline.score_predictions`. Every case is scored through `score_predictions`, so the pipeline and the
   oracle check see the same metrics.
3. Declare its status in `metrics/__init__.py`. A name ending in one of the `_LOWER_IS_BETTER` suffixes is
   lower-is-better. A metric that is reported but not compared goes into `INFORMATIONAL`.
4. Extend the oracle tests in `tests/alignair_assay/metrics/` so the oracle scores perfectly on it, and add a test
   for a wrong prediction.
5. For a headline metric, add it to `core.headline.HEADLINE`.

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
   `testing.assert_oracle_perfect` on its truth (with `genotypes=workflow.scoring.read_genotypes(donors.json)` for
   a suite with donors, as `all` scores it). A suite that fails the oracle check has a truth defect.

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
| `S2` | `siblings` | `shm_0`, `shm_0.03`, `shm_0.08`, `shm_0.15`, `shm_0.25` (SHM rate); each 600, full; V restricted to `sibling_alleles(reference, "v", 5)` | rc 0 | none | `sibling_distance` (per case: `1`, `2`, `3`, `4-5`) | `v.top1_in_set`, `v.set_precision`, `v.gene_top1`, `j.top1_in_set` |
| `S3` | `junction` | `{sweep}_{value}`: `np1`, `np2` ∈ {0, 3, 6, 12, 20}, `v3` ∈ {0, 2, 5, 10}, `j5` ∈ {0, 4, 8, 14}, `d_both` ∈ {0, 4, 8} (nt); each 150, full, rate 0.02 | rc 0 | none | `sweep`, `value`, `p_nt` (per case: `yes`/`no`) | `junction.nt_exact`, `junction.length_exact`, `clone_key.exact`, `junction.start_abs_err`, `junction.end_abs_err`, `v.seq_end_exact`, `j.seq_start_exact` |
| `S4` | `truncation` | `5p_{k}` (exact 5′ loss, k = 0, 25, …, 300), `3p_{k}` (exact 3′ loss, k = 0, 10, 20, 30, 40); each 150, rate 0.05 | rc 0 | none | `sweep` (`5p`/`3p`), `loss` | `v.top1_in_set`, `v_short.abstain_ok`, `v.germ_start_err`, `j.top1_in_set`, `j.germ_end_err`, `v.spurious_call`, `junction.nt_exact` |
| `S5` | `indels` | `v_{k}` (exactly k polymerase indels, all in V; k = 1, …, 6), `j_1` (exactly one, in J); each 300, full, rate 0.03; rejection sampling on `n_v_indels` / `n_j_indels` and `n_indels` | rc 0 | none | `indels` (the stratum name) | `v.indel_found`, `v.indel_false`, `v.seq_end_exact`, `junction.nt_exact`, `v.top1_in_set` |
| `S6` | `d_observability` | `trim_{k}` (k nt at both D ends, k = 0, 3, 6, 9, 12) 400 each; `inverted` (every D inverted, default trims) 500; full, rate 0.02 | rc 0 | none | `d_trim` (k or `default`), `inverted` (`yes`/`no`) | `d.top1_in_set`, `d_short.abstain_ok`, `d.seq_start_exact`, `d.seq_end_exact` |
| `S7` | `novel` | `cohort` 3000; full, rate 0.075; cohort over donors that each carry 25 V genes as a homozygous novel allele (1–6 SNPs, cycling); arms `shared` and `genotype` | rc 0.5 | `DonorSpec(10, "S7", DonorParams(novel_v_genes=25))` | `novel_snps` (per case: the SNP count of its generative V allele, or `none`) | `v.top1_in_set`, `v.gene_top1`, `v.mutation_count_err`, `clone_key.exact`, `junction.nt_exact` |
| `S8` | `genotype` | `cohort` 3000; full, rate 0.075; cohort over donors with heterozygous and deleted genes; arms `shared` and `genotype` | rc 0.5 | `DonorSpec(10, "S8")` | — | `v.outside_genotype`, `v.deleted_gene_call`, `j.outside_genotype`, `v.top1_in_set`, `clone_key.exact` |
| `S9` | `clonal` | `lineages` 60 clones (`clonal_lineage`: up to 25 sampled cells per clone, per-base SHM rate 0.008 per generation; median V mutation rate 0.075); full; clonal | rc 0.5 | none | — | `clone_key.exact`, `clone.agree_modal`, `junction.nt_exact`, `v.top1_in_set` |
| `S10` | `heldout` | `uniform` (uniform SHM model), `hkl_s5f` (S5F kernel `hkl_s5f`), `np_shift` (NP lengths +4), `trim_shift` (trims +3); each 600, full, rate 0.075; cohort | rc 0.5 | `DonorSpec(20, "S10")` | `heldout` | `HEADLINE` |
| `S11` | `orientation` | `full_igg` 500, `fr3_igg` 500 | both orientations (pairs) | none | `design` | `orientation.correct` + `HEADLINE` |
| `S12` | `noise` | `seq_err_0.005`, `seq_err_0.01` (`sequencing_errors` rate), `pcr_0.002` (`pcr_amplify` rate), `n_bases_3`, `n_bases_8` (`ambiguous_base_calls` count), `pol_indel_1`, `pol_indel_2` (`polymerase_indels` count); each 300, full, rate 0.075 | rc 0.5 | none | `noise` (the stratum name) | `HEADLINE` |

`HEADLINE` is `core.headline.HEADLINE` (clone key, junction, V, J, V mutation count, D, productive), the one list
the suites and the reports share.

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
GenAIRR's record `np1_length` / `np2_length` can be a base shorter than the sampled NP length (see S3 below).

The targeted suites (S2, S3, S4, S6) build their own chains from `recipes.mutated(experiment, rate, ...)` (the SHM
step, skipped at rate 0) and `recipes.symmetric_d_trim(ctx, per_side)`; S4 extends `repertoire(ctx, rate=0.05)`
with an exact `end_loss_5prime(length=(k, k))` or `end_loss_3prime(length=(k, k))`. All of them present every
read forward (rc 0) and mutate with the S5F `hh_s5f` kernel; S2, S3 and S6 reads have no end loss.

- **S2.** `s02_siblings.sibling_alleles(reference, letter="v", max_distance=5)` gives each allele's minimum Hamming
  distance to another allele of the same segment and equal length, keeping those within `max_distance` (164 of
  the 198 human IGH OGRDB V alleles). V is sampled uniformly from them (`restrict_alleles(v=...)`), and each case
  is labelled with its generative V allele's distance (`0` would mark an identical sequence under another name).
- **S3.** One variable per stratum, the others at GenAIRR's defaults: `recombine(np1_lengths=[(v, 1.0)])` (or
  `np2_lengths`), `trim(v_3=[(v, 1.0)])` (or `j_5`; `trim()` keeps the default distribution of every side it is
  not given), or `symmetric_d_trim` for `d_both`. Trims are exact on every record. The record's NP length is
  GenAIRR's annotation, which is one base shorter than requested on about 1% (NP1) and 2.5% (NP2) of records,
  without SHM or end loss as well: GenAIRR claims an NP base for the neighbouring V, D or J when it matches that
  segment's trimmed germline (or, for a segment GenAIRR re-called, the re-called allele's). Strata are labelled
  with the requested value. GenAIRR 2.3.1 cannot set P-nucleotide lengths and human IGH OGRDB has no P-nucleotide
  model, so `p_nt` is `no` on every case of that reference.
- **D trims beyond GenAIRR's cap (S3 `d_both`, S6 `trim_{k}`).** `trim()` applies an explicit length as given, and
  GenAIRR's engine aborts the whole run when the two D trims exceed the sampled allele. `symmetric_d_trim` therefore
  restricts D (`restrict_alleles(d=...)`) to the alleles at least `2k + 2` nt long whenever a shorter one exists
  (`recipes.d_alleles_for_trim`): on human IGH, all 33 D alleles for k ≤ 4, 32 for k = 6, 25 for k = 8, 22 for
  k = 9 and 14 for k = 12. Every record is then trimmed exactly k at both ends and keeps at least 2 D bases, so a
  stratum's label is always its records' real trim; the D allele pool narrows with k.

S5 builds on `repertoire(ctx, rate=0.03, noise=("polymerase_indels", {"count": (k, k)}))`, and S9 on its own
`clonal_lineage` chain:

- **S5.** GenAIRR's `polymerase_indels(count=(k, k))` puts exactly k indels in a read but places them anywhere
  (V, D, J or NP; record fields `n_v_indels`, `n_d_indels`, `n_j_indels` and the total `n_indels`), so each stratum
  keeps, by rejection sampling, the records whose indels all fell in its segment. At the default seed (dev, scale
  1.0) acceptance falls from 0.73 (`v_1`) to 0.23 (`v_6`) and is 0.15 for `j_1`, in 1 to 3 rounds. An indel at a
  segment edge cannot be told from trimming, and adjacent indels can merge or cancel, so a read can show fewer
  indels than GenAIRR placed. At that seed every `v_k` read still shows an indel in V (`v.indel_found` applies to
  all 1800), while a few `j_1` reads show none in J and are scored by `j.indel_false` instead.
- **S9.** `recombine().clonal_lineage(n_clones=ctx.n, n_sample=25, rate=0.008)`. Each clone is one lineage tree
  grown from a recombined founder. Up to 25 of its cells are sampled, and identical genotypes are collapsed, so a
  clone yields at most 25 records. Members share their alleles, trims and NP regions, and differ only by the
  mutations of their branch. Every case gets `groups["clone"]` (the record's `clone_id`); `build.json` counts the
  stratum's `requested` in `clones`. SHM is internal to `clonal_lineage`, and GenAIRR rejects `mutate()` before
  or after it. Its `rate` is per base and per generation (default `max_generations` 10), not per read: 0.05
  leaves reads about 40% mutated, and 0.008 puts them at IgG level. At the default seed (dev, scale 1.0), the 60
  clones give 1130 cases with clone sizes 4 to 25 (median 21). The per-read V mutation rate (truth V mismatches
  over observed V germline bases) has median 0.075 and IQR 0.065 to 0.088. One clone (23 cases), whose J 5′
  trim cuts the J anchor codon, has no junction, so neither a clone key nor `clone.agree_modal`.

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

**Novel alleles.** `DonorParams(novel_v_genes=m, novel_snp_cycle=(1, 2, 3, 4, 5, 6))` gives each donor `m` V genes
carried as a homozygous novel allele (default `m` = 0: none). They are drawn from a second stream of the donor's
own (seeded by the seed, its id and `novel`) after the rest of its genotype, which is therefore the genotype
drawn without them. The donor's carried V genes are shuffled and taken in turn: the i-th novel allele,
`{gene}*{900 + i}`, derives from one of the gene's carried alleles (uniform) by
`novel_snp_cycle[i % len(novel_snp_cycle)]` SNPs and replaces the gene's alleles.
`suites.novel.make_novel_allele(reference, base, k, rng, name, *, genotype)` places the SNPs: `k` distinct
positions from `invariant_positions(reference, base)` (positions where every other allele of the gene that is
long enough has the base's nucleotide, outside the anchor codon `[anchor, anchor + 3)`), each given a different
nucleotide uniformly. GenAIRR's `Genotype.add_novel_allele` (on a `Genotype` of the
DataConfig, passed as `genotype`) is the validity oracle: it rejects a variant whose anchor codon changes residue
or whose coding frame gains a stop codon, and the draw is repeated up to `MAX_ATTEMPTS` = 50 times before a
RuntimeError. A gene whose base yields no accepted variant (six human IGH OGRDB V alleles have a stop codon in
their coding frame) is passed over, and a donor with too few eligible genes raises. `sample_donors(...,
config=)` takes the DataConfig for this; `suites.engine` passes the one the reference is built from.
Novel names are per donor: two donors may each have an `IGHV...*900` with different sequences.

A `Donor` holds `alleles` (gene -> its one or two alleles, sorted; deleted genes absent), `deleted` (gene names)
and `novel` (name -> `core.genotype.NovelAllele`, each carried under its base's gene). A `NovelAllele`'s `snps`
must be distinct sorted positions in its sequence, and `NovelAllele.check_base(base_sequence)` checks that it
differs from its base exactly there (`tool_genotype` and the truth builder run it).
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
- `fingerprint() -> dict`: what identifies the tool's predictions by content as well as by name, without preparing
  a run directory: its settings plus the digests of what it runs (binaries, the resolved image ID, the model
  file). `all` puts it in the run directory's configuration and in every run's `settings`, so a tool changed in
  place is never mistaken for the one that made earlier runs. Raise (FileNotFoundError, RuntimeError) when the
  tool cannot be identified.
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
| `oracle` | `runners.oracle.OracleRunner` (fingerprint: none; each run records the truth's SHA-256) | binds the run directory | writes `truth.oracle.oracle_prediction` rows for the reads of the FASTA, in its order, from the suite's truth (in-process: no command, no RSS) |
| `igblast` | `runners.igblast.IgBlastRunner` (`--igblast-dir`, `--threads`, `--cpus`; fingerprint: the directory and the SHA-256 of `igblastn`, `makeblastdb` and the auxiliary file) | checks the release; writes `germline_{v,d,j}.fasta` and builds the BLAST databases (`makeblastdb -parse_seqids -dbtype nucl`) in `runs/igblast/_work/` | `IGDATA=<dir> igblastn -germline_db_V/D/J <db>/V,D,J -auxiliary_data optional_file/human_gl.aux -organism human -ig_seqtype Ig -num_threads <threads> -outfmt 19 -query <fasta> -out <tsv>`; provenance: version text, SHA-256 of `igblastn`, `makeblastdb`, the auxiliary file and every database file, threads and CPUs |
| `alignair` | `runners.alignair.AlignAIRImage` (`--image`, `--model`, `--cache-dir`, `--threads`, `--cpus`; fingerprint: the image reference and resolved image ID, the model spec and the installed model file's SHA-256) | inspects the image (pulling it when it is not local); installs the model into the cache (see AlignAIR model cache) with `models get`, then `models verify` and `models path`; records `models info` and `doctor --json`; exports the model's reference with `reference export <model file> --fasta` (3.0.0 crashes on a registry id there) into `runs/alignair/_work/` and refuses it unless it is the truth reference | `docker run --rm -u <uid>:<gid> -v <cache>:/home/appuser/.cache/alignair -v <log dir>/input:/in:ro -v <tsv dir>:/out [--cpuset-cpus <cpus>] [-e OMP_NUM_THREADS=<t> -e MKL_NUM_THREADS=<t>] --entrypoint sh <image> -c 'alignair "$@" 2>&1; rc=$?; cat /sys/fs/cgroup/memory.peak 2>/dev/null; exit $rc' sh predict --model <model> --input /in/<fasta> --out /out/<tsv> --device cpu --quiet [--genotype /in/genotype.json]`; `<log dir>/input` is emptied and holds only a copy of the reads FASTA (and the genotype JSON), so the container never sees the suite's cases or truth; the wall time includes the container's start; the peak RSS is the container's cgroup v2 `memory.peak` (resident memory plus page cache; None with an `rss_note` when unreadable); provenance: image reference, ID and RepoDigests, the cache record, model spec, path and SHA-256, the model card, the doctor output, the exported reference, the moved `predict` sidecar (`logs/alignair.run.json`), the genotype file, threads and CPUs |
| `alignair-local` | `runners.alignair.AlignAIRLocal` (`--checkpoint`, `--device`, `--batch-size`, `--threads`, `--cpus`; fingerprint: the checkpoint path and SHA-256, the device) | checks the checkpoint; probes the torch version and CUDA device the subprocess sees; exports the checkpoint's reference into `runs/alignair-local/_work/` with this checkout's CLI and refuses it unless it is the truth reference | `PYTHONPATH=<checkout>/src [OMP_NUM_THREADS=<t> MKL_NUM_THREADS=<t>] <python> -c "import sys; from alignair.cli.main import main; sys.exit(main())" predict --model <checkpoint> --input <fasta> --out <tsv> --device <device> --batch-size <n> --quiet [--genotype <log_dir>/genotype.json]` under `timed_command` (peak RSS of the CLI process); provenance: checkpoint path and SHA-256, the checkout's path and commit state, interpreter, torch version, CUDA device name, device, batch size, the exported reference, the moved sidecar, the genotype file, threads and CPUs |

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

The donors' TSVs are then concatenated into the arm's `predictions.tsv` under one header, each case once: a
later row for a read already written is dropped (the first row wins, as the AIRR reader keeps it). A donor whose header
differs from the first donor's, or whose TSV has a row for a read that is not its own, stops the arm with a
ValueError naming the donor, and no run record is written. The arm's `run.json` sums the donors' wall times,
takes their largest peak RSS and lists their commands in order. Its provenance holds per donor the reads, wall
time, peak RSS, return code, allele counts and the run's provenance, plus `skipped_donors`, `donors_json`
(path, SHA-256) and `reads`. The donors run in `donors.json` order and stop at the first failed run. The arm is
then recorded with that return code, the `failed_donor` and its `error` line, and is not scored.

AlignAIR 3.0.0's `predict --genotype` crashes (`AttributeError: 'tuple' object has no attribute 'items'`):
`alignair/cli/predict.py` passes `load_genotype`'s `(genotype, unknown)` tuple on as the genotype. Until a fixed
AlignAIR is released, every AlignAIR genotype-arm run is recorded as failed with that error line.
