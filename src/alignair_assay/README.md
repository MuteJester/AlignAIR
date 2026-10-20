# alignair_assay

## Purpose

`alignair_assay` scores immune-receptor aligners (AlignAIR, IgBLAST, ...) against **observable truth**: what a
perfect aligner could recover from each simulated read. It is a development package, excluded from the wheel, and
the instrument for improving AlignAIR's architecture, wiring and loss. GenAIRR is the only simulator.

The flow is: suites (declarative GenAIRR recipes) → simulated cases → observable truth → each tool's AIRR
rearrangement TSV → per-case scores → coverage-aware summaries and paired, Holm-corrected comparisons.

Units everywhere: coordinates are 0-based and half-open, rates run from 0 to 1, and wall times are in seconds.

## Model iteration quick start

Iterate on the `dev` split (the default); `test` is locked and used only for claims. From the repository root,
compare a checkpoint of this checkout against IgBLAST:

```
PYTHONPATH=src python -m alignair_assay all --run-dir .private/assay/iter --suites S1,S3,S4,S6 \
    --tools igblast,alignair-local --igblast-dir IGBLAST_RELEASE_DIR \
    --alignair-checkpoint MODEL.alignair --device cuda [--quick]
```

- **Results** land in the run directory: `report.md` (with `report.json`) first, then per suite and arm
  `scores/<S>/<arm>/comparison.md` (IgBLAST against the checkpoint, paired and Holm-corrected) and each tool's
  `<tool>.report.md`; `runs/<tool>/<S>/<arm>/` holds the predictions, `run.json` and logs. `--quick` is a smoke run
  at scale 0.02.
- **Rerun the same command after a change.** Resume keeps what is still current and redoes the rest:
  - A suite is rebuilt when its identity changes (see What invalidates a build): the configuration, its catalog
    module or the shared suite code, the truth code, or the GenAIRR in use. No `--force` is needed.
  - An IgBLAST run is kept while IgBLAST's binaries and the suite's reads are unchanged. An alignair-local run is
    kept while the checkpoint's SHA-256, the code the CLI runs (the digest of `src/alignair`'s `.py` and `.pyx`
    files, the torch version) and the reads are unchanged. A new checkpoint or a code edit is a new
    configuration: `all` stops and names what changed (`alignair-local.checkpoint_sha256`,
    `alignair-local.code_sha256`, `.torch_version`), and `all --force` then redoes the alignair-local runs only,
    keeping IgBLAST's. A commit, or uncommitted changes, that leave `src/alignair`'s sources alone do not count:
    the checkout's commit and dirty flag are provenance only.
  - Scores, comparisons and the report are always recomputed from the current runs, so a scoring change takes
    effect on the next `all` without rebuilding or rerunning anything.
- **In-process evaluation.** alignair-local runs the AlignAIR CLI in a subprocess and scores its AIRR output. The
  `diagnostics` layer is where in-process code lives (the stage replay, head-level scoring, truth-substitution
  ablations and speed profile, see Diagnostics); it and `core/reference.py` are the only code that imports `alignair`.

### Model diagnostics

`all` tells you how a checkpoint compares with IgBLAST. `diagnose` tells you where its errors and its time come
from: each head's raw output against the truth, how much error each pipeline stage adds, the allele calibration and
a speed profile. It runs in process (no subprocess, no AIRR files) on suites already built on the `dev` split:

```
PYTHONPATH=src python -m alignair_assay build --run-dir .private/assay/dev --suites S1,S3 --scale 0.1
PYTHONPATH=src python -m alignair_assay diagnose --run-dir .private/assay/dev --checkpoint MODEL.alignair \
    --suites S1,S3 [--device cuda] [--batch-size 256] [--variants baseline,true_spans,...] [--no-profile] \
    [--max-cases 500] [--allow-unverified-replay]
```

It writes a one-page summary, `diagnostics/<model_key>/diagnostics.md`, with each suite's detail in
`<S>/report.md`, the speed profile in `profile.md` and everything as data in `diagnostics.json`, and prints one line
per suite. A suite built on the `test` split is refused: `test` stays locked for claims. So is an AlignAIR checkout
whose predict pipeline has changed since the replay was pinned, unless `--allow-unverified-replay` (see The pin
under Replay).

**From a training loop.** Call the evaluator with the in-memory model and its `ReferenceSet` (for example every few
thousand steps); `max_cases` takes that many reads per suite, evenly spaced. On a GPU, 500 S1 reads with every
variant take about 5 s:

```python
from alignair_assay.core.layout import RunLayout
from alignair_assay.workflow.diagnose import evaluate

result = evaluate((model, reference_set), RunLayout(".private/assay/dev"), ["S1", "S3"], max_cases=500)
for name, value in result.scalars().items():                          # the summary's numbers, flat
    writer.add_scalar(f"assay/{name}", value, step)                   # e.g. S1/baseline/junction.nt_exact
s1 = result.suites["S1"]
s1["heads"]["head.j.end_exact.soft"]["mean"]                          # a raw head, against the truth
s1["variants"]["baseline"]["scores"]["junction.nt_exact"]["mean"]     # the product pipeline, end to end
[step["metrics"]["junction.nt_exact"]["delta"] for step in s1["contributions"]]  # error added per stage
```

- The model's reference must be the truth reference (`HUMAN_IGH_OGRDB` for the built-in suites); another is
  refused, naming the first differing allele, before the model is moved.
- The model runs on its own device unless `device` names another. A model moved to another device is moved back
  to its own afterwards, even when a suite fails, and every replay puts it back in its training mode, so the
  optimizer's next step works. A `torch.compile`, `DataParallel` or `DistributedDataParallel` wrapper is
  diagnosed as the model it wraps.
- An in-memory model is not profiled unless `profile=True` (a model file is, unless `profile=False`): every call on
  changed weights writes a new model-key directory, and a profile there would replay the model about seven more
  times and add a few megabytes each time.

**What is cached.** Results live under `diagnostics/<model_key>/` and are recomputed whenever anything they depend
on changes, never served stale; see Model key and cache under The evaluator. `diagnose` warns when a suite's build
is not current for this checkout (rebuild it to diagnose against the current truth code). One record is kept per
suite and model key, so diagnosing the same checkpoint on the CPU and on the GPU in one run directory evicts each
other's results: use a second run directory, or call `diagnostics.profile.profile_model` directly for a second
device's speed profile.

**Reading the summary** (`diagnostics.md`, `result.report_md`; the same numbers as data in `result.summary`, flat
in `result.scalars()`). Three header lines: the model, device, batch size and reads per suite; the suites computed
and reused, and the wall time; which comparisons are cluster-aware. Then:

1. *Where the end-to-end error comes from*: one row per suite for `clone_key.exact` and for `junction.nt_exact`.
   The baseline (n, coverage); the Δ of each stage's truth in percentage points (orientation, segmentation, calls,
   germline), bold when significant and marked "(worse)" when significantly worse; `true_germline` with the gap
   left to 100% (the metadata heads and the AIRR assembly); the `argmax_decoder` swap's Δ; n paired (K clusters).
2. *Calls*: V / D / J top-1 in set at baseline, the true calls' Δ, the allele head's top-1 → top-3 and its ECE.
3. *Boundaries: decoder vs network*: each edge's exact rate under the product's soft-argmax → the logits' argmax,
   as "7→44". A large jump points at the decoder, not the network.
4. *Metadata heads*: orientation and productive accuracy, mutation-rate and indel-count absolute error.
5. *Speed*: reads per second, and the top three stages and modules with their shares.
6. *Flags* (at most 8): an unverified replay, stale builds, baseline coverage below 95%, fallbacks above 5% of
   reads, significantly worse steps, comparisons with too few clusters, cropped and misoriented reads, and the
   Holm-family note (Holm is per comparison, not across steps × suites).

**Each suite's detail** (`<S>/report.md`, linked from the summary): its reads, then the head metrics (each edge under
both decoders side by side, with exact rate, absolute and signed error; the allele ranks; the metadata heads), the
stage contributions (each step's Δ with its verdict and "n paired (K clusters)", and its table of the paired means
whose difference is Δ beside each variant's own mean and coverage), the calibration (ECE and reliability bins), the
single swaps, and the fallbacks. `profile.md` holds the profile's tables (per stage and per network module).

## Layers and dependency rule

Each subpackage is a layer and may import only the layers in its row.

| Layer | Holds | May import from `alignair_assay.` |
|---|---|---|
| `core` | shared primitives: `reference` (the germline reference from a dataconfig or a FASTA, with a dataconfig's IMGT-gapped V sequences as `gapped(name)`, `DEFAULT_REFERENCE`, `dataconfig`), `cigar`, `seqops`, `prediction` (normalized prediction records), `jsonl`, `cases` (case schema, case and FASTA files, `evenly_spaced` samples), `provenance` (git state, file and source digests, `code_identity` of a torch model package), `layout` (`RunLayout`, the run-directory path schema), `genotype` (`NovelAllele`, `DonorGenotype`: donor genotypes as plain data), `headline` (`HEADLINE`, the headline metrics shared by suites and reports), `container` (pinned, hardened container runs: `check_image`, `hardened_run`, `run_container`, `ContainerError`) | core |
| `truth` | observable truth (pure): `model`, `segment`, `boundaries`, `alleles`, `genairr` (record decoding), `build`, `bench`, `case` (truth from a case), `io`, `oracle`, `summary` (what a truth set could not score) | core, truth |
| `airr` | AIRR TSV `reader` (`load_airr_tsv`, and `rows_to_predictions` for rows already in memory) and `writer` | core, airr |
| `metrics` | per-case scorers (pure), the group scorers that read a case's whole clone (`groups`), the scorers against a read's donor genotype (`genotype`), and the metric registries | core, truth, metrics |
| `stats` | aggregation with coverage and the paired comparison (pure) | core, metrics, stats |
| `report` | JSON reports and markdown summaries (`build`, `markdown`), the run report rendered from a run summary (`run`) and the diagnostics report (`diagnostics`: its one-page summary, scalars, per-suite detail and profile) | core, stats, report |
| `scoring` | `pipeline` (score one tool against a truth file; `truth_reference`, the reference built suites' truth was built against), `slices`, `oracle` (the oracle rule on per-case scores) and `clusters` (`cluster_ids`: the clone or orientation pair a comparison clusters a case by) | core, truth, airr, metrics, stats, scoring |
| `suites` | the suite registry (`__init__`), suite types (`base`), simulated donors (`donors`, with `read_genotypes` for a `donors.json`), novel alleles (`novel`: invariant positions and validated SNP placement), the deterministic generator (`engine`), shared repertoire recipes (`recipes`) and the catalog (`catalog/`, one module per suite) | core, suites |
| `runners` | one runner per tool (`igblast`, `alignair`: the AlignAIR image and a local checkpoint, `oracle`) and the registry (`__init__`: `RUNNERS`, `make_runner`), the runner interface, run records and timed commands (`base`), the shared reference (`reference`), the genotype arm (`arms`) | core, airr, truth, suites, runners |
| `workflow` | build → run → speed → score → report orchestration over a run directory: `build` (one suite), `scoring` (`score_file`, score and comparison reports and their files, shared by `score`, `compare` and `all`), `speed` (the speed protocol: `speed_set`, `measure`, `speed_configs`, `host_info`, `run_speed`), `summary` (the run summary, read back from the run directory), `orchestrate` (`RunConfig`, `run_all`, `score_and_report`) and `diagnose` (`evaluate`, the evaluator behind `diagnose`: the diagnostics layer orchestrated over a run directory, cached per suite, with its report) | everything except cli |
| `diagnostics` | model diagnostics for AlignAIR, run in process (see Diagnostics): `replay` (the predict pipeline replayed stage by stage, with substitutions), `heads` (head-level metrics), `ablation` (truth-substitution variants and stage contributions), `profile` (per-module and per-stage speed) and `identity` (the model key: `weights_sha256` and the AlignAIR code identity) | core, truth, airr, metrics, stats, scoring, suites, diagnostics |
| `real.genotype` | real donors' genotypes from genomic sequencing (see Real data): `RealAllele`, `RealGenotype`, the VDJbase loader (`load_vdjbase_csv`), the OGRDB IUIS map (`IuisMap`), the D-call filter (`filter_d`), `merge_identical`, `upto` (IMGT position 318) and the paper's novel alleles (`sd3_novel`, `sd3_only`) | core, real.genotype |
| `real.fetch` | verified downloads under the data root (see Real data): each pilot run's reads (`fetch_reads`: AWS ODP at full quality, converted in the pinned sra-tools container, else ENA's FASTQs), the genotype files (`fetch_genotype_files`), `quality_kind` (full or SRA-Lite) and `fetch_manifest`; the pilot pins (`PILOT_DONORS`, `PILOT_SPLIT`, `VDJBASE_SAMPLE`, `SRA_TOOLS_IMAGE`) | core, real.fetch |
| `real.locus` | the genes each donor's genomic contigs capture (see Real data): the contig BAM read in the pinned samtools container (`contig_spans`, `parse_sam_spans`, `SAMTOOLS_IMAGE`), IGenotyper's gene coordinates (`read_gene_coords`), the capture rule (`captured_genes`, `with_capture`) and `donor_capture`, which records how it was computed | core, real.genotype, real.locus |
| `real.preprocess` | annotation-free preprocessing of a real run (see Real data): the protocol classifier (`classify_protocol`), the pRESTO recipes (`recipe`) run in the pinned Immcantation container (`IMMCANTATION_IMAGE`), the primers (`VH_PRIMERS`, `C_PRIMERS_MULTIPLEX`, `CREGION_RC`), the cap (`cap_sequences`), the primer spans (`primer_spans`) and `preprocess_run` | core, real.preprocess |
| `cli` | the `python -m alignair_assay` commands | everything |
| `testing.py` | test-only helpers: `assert_oracle_perfect` (the oracle rule of `scoring.oracle`, with the donor genotypes when given; returns the oracle's per-case scores) | everything |

Further rules:

- `alignair` is imported only by `core/reference.py` and `diagnostics/`, and `alignair_benchmark` is never imported.
- `truth`, `metrics` and `stats` are pure: no file IO, no subprocess, no CLI, no network. `truth/io.py` reads and
  writes truth files through `core.jsonl`. So are `real.truth` and `real.metrics`.
- `real` is the one nested package: each module `real/<sub>.py` is a layer of its own, `real.<sub>`, with its own
  row (Real data lists them). `workflow` and `cli` may import any of them; `report` none, as it renders plain
  dicts.
- GenAIRR is imported only from `core`, `truth` (record decoding), `suites` and `workflow`.

`tests/alignair_assay/test_architecture.py` enforces the table and the first three rules by reading every module's
imports, function-level ones included. A new subpackage, or a new module of `real/`, fails that test until it has a
row in the test's `ALLOWED` map. Tests live in `tests/alignair_assay/<layer>/`, mirroring the package;
`tests/alignair_assay/conftest.py` provides the `genairr` guard, the session `reference` (`HUMAN_IGH_OGRDB`) and
`load_fixture(name)` for the committed bench cases in `tests/alignair_assay/fixtures/`.

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
  simulates with `expose_provenance=True`). GenAIRR also re-calls each segment after SHM (its best match to the
  read, which differs from the generative allele in a few percent of mutated reads): that call list is `{g}_call`
  up to the assembled encoding, and `recalled_{g}_call` in the generating-reference encoding, where `{g}_call` names
  the generative allele (`truth.genairr.record_calls` reads either). A record that does not name the generative
  allele (bench cases, records of the earlier encodings simulated without provenance) falls back to the first
  entry of the call list, and the truth summary counts those fallbacks per gene (`primary_fallbacks`).
- Coordinates, CIGAR and SHM truth are observed against the generative allele. When it is not in the call list,
  GenAIRR projects the CIGAR edges onto the re-called allele (fills to that allele's ends, NP bases it matches
  claimed as germline) while the core alignment stays generative. `truth.genairr` undoes this: read bases past the
  end of a shorter re-called allele become the generative allele's positions there, a re-called segment is clipped
  to the generative allele's trimmed germline `[trim_5, len - trim_3)` (reverse-complemented for an inverted D),
  and a re-called inverted D in the walker encoding (below) has its fills decoded with the re-called allele's
  length. A re-called segment whose alignment these do not determine (fills cut short by the re-called allele's
  end; past-the-end read bases that do not match the generative positions there one for one, from an indel) is
  dropped as unscorable. That this reproduces GenAIRR's own generative alignment is measured, not proven: on 3000
  records at SHM rate 0.15 every re-called segment matched it (V 187, D 182, J 8; inverted D 190 with every D
  inverted), and `tests/alignair_assay/truth/test_generative_truth.py` measures it on the installed engine. Known
  limitation: GenAIRR's fills stop at the projected allele's length, so when the re-called allele is shorter than
  the generative one, a generative deletion at or past the re-call's length is not in the CIGAR and cannot be
  recovered (only records with indel noise can have one; SHM makes no indels).
- **GenAIRR record encodings.** GenAIRR up to 2.3.1 (commit `1ededa6`, the *walker* encoding) and later engines
  (the *assembled* encoding) record two things differently, and `truth.genairr` reads both. An inverted D: the
  walker encoding gives forward-allele coordinates walked by a forward-only CIGAR walker (its `D` ops are fills,
  deletions inside the D are not recorded and are placed by minimum mismatch, with a warning, and its NP claims are
  scored against the forward allele, so they are dropped); the assembled encoding gives the forward-allele bounds
  `[d_germline_start, d_germline_end)` and a plain CIGAR against the reverse-complemented allele, internal
  deletions and NP claims included (claims are kept, as for a forward segment). The decoder tells them apart by
  shape: the assembled CIGAR consumes exactly the recorded germline and read spans (the walker's fills consume
  more, except when at most one D base is in the read, where both decode alike), and the assembled decoding is kept
  only when it passes the segment invariants; anything else is decoded as the walker encoding. A re-called segment
  with a shorter re-called allele: the walker encoding aligns the read bases past its end to the generative
  positions, the assembled encoding records them as insertions. And a segment removed completely by end loss: the
  assembled encoding gives it an empty CIGAR and no germline coordinates, which the builder treats as not observed.
  The committed fixtures `tests/alignair_assay/fixtures/old_engine_*.json` (bench_v2 inverted Ds and dev-split
  re-calls simulated by `1ededa6`) keep the walker decoding tested whichever GenAIRR is installed. The newest
  engines use the *generating-reference* encoding, told apart by its `recalled_{g}_call` fields: `{g}_call` is the
  generative allele and `recalled_{g}_call` the post-SHM call list; every alignment is against the generative
  allele (nothing to undo for a re-called segment), no NP base is claimed for a segment, end loss is missing
  coverage instead of `D` columns, and the inverted D is assembled.
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
  non-informational metric either tool produced; the informational ones are reported beside it (the comparison
  report's `informational` block and its "Informational (not compared)" table: each tool's own mean, n and
  coverage, untested).
- Holm is per comparison: one pair of tools on one suite and arm, or one step or swap of one suite in the
  diagnostics, over that comparison's metrics. It is not corrected across steps × suites.
- **Clustered comparisons.** Reads that are not independent are compared by cluster
  (`scoring.clusters.cluster_ids`, `CLUSTER_KEYS`): the members of one clone (`groups["clone"]`, S9) and the
  forward and reverse-complemented presentations of one record (`groups["pair"]`, S11), within their stratum.
  Donors (`groups["subject"]`) are not clustered by default. When a metric's pairs fall into fewer clusters than
  pairs, `stats.compare(..., clusters=...)` tests the per-cluster sums `D_k` of the paired differences
  (`clustered_mcnemar` for 0/1 metrics, `clustered_z` otherwise; z = ΣD_k / √ΣD_k²) with a cluster-robust (CR1)
  CI, and gives no verdict below 10 clusters (`too_few_clusters`, p = 1). One clone carrying a difference is then
  never significant on its own. Reports show "n paired (K clusters)". S9 has 60 clones at scale 1.0, so it needs
  scale ≥ 0.2 (≥ 10 clones) for any verdict. With every cluster a singleton, the comparison is the read-level one
  (exact McNemar or paired z), unchanged. `all` recomputes every score and comparison on each call, so the
  clustered tests apply to an existing run directory on its next `all` or `report`.

**Allele metrics** (`metrics.alleles`, per observed segment `g`):

- `{g}.top1_in_set`: the top-1 call is in the scored set (generative set plus the simulator's re-call and
  the as-near alleles).
- `{g}.top1_generative` (informational): the top-1 call is in the generative set, i.e. true-allele accuracy. It is
  reported, with coverage, beside `{g}.top1_in_set` in every score report and in each comparison's informational
  table, but it is not in the Holm family: when SHM has made a read a sibling allele's, a perfect sequence-only
  aligner calls the sibling, so this metric cannot reach 1.0 for it and a verdict on it would penalise a defensible
  call. `{g}.top1_in_set` is the compared allele metric.
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

`core.layout.RunLayout` names every run-directory path the workflow and the commands use. Runners and the speed
protocol write inside the directories it gives them, under names of their own: a run's `logs/` (the tool's log,
IgBLAST's restricted `db/`, the AlignAIR image's staged `input/`, the genotype JSON), a tool's `_work/`
(databases, the model cache, the exported reference) and a speed configuration's `speed/runs/<config>/` (`tiny/`
and `full/`, each with its predictions and logs). `RunLayout` only builds paths (no IO), and suite, tool and arm
names must each be one path component.

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
  speed/speed.json                    the speed measurements (`all --speed`, `speed`)
  speed/reads_<n>.fasta               n reads evenly spaced across S1, timed
  speed/runs/<config>/                each timed configuration's last tiny/ and full/ run (predictions, logs)
  report.json, report.md              the run summary and the run report (`all`, `report`)
  real/<donor>/genotype.json          a real donor's genotype, with the genes its genomic contigs capture
  real/<donor>/reads.fasta            the donor's preprocessed unique sequences (+ reads.tsv, one row each)
  real/<donor>/preprocess.json        the preprocessing record, written last (+ preprocess_logs/; preprocess_work/
                                      while it runs)
  diagnostics/<model_key>/            one model's diagnostics (`diagnose`, `workflow.diagnose`)
    <S>/suite.json                    one suite's cached result, with the key it was computed for
    <S>/report.md                     one suite's detail of the last report
    profile.json                      the cached speed profile, with its key
    profile.md                        the speed profile of the last report
    diagnostics.json, diagnostics.md  the model's last diagnostics report and its one-page summary
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
| `speed/reads_<n>.fasta`, `speed/runs/<config>/` | `speed_reads(n)`, `speed_work_dir(config)` |
| `report.json`, `report.md` | `report_json`, `report_md` |
| `diagnostics/`, `diagnostics/<model_key>/`, `<S>/` | `diagnostics_root`, `diagnostics_dir(key)`, `diagnostics_suite_dir(key, s)` |
| `diagnostics/<model_key>/<S>/suite.json`, `profile.json` | `diagnostics_suite_json(key, s)`, `diagnostics_profile_json(key)` |
| `diagnostics/<model_key>/<S>/report.md`, `profile.md` | `diagnostics_suite_md(key, s)`, `diagnostics_profile_md(key)` |
| `diagnostics/<model_key>/diagnostics.json`, `diagnostics.md` | `diagnostics_json(key)`, `diagnostics_md(key)` |
| `real/`, `real/<donor>/`, `real/<donor>/genotype.json` | `real_root`, `real_donor_dir(donor)`, `real_genotype_json(donor)` |
| `real/<donor>/reads.fasta`, `reads.tsv`, `preprocess.json` | `real_reads(donor)`, `real_reads_meta(donor)`, `real_preprocess_json(donor)` |
| `real/<donor>/preprocess_logs/`, `preprocess_work/` | `real_preprocess_logs(donor)`, `real_preprocess_work(donor)` |

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
python -m alignair_assay speed --run-dir RUN [--tools igblast,alignair] [--threads 8] [--configs NAME,...] \
    [--n-tiny 50] [--n-full 4000] [--repeats 3] [--igblast-dir DIR] [--alignair-image IMAGE] \
    [--alignair-model alignair-igh-human@1.0.0] [--alignair-checkpoint MODEL.alignair]
python -m alignair_assay diagnose --run-dir RUN --checkpoint MODEL.alignair --suites S1,S3 [--device cuda] \
    [--batch-size 256] [--variants baseline,true_spans,...] [--no-profile] [--max-cases N] \
    [--allow-unverified-replay]
python -m alignair_assay truth --bench-cases CASES.jsonl[.gz] [--reference HUMAN_IGH_OGRDB] --out TRUTH.jsonl.gz
python -m alignair_assay score --truth TRUTH --predictions AIRR.tsv --tool NAME --out-dir DIR [--donors DONORS.json]
python -m alignair_assay compare --truth TRUTH --a A.tsv --a-name A --b B.tsv --b-name B --out-dir DIR [--alpha 0.05] \
    [--donors DONORS.json]
```

- `all` (`workflow.orchestrate.run_all`) does everything over one run directory, and can be rerun to resume. See
  The all command below. The run directory and the tool paths (`--igblast-dir`, `--alignair-checkpoint`) are made
  absolute, `~` expanded, before the configuration is built (`cli.common.tool_path`), so the manifest never
  records a path relative to where `all` was started; `speed` does the same.
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
  `n_cases`. `build.json` records the same configuration, its `identity` and `config_hash` (SHA-256 of the
  canonical JSON of the identity; see What invalidates a build), the per-stratum generation stats (`requested` in
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
- `speed` times tools with the speed protocol (see The speed protocol) on the S1 reads of a run directory where
  S1 is built (`build --suites S1`, at scale 0.8 or more for the default 4000 reads). The tools are `--tools`
  (default `igblast,alignair`), plus alignair-local when `--alignair-checkpoint` is given, with the tool options
  of `all`; `--threads` is N, and `--configs` times only the named configurations. It writes `speed/speed.json`
  with no config hash (it belongs to no `all` configuration, so a run report does not show it) and prints the speed
  table. A run directory made by `all` is refused: time it with `all --speed`, so the measurement belongs to its
  manifest. S1 not built, an unknown configuration and a tool without its options are input errors (status 2)
  before anything is timed; a configuration that fails is recorded with its error, the others are still timed,
  and the command exits with status 2.
- `diagnose` runs `workflow.diagnose.evaluate` on a checkpoint and suites built on the dev split of `RUN` (see
  Model diagnostics and The evaluator below). The checkpoint path is made absolute. It prints one line per suite
  (computed, or reused from the cache) and the profile, and names the one-page summary
  `diagnostics/<model_key>/diagnostics.md`. A suite whose build is not current for this checkout
  (`workflow.diagnose.stale_builds`, from `workflow.build.build_changes`) is diagnosed as built, with a warning and a
  flag in the summary. An unbuilt suite, a suite on the `test` split, an unknown variant, a missing checkpoint, a
  model on another reference, a CUDA device torch does not see, or AlignAIR code the replay is not pinned to
  (without `--allow-unverified-replay`) exits with status 2 before anything is written. The other commands never
  import torch: `diagnose` imports the diagnostics layer when it runs.
- `truth` writes the truth file. Its header records `kind`, `version`, `reference`, `reference_digest` and the
  GenAIRR provenance: `genairr_version`, `genairr_source`, `genairr_git_commit`, `genairr_git_dirty` and
  `genairr_source_sha256` (the package directory's content: sources, compiled engine and data). The
  command then prints the excluded cases, warnings, unscorable genes, the two simulator-call counts ("simulator
  call list differs from the generative set" and "simulator first call is not the generative allele", see Truth
  conventions) and primary fallbacks.
- `score` writes `<tool>.scores.jsonl.gz`, `<tool>.report.json` and `<tool>.report.md`.
- `compare` writes `comparison.json` and `comparison.md`, and prints the headline table. Its 95% and simultaneous
  CIs of 0/1 metrics are clipped to [-1, 1]. The informational metrics (true-allele accuracy among them) follow
  the tested ones, untested, under `informational`.
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
   by building its runner, which runs, pulls and downloads nothing. With `--speed` (`workflow.speed.check_speed`),
   S1 must be among the suites with at least 4000 reads at the scale (scale 0.8 or more), the threads must not
   exceed the host's logical CPUs, every timed configuration's runner must build, and a checkpoint timed as
   alignair-local must be a file. Each tool's runner is then
   built and fingerprinted (`Runner.fingerprint`, see Adding a runner), which
   identifies the tool by content: IgBLAST's directory and the SHA-256 of `igblastn`, `makeblastdb` and the
   auxiliary file; the AlignAIR image reference and the image ID it resolves to (pulling it when it is not local),
   and the model spec and the SHA-256 of the model file it resolves to (installing it into the run's model cache);
   the local checkpoint's path, SHA-256 and device, and the code the local CLI runs
   (`runners.alignair.local_code_identity`: `code_sha256`, the SHA-256 of the relative paths and bytes of this
   checkout's `src/alignair/**/*.py` and `*.pyx`, and the `torch_version`); nothing for the oracle. Then
   `manifest.json` is written, holding the `config`, its `identity`, the `config_hash` (SHA-256 of the canonical
   JSON of the identity), the split, the host, the checkout's git state, the GenAIRR provenance and `warnings` (an
   AlignAIR image not pinned by digest, or a model not pinned to a version). The identity is what the outputs
   depend on: suites, tools, split, reference, seed, scale and the tools' fingerprints (`settings`). Workers, threads
   and `--speed` are not part of it, so an upgrade in place, a moved image tag, a retrained checkpoint, an edit of
   the AlignAIR code alignair-local runs (any change to a `.py` or `.pyx` file under `src/alignair`, committed or
   not) or another torch version is a new configuration even when every path and name is the same. A commit, or an
   uncommitted change elsewhere in the checkout, leaves the fingerprint as it is.
2. **Build.** Each suite is built, unless its `build.json` already has the suite's identity (see What invalidates a
   build) and its truth file the current truth version. A rebuilt suite prints which identity fields changed.
3. **Shared reference.** With `alignair` among the tools and no `reference/germline.fasta` yet, the image exports
   its model's reference there. `ensure_shared_reference` then verifies the file against the truth reference, or
   writes it from the truth. A model with another reference stops the run before any tool runs, and its export is
   removed.
4. **Runs.** The oracle runs on every suite's shared arm. Then each tool runs on each suite and each of the
   suite's arms, the shared arm through `run_suite` and the genotype arm through `run_genotype_arm`. Every tool
   with a run to make is prepared before the first run. Each run records its tool's fingerprint as the
   provenance's `settings`.
5. **Speed** (`--speed` only). `workflow.speed.run_speed` times every configuration of the run on S1 (see The
   speed protocol) and writes `speed/speed.json` with the manifest's config hash.
6. **Scoring.** Every suite and arm is scored afresh (`score_and_report`), after its old `scores/<S>/<arm>/` is
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
7. **Report.** `report.json` holds the run summary (`workflow.summary.collect_run_summary`) and `report.md` its
   rendering (`report.run.run_report_markdown`).

`all` exits with status 0, or with status 2 when any run failed, is missing, stale or could not be scored, the
oracle is not perfect on a suite, or a speed measurement failed. The report names which.

**Resume.** Rerunning `all` with the same configuration skips each suite whose build is current and each run whose
`run.json` shows return code 0 with predictions, the same `settings` (the tool's current fingerprint) and
unchanged input files. A failed run is redone. With `--speed`, a configuration's measurement is kept when
`speed.json` was measured on the same host (`host_info`), with the same protocol version and settings, S1 reads,
timed read sets (their SHA-256) and config hash, and the configuration succeeded with the same tool, threads, CPUs,
device and, for alignair-local, checkpoint digest and code (`local_code_identity`); a failed or changed one is
measured again. Scores, comparisons and the report are always recomputed from the current runs, so a change to the
scoring code takes effect on the next `all`. It never mixes in stale outputs.

**What invalidates a build.** A suite's build is redone by the next `all` (no `--force` needed: the manifest does not
change) whenever a field of its identity (`workflow.build.build_identity`) differs from the one its `build.json`
records:

- the configuration: `suite`, `split`, `reference_digest`, `seed` and `scale`;
- `suite_source`: the SHA-256 (`core.provenance.files_digest`) of the suite's catalog module and the code every
  suite's cases come from (`suites/base.py`, `engine.py`, `donors.py`, `recipes.py`, `novel.py`, `core/cases.py`,
  `core/seqops.py`), so an edited recipe, stratum count, rate, label, `accept` or donor parameter rebuilds that
  suite (and an edit of the shared modules every suite);
- `truth_source`: the SHA-256 of every `truth/*.py` and the core modules truth uses (`core/cigar.py`,
  `genotype.py`, `reference.py`, `seqops.py`), so a truth rule changed in place rebuilds every suite without a
  version bump; and `truth_version` (`truth.io.TRUTH_VERSION`, kept for a changed file format);
- the GenAIRR in use: its version, git commit and dirty flag, and `genairr_source_sha256`, the content of its
  package directory, so a checkout rebuilt in place while dirty is a new simulator too.

Any byte counts, a comment or whitespace edit included: a few needless rebuilds (a full build takes about 36 s) are
the price of never reusing a stale one. A rebuild rewrites the suite's files; a tool's run on it is kept when the
suite's reads are byte-identical (its predictions depend on nothing else), and redone otherwise, while a run that
recorded the truth or donors file (the oracle, the genotype arm) is redone when that file changed.

**Another configuration.** In a run directory whose manifest has another config hash, `all` stops before changing
anything. The error names every field that differs, as `field old -> new`, for example `seed 20261019 -> 7`.
`--force` replaces the manifest; its `replaced` records the old hash and the changed fields. The steps then redo
what the change affects. A new seed, split, scale or reference rebuilds the suites, which changes their reads, so
every run is redone. A changed tool fingerprint (`<tool>.<key>` in the error) redoes that tool's runs only.
Outputs of suites or tools no longer in the configuration stay on disk but are not reported. `--quick` is scale
0.02 (S1 then has 100 cases); an explicit `--scale` wins over it.

**The run report** (`report.md`) has these sections; a section without data is left out.

1. **Failed runs.** Every run that failed, is missing, stale or could not be scored: tool, suite, arm, status,
   return code, failed donor, the tool's error line and the run record. These runs are not scored, and the
   comparisons that need them are left out.
2. **Oracle check failures.** Suites where the oracle is `imperfect` (it breaks the oracle rule; listed with the
   first offending cases), which have a truth defect, and suites where it was `not scored`, listed with the reason
   from the oracle's failed run (failed, missing, stale, or not scored, e.g. an unreadable `donors.json`): their
   truth is unchecked, not known to be wrong. Both make `all` and `report` exit with status 2.
3. **Truth warnings.** Suites with unscorable segments (truth that could not be built, `ObservableTruth.unscorable`):
   per gene the count, the cases with warnings and the build record whose `truth.warnings` give each case's
   reasons. Those segments are not scored, so the suite's metrics cover fewer reads than it simulated; `all` and
   `report` print a warning for each, but it does not change the exit status.
4. **Summary.** Per suite and arm: cases, excluded cases, the oracle check, and how many of the suite's primary
   metrics each tool wins and how many are tied or inconclusive.
5. **Truth.** Per suite, from its build record's truth summary: cases, excluded cases, cases with warnings, and per
   gene the unscorable segments and the re-called reads (`simulator_calls`' `recalled`).
6. **Suites.** Per suite: title, purpose, and per arm the primary-metric rows of its comparison, linked to its
   `comparison.md`. A clustered row shows "n paired (K clusters)" (see Clustered comparisons), and the section
   explains them once.
7. **S10 vs S1 (held-out settings).** Per tool, S10's primary-metric means against those of S1's `full_igg`
   stratum, which has S10's read design and SHM rate under the default distributions. The table shows the
   difference and the mean per held-out setting, each mean with its n and coverage. These are unpaired and
   informational: no test is made.
8. **S11 orientation parity.** Per tool, the forward reads against their reverse-complemented twins, paired by
   `groups["pair"]` and compared with `stats.compare` (Holm at 0.05 across the tool's metrics). The section lists
   the metrics that differ and the primary-metric rows.
9. **Genotype arms.** For suites with a genotype arm, each tool's primary-metric means in each arm, with n and
   coverage.
10. **Excess alleles.** For suites with donors, the excess-alleles table (see Genotype metrics) of each run: the
    oracle's shared arm (always 0%), then each tool in each arm: for V and J, the share counting all calls and the
    avoidable share, each with its n.
11. **Speed.** Present when `speed/speed.json` exists (`workflow.summary.speed_summary`). The host, how the
    protocol measures, and per configuration its cold start, steady reads/s, median full run and peak RSS, then its
    threads, CPU list and device; a failed configuration shows dashes and its error is listed under the table. Only
    measurements of the current configuration are shown; the others are named, with why, under the table or in place
    of it (and in a warning). A `speed.json` of another protocol version, whose config hash is not the manifest's
    (measured for another configuration, or by the `speed` command), or whose reads' SHA-256 is not that of S1's
    current `reads.fasta`, is stale as a whole. A planned configuration the file has no result for (an interrupted
    `all --speed`) is listed as not measured, so a partial file never looks complete. An alignair-local
    configuration whose `checkpoint_sha256` is not the SHA-256 of the configuration's `--alignair-checkpoint` now
    (retrained in place or replaced), or whose code
    (`code_sha256`, `torch_version`) is not this checkout's now, is stale alone: the config hash
    does not cover either when alignair-local is not a compared tool. `all --speed` measures stale ones again.

To add a command, write one module in `cli/` with `register(subparsers)` and `run(args) -> int`, and add it to
`cli.COMMANDS`. The module owns its own input checks: `run` raises ValueError (or FileNotFoundError) for invalid
input, which `cli.main` prints as `error: ...` with exit status 2. A failing environment raises RuntimeError (a
stratum rejection sampling cannot fill, a docker pull or inspect, the IgBLAST version check, `makeblastdb`,
AlignAIR's `models get`/`verify`), which `cli.main` reports the same way, never as a traceback.

### The speed protocol

`workflow.speed` measures how fast each tool annotates, on one host and on the same reads, the way the run
directory's tools are run (the same runners, through `Runner.run`):

- **Configurations** (`speed_configs(cfg)`). Each tool of the run but the oracle runs at 1 thread pinned to CPU 0
  (`<tool>_1t`) and at N threads (`--threads`) pinned to CPUs 0 to N−1 (`<tool>_<N>t`). alignair-local, timed
  when it is a tool or `--alignair-checkpoint` is set, runs on the GPU with N threads (`alignair-local_cuda`) when
  the interpreter's torch sees a CUDA device, and on CPU at N threads. Pinning is `taskset` for IgBLAST and the
  local CLI and `--cpuset-cpus` for the AlignAIR image; the threads are IgBLAST's `-num_threads` and AlignAIR's
  `OMP_NUM_THREADS`/`MKL_NUM_THREADS`.
- **Read sets** (`speed_set(layout, n)`). `speed/reads_<n>.fasta` holds n reads evenly spaced across S1's N
  `reads.fasta` reads: those at positions floor(i · N / n) for i = 0 … n−1, in file order (every (N/n)-th read when
  that is whole). There is a tiny set (`n_tiny`, default 50) and a full set (`n_full`, default 4000). S1 lists its
  reads stratum by stratum, so its first n reads would be mostly its first read designs (the first 4000 at scale
  1.0 hold only 250 FR3 reads, the first 50 only full-length IgM reads); the evenly spaced sets hold every stratum in
  about its share of S1, so both reflect S1's mix of read designs and SHM levels. S1 with fewer than n reads is an
  error.
- **Measurement** (`measure`). Each configuration is prepared once (not timed), then runs the tiny and the full set
  alternately, `repeats` times each (default 3). Every run is a fresh process, or a fresh container for the
  AlignAIR image, whose wall time includes the container's start. **Cold start** is the median tiny-run wall time:
  starting, loading the model or the databases, and n_tiny reads. **Steady reads/s** is (n_full − n_tiny) /
  (median full − median tiny): the rate once started (None when the full runs were not slower). **Peak RSS** is
  the largest over the runs: GNU time's maximum resident set size for IgBLAST and the local CLI, the container's
  cgroup `memory.peak` (resident memory plus page cache) for the AlignAIR image. A run that fails (non-zero return
  code, no predictions) fails the configuration with its error line; the other configurations go on.
- **Host** (`host_info()`): CPU model, logical CPUs, RAM in GiB, GPU name (nvidia-smi), docker daemon version and
  kernel release. Nothing else should run on the host, and on the GPU, while it is timed.
- **`speed.json`**: `kind`, `version` (`SPEED_VERSION`, bumped whenever the sampling, the measurement or the
  file's meaning changes), `host`, `n_tiny`, `n_full`, `repeats`, `reads` (suite and SHA-256 of its
  `reads.fasta`), `sets` (the SHA-256 of the timed `tiny` and `full` read sets), `config_hash` (the `all`
  configuration it belongs to; None from the `speed` command), `planned` (every configuration to time, in order;
  written before the first is measured, so an interrupted run's file lists what it did not reach) and
  `results`: per configuration measured so far its `tool`, `threads`, `cpus`, `device` (and, for alignair-local,
  `checkpoint_sha256`, `code_sha256` and `torch_version`), `measured_utc` and either its
  `settings` (the tool's fingerprint), `cold_start_s`,
  `steady_reads_per_s`, `median_tiny_s`, `median_full_s`, `peak_rss_kb`, `walls` (every run's wall time, by set)
  and `command` (the last full run's), or its `error`. `speed/runs/<config>/` keeps each set's last run
  (predictions and logs).

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
4. Register the suite in `_CATALOG` in `suites/__init__.py`, and its module in `_MODULES` (`suites.catalog_module`
   finds the module whose source is part of the suite's build identity).
5. Test it in `tests/alignair_assay/suites/` at a small scale: counts, labels and groups, and the oracle check
   `testing.assert_oracle_perfect` on its truth (with `genotypes=suites.donors.read_genotypes(donors.json)` for
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
  worker count. The workers use the `spawn` start method, which re-imports the calling script: a script that
  calls `generate_suite`, `build_suite` or `run_all` with workers > 1 needs an `if __name__ == "__main__":` guard
  around the call (`python -m alignair_assay` needs none).

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
  GenAIRR's annotation. Up to the assembled encoding it is one base shorter than requested on about 1% (NP1) and
  2% to 2.5% (NP2) of records, without SHM or end loss as well: GenAIRR claims an NP base for the neighbouring V, D
  or J when it matches that segment's trimmed germline (or, for a segment GenAIRR re-called, the re-called
  allele's). The generating-reference encoding claims none, so it is the sampled length. Strata are labelled
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
cache directory); a factory reads the fields it needs and raises ValueError naming a missing one, without running,
pulling or downloading anything (`all` builds every runner to check its options before it writes anything). Expose
new options as `run` flags. The speed protocol times a registered tool at 1 and N threads through its `threads`
and `cpus` options, so honour both, and report `n_reads` and `wall_seconds` truthfully. Test the runner in
`tests/alignair_assay/runners/`; a test that needs the tool installed skips unless its environment variable is
set (`ALIGNAIR_ASSAY_IGBLAST_DIR`, `ALIGNAIR_ASSAY_DOCKER`, `ALIGNAIR_ASSAY_MODEL`).

Registered runners:

| Tool | Runner | Setup (`prepare`) | Run |
|---|---|---|---|
| `oracle` | `runners.oracle.OracleRunner` (fingerprint: none; each run records the truth's SHA-256) | binds the run directory | writes `truth.oracle.oracle_prediction` rows for the reads of the FASTA, in its order, from the suite's truth (in-process: no command, no RSS) |
| `igblast` | `runners.igblast.IgBlastRunner` (`--igblast-dir`, `--threads`, `--cpus`; fingerprint: the directory and the SHA-256 of `igblastn`, `makeblastdb` and the auxiliary file) | checks the release; writes `germline_{v,d,j}.fasta` and builds the BLAST databases (`makeblastdb -parse_seqids -dbtype nucl`) in `runs/igblast/_work/` | `IGDATA=<dir> igblastn -germline_db_V/D/J <db>/V,D,J -auxiliary_data optional_file/human_gl.aux -organism human -ig_seqtype Ig -num_threads <threads> -outfmt 19 -query <fasta> -out <tsv>`; provenance: version text, SHA-256 of `igblastn`, `makeblastdb`, the auxiliary file and every database file, threads and CPUs |
| `alignair` | `runners.alignair.AlignAIRImage` (`--image`, `--model`, `--cache-dir`, `--threads`, `--cpus`; fingerprint: the image reference and resolved image ID, the model spec and the installed model file's SHA-256) | inspects the image (pulling it when it is not local); installs the model into the cache (see AlignAIR model cache) with `models get`, then `models verify` and `models path`; records `models info` and `doctor --json`; exports the model's reference with `reference export <model file> --fasta` (3.0.0 crashes on a registry id there) into `runs/alignair/_work/` and refuses it unless it is the truth reference | `docker run --rm -u <uid>:<gid> -v <cache>:/home/appuser/.cache/alignair -v <log dir>/input:/in:ro -v <tsv dir>:/out [--cpuset-cpus <cpus>] [-e OMP_NUM_THREADS=<t> -e MKL_NUM_THREADS=<t>] --entrypoint sh <image> -c 'alignair "$@" 2>&1; rc=$?; cat /sys/fs/cgroup/memory.peak 2>/dev/null; exit $rc' sh predict --model <model> --input /in/<fasta> --out /out/<tsv> --device cpu --quiet [--genotype /in/genotype.json]`; `<log dir>/input` is emptied and holds only a copy of the reads FASTA (and the genotype JSON), so the container never sees the suite's cases or truth; the wall time includes the container's start; the peak RSS is the container's cgroup v2 `memory.peak` (resident memory plus page cache; None with an `rss_note` when unreadable); provenance: image reference, ID and RepoDigests, the cache record, model spec, path and SHA-256, the model card, the doctor output, the exported reference, the moved `predict` sidecar (`logs/alignair.run.json`), the genotype file, threads and CPUs |
| `alignair-local` | `runners.alignair.AlignAIRLocal` (`--checkpoint`, `--device`, `--batch-size`, `--threads`, `--cpus`; fingerprint: the checkpoint path and SHA-256, the device, and the code the CLI runs: the SHA-256 of `src/alignair`'s `.py` and `.pyx` files, the checkout's git commit and dirty flag, the torch version) | checks the checkpoint; probes the torch version and CUDA device the subprocess sees; exports the checkpoint's reference into `runs/alignair-local/_work/` with this checkout's CLI and refuses it unless it is the truth reference | `PYTHONPATH=<checkout>/src [OMP_NUM_THREADS=<t> MKL_NUM_THREADS=<t>] <python> -c "import sys; from alignair.cli.main import main; sys.exit(main())" predict --model <checkpoint> --input <fasta> --out <tsv> --device <device> --batch-size <n> --quiet [--genotype <log_dir>/genotype.json]` under `timed_command` (peak RSS of the CLI process); provenance: checkpoint path and SHA-256, the checkout's path and commit state, interpreter, torch version, CUDA device name, device, batch size, the exported reference, the moved sidecar, the genotype file, threads and CPUs |

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

## Diagnostics

### Replay

`diagnostics.replay.replay(model, reference_set, sequences, ids, *, device, batch_size,
substitutions=Substitutions())` runs AlignAIR's predict pipeline one stage at a time and returns `StageOutputs`: the
raw forward outputs (logits included, concatenated over batches), the cleaned and calibrated `Predictions`, the
forward-frame reads, the corrected `Segments`, the allele calls, the germline alignments, the light records, the
AIRR rows, the wall time per stage (`STAGES`), each allele head's allele names (`allele_names`) and the per-read
allowed-allele masks the selector applied (`allowed`: a multi-locus model's locus masks, None for a one-locus
model). Each stage is AlignAIR's own function, called in the order `alignair.predict.pipeline.predict` calls it,
with the `PredictConfig` that `alignair.api.predict_sequences` builds (`predict_config`); the forward pass builds
each batch dict itself, so orientation can be teacher-forced.

`replay(..., substitutions=Substitutions(...))` feeds values in between the stages, never inside one; the default
substitutes nothing. Each per-read list has one entry per read, and a None entry keeps the model's value for that
read.

- `orientation`: the ids are teacher-forced in the forward pass (`batch["orientation"]`) and replace the predicted
  orientation before `canonicalize`. The orientation head's own logits are unchanged.
- `decoder="argmax"`: `correct_segments` receives `argmax_boundaries` of the position logits instead of the
  soft-argmax expectations. This is the same decoder as the head metrics' `argmax`, with ends on `0..n`. The
  classifier still reads the expectations.
- `spans` (gene -> starts and ends, in the model's forward frame, 0-based and end-exclusive):
  - With `segment_spans` (the default), they replace the corrected `Segments`. `low_quality` keeps
    `correct_segments`' meaning (a V or J of zero length). A given span is never re-ordered against the model's
    span of another gene.
  - With `classifier_spans`, they go into each gene's classifier (`model.branches[g].classify` on the read
    re-embedded in the forward pass's frame) in place of the expectations. The new probabilities then go through
    the calibration, the masks and the selector.
- `calls` (`GeneCall`s) replace the selector's calls before the germline alignment.
- `germline` (`GermlineAlignment`s on the read's call) replace the matcher's alignments. The matcher runs only for
  reads that still have a gene left to it.

Bad values are a ValueError: a list of the wrong length, a gene the model lacks, an orientation id outside 0–3, an
empty span or one outside its read, a call to an allele the model cannot call, or an alignment on another allele
than the call. Tests pin each substitution against the product:
- forcing the model's own orientation gives the product's rows and forward outputs;
- the classifier seam fed the model's own spans gives the product bit for bit, on the small models and on the
  released checkpoint;
- the argmax decoder equals `heads.decode_boundaries`' argmax.

The replay is proven equal to the product: `tests/alignair_assay/diagnostics/test_replay.py` requires its AIRR
rows and light records to equal `predict_sequences`' on S0 reads (plus an over-length, a lower-case and an N-bearing
read) with small random-init models shaped to cover every orientation, indels and two loci, and on 200 S1 reads
with the released checkpoint (`ALIGNAIR_ASSAY_MODEL`).

**The pin.** `diagnostics.pin.PINNED` holds a digest of the code of the product functions the replay mirrors
(`pipeline.predict`, `predict_sequences`, `run_model`; `pin.code_digest`), and `predict_sequences` must still run
`pipeline.predict` (`api._predict`). `pin.pin_changes()` names what no longer holds, with the new digests:
- the equivalence test fails on any change, and stays the place that re-pins;
- `diagnostics.identity.replay_pin()` takes it once, when `diagnostics.identity` is imported, beside the model
  key's code digest, so the two describe the same code. `workflow.diagnose.evaluate` refuses to run on a changed
  pin, naming the changed functions and their new digests, unless `allow_unverified_replay=True`
  (`diagnose --allow-unverified-replay`). Then it warns (`UnverifiedReplayWarning`) and stamps
  `replay_verified: false` and `replay_changes` into the result, every `suite.json` and the report. Whether the
  replay is verified, and the digest of the product code, are part of each suite's cache key, so verified and
  unverified results never mix.

To re-pin after an AlignAIR change: update `diagnostics/replay.py` to the new code, run the equivalence tests
(`tests/alignair_assay/diagnostics/test_replay.py`, with `ALIGNAIR_ASSAY_MODEL` set), then put the new digests in
`PINNED`.

The small random-init models (`make_small_model`, `small_model`) come from `tests/alignair_assay/conftest.py`;
`tests/alignair_assay/diagnostics/conftest.py` adds the shaped ones (`make_shaped_model`) and a built S0
(`s0_layout`, `s0_reads`).

`load_model_checked(checkpoint_or_model, truth_reference, *, device)` loads a `.alignair` file with
`alignair.api.load_model`, or passes a `(model, reference_set)` pair through, and refuses a model whose reference is
not the truth reference (unequal digests), naming both references' allele counts and the first allele that is
missing from the model's reference, extra in it, or has a different sequence there. A pair's model is unwrapped
first (`unwrap_model`: `torch.compile`'s `_orig_mod`, `DataParallel`'s and `DistributedDataParallel`'s `module`),
then checked, and only then moved to `device` in place; moving it back is the caller's (`model_device` gives the
device it was on; the evaluator restores it).

### Head metrics

`diagnostics.heads.head_scores(outputs, truths, cases=None)` scores the raw head outputs of a replay against the
observable truth: one `CaseScores` per non-excluded truth, keyed by case id, with the NaN rules above (an absent key
does not apply). Cases are matched to replayed reads by id. A scored case without a replayed read, or with a
replayed read of the wrong length, is a ValueError; so is a `cases` list that lacks a scored case.
`diagnostics.heads.METRIC_NAMES` lists every name, and `metrics/__init__.py` classifies them.

**Frame.** The segment heads predict positions in the model's forward frame, on the read padded on the right to the
logits width (576). AlignAIR crops a longer read to the first 576 bases of the presented read: the canonical prefix
of a forward read, and the canonical suffix of a reverse-complemented one. Positions are moved into the truth's
canonical frame by that window's start.

**Misoriented reads.** When the model's orientation (the frame the replay canonicalized the read in) is not the
truth's, its positions are on another strand or in another direction, so none of the read's boundaries is scored:
comparing them would count a frame mismatch, which can also land on the truth by chance, as a boundary error. The
read is counted instead: `head.misoriented` (informational) is 1.0 for it, and the evaluator's `reads` count
`misoriented` reads beside the cropped ones. The orientation head's own error is `head.orientation.correct`, and the
end-to-end cost of a wrong orientation is the `true_orientation` step. The read's allele and metadata heads are
still scored.

**Boundaries** (`head.{g}.{start,end}_{exact,abs_err,err}.{soft,argmax}`). Coordinates are 0-based and end-exclusive,
like `correct_segments`.

- `soft`: the floor of the head's soft-argmax expectation, clipped to the read. This is `correct_segments`'
  per-segment step, without its min-span and ordering repair.
- `argmax`: the argmax of the position logits over the read's coordinates. A start lies on a read base (`0..n-1`).
  An end-exclusive end lies on `0..n`; position `n` exists when the read is shorter than the window. A segment that
  runs to the read end has end `n`. The model's soft-argmax is masked to `0..n-1`, so it can never reach that end.
- `exact`: the decoded edge is inside the truth equivalence interval. `abs_err` (lower is better): the distance to
  the interval. `err` (informational): the signed distance, negative before the interval.

Edges are scored for observed segments with at least `MIN_OBSERVED` germline bases, the end-to-end metrics' rule.
On a cropped read, an edge is scored only when its whole interval lies inside the window and does not touch the cut.
The allele and metadata heads of a cropped read are scored against the truth of the whole read, although the model
saw only its window. An end-exclusive edge at the window's last position, 576 (a read of exactly 576 bases, or a
cropped reverse-complemented read whose segment runs to the canonical end), cannot be decoded by either decoder but
is still scored, so it is always at least one base off; it is rare and left in.

**Alleles.** Alleles are ranked by the calibrated probabilities the selector sees, over the read's `allowed` alleles,
with ties going to the lower index. They apply to the same segments as the boundaries.

- `head.{g}.allele_top{1,3,5}`: the top k meet the scored truth set.
- `head.{g}.allele_top1_generative` (informational): the top-1 is in the generative set.
- `head.{g}.n_above_threshold` (informational): the number of allowed alleles at probability >= 0.5.
- `head.d.short_d_prob_ok`: P(Short-D) >= 0.5 exactly when the D is absent or shorter than `MIN_OBSERVED["d"]`. It
  applies only to a model whose D head has a `Short-D` output. The released IGH model has none: its 33 D outputs are
  the 33 reference D alleles.

**Metadata heads.**

- `head.mutation_rate_abs_err.read`: against the truth V mismatches over the observed V germline bases. It applies
  when V is observed with at least `MIN_OBSERVED["v"]` bases.
- `head.mutation_rate_abs_err.record` (informational): against the GenAIRR record's `mutation_rate`, the training
  target. It applies only when `cases` are given.
- `head.indel_count_abs_err`: against the inserted plus deleted bases of the truth's observed V, D and J CIGARs
  (GenAIRR's indels are single bases). It applies when every gene's truth was built.
- `head.productive.correct`: the head's probability >= 0.5 matches the truth, where the truth defines it.
- `head.orientation.correct`: the orientation logits' argmax is the truth orientation (0 or 1).
- `head.cropped` (informational): 1.0 for a read longer than the window. Its sum counts the crops.
- `head.misoriented` (informational): 1.0 for a read in another frame than the truth's, whose boundaries are not
  scored (see Misoriented reads). Its sum counts them.

`decode_boundaries(outputs, gene, decoder)` and `allele_ranking(outputs, gene)` expose the decoders and the ranking.
The tests check them against the product: the soft decoder gives `correct_segments`' coordinates wherever its repair
does nothing, and the ranking's first allele is the selector's top-1 call.

`calibration_table(outputs, truths, gene, bins=10)` measures the reliability of the top-1 calibrated probability
against the top-1 being in the scored set, over the cases the allele metrics apply to. It has equal-width bins,
closed on the right, as in AlignAIR's `calibrate.ece`. Each bin gives `lo`, `hi`, `n`, the mean `confidence` and the
`accuracy`, None when empty. The table also gives `n`, the overall `confidence` and `accuracy`, and the expected
calibration error `ece`.

### Truth-substitution ablations

`diagnostics.ablation` measures how much of AlignAIR's end-to-end error each stage contributes. Each variant
replays the pipeline with observable truth fed in place of some stages' output. Its AIRR rows are then scored
exactly as a tool's are.

**Variants.** The cumulative chain (`VARIANTS`) adds the truth of one more stage at each step:

| Variant | Adds |
|---|---|
| `baseline` | nothing: the product pipeline |
| `true_orientation` | the truth orientation, teacher-forced and used to canonicalize the read |
| `true_spans` | the truth read spans of the observed segments, as the segments and as the classifier's input |
| `true_calls` | the truth primary allele as each observed segment's call |
| `true_germline` | the truth germline alignment of each observed segment (allele, read and germline coordinates, CIGAR) in place of the matcher |

The single swaps (`SINGLE_SWAPS`) each change one thing on the baseline:
- `argmax_decoder`: segments from the argmax of the position logits; the classifier is unchanged.
- `true_classifier_spans`: truth spans into the classifier only; the segments stay the model's.

`DEFAULT_VARIANTS` is both lists.

**Truth in AlignAIR's terms** (`substitutions_for(variant, truths, outputs_baseline)`, which returns the
`Substitutions` and the fallback counts):
- AlignAIR works on the read it saw: the first 576 bases of the presented read, in its forward frame, 0-based and
  end-exclusive, with ungapped germline coordinates.
- A segment's observed read span is moved into that window: the canonical prefix of a forward read, the canonical
  suffix of a reverse-complemented one.
- A call is `GeneCall((primary,), (1.0,))`.
- An alignment is `GermlineAlignment(allele, seq_start, seq_end, germ_start, germ_end, cigar)` from the observed
  segment.

**Fallbacks.** Truth the read does not have is never invented. The segment keeps the model's value, and the
variant counts it as `{field}.{gene}.{reason}`, for example `germline.d.inverted`. The field is `spans`, `calls` or
`germline`. The reason is the first that applies:

| Reason | When |
|---|---|
| `unscorable` | the segment's truth could not be built |
| `unobserved` | the read shows no base of the segment |
| `orientation` | spans and germline only: the variant keeps the model's orientation and it is wrong, so the truth coordinates are not in the model's frame |
| `cropped` | the segment is not wholly inside the window AlignAIR saw of an over-length read |
| `not_in_model` | calls and germline only: the model cannot call the primary allele |
| `inverted` | germline only: an inverted D, whose reverse-complemented germline frame AlignAIR's alignments and assembly cannot represent; its span and call are still substituted |

Excluded cases and reads without a truth keep every model value and are not counted.

**Running and scoring.** `run_variants(model, reference_set, cases, truths, *, reference, device, batch_size,
variants=DEFAULT_VARIANTS, genotypes=None)` gives one `VariantResult` per variant: `rows`, `scores`, `fallbacks`,
`seconds` (the replay's per-stage wall time) and `substituted`.
- `reference` is the truth reference the rows are scored against. A model whose own reference is not that one is
  refused before anything is replayed (`replay.check_reference`, naming the first differing allele), so an
  in-memory model never reaches scoring unchecked.
- A pure replay comes first. It gives the substitutions their reads and the model's own values. The evaluator
  passes the replay its head metrics read as the private `_baseline`, which must be this model's pure replay of the
  same reads (only its reads are checked), so it is not run twice.
- The rows become the text rows AlignAIR's AIRR writer would put in a TSV (`airr_text_rows`). The assay's reader
  parses them (`airr.reader.rows_to_predictions`, the parser `load_airr_tsv` uses). They are scored with
  `scoring.pipeline.score_predictions`, with the donor genotypes when given, against the truths of the replayed
  cases.
- A test pins this in-memory path to a TSV written by `alignair.io.airr.write_airr`.

**Reading stage contributions.** `stage_contributions(results, metrics, clusters=None)` compares each consecutive
pair of the chain with `stats.compare`, Holm-corrected over `metrics` and, given `clusters`, tested by cluster (see
Clustered comparisons; the evaluator passes `scoring.clusters.cluster_ids` of the suite's truths). Each entry gives:
- `from`, `to`, and the `stages` the step adds (`orientation`, `segmentation`, `allele_calls`,
  `germline_alignment`; a variant missing from `results` makes its pair span both stages);
- per metric: each variant's own mean, value count and coverage over its own cases (`mean_from`, `n_from`,
  `coverage_from` and the `_to` twins); the means over the `n_paired` cases both cover, in `n_clusters` clusters
  (`mean_from_paired`, `mean_to_paired`) and the paired `delta`, their difference (to minus from), with its `ci95`;
  the test and p-value; and a `verdict` of `to_better`, `from_better` or `tie_or_inconclusive`. A table shows the
  paired means beside the delta, so the delta is always the difference of the numbers next to it, and "n paired (K
  clusters)" beside each Δ.

`swap_effects(results, metrics, clusters=None)` compares each single swap with the baseline the same way.
`CONTRIBUTION_METRICS` is a default list: the headline metrics, the junction translation, the read coordinates and
the orientation.

A step's delta is the error that stage contributes **given perfect upstream stages**. It is not a share of the
baseline error. Two consequences:
- The chain can go down. On S1, `true_calls → true_germline` lowers `junction.nt_exact`. The assembly drops the
  junction of a read predicted non-productive with more than one indel unless one of its CIGARs has an indel. The
  matcher's spurious D deletions happen to route those reads through the CIGAR junction path. The clean true
  alignment does not.
- What still differs from the oracle after `true_germline` belongs to the metadata heads (the productive and indel
  predictions that gate the assembly) and to the AIRR assembly itself. The tests show that, given the true germline
  alignment, the assembly writes the V germline coordinates in IMGT-gapped numbering. It also writes the CDR3
  coordinates, from which the junction coordinates are read, as columns of the gapped sequence alignment.

### Speed profile

`diagnostics.profile.profile_model(model, reference_set, sequences, *, device, batch_size, warmup_batches=1,
repeats=3)` shows where AlignAIR spends its time. It replays the predict pipeline on the reads and returns the
`device`, `batch_size`, `reads`, `reads_per_second` and two breakdowns. Each entry of a breakdown gives `ms_per_read`
and its `share` of that breakdown's total.

- `modules`: the network's leaf submodules, timed with forward hooks: `embedding` (two calls per forward, summed),
  `orientation_head`, `meta_tower`, `meta_heads.<name>`, and for each gene `<g>.seg_tower`, `start_head`,
  `end_head`, `cutout`, `cls_tower`, `cls_mid` and `cls_head`. `other` is the rest of the forward stage (orientation
  transform, soft-argmax, tokenizing, copies to the host). The shares sum to 1 over the forward stage's time, so
  compare them with the `forward` share of `stages` to get a module's share of the whole run.
- `stages`: the replay's stages (`replay.STAGES`). `forward` is the network call; `germline`, `airr` and the other
  stages are the Python post-processing. The shares sum to 1 over the run.

How to read it. Towers that repeat across genes (`v.seg_tower`, `d.seg_tower`, `j.seg_tower`, the three
`cls_tower`s and `meta_tower`: seven conv towers) cost about the same each, so their combined share is the cost of
not sharing the convolution. The flatten-to-dense heads (`start_head`, `end_head`, `cls_mid`, `meta_heads`) are the
cost of the fixed 576-position window. Post-network cost is the `germline` and `airr` stages.

The module timings come from separate, hooked runs (on CUDA the hooks synchronize the device around each module,
which slows the run), and the stage timings and `reads_per_second` from unhooked runs; each is the median of
`repeats`. A warm-up replay of `warmup_batches` batches comes first, so CUDA start-up and first-call costs are
excluded. The hooks are removed in a `finally`, and a profile never runs inside an accuracy replay: rows replayed
after a profile equal rows replayed before it (a test pins this). The model is moved to `device` for the runs and
back to its own device afterwards, also in a `finally`.

### The evaluator

`workflow.diagnose.evaluate(model_or_checkpoint, layout, suites, *, device=None, batch_size=256,
variants=DEFAULT_VARIANTS, profile=None, max_cases=None) -> DiagnosticsResult` composes the pieces above over a run
directory; it is the `diagnose` command and the training-loop entry point (see Model diagnostics). It lives in
`workflow` because it reads and writes the run directory and renders the report; the `diagnostics` layer holds the
pieces and the model identity (`diagnostics.identity`). `model_or_checkpoint` is a `.alignair` file or an in-memory
`(model, reference_set)` pair; either goes through `load_model_checked`.
- `device`: None runs a model file on `DEFAULT_DEVICE` (`"cuda"`) and an in-memory model on its own device. An
  in-memory model moved to another device is moved back to its own in a `finally` that wraps every suite and the
  profile. A CUDA device torch does not see is refused, naming `device="cpu"` (the command names `--device cpu`).
- `profile`: None profiles a model file and not an in-memory model; True or False always wins.

For each suite, it:
1. loads the suite's cases and truth (`scoring.pipeline.load_truth`, against `scoring.pipeline.truth_reference`)
   and takes all cases, or `max_cases` evenly spaced (`select_cases`: positions `floor(i * N / n)`, the speed
   protocol's rule, `core.cases.evenly_spaced`);
2. replays the product pipeline once and scores the heads (`head_scores`, with the cases' records) and the
   calibration of each gene (`calibration_table`);
3. runs the variants on the same reads with that replay as the baseline (`run_variants`, with the donors'
   genotypes for a suite with `donors.json`, as `all` scores it) and compares them over `compared_metrics`: the
   suite's primary metrics, then `CONTRIBUTION_METRICS`, leaving out informational ones and those no variant scored.

`variants` run in `DEFAULT_VARIANTS` order; the baseline is added to any other (`variant_set`), and none runs the
heads, the calibration and the profile only. The profile (`profile_model`) runs last, after every accuracy replay,
on up to `PROFILE_READS` (1000) of the diagnosed reads, evenly spaced.

`DiagnosticsResult` holds the `model_key`, the per-suite results (`suites`), the `profile`, the one-page summary as
markdown (`report_md`) and as data (`summary`: `report.diagnostics.diagnostics_summary`), `replay_verified` and
`replay_changes` (see The pin), and `stale_builds` (suite -> what changed). `scalars()` flattens the summary for a
training dashboard: `{"S1/baseline/junction.nt_exact": 0.77, "S1/delta/segmentation/junction.nt_exact": 0.21,
"S1/heads/head.j.end_exact.soft": 0.003, "S1/calibration/v.ece": 0.05, "speed/reads_per_second": 1103.4, ...}`
(`report.diagnostics.diagnostics_scalars`; a missing value is left out).

A suite's result holds `suite`, its cache `key`, `reads` (`built`, `diagnosed`, `scored`, `excluded`, `cropped`,
`misoriented`: scored reads the model put in another frame than the truth's, see Head metrics), `clusters`
(`cluster_counts`: the clusters of two or more diagnosed reads, the `CLUSTER_KEYS` their ids cluster by, and how
many `clusters` and `cases`; a cluster of one read is compared as a read on its own, so S11 with only its `:fwd`
reads diagnosed is not cluster-aware), `heads` and each variant's `scores` (per metric: `mean`, `n`,
`n_applicable`, `coverage`), `calibration` (per gene), `metrics` (the compared family), `variants` (`scores`,
`fallbacks`, per-stage `seconds`, `substituted`), `contributions` (`stage_contributions`), `swaps`
(`swap_effects`), the `seconds` it took, `replay_verified` and `replay_changes` and, in the result only, `reused`.

**Refusals.** Before the model is loaded or anything is written: no suite, an unknown suite or variant, a batch size
or `max_cases` below 1, a suite without `build.json`, a suite whose build record or truth header names another
split than `dev` (`check_dev_build`), suites built against different references, a device torch cannot use
(ValueError); a missing model file (FileNotFoundError). A model whose reference is not the truth reference is
refused by `load_model_checked`, naming the first differing allele. A suite whose diagnosed cases have no scored
truth is a ValueError naming the suite.

**Model key and cache.** The model key is `<weights>-<code>`: 12 hex digits of `weights_sha256` (a model file's
bytes; an in-memory model's class, module tree, configuration, allele order and state tensors) and of
`diagnostics.identity.alignair_code()["code_sha256"]` (`core.provenance.code_identity` of the imported `alignair`
package, taken when `diagnostics.identity` is imported, after it has loaded the modules AlignAIR imports lazily
(the germline matcher, the model-file loader), so it describes the code the process runs; in this checkout it
equals the alignair-local runner's `local_code_identity()`, one implementation). A suite's `suite.json` holds its
result and its `key` (`suite_key`): `kind` and `version`, the full `weights_sha256`, `code_sha256` and
`torch_version`, `assay_code_sha256` (the Python files of the core, truth, airr, metrics, stats, scoring, suites and
diagnostics layers and of `workflow/diagnose.py`), the suite build (`name`, the build record's `identity` and
`config_hash`, the SHA-256 of `cases.jsonl.gz`, `truth.jsonl.gz` and `donors.json`), `variants`, `max_cases`,
`device`, `batch_size` and `replay` (whether the replay is verified, and `diagnostics.pin.replay_digest` of the
product code it mirrors). A record is reused only when its key equals the current one; an unreadable record or any
difference removes it and recomputes the suite. `profile.json` holds the profile and its key: the model identity,
`assay_code_sha256` (the replay it times is assay code), the device with its name, the host and torch's CPU
threads, the batch size and the SHA-256 of the profiled reads. `diagnostics.json`
(`report.diagnostics.build_diagnostics_report`: `model`, `config`, `suites`, `profile`, `summary`), the summary
`diagnostics.md` (`diagnostics_markdown`), each diagnosed suite's `<S>/report.md` (`suite_markdown`) and, with a
profile, `profile.md` (`profile_markdown`) are rewritten on every call. Every file is written through a temporary
file of its own in the same directory (`tempfile.NamedTemporaryFile`) and renamed into place, so a half-written
one is never read, even with two writers.

The tests (`tests/alignair_assay/workflow/test_diagnose.py`) run end to end on S0 and S1 at scale 0.02 with the
small random-init model, and pin the cache: a second call reuses every suite, and a new checkpoint, another code
digest, another torch or assay code, a rebuilt suite, another variant set, case count, batch size or device, or a
record whose key was edited is recomputed.

## Real data

P3 scores the tools on real human IGH repertoires (BioProject PRJNA555323; Rodriguez et al. 2023, Nat Commun
14:4419) against each donor's own alleles from genomic long-read sequencing, published by VDJbase (genomic study
P25, CC0). This section grows with P3a; so far it covers the downloads, the donor genotypes and the preprocessing of
the reads.

**Data root.** Downloads live under a data root given to the commands, never in a run directory or the repository.
Each source has its own directory there, with an `MD5SUMS.txt`:

- `airrseq/<donor>/<run>/<run>_{1,2}.fastq.gz`: ENA's FASTQs;
- `airrseq_odp/<donor>/<run>/`: the run's SRA object from NCBI's AWS Open Data mirror and the pair converted from
  it;
- `genotypes/vdjbase_genomic_samples/<sample>.csv` and `.bam` (one VDJbase sample per donor, e.g. `P25_I58_S1` for
  SC-23: its alleles and its contigs on IGenotyper's `igh` reference);
- `genotypes/ogrdb_human_IGH_VDJ/Homo_sapiens_IGH_VDJ_rev_8.json` (OGRDB's IGH_VDJ germline set G00060, CC0);
- `genotypes/igenotyper/gene_coords.bed` (IGenotyper's gene coordinates on `igh`, at commit `40a00015`);
- `genotypes/rodriguez2023_suppl/41467_2023_40070_MOESM4_ESM.xlsx` (the paper's supplementary data).

Tests that read it run only when `ALIGNAIR_ASSAY_DATA` names it; those that download need `ALIGNAIR_ASSAY_NETWORK=1`
too, and those that run a container (sra-tools, samtools, Immcantation) `ALIGNAIR_ASSAY_DOCKER=1`.

**Fetch (`real.fetch`).** Every file is pinned in the module (size, md5 or S3 ETag, read count, commit, image
digest), so a fresh data root gets the same bytes or a refusal. A file already there is reused only when it is the
pinned file (and its `MD5SUMS.txt` agrees); a download goes to `<name>.part` and is renamed once verified. Downloads
are https only, with a 60 s timeout and at most two retries; nothing downloaded is executed.

- `fetch_reads(run, data_root)` returns a pilot run's pair and its `FetchRecord` (`path`, the pair's directory;
  `md5`, of the pair's `MD5SUMS.txt` lines; `bytes`, `source_url`, `kind` `"reads"`, and `quality`, `"full"` or
  `"lite"`; `detail` says what was checked). It prefers ODP: the object is checked against its pinned size and S3
  ETag, then in `SRA_TOOLS_IMAGE` (ncbi/sra-tools 3.4.1, pinned by digest; as the current user, with no network,
  on the run's directory only) by `vdb-validate` and converted by `fasterq-dump --split-files` with ENA's read
  names. The mates must hold ENA's read count and, where ENA's FASTQs are present, their first 2,000 records must
  be ENA's names and bases. When the object is missing or its qualities are SRA-Lite, the ENA FASTQs (ENA's md5s)
  are used and the record says why. All eight pilot runs convert at full quality, while ENA serves SRA-Lite
  qualities (`?` and `$` only) on every pilot run but SC-23's; on W-97 all 103,049 pairs have ENA's names and
  bases, with 27 quality symbols.
- `quality_kind(fastq_gz, n=2000)` is `"lite"` when every quality symbol of the first `n` records is `?` or `$`,
  else `"full"`.
- `fetch_genotype_files(donor, data_root)` returns `vdjbase_csv`, `vdjbase_bam`, `ogrdb` and `gene_coords` records.
- A run's record is `fetch/reads/<run>.json` and a donor's `fetch/genotypes/<donor>.json`; a reads record is
  reused while its files are unchanged, and a changed file stops the run with its name. `fetch_manifest(data_root)`
  gathers them into `fetch_manifest.json`, with the pins, the module's code digest, each run's `quality` and what
  is `pending`.
- It stops with `FetchError` before any download when Docker or the pinned image (by digest) is not available, and
  on a download that is not the pinned file, a conversion that is not the run's reads, or a file already there
  that is not the pinned one, which it leaves as it is.

**Sub-layers.** Each module of `real/` is a layer of its own (see Layers and dependency rule). The rows, some ahead
of their modules:

| Layer | May import from `alignair_assay.` |
|---|---|
| `real.genotype` | core, real.genotype |
| `real.truth` (pure) | core, real.genotype, real.truth |
| `real.metrics` (pure) | core, metrics, real.genotype, real.truth, real.metrics |
| `real.fetch` | core, real.fetch |
| `real.locus` | core, real.genotype, real.locus |
| `real.preprocess` | core, real.preprocess |

**Donor genotypes (`real.genotype`).** Identity is the ungapped sequence, compared exactly; names are never matched
across sources. The shared reference's V names are OGRDB F/G labels (`IGHVF6-G21*06`) without IUIS names, so:

- `IuisMap.from_ogrdb(json_path, reference)` maps each shared allele to the OGRDB IUIS labels with its exact
  sequence (`labels`, `genes`; `shared_for_sequence` and `labels_for_sequence` go the other way). On
  `HUMAN_IGH_OGRDB` against rev 8, 236 of 238 shared alleles map; `unmapped()` is `IGHVF10-G33*01` and
  `IGHVF10-G42*01`, each one base shorter than OGRDB's IGHV3-13*06 and IGHV3-43D*05. Eleven carry two labels:
  identical-sequence paralogs such as `IGHVF6-G21*06` = IGHV1-69*01 / IGHV1-69D*01, and the identical D pairs.
- `load_vdjbase_csv(path, donor, reference, iuis_map)` gives a `RealGenotype` (`donor`, `alleles`, `captured`,
  `source`) of `RealAllele` records (`allele_id`, `gene`, `segment`, `sequence`, `gapped`, `haplotype`,
  `functional`, `shared`, `iuis`, `imgt_identity`), one per haplotype per gene. Haplotype 0 is unphased, not
  homozygous. `gapped` is VDJbase's V-REGION-GAPPED, whose gaps are `.` columns and `-` for a deleted base.
  `RealGenotype.carried(gene)`, `genes(segment)` (every carried gene, functional or not) and `absent_genes()`
  (genes with an allele absent from the shared reference) read it; `captured` stays None until the contig spans
  are known. The shared reference's own gapped V sequences are `Reference.gapped(name)`, from the dataconfig.
- `functional` is False when VDJbase's `notes` flag a V-REGION defect: a stop codon, a first or second cysteine or
  the conserved tryptophan not found, or a V-REGION length mismatch. On the eight pilot donors this reproduces
  VDJbase's genotype API allele for allele. Their functional V genes are the API's counts (SC-23 47, SC-4 49,
  SC-11 50, SC-6 49, W-97 39, W-89 53, W-133 40, W-131 48), out of 51, 53, 54, 52, 42, 57, 44 and 52 V genes in
  their CSVs.
- `filter_d` drops a D call whose id has a gap-indel token (`g8-`, `8---a11`, an `i15ac` insertion) or whose
  identity to its closest IMGT allele is under 90%; the loader applies it and lists what it dropped in `source`.
  `merge_identical(reference)` maps each shared allele to those with its sequence (D4-4*01 / D4-11*01,
  D5-5*01 / D5-18*01). `upto(gapped, 318)` is an allele through IMGT position 318, past which rearranged reads
  cannot tell alleles apart (Peres et al. 2023).
- `sd3_novel(path)` reads the paper's 137 novel alleles (Supplementary Data 3; the workbook or a tab-separated
  export of its sheet), reverse-complementing each sequence, which the sheet gives reversed, as
  `Sd3Allele(closest, sequence, gene, closest_all)`. `sd3_only(genotypes, sd3)` is the genes touched by those in
  no given donor's alleles: the sheet's gene and every closest allele's gene. Only the donors given count, so pass
  the cohort: the eight pilot donors alone leave 45 genes touched.

**Captured genes (`real.locus`).** R1 counts a call to a gene the donor lacks as `unsupported_gene` only where the
donor's long reads resolve that gene's region; a call to a gene in a region they do not resolve is excluded. A gene
is captured when the donor carries an allele of it (any segment, any haplotype), or when its interval on
IGenotyper's `igh` reference (`gene_coords.bed`) lies wholly inside one contig of haplotype 0 (unphased), or inside
a contig of haplotype 1 and a contig of haplotype 2. Two contigs that cover a gene only together, or a partial
overlap, do not capture it.

- The haplotype is the contig name's `h=` (`c=igh:<start>-<end>_h=<0|1|2>_...`), the block haplotype IGenotyper
  assembled it from. The `RG` tag is per-alignment SNV phasing (RG 0: no phased SNV under the alignment), not the
  contig's haplotype. IGenotyper's allele extraction uses `h=`, and the VDJbase CSVs' haplotype equals it on all 840
  pilot rows, while RG differs on 60. A contig name without an `h=` of 0, 1 or 2 stops the donor.
- Only the alignments IGenotyper uses count (its `skip_read`): unmapped, secondary and supplementary records are
  skipped, with no MAPQ filter. Most records are secondary (571 of SC-23's 674), and counting them would add
  genes the contigs do not span.
- A span is `[POS - 1, POS - 1 + L)`, L the CIGAR's reference-consuming length (`M`, `D`, `N`, `=`, `X`). A long
  `D` therefore spans the genes a haplotype deletes (W-89: 9.4 kb over IGHD1-7 to IGHD3-3).
- The BED's 92 `igh` genes (59 IGHV, 27 IGHD, 6 IGHJ) carry IUIS gene names: OGRDB rev 8's, plus IGHV1-38-4,
  which VDJbase uses too. It has no IGHV3-25, IGHV3-62 or IGHV8-51-1; a donor that carries them has them captured
  all the same.
- `contig_spans(bam)` runs `samtools view` in `SAMTOOLS_IMAGE` (biocontainers samtools 1.23.1, pinned by digest)
  as the current user, with no network and no capabilities, on the BAM's directory alone mounted read-only, and
  kills the container by name on a timeout; only its SAM text is parsed (`parse_sam_spans`).
  `donor_capture(genotype, bam, bed)` returns the genotype with `captured` set and, in `source["capture"]`, the
  image, the whole command, the SHA-256s of the inputs (the genotype's CSV among them, since every carried gene is
  captured) and of the module, a `fingerprint` of those, the spans per haplotype,
  and the gene lists `spanned`, `carried_not_spanned` and `spanned_not_carried` (candidate deletions or assembly
  gaps). Its BAM must be the genotype's VDJbase sample.
- On the pilot donors the contigs span 40 to 52 V genes. Calls to V genes the donor lacks can be `unsupported_gene`
  only for 0 to 3 genes per donor (IGHV4-30-2 on SC-23; IGHV3-11, IGHV3-15 and IGHV4-30-2 on W-97; IGHV4-31 on
  SC-4); calls to the donor's other missing genes are excluded.

**Preprocessing (`real.preprocess`).** Each run's pairs become the unique sequences the tools are given, with pRESTO
0.7.9 in `IMMCANTATION_IMAGE` (immcantation/suite 4.8.0, pinned by digest), one container per step: as the current
user, with no network and no capabilities, on at most 8 CPUs, the reads mounted read-only and one writable work
directory. No V(D)J tool and no germline reference is used; the only sequences given are the primers.

- The protocol is read from the first 2,000 pairs (`classify_protocol`): `"race"` when R2[12:16] is the `GTAC`
  linker in at least 80% of them, `"multiplex"` when at least 80% of R1 start with one of the paper's VH primers
  (`VH_PRIMERS`) and 80% of R2 with an IgM/IgG/IgA primer (`C_PRIMERS_MULTIPLEX`), each within 2 mismatches;
  otherwise, or when both hold, `"unknown"`, and the run is refused with the fractions in its record.
  All eight pilot runs classify as their group: the SC runs `race` (linker in 94.4-96.1% of pairs), the W runs
  `multiplex` (VH primer in 94.7-95.8%, C primer in 90.3-93.5%).
- Full-quality reads are first trimmed at Q20 and kept from 125 nt (the paper); SRA-Lite reads skip that, and the
  record says so. 5'RACE: UMI (R2's first 12 nt) and the `GTAC` + `GGG` cut; the isotype from Immcantation's IGH
  C-region primers on R1 (`MaskPrimers align`, within 70 nt, error 0.2); UMI consensus as the paper
  (`ClusterSets` within each UMI, `AlignSets muscle`, `BuildConsensus`); `AssemblePairs align`; `CollapseSeq`;
  consensus count of at least 2. Multiplex: `MaskPrimers score` masks the VH primer on R1 and the C primer on R2;
  both must be found; `AssemblePairs align`; at least 250 nt after primer removal (the paper's 400 nt before it is
  `paper_length_filter`); `CollapseSeq`. Reference-guided assembly (`AssemblePairs sequential`, which the image's
  `presto-clontech-umi` uses) is ruled out; pairs that do not overlap are dropped and counted. The module docstring
  lists every difference from `presto-clontech-umi`.
- `preprocess_run(donor, r1, r2, layout, quality=...)` writes `real/<donor>/reads.fasta` (at most `cap`, 50,000,
  sequences, ordered by sequence id and chosen evenly spaced), `reads.tsv` (`sequence_id`, `isotype`, `protocol`,
  `consensus_count`, `dupcount`, `primer_masked`: the `[start, end)` spans of primer bases, which carry no germline
  evidence), each step's pRESTO output in `preprocess_logs/` and, last, `preprocess.json`: the counts and pass rate
  of every step, the classifier's evidence, the image, the length distribution, the isotype split, the wall time
  and a fingerprint of the inputs' md5s, the module, the image and the parameters. A current record is reused.
- On the pilot, W-97 (103,049 pairs) gives 41,161 unique sequences in 174 s, 66% of its primer-matched pairs
  assembling; SC-23 (164,664 pairs) gives 6,601 in 317 s, 51% of its consensus pairs assembling. The pairs lost
  are those whose mates, shortened by the Q20 trimming, no longer overlap; without the quality steps 83% and 71%
  assemble.

**Splits.** The pilot dev donors are SC-23, SC-11, W-97 and W-133; the test donors SC-4, SC-6, W-89 and W-131 are
locked for claims.
