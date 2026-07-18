# Migrating from IgBLAST

AlignAIR writes the **same AIRR rearrangement schema** IgBLAST does with `-outfmt 19`, so most of
a downstream pipeline (Change-O, Immcantation, Scirpy, nf-core/airrflow) keeps working unchanged.
This example maps the two tools field for field, shows a real side-by-side on one read, and ends
with a migration checklist.

Runnable AlignAIR commands here are checked against the real CLI in CI. The `igblastn` command is
IgBLAST's own and is shown for reference only.

## Invocation

IgBLAST needs BLAST databases built from germline FASTAs and an organism flag. AlignAIR carries its
germline reference inside the model file, so there is no database to build and no reference to select
at the command line - you pick a model instead.

```bash
# IgBLAST (your existing command; -outfmt 19 emits the AIRR schema)
igblastn -germline_db_V human_gl_V -germline_db_D human_gl_D -germline_db_J human_gl_J \
  -auxiliary_data human_gl.aux -organism human -ig_seqtype Ig \
  -outfmt 19 -query reads.fasta > igblast.tsv
```

```bash
# AlignAIR: one command, reference embedded in the model, downloaded and hash-verified on first use
alignair predict --input reads.fasta --out alignair.tsv --model alignair-igh-human@1.0.0
```

Pin the model version (`@1.0.0`) whenever a result needs to be reproducible. See all installed and
available models with `alignair models list`.

## Reference selection

| IgBLAST | AlignAIR |
| --- | --- |
| `-germline_db_V/D/J` point at BLAST DBs you build | The germline reference is embedded in the model and fingerprinted on load |
| `-organism human` | Choose the model: `alignair-igh-human`, `alignair-igkl-human`, `alignair-tcrb-human` |
| Add alleles by rebuilding the DB | Adding alleles changes the model's fixed output space - train a new model (`alignair train`) |
| Restrict to a subset by editing the DB | Restrict to a donor's alleles at predict time: `--genotype donor.yaml` (no retraining) |

An AlignAIR model can only call the alleles it was trained on; a genotype **subsets** that reference,
it cannot add to it. This is the one behaviour that differs in kind from editing a BLAST database.

## A real read through both tools

The same 348 nt human IGH read, aligned by IgBLAST (`-outfmt 19`) and by
`alignair predict --model alignair-igh-human@1.0.0`. Both rows are genuine tool output, not
illustrations.

| Field | IgBLAST | AlignAIR | Note |
| --- | --- | --- | --- |
| `v_call` | `IGHVF10-G37*08` | `IGHVF10-G37*08` | agree |
| `j_call` | `IGHJ6*03` | `IGHJ6*03` | agree |
| `d_call` | `IGHD2-8*02` | `IGHD1-26*01` | genuine disagreement - D is short and heavily trimmed, the least certain call in both tools |
| `v_call_set` / `d_call_set` / `j_call_set` | *(not emitted)* | `IGHVF10-G37*08` / `IGHD1-26*01` / `IGHJ6*03` | AlignAIR extension: the candidate set (`p >= 0.5`, capped at 3). Single-member here; widens on ambiguous reads |
| `junction` | `TGTGCGAAAGGAGTCATACTGGCAGTTACTGG` | `TGTGCGAAAGGAGTCATACTGGCAGTTACTGG` | agree (nucleotides) |
| `junction_aa` | `CAKGVILAVTG` | `CAKGVILAVT.` | AlignAIR's J boundary is 2 nt shorter here, leaving the last codon incomplete (`.`). See "junction assembly" below |
| `v_sequence_start` / `_end` | `1` / `295` | `1` / `296` | both 1-based; boundaries can differ by 1-2 nt |
| `j_sequence_start` / `_end` | `314` / `348` | `314` / `346` | same |
| `v_cigar` | `295M53S1N` | `296M` | different CIGAR conventions; both consume the query correctly |
| `productive` / `vj_in_frame` / `stop_codon` / `rev_comp` / `locus` | `F` / `F` / `F` / `F` / `IGH` | `F` / `F` / `F` / `F` / `IGH` | agree |
| `airr_assembly_status` | *(not emitted)* | `complete` | AlignAIR extension: honest per-record quality flag |

## Output fields

The shared AIRR columns (`sequence_id`, `v/d/j_call`, `*_sequence_start/end`, `junction`,
`junction_aa`, `productive`, `rev_comp`, `sequence_alignment`, `germline_alignment`, `*_cigar`, ...)
carry the same meaning, so a reader written for IgBLAST output reads AlignAIR output.

AlignAIR **adds** columns IgBLAST does not emit:

- `v/d/j_call_set` - the candidate set per gene (see "call sets" below).
- `airr_assembly_status` / `airr_assembly_reason` - a per-record quality flag; filter to `complete`
  before junction or productivity analysis.
- `mutation_rate`, `productive_prediction` - neural estimates (advisory).
- `segmentation_low_quality`, `length_cropped`, `input_sequence` - provenance and quality flags.

Some IgBLAST-specific columns (`v_frameshift`, `d_frame`, `complete_vdj`) are not part of AlignAIR's
output.

## Semantic differences worth knowing

- **Call sets, not a single guess.** When a read cannot distinguish alleles, AlignAIR reports every
  allele it scored at `p >= 0.5` in `*_call_set` (ranked, capped at 3), instead of committing to one
  top call. A multi-member set means "do not read this at allele resolution", not "these are the only
  possibilities". IgBLAST reports a single ranked list; there is no direct equivalent.
- **Coordinates.** Same 1-based AIRR convention as IgBLAST. Boundaries can jitter by 1-2 nt,
  especially on the J side - group clones on `junction_aa`, not single-nt positions.
- **Junction assembly.** AlignAIR derives the junction from its own coordinates; a boundary that lands
  mid-codon yields an incomplete final residue (`.`), as above. The nucleotide `junction` is usually
  exact even when the last `junction_aa` residue is not.
- **Confidence.** AlignAIR does not emit a per-call e-value. Pretrained models carry post-hoc
  temperature calibration on the allele heads, but the reserved confidence columns are left blank -
  do not filter on them.
- **Partial predictions.** A row can be `airr_assembly_status = partial` (valid calls, but the
  junction or another product could not be assembled). These are emitted with their calls; IgBLAST has
  no equivalent flag. Filter to `complete` for junction-level work.

## CPU and GPU

IgBLAST is CPU-only and multithreaded. AlignAIR runs on CPU or GPU:

- **GPU** (CUDA, or Apple Silicon MPS) is selected automatically when available and is the fast path.
- **CPU-only** works with no change; install the CPU build of PyTorch first to avoid pulling CUDA
  (see Troubleshooting). Throughput on the documented hardware is on the Speed & throughput page - it
  states the hardware it was measured on, so measure on yours before capacity planning.

## Runnable side-by-side

No ground truth needed - `alignair compare` reports agreement between two AIRR TSVs on the same reads,
including set-rescue (how often the other tool's call falls inside AlignAIR's `*_call_set`).

```bash
# 1. run both tools on the same reads (igblastn is your existing command, above)
alignair predict --input reads.fasta --out alignair.tsv --model alignair-igh-human@1.0.0

# 2. structural check of the AlignAIR output
alignair validate-airr alignair.tsv

# 3. where do they agree? (set-rescue distinguishes shared ambiguity from real conflict)
alignair compare --a alignair.tsv --b igblast.tsv --a-name AlignAIR --b-name IgBLAST
```

## Migration checklist

- [ ] Pick the model for your locus (`alignair-igh-human`, `alignair-igkl-human`,
      `alignair-tcrb-human`); pin `@version` for reproducibility.
- [ ] Replace the `igblastn ... -outfmt 19` call with `alignair predict --input ... --out ... --model ...`.
- [ ] Point your downstream reader at the new TSV - the shared AIRR columns are unchanged.
- [ ] Add a filter on `airr_assembly_status == complete` before junction / productivity analysis.
- [ ] Decide how to use `*_call_set`: treat multi-member sets as below allele resolution.
- [ ] If you constrained IgBLAST's DB to a donor, pass `--genotype donor.yaml` instead.
- [ ] If you relied on novel or added alleles, plan an `alignair train` run - a model's allele set is fixed.
- [ ] Sanity-check a batch with `alignair compare` before switching the pipeline over.
