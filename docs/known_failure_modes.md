# Known failure modes & when to prefer another tool

AlignAIR is a neural aligner with a runtime (dynamic) reference. It is strong on allele
calling and segmentation across the widest input range, but it is not the right tool for
every job. This page is the honest list of where it struggles and what to use instead —
because trusting a result means knowing when not to.

## Where AlignAIR can be wrong

| Failure mode | What happens | What to do |
|--------------|--------------|------------|
| **Junction boundary jitter** | The CDR3/junction coordinates can be off by ~1–2 nt, and the J side is a little worse than the V side. The `junction` string is usually fine, but exact `*_sequence_end`/`junction_start` may wobble. | Treat junction *coordinates* as approximate; use the equivalence-set/`junction_aa` rather than single-nt positions for clonal grouping. |
| **Missing anchors → no junction** | If the reference lacks the conserved Cys-104 (V) / Trp-Phe-118 (J) anchors for an allele, that read gets V/D/J calls but an **empty** junction (honest absence). | Add anchors to the reference, or accept junction-free rows for those alleles. `alignair reference validate` reports anchor coverage. |
| **Short-read / fragment ambiguity** | A short read carries little V information, so the true allele is genuinely indistinguishable from many others. | This is handled, not hidden: `*_call_set` lists the indistinguishable alleles and `*_resolved_call`/`*_call_level` backs off to gene → family → abstain. Use the resolved call, not the single top call. |
| **Full-length, heavily-mutated V** | At high SHM on full-length reads, V allele accuracy is a **tie** with IgBLAST, not a win — the hardest regime. | Cross-check with IgBLAST for SHM-heavy lineage work; the `*_call_set` still narrows the candidates. |
| **Model / reference mismatch** | Running a model on the wrong locus (e.g. an IGH model on IGK reads) yields plausible-but-meaningless calls. | This is a **hard error by default** (locus check); only `--force-locus-mismatch` overrides. Use a model trained for your locus. |
| **Out-of-scope / contaminant reads** | Non-target sequences still get calls. | `is_contaminant` is an advisory flag (calls are retained, not dropped); filter on it if you need a clean set. |
| **Light chains / no-D loci** | No D segment exists. | `d_call` is empty and `np2` is empty by design — not an error. |
| **Tiny/demo models** | `alignair demo` and the `smoke` train preset produce models whose calls are **not accurate** — they only prove the pipeline runs. | Train `desktop`/`standard`, or use a published bundle, for real calls. |

## Fields that are *not* emitted yet

CDR/FWR region coordinates and read-derived `c_call` are **not** produced (AlignAIR aligns
V(D)J only and does not model IMGT region numbering or the constant region). See the
[AIRR field coverage matrix](airr_fields.md) for the full list and the metadata workarounds.

## When to prefer IgBLAST / MiXCR / partis

- **Raw throughput on bulk data** — AlignAIR is currently ~2–3× slower than IgBLAST
  (the gapped-alignment/decode stage dominates). For very large bulk repertoires where the
  default reference suffices, IgBLAST may finish sooner. Try `--no-full-alignment` and
  `--v-reader parasail` to claw back time (see the [benchmarks](benchmarks.md)).
- **A long-established, citation-stable standard** — if a pipeline or reviewer expects
  IgBLAST/IMGifeT output specifically, use it.
- **End-to-end clonotype/repertoire assembly** — MiXCR and the Immcantation suite cover
  clustering, lineage, and downstream stats that AlignAIR does not; AlignAIR's job is the
  per-read alignment (its AIRR output feeds straight into them).

## Where AlignAIR is the better choice

- A **donor/study-specific reference** (allele subset and/or **novel** alleles) supplied at
  runtime, with no retraining — the dynamic-genotype feature.
- **Calibrated ambiguity**: equivalence sets + resolved calls instead of a single, possibly
  wrong, top allele — especially on short or ambiguous reads.
- **Non-human / custom species** via `alignair train` on your own FASTA reference.
- Broad input coverage (fragments, reverse-complement, indels) in one model.
