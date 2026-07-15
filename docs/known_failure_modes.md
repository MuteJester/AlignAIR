# Known failure modes & when to prefer another tool

AlignAIR is a neural aligner that is strong on allele calling and segmentation across a wide input
range, but it is not the right tool for every job. This page is the honest list of where it struggles
and what to use instead — because trusting a result means knowing when not to.

## Where AlignAIR can be wrong

| Failure mode | What happens | What to do |
|--------------|--------------|------------|
| **Junction boundary jitter** | The CDR3/junction coordinates can be off by ~1–2 nt, and the J side is a little worse than the V side. The `junction` string is usually fine, but exact `*_sequence_end`/`junction_start` may wobble. | Treat junction *coordinates* as approximate; use the equivalence-set/`junction_aa` rather than single-nt positions for clonal grouping. |
| **Missing anchors → no junction** | If the reference lacks the conserved Cys-104 (V) / Trp-Phe-118 (J) anchors for an allele, that read gets V/D/J calls but an **empty** junction (honest absence). | Add anchors to the reference, or accept junction-free rows for those alleles. Built-in dataconfigs carry anchors. |
| **Short-read / fragment ambiguity** | A short read carries little V information, so the true allele is genuinely indistinguishable from many others. | This is surfaced, not hidden: `*_call_set` lists the indistinguishable alleles. When the set spans more than one allele, treat the result as gene/family-level rather than a confident single call. |
| **Full-length, heavily-mutated V** | At high SHM on full-length reads, V allele accuracy is a **tie** with IgBLAST, not a win — the hardest regime. | Cross-check with IgBLAST for SHM-heavy lineage work; the `*_call_set` still narrows the candidates. |
| **Model / reference mismatch** | Running a model on the wrong locus (e.g. an IGH model on IGK reads) yields plausible-but-meaningless calls. | Use a model trained for your locus; a multi-locus model attributes each read to a locus and cannot make cross-locus calls. |
| **Out-of-scope / contaminant reads** | Non-target sequences still get calls. | `is_contaminant` is an advisory flag (calls are retained, not dropped); filter on it if you need a clean set. |
| **Light chains / no-D loci** | No D segment exists. | `d_call` is empty and `np2` is empty by design — not an error. |
| **Tiny/demo models** | `alignair demo` and the `quick` train preset produce models whose calls are **not accurate** — they only prove the pipeline runs. | Use a [pretrained model](models.md), or train the `desktop`/`full` preset, for real calls. |

## Fields that are *not* emitted yet

CDR/FWR region coordinates and read-derived `c_call` are **not** produced (AlignAIR aligns
V(D)J only and does not model IMGT region numbering or the constant region). See the
[AIRR field coverage matrix](airr_fields.md) for the full list and the metadata workarounds.

## When to prefer IgBLAST / MiXCR / partis

- **Raw throughput on bulk data** — AlignAIR is currently ~2–3× slower than IgBLAST
  (the gapped-alignment/decode stage dominates). For very large bulk repertoires where the
  default reference suffices, IgBLAST may finish sooner. A lighter `--columns` preset (which skips
  the gapped-alignment assembly) claws back time (see [Performance](performance.md) for measured
  throughput and [Benchmarks](benchmarks.md) for accuracy).
- **A long-established, citation-stable standard** — if a pipeline or reviewer expects
  IgBLAST/IMGifeT output specifically, use it.
- **End-to-end clonotype/repertoire assembly** — MiXCR and the Immcantation suite cover
  clustering, lineage, and downstream stats that AlignAIR does not; AlignAIR's job is the
  per-read alignment (its AIRR output feeds straight into them).

## Where AlignAIR is the better choice

- A **donor/study-specific genotype** — a subset of the model's reference — supplied at predict time
  with `--genotype`, no retraining, to sharpen calls on ambiguous reads.
- **Surfaced ambiguity**: an equivalence set (`*_call_set`) instead of a single, possibly wrong, top
  allele — especially on short or ambiguous reads.
- **Non-human / custom species** via `alignair train` on your own FASTA reference.
- Broad input coverage (fragments, reverse-complement, indels) in one model.
