# AlignAIR-AA — Gate 1: Supported-Input Product Contract

> **AIRR Utility & Trust Gate, Gate 1 of 6.** This contract defines who the AA product is for, the
> inputs it supports, the inputs it must refuse or hedge, the minimum evidence each result needs to be
> trustworthy, and when a user should reach for a different tool instead. **No HF/public-registry
> publication until Gates 1–4 pass** (Gate 1 contract, Gate 2 semantics audit, Gate 3 adversarial/OOD
> applicability, Gate 4 comparative scientific utility).

**Status:** DRAFT for review · **Date:** 2026-07-24 · **Model in scope:** `alignair-aa-igh-human 2.0.0`
(v2 = `cont_step318000`; 149 V protein-equivalence classes, 33 D allele-head indices, 7 J classes; input
≤ 192 aa; simulator-domain conformal calibration only).

**Standing decisions that bound this contract** (owner-approved 2026-07-24):
1. **Native JSON, no AIRR claim.** The product emits AlignAIR-AA JSON as its own schema and does **not**
   claim AIRR-Rearrangement compliance or reuse standard AIRR field names with different semantics. An
   AIRR-adjacent adapter is deferred to *after* community validation (Gate 6), not v1.
2. **Trust-signaling lives in documentation, not new output fields.** Evidence levels, D's experimental
   status, and applicability are captured in this contract, the Gate 2 audit, the model card, and the
   field-evidence table — the v1 result schema is **not** extended with `evidence_level`/`applicability`
   fields until Gate 3 validates their semantics.
3. **D remains experimental**, documented in the card and audit (not as a per-result field).

---

## 1. Product thesis (the narrow, defensible claim)

> **Human IGH variable-domain amino-acid annotation for users who possess protein sequences but not
> trustworthy matching nucleotide sequences** — especially therapeutic, engineered, historical, or
> protein-only antibody datasets.

The product predicts, from a **human immunoglobulin heavy-chain variable-domain (VH) amino-acid
sequence**, the most likely germline **V** (as an OGRDB-ASC protein-equivalence class, expanded to its
IMGT alleles), the **J**, an **experimental D**, approximate segment coordinates, and a **calibrated
alternative set** whose coverage guarantee holds *only under the simulator calibration distribution*.

It exists because the standard germline-annotation pipeline (IgBLAST, nucleotide AlignAIR, IMGT/V-QUEST)
is nucleotide-oriented and cannot be applied when only protein is available.

## 2. Intended users

- Antibody engineers / therapeutic-antibody scientists holding **protein VH sequences** (patents,
  literature, in-house constructs) with no reliable matching nucleotide.
- Repertoire/AIRR researchers working with **protein-only or historical datasets** (e.g. mass-spec-derived
  or synthesized sequences) who need a germline-ancestry starting point.
- Curators who want a **fast, uncertainty-aware first-pass** germline hypothesis for human VH proteins,
  to be confirmed by orthogonal methods.

**Not the intended user:** anyone who has nucleotide reads (use nucleotide AlignAIR / IgBLAST); anyone
needing IMGT numbering, CDR delimitation, or a true AIRR junction; anyone annotating light chains or TCRs.

## 3. Three concrete supported workflows

1. **Germline-ancestry hypothesis for engineered/therapeutic VH proteins.** Input a set of human VH
   protein sequences; for each get the top V protein-class → IMGT alleles, its family/gene, and the
   calibrated alternative set, to support design-ancestry / lineage discussion. *(Closest to the existing
   16-antibody descriptive check — the one place real-domain evidence exists, at family level only.)*
2. **Uncertainty-aware triage of a protein VH dataset.** Run a protein FASTA; use `set_size` and the
   calibrated class/allele set to separate confident calls from ambiguous ones, and route the ambiguous
   ones to manual review. The *value* here is the honest uncertainty signal, not a single top-1 label.
3. **Approximate V/J boundary extraction for manual inspection.** Use the segment coordinates and
   `inter_vj_aa` (the method-defined V-end..J-start region — **not** a validated CDR3/junction) to locate
   regions of interest for downstream manual analysis in another tool.

## 4. Supported-input matrix (in-contract)

| Dimension | Supported |
|---|---|
| Locus | Human **IGH** heavy chain only |
| Domain | **Variable domain (VH)** — framework + CDR structure of V, ideally through J |
| Molecule | **Amino-acid** sequence, single chain |
| Alphabet | 20 standard aa + `X`; no `*`, no gaps, no whitespace |
| Length | ≤ **192** aa (longer is rejected, never truncated) |
| Orientation | Handled in-model (forward/again — orientation head) |
| Coverage | Full-length VH **and** truncations/fragments **within the trained coverage contract** (`full`/`trunc`/`short` simulator strata) |

An in-contract input is one a user could plausibly have as "a human antibody VH protein": a full VH, or a
fragment that still contains a substantial contiguous V region.

## 5. Explicitly unsupported inputs (out-of-contract)

The model will still emit IGH V/D/J calls for **any** in-alphabet ≤192-aa string, because the v1 validator
checks only alphabet and length. Until **Gate 3** adds an applicability guard, these produce
**precise-looking but meaningless** annotations. Treat every result on the following as invalid:

- **Light chains** — IGK / IGL.
- **T-cell receptors** — TRA / TRB / TRG / TRD.
- **Non-antibody proteins** — any unrelated ~120-aa protein.
- **Isolated CDR3 / short peptides** — a CDR3-only or otherwise anchor-less fragment.
- **Signal-peptide + VH, or full heavy chains** — inputs containing leader and/or constant-domain
  residues outside the trained VH window.
- **Camelid VHH / single-domain antibodies** — not in the human IGH training distribution.
- **Non-human** (murine, rabbit, chimeric) — unless a specific variant is separately validated.
- **Sequences lacking sufficient V or J evidence** — too short, or covering a region the model was not
  trained to anchor.
- **Severe OOD** — anything far from the human-VH manifold.

> **A widened conformal set is NOT an OOD detector.** Conformal set size is calibrated under the simulator
> distribution; on genuinely out-of-domain input the model can be confidently wrong with a *small* set.
> The set quantifies simulator-domain ambiguity, not "is this even an IGH VH protein."

## 6. Minimum evidence required per result (asserted; enforcement is Gate 3)

| Result | Minimum evidence for the call to mean anything |
|---|---|
| **V** call | Input contains a substantial contiguous portion of the V domain (framework + CDR structure), within the trained coverage window. |
| **J** call | The J region is actually present in the input. |
| **D** call | **Experimental in all cases** — no real-domain evidence; AA D signal is intrinsically weak. Do not rely on it. |
| **coordinates** | Only meaningful when the corresponding segment is present; they are model estimates, not validated boundaries. |
| **`inter_vj_aa`** | Only a substring between V-end and J-start; meaningful only if both boundaries are real. Not a CDR3/junction. |
| **calibrated set** | Coverage guarantee holds only under simulator exchangeability and the applied group; on real/OOD input it is descriptive. |

These are **contract requirements the product does not yet enforce.** v1 cannot detect their absence — that
detector is Gate 3. Until then, the burden is on the user (and this documentation) to stay in-contract.

## 7. When to use a different tool instead

- **You have nucleotide reads** → **nucleotide AlignAIR** or IgBLAST (fuller AIRR output, IMGT numbering,
  real junction, productivity).
- **You need IMGT numbering / CDR delimitation / a true AIRR junction** → ANARCI, IMGT/DomainGapAlign,
  IMGT/V-QUEST.
- **You need general AIRR-seq repertoire annotation** → IgBLAST / nucleotide AlignAIR.
- **Light chain or TCR** → not this tool (no light-chain / TCR support).
- **Non-human or camelid** → a species-appropriate tool; this model is human-IGH only.

## 8. Non-goals / explicit non-claims

AlignAIR-AA is **not**:
- a replacement for nucleotide AlignAIR;
- AIRR-Rearrangement-compliant output;
- an IMGT-numbering or CDR-delimitation tool;
- a general AIRR-seq / repertoire annotation tool;
- a CDR3 or AIRR-junction caller (`inter_vj_aa` is method-defined; see Gate 2);
- an OOD-safe classifier (v1 has no applicability guard — Gate 3);
- validated on real proteins beyond descriptive V-**family** transfer (n=16, inconclusive; see Gate 2 /
  the field-evidence table).

## 9. Release gate

This contract is **release-gating**. The AA bundle must not be pushed to Hugging Face or a public registry
until **Gate 1** (this contract, approved), **Gate 2** (semantics audit, approved), **Gate 3**
(adversarial/OOD applicability validated or the product refuses out-of-domain claims), and **Gate 4**
(comparative scientific utility vs established tools) have all passed. Gates 5–6 (community usability,
evidence-labelled private beta) precede any broad release.
