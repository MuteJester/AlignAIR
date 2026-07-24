# AlignAIR-AA — Gate 1: Supported-Input Product Contract

> **AIRR Utility & Trust Gate, Gate 1 of 6.** This contract defines who the AA product is for, the
> inputs it supports, the inputs it must refuse or hedge, the minimum evidence each result needs to be
> trustworthy, and when a user should reach for a different tool instead. **No HF/public-registry
> publication until Gates 1–4 pass** (Gate 1 contract, Gate 2 semantics audit, Gate 3 adversarial/OOD
> applicability, Gate 4 comparative scientific utility).

**Status:** ✅ **APPROVED (2026-07-24)** · **Model in scope:** `alignair-aa-igh-human 2.0.0`
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
sequence presented in N→C order**, the most likely germline **V** (as an OGRDB-**ASC**
protein-equivalence class, expanded to its IMGT alleles via the frozen ASC→IMGT bridge), the **J** (a
protein-equivalence class over IMGT-named J alleles — **no** ASC bridge), an **experimental D**,
approximate segment coordinates, and a **calibrated alternative set** whose coverage guarantee holds
*only under the simulator calibration distribution*.

**Why it might be useful — stated honestly.** Protein germline annotation is **not** unavailable
elsewhere: NCBI IgBLAST accepts protein input via `blastp`
(https://www.ncbi.nlm.nih.gov/igblast/igblast.cgi), with protein-specific behavior and output.
AlignAIR-AA's *prospective* differentiation is therefore **not** "other tools cannot read protein"; it is
**uncertainty-aware protein-class prediction sets, learned robustness to SHM/indel/truncation, and safe,
provenance-pinned (pickle-free) artifact loading**. Those are claims that **still require Gate 4
(comparative utility) to substantiate** — they are the reason to *test* the tool, not yet a proven
advantage.

## 2. Intended users

- Antibody engineers / therapeutic-antibody scientists holding **protein VH sequences** (patents,
  literature, in-house constructs) with no reliable matching nucleotide.
- Repertoire/AIRR researchers working with **protein-only or historical VH datasets** (patents,
  literature, synthesized constructs) — **provided a complete, contiguous VH protein has been assembled**
  — who need a germline **reference-assignment** starting point.
- Curators who want a **fast, uncertainty-aware first-pass** germline reference-assignment hypothesis for
  human VH proteins, to be confirmed by orthogonal methods.

**Mass-spectrometry evidence is out of scope for now.** Mass-spec antibody data is typically
*discontinuous peptide* evidence, not a contiguous VH sequence. Admit it **only** once a complete,
contiguous VH protein has been defensibly assembled by other means, and validate that assembly workflow
separately before trusting any output.

**Not the intended user:** anyone who has nucleotide reads (use nucleotide AlignAIR / IgBLAST); anyone
needing IMGT numbering, CDR delimitation, or a true AIRR junction; anyone annotating light chains or TCRs.

## 3. Candidate workflows and current evidence

Only **one** workflow has any real-domain evidence, and even that is provisional. The other two are
**candidate** workflows supported by simulator evidence only. Do not present them as established.

**Provisional (has limited real-domain evidence):**

1. **Frozen-reference V-family assignment hypothesis for engineered/therapeutic VH proteins.**
   Input human VH protein sequences; for each get the top V protein-class → IMGT alleles and the calibrated
   alternative set. **Only the V family carries any real-domain evidence; the class, gene, and allele
   remain simulator-only** and must not be presented as validated. The output is a **reference/similarity
   assignment against the frozen germline set** — potentially informative when discussing a construct's
   *declared* design ancestry, but it **cannot independently prove** the historical template or lineage.
   *Evidence (family only): descriptive on 16 therapeutic antibodies, **three families, IGHV3-dominated,
   inconclusive** (n=16); not certified for general human IGHV-family assignment.*

**Candidate (simulator evidence only — not validated on real proteins):**

2. **Uncertainty-aware triage of a protein VH dataset.** Use `set_size` and the calibrated class/allele
   set to separate **narrower from broader simulator-calibrated hypothesis sets**, and route the broad
   ones to manual review. Note: a small or singleton set on real/OOD input is **not** established
   confidence — the coverage guarantee is simulator-domain only.
3. **Experimental — approximate V/J boundary + `inter_vj_aa` inspection.** Use the segment coordinates and
   `inter_vj_aa` (the method-defined V-end..J-start region — **not** a validated CDR3/junction) to locate
   regions for manual analysis elsewhere. **Simulator-only and biologically unvalidated.**

## 4. Supported-input matrix (in-contract)

| Dimension | Supported |
|---|---|
| Locus | Human **IGH** heavy chain only |
| Domain | **Variable domain (VH)** — framework + CDR structure of V, ideally through J |
| Molecule | **Amino-acid** sequence, single chain |
| Alphabet | 20 standard aa + `X` (uppercased); no `*`, no gaps. Leading/trailing whitespace is **stripped**; **interior** whitespace is **rejected** |
| Length | ≤ **192** aa (longer is rejected, never truncated) |
| Orientation | **Input must be N→C.** The AA model has **no orientation head** (proteins have no reverse complement). Reversed residue order is **unsupported** and is a Gate-3 negative control |
| Coverage | **Complete VH domain only** is provisionally in-contract. Truncations/fragments are **not yet operationally supported** — the coverage rules are un-operationalized biological prerequisites (see §6) |

An in-contract input for v1 is a **complete human VH domain protein in N→C order**. Fragment/truncation
support is an open Gate-3 question, not a current guarantee (§6).

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

## 6. Minimum evidence required per result (biological prerequisites — NOT yet operational)

| Result | Biological prerequisite for the call to mean anything |
|---|---|
| **V** call | Input contains a substantial contiguous portion of the V domain (framework + CDR structure). |
| **J** call | The J region is actually present in the input. |
| **D** call | **Experimental in all cases** — no real-domain evidence; AA D signal is intrinsically weak. Do not rely on it. |
| **coordinates** | Only meaningful when the corresponding segment is present; they are model estimates, not validated boundaries. |
| **`inter_vj_aa`** | Only a substring between V-end and J-start; meaningful only if both boundaries are real. Not a CDR3/junction. |
| **calibrated set** | Coverage guarantee holds only under simulator exchangeability and the applied group; on real/OOD input it is descriptive. |

**These are prose prerequisites, not operational rules.** Phrases like "substantial contiguous portion"
and "trained coverage window" cannot guide a user or a validator. **Gate 3 must convert them into
measurable, testable rules**, at minimum:

- minimum residue coverage (absolute and/or as a fraction of the VH domain);
- the required V/J evidence or anchor positions;
- which truncation positions are permitted;
- the minimum domain fraction;
- the behavior when only V, or only J, is present.

**Until Gate 3 operationalizes these, only a complete VH-domain input is considered provisionally
in-contract.** The v1 product cannot detect a prerequisite's absence — it will emit a confident-looking
call regardless. That detector is the entire point of Gate 3.

## 7. When to use a different tool instead

These are **distinct capabilities** — germline annotation, IMGT numbering, and repertoire analysis are not
the same task:

- **You have nucleotide reads** → **nucleotide AlignAIR** or IgBLAST (fuller AIRR-style annotation, real
  junction, productivity).
- **You want germline annotation of a protein and are comparing tools** → **IgBLAST also accepts protein
  via `blastp`** (different behavior/output from its nucleotide mode); it is a legitimate baseline, not an
  unavailable capability. This is the Gate-4 comparison, not a reason the AA product is unique.
- **You need IMGT numbering / CDR delimitation / a true AIRR junction** → ANARCI, IMGT/DomainGapAlign,
  IMGT/V-QUEST. AlignAIR-AA does **none** of these (it emits its own 0-based AA coordinates, not IMGT
  numbering).
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
