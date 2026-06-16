# DNAlignAIR R4 — Model Assembly + Gym + Training — Design

**Date:** 2026-06-16
**Status:** Approved (decisions + decomposition); build R4a → R4b → R4c.
**Depends on:** R1 (matching), R2 (backbone/orientation/region), R3 (germline coords + state) — all complete.
**Parent:** `docs/superpowers/specs/2026-06-16-dnalignair-architecture-design.md`

## 1. GenAIRR provenance investigation (what per-position GT is available)

From `Experiment.stream()` → `Outcome.final_simulation()` (a `Simulation`):
- `regions()` → `[Region(segment, start, end, frame_phase)]` in **observed-sequence coordinates**
  (e.g. `V [0..278)`, `NP1`, `D`, `NP2`, `J`). → per-position **region** GT.
- `germline_position(index)` → germline coordinate for an observed index (handles 5' trim: index 0 →
  germline 10 means 10 trimmed). → **germline start/end / trims** GT.
- `germline_bases` / `bases` → equal-length int (char-code) arrays; `bases[i] != germline_bases[i]`
  marks **changed** positions (mutations ∪ noise — NOT separable per position).
- Aggregate counts on the record: `n_mutations`, `n_quality_errors` + `n_pcr_errors` (noise),
  `n_indels`, `mutation_rate`, `productive`.
- NOT available: a per-position mutation-vs-noise label (`events()` names the pass only; `pre/post`
  are summary counts).

## 2. Decisions (confirmed)

| Topic | Decision |
| --- | --- |
| Per-position state head | **4 classes: {germline, substitution, insertion, deletion}** (substitution = any base change). Supersedes the R3 5-class scheme. |
| Noise vs mutation | Delivered by **scalar aggregate heads**: `noise_count` (← `n_quality_errors`+`n_pcr_errors`), `mutation_rate` (← `mutation_rate`/`n_mutations`), `indel_count` (← `n_indels`), `productivity` (← `productive`). |
| GenAIRR changes | **None** for now (a per-position change-type accessor is a possible future enhancement for full per-position noise/mutation split). |
| Decomposition | **R4a (model assembly) → R4b (gym + provenance GT) → R4c (composite loss + training/eval)**, each its own plan. |

## 3. R4a — unified `DNAlignAIR` model assembly

A single `DNAlignAIR(nn.Module)` wiring the R1–R3 components:
- `OrientationHead` → orientation logits (canonicalize at inference; trainer teacher-forces the true
  transform).
- `SequenceBackbone` → per-position reps `H (B,L,d)`.
- `RegionTagger(H)` → region logits; `PerPositionStateHead(H)` → 4-class state logits.
- Scalar heads from a masked-pooled `H`: `noise_count`, `mutation_rate`, `indel_count`, `productivity`.
- **Matching:** per gene, pool `H` over that gene's region positions → segment rep, project to the
  germline-embedding space → score against the reference's pooled germline embeddings (`AlleleMatchingHead`).
- **Germline coordinates:** a `germline_coords(seg_reps, seg_mask, germ_reps, germ_mask)` method runs the
  `GermlineAligner` against a chosen allele's per-position germline reps (trainer supplies the true
  allele; inference uses the top-1 match).
- Reference handling: `encode_reference(reference_set)` encodes all alleles' germline sequences once
  (pooled embeddings + per-position reps), cached; recomputed during training so the germline encoder
  learns. Genotype = candidate-row mask.
- `forward(tokens, mask, reference_embeddings)` returns a typed output with all dense + matching + scalar
  outputs. *Done when:* forward on a dummy batch with a small `ReferenceSet` returns all outputs with
  correct shapes; `germline_coords` returns exact-shaped logits.

## 4. R4b — GenAIRR gym + provenance ground truth

- `ReferenceSet` from 1..N dataconfigs (already built).
- `provenance.py` — extract from a streamed `Outcome`, per sample, the aligned arrays: per-position
  region labels, per-position state labels (germline/substitution/insertion/deletion from
  `bases`/`germline_bases`/`germline_position`), germline start/end per gene, the multi-allele call set
  per gene, the applied orientation, and the aggregate counts. *Done when:* a unit test extracts these
  GT arrays for a known seed and they are internally consistent (region spans match the call coords,
  state counts match aggregates within tolerance).
- `gym.py` — an `IterableDataset` that, per step, samples a dataconfig + a **curriculum** corruption
  profile (clean → progressively harder), streams Outcomes, and yields tokenized inputs + the GT bundle.
  *Done when:* a batch trains-steps the R4a model without error and shapes line up.

## 5. R4c — composite loss + training/eval

- Composite, **Kendall uncertainty-weighted** loss: orientation CE + region per-position CE +
  state per-position CE + multi-label allele BCE (full candidate set) + germline-coord soft-target CE +
  scalar regressions (noise/mutation/indel) + productivity BCE.
- Extend the existing `training/Trainer` (or a thin subclass) for the gym + this loss.
- Eval harness: boundary deviation (in-seq + germline), allele-call agreement, per-position accuracy,
  count MAEs. *Done when:* the full model trains on the gym and the metrics improve over steps; later
  extended to compare vs IgBLAST/MiXCR/partis.

## 6. Risks / iterate-on

- Germline-coord wiring (gather the matched allele's per-position reps, pad to a common length per batch)
  is the fiddliest assembly piece; teacher-force the true allele in training first.
- Curriculum schedule is empirical; start with a simple linear ramp.
- Per-position substitution can't distinguish noise vs mutation; the scalar heads cover the aggregate
  need. Revisit a GenAIRR accessor if per-position split becomes important.
- Compute: encoding the full reference each training step + a Transformer at full resolution — cache
  reference embeddings and consider mixed precision; monitor step time.
