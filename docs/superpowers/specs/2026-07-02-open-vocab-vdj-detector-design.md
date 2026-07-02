# Open-Vocabulary VDJ Detector — Design Spec

**Date:** 2026-07-02
**Status:** design, pending review → implementation plan.

## Goal

A deep model that, given a query read **and** a reference (V/D/J allele names + DNA
sequences supplied at inference), **detects which reference alleles are present in the read
and localizes them**, predicting the minimal set of quantities from which a deterministic
post-processor reconstructs the full GenAIRR annotation JSON. The reference is a true input:
add/remove/rename alleles or introduce novel sequences (a few SNPs off a known allele) at
inference, and the model re-targets its alignment to *that* reference with the same
correctness — no retraining.

## Why this design (what the experiments proved)

Three trained runs settled the architecture question empirically:

1. **Stuffed-prompt LLMs memorize, they don't align.** A decoder-only model with the reference
   in its prompt reached 0.94 V-call on canonical names but **0.0–0.12 when the alleles were
   renamed** — it learned `read-fingerprint → memorized name`, ignoring the reference. 100%
   name-randomization drove call accuracy to ~0: the stuffed-prompt architecture cannot learn
   in-context sequence comparison.
2. **BPE hides the very similarity alignment needs.** At 5% SHM (94% DNA identity) the read and
   its own germline share ~56% of BPE tokens and **0% positional match** — a SNP reshuffles the
   downstream merges. Compression sabotages comparison.
3. **Pooled matching washes out siblings.** A single pooled embedding per allele averages away
   the 1–2 SNPs that separate sibling alleles — the measured accuracy ceiling.

The task is therefore **open-vocabulary detection**: localize objects (V/D/J segments) and
classify them against a class set given at inference (the reference). The industry-SOTA family
for this is **YOLO-World / GLIP** = a detector fused with a CLIP-style open-vocabulary head.
This spec adapts that family to DNA, with two properties as first-class design goals:

- **Dynamic reference by construction** — the answer is *computed against* the provided
  sequences, so novel/renamed alleles are just new entries in the compared set.
- **Never converges on a shortcut** — both the architecture (forced reference use, non-poolable
  discrimination) and the training (self-escalating hard negatives, regret-based curriculum,
  shortcut-removing augmentation) keep the infinite gym at the model's competence frontier.

## Architecture

```
read tokens ─►[shared nucleotide encoder]─► read token embs (L,d) ─┐
reference (V/D/J: name+seq) ─►[SAME encoder]─► per-candidate token embs (Lg,d) + pooled (d)
                                                     │                │
                                          [deep reference fusion] ◄───┘  (read ⇄ candidate pooled embs)
                                                     ▼
                                          reference-conditioned read tokens
                                                     ▼
                                   [typed VDJ query decoder]  ← 3 fixed queries {V,D,J}
                                                     ▼
                        per query q ∈ {V,D,J}:
                          • span head    → in-read start/end (+ objectness: present?)
                          • allele head  → token-level match of read-span tokens vs that gene's
                                           candidate germline tokens → allele logits over candidates
                          • trim head    → germline start/end offsets
                                                     ▼
                        [deterministic post-process] → full GenAIRR JSON
```

### 1. Shared nucleotide encoder (reuse existing `SharedNucleotideEncoder`)

One bidirectional nucleotide transformer (RoPE / SDPA / SwiGLU) encodes **both** the read and
every candidate germline into per-token embeddings in a single space; a token-type embedding
distinguishes read from germline. Shared weights ⇒ a mutated read segment lands near its
germline. This is our existing, working encoder — kept as-is. It provides, per gene, each
candidate's **per-token** embeddings (`Lg,d`) and a pooled normalized embedding (`d`).

Character/nucleotide-level tokens (not BPE): a SNP changes exactly one token, so read↔germline
correspondence survives (finding #2).

### 2. Deep reference fusion (adapt GLIP `BiAttentionBlock`)

`N_fuse` bidirectional cross-attention layers between the read tokens and the **pooled**
candidate embeddings (per gene, ≤ tens of candidates → cheap). The read representation becomes
conditioned on *which alleles are present*, and vice-versa. This is the architectural lever
that makes reference-use non-optional (fixes finding #1). Fusion uses pooled candidate vectors;
fine SNP detail is used later, in the token-level allele head (§4b).

### 3. Typed VDJ query decoder (adapt DETR decoder; no Hungarian)

Three learned query embeddings with fixed roles — `V`, `D`, `J`. A transformer decoder lets
each query cross-attend the fused read tokens → one query representation per gene. Roles are
fixed by the recombination structure, so **no Hungarian matching** is needed (simpler than
DETR). D is optional (light/kappa/lambda loci): its **objectness** head predicts presence.

### 4. Per-query heads

**(a) Span (adapt YOLOX decoupled head).** From each query rep: in-read `start`/`end`
(boundary regression) + `objectness` logit (presence). Localizes the gene segment in the read.

**(b) Allele — open-vocabulary, token-level (adapt open_clip `ClipLoss` + GLIP region-word).**
Gather the read tokens inside the query's predicted span; score them against each candidate
germline's tokens by **late interaction (MaxSim)**: for each read-span token, take its max
cosine over the candidate's tokens, sum/mean over the span → one score per candidate. The
discriminating SNP survives because it's a per-token match, not a pooled average (fixes finding
#3). A learnable temperature (`logit_scale`) scales the scores. Candidate set = that gene's
alleles in the reference ⇒ the call is a valid reference allele **by construction**, and a novel
allele is simply another candidate. Per-position SHM reliability (from the state head, §6) can
down-weight likely-mutated read positions in the MaxSim.

**(c) Germline trims (regression).** From the query rep + matched germline: germline
`start`/`end` offsets, replacing the differentiable DP with direct regression (retires the
soft-DP; keep the banded DP only as an optional refinement, off by default).

### 5. Deterministic post-processing (no learning)

From the minimal predicted set `{per-gene: allele, read span, germline trims, objectness}` plus
the reference, derive **everything else in closed form**: the germline alignment string, junction
(via conserved anchors), productivity (in-frame + no stop), np1/np2 (segment gaps), CIGAR,
identity, and the full AIRR field set. This is ~a module of rules, no model involved — reusing
the existing AIRR-assembly + `decode_germline_coords` utilities.

### 6. Auxiliary heads (kept from DNAlignAIR)

- **Orientation** (4-class) → canonicalize the read to forward before the encoder (involutions).
- **Per-position state** (germline/substitution/insertion/deletion) → feeds SHM reliability into
  the MaxSim (§4b) and is a useful auxiliary signal.
- Scalar count heads (noise/mutation/indel) — optional auxiliaries.

## The minimal predicted set

The model predicts **only** what is not derivable: per gene → `allele` (open-vocab match),
in-read `start`/`end`, germline `start`/`end` (trims), `objectness`; plus `orientation`. Every
other GenAIRR field is computed in §5. Smaller output = smaller error surface + guaranteed
internal consistency.

## Losses

Kendall-weighted composite (reuse `UncertaintyWeight` balancing; protect the allele + coord
heads from being abandoned):

- **Allele match:** symmetric InfoNCE (open_clip `ClipLoss`) over the candidate scores from §4b,
  temperature-scaled, **with hard-negative siblings** injected per example. Per gene.
- **Span:** smooth-L1 (or 1D-IoU) on in-read start/end per present query.
- **Objectness:** BCE on presence (drives D-absent on light loci and mislocalized queries).
- **Germline trims:** smooth-L1 on germline start/end (or the soft-argmax coord loss for a
  banded-DP refinement arm).
- **Orientation / state:** cross-entropy (auxiliary).

## Training regime — the adversarial infinite gym

The online GenAIRR gym, made **self-escalating** so it never runs dry:

- **Hard-negative sibling mining.** For each true allele, add its nearest siblings (from the
  existing `build_sibling_index`) as explicit contrastive negatives. As the encoder sharpens,
  the hardest siblings get harder → the task tracks the model.
- **Regret / competence curriculum.** Reuse the lattice/control infra to preferentially sample
  the task-space cells where the model is *currently weakest* (heavy-SHM, short fragments,
  inverted-D, sibling-dense genes) — always training at the frontier, never replaying solved
  cases.
- **Shortcut-removing augmentation.** (i) **Name randomization** — relabel candidates with random
  names in both the reference and the target, so names carry no signal (only sequence match
  works). (ii) **Germline SNP perturbation** — perturb a candidate germline (and the derived read)
  so the model routinely matches sequences it has never seen — trains the novel-allele contract
  directly.
- **Teacher forcing → scheduled sampling.** Early: supervise the allele/trim heads on the true
  spans. Ramp: supervise on the model's own predicted spans, closing the teacher-forced→deployed
  gap. `coord_tau` annealed as today.

## Anti-shortcut design (explicit)

| Failure mode we saw | Mechanism that prevents it |
|---------------------|----------------------------|
| Memorize `read→name`, ignore reference | **Fusion** (§2) — the answer is computed against the reference |
| Average away sibling SNPs | **Token-level MaxSim** (§4b) — discrimination can't be pooled out |
| Solve easy cases, stall | **Hard-negative siblings + regret curriculum** — difficulty co-evolves |
| Rely on canonical names/sequences | **Name + SNP augmentation** — only the real task remains |

## Adapted open-source components (permissive; attribution preserved)

Cloned to `.private/reference/` (git-ignored) for study; we adapt the ideas from the specific
files below into a new `src/alignair/nn/sota/` package as clean 1‑D PyTorch (no framework code
vendored), naming the source in each module docstring and in `sota/ATTRIBUTION.md`. As built
(`clip_contrastive` + `region_word` consolidated into `matching.py`; a `loss.py` and `detector.py`
assembly added):

| Target module (built) | Adapted from | License |
|-----------------------|--------------|---------|
| `sota/matching.py` (MaxSim late-interaction + symmetric InfoNCE, `logit_scale`) | ColBERT; `GLIP` region-word (`modeling/rpn/loss.py`, `vldyhead.py`); `open_clip` (`loss.py`, `model.py`) | MIT |
| `sota/fusion.py` (read⇄reference cross-attention, `BiAttentionBlock`; uni/bi-directional) | `GLIP/maskrcnn_benchmark/utils/fuse_helper.py` | MIT |
| `sota/query_decoder.py` (typed V/D/J object-query decoder) | `detr/models/transformer.py` | Apache-2.0 |
| `sota/span_head.py` (decoupled reg/obj head) | `YOLOX/yolox/models/yolo_head.py` | Apache-2.0 |
| `sota/loss.py` (L1 + generalized-IoU span, objectness BCE, contrastive allele) | `detr` `SetCriterion`; `YOLOX` head loss | Apache-2.0 |
| `sota/detector.py` (assembly: encoder → fusion → typed queries → decoupled heads) | our composition (YOLO-World / GLIP shape) | — |

We do **not** use Ultralytics YOLO (AGPL-3.0) or Nucleotide-Transformer weights (CC-BY-NC).

## Inference

Encode read + reference (reference encoding cached per genotype). Fuse. Query decoder → per
gene: span, allele (top-1 or a calibrated set via MaxSim), trims, objectness. Deterministic
post-process → AIRR JSON. Novel/renamed alleles need no special handling — they are candidates.

## Testing & evaluation

- **Unit:** encoder/germline shapes, fusion block, query decoder, span head, MaxSim + `ClipLoss`,
  post-processor field derivation.
- **Overfit sanity:** a tiny model memorizes a handful of examples end-to-end.
- **The dynamic-genotype contract (the gate the LLM failed):** eval on (i) canonical, (ii) renamed
  candidates, (iii) **novel SNP-perturbed** alleles injected into the reference — call accuracy must
  hold across all three. This is the regression that distinguishes "aligns to the reference" from
  "memorized."
- **Benchmark vs IgBLAST** via the existing benchmark module (per-gene accuracy, coord exactness,
  throughput), on the same strata.

## Relationship to the clean DNAlignAIR base

- **Reuse:** `SharedNucleotideEncoder`, orientation + state heads, the gym + lattice/control
  curriculum, `UncertaintyWeight` loss balancing, `build_sibling_index` + reader infra, the
  AIRR-assembly + `decode_germline_coords` post-processing.
- **Replace:** pooled retrieval → token-level MaxSim + fusion; the dense region tagger → typed
  VDJ query decoder + YOLOX span head; the soft/banded DP coordinate path → trim regression
  (banded DP kept only as an optional refinement arm).
- **New:** `src/alignair/nn/sota/` (adapted components), the fusion block, the query decoder, the
  name/SNP augmentation, the contract eval.

## Scope

- **v1:** the architecture above + adversarial gym, human IGH, trained from scratch in the gym.
- **Out of scope / follow-ons:** a permissive pretrained DNA encoder as backbone; multi-locus
  (IGK/IGL/TCR); RLVR fine-tuning on the verifiable reward; the optional banded-DP refinement arm.
