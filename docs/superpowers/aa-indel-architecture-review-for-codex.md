# Architecture Review Request — Amino-Acid AlignAIR, indel/frameshift robustness

## Your role

You are a senior machine-learning architect with deep expertise in sequence models, differentiable
alignment, signal processing, and computational immunology (immunoglobulin/TCR V(D)J recombination).
Your job is to **critically evaluate** an amino-acid variant of the AlignAIR aligner and **propose
concrete, implementable architectural improvements** for one specific failure mode: **nucleotide
indels that frameshift the amino-acid input**. Be rigorous and skeptical. Call out where we are
solving the wrong problem, where an idea is information-theoretically impossible, and where a simpler
approach beats a fancy one. Then be genuinely creative.

Before answering, **read the code and the frozen result artifacts listed at the end** — do not take
our summary on faith; verify it.

## What AlignAIR is (nucleotide baseline)

AlignAIR aligns antibody/TCR reads to germline V/D/J alleles. The PyTorch model (`src/alignair/core/`):
- token+position embedding → an **in-model orientation head** that detects which of 4 orientations
  (forward / reverse-complement / complement / reverse) the read is in and applies the inverse
  transform, then re-embeds the canonicalized read (`src/alignair/nn/heads/orientation.py`,
  `core/model.py`). This is a *reversible* correction of a *known, invertible* corruption.
- shared conv feature towers (`core/layers.py`), then per-gene `GeneBranch` (`core/gene_branch.py`):
  a segmentation tower → start/end position logits → soft-argmax boundary → soft-cutout of the span →
  a classification tower → a **sigmoid multi-label allele head over the full allele set**
  (198 V / 33 D / 7 J for human IGH). Multi-label because a read can be consistent with several alleles.
- meta heads off a shared tower: mutation-rate, indel-count, productivity (`core/gene_branch.py`,
  `core/losses.py`).
- The nucleotide model handles indels because it aligns nucleotides and has the read + a per-position
  edit-state notion (CIGAR I/D). The shipped human IGH model trained ~490k steps.

## The amino-acid experiment (what we built)

Goal: an aligner whose **input is a protein sequence** (mass-spec / Edman / antibody-engineering
sequences have no underlying nucleotide read). Design decision (deliberately minimal): **reuse the
exact nucleotide model, loss, and GenAIRR training gym unchanged; only change the input to amino
acids.** See `src/alignair/aa/`:
- input tokens = the AA-tokenized `sequence_aa` (vocab 23) instead of nucleotides (`aa/tokenizer.py`,
  `aa/gym_bridge.py`).
- segment coordinates projected into AA-residue space — forced, because the masked soft-argmax can
  only point inside the AA-length input (`aa/translation.py::project_codons`).
- **allele targets stay the full *nucleotide* allele multi-hot** (not collapsed protein classes), so
  that synonymous-allele ambiguity must *emerge* in the sigmoid heads rather than being hand-encoded.
- orientation head **off** (`use_orientation=False`, `core/config.py`) — reverse-complement is
  undefined for protein and its complement table is nucleotide-only.
- everything else identical: same `hierarchical_loss`, same gym curriculum/strata, same conv towers.

We also translate germline alleles to protein and built an **AA-nearest-germline oracle**
(`aa/forensics.py`) = the best any classifier could do from protein alone (the provable information
ceiling), anchored at the conserved Cys landmark. And an AA benchmark adapter over the real
stratified benchmark (`aa/benchmark.py`, `src/alignair_benchmark/`).

## What we proved empirically (verify against the artifacts)

Trained 20k steps (24× less than the 490k-step shipped DNA model — under-trained). Evaluated on a
large stratified sample. Key, *proven* findings:

1. **V allele identity is mostly a training gap, not an information limit.** Model V protein-class
   accuracy **0.73** vs the AA oracle ceiling **0.90**. Decomposing V errors with the oracle:
   **83% are model-deficiency (the oracle recovers the true allele from the same AA, the model does
   not → fixable), 17% are information-ceiling (unrecoverable from AA).** When wrong, the true allele
   is **median rank 2**. (`forensics.json`)
2. **Indels are the amino-acid-specific failure.** A single nucleotide indel frameshifts the protein
   downstream; multiple indels compound (the first ruins the frame, later indels sit in already-
   corrupted territory). Of indel reads: **12% have the indel downstream of V (in D/J) → V frame
   intact, V undamaged**; **88% have the indel inside V → V frameshifted.** Measured from the V CIGAR
   (`mask_recovery.json`): pure **masking** (keep the first in-frame run) recovers enough V in ~**52%**
   of indel reads; full **frame correction** (re-frame at each indel) would recover ~**94%**; ~**5%**
   truly lost. So masking is a partial fix; correction is the real one — but correction needs
   nucleotide-level indel localization the pure-AA model does not have.
3. **Ambiguity did not emerge cleanly.** With single-allele targets, the sigmoid heads produce a
   diffuse low-confidence smear across many alleles rather than a crisp spike on the synonymous
   equivalence class (7% of probability mass lands on the true class; top-|class| recall 0.63). J is
   solved (0.94, and J has no synonymous ambiguity); D is moderate (~0.65). (`metrics.json`)

## The core problem for you to attack

Make the amino-acid model **robust to nucleotide indels / frameshifts**: recover V/D/J allele calls
(and ideally coordinates) with **calibrated uncertainty**, given only the (possibly frameshifted)
protein — or via a principled mechanism that does not require handing the model the nucleotides.

The hard constraint we already understand: **frameshift is information-theoretically irreversible at
the amino-acid level.** Translation is many-to-one (61 codons → 20 AAs), so from the frameshifted
protein alone you cannot recover the nucleotides and re-frame. A layer that operates on the shifted
protein cannot *invert* the shift the way the orientation head inverts reverse-complement. Any "fix"
must therefore be **reconstruction/marginalization from a prior**, not inversion.

## A seed idea from us — evaluate it critically, then improve or replace it

We want you to react to (and strengthen or discard) the following, and propose better if you can:

> Instead of translating once and committing to a single reading frame, represent **inside the model
> a compact "superposition" over the latent nucleotide sequences / reading frames / indel-correction
> paths** that could have produced the observed protein. Because we never see the DNA, and an indel
> makes the downstream frame ambiguous, the model would **carry a distribution over frame-paths**
> rather than collapsing to one. Proposed mechanism: encode positional/phase information with a
> **complex-valued, Fourier/wave-like representation**, so many possible offsets/alignments are
> represented **compactly and cheaply (superposition)**, and the allele heads read out from it —
> yielding **calibrated uncertainty**: under frameshift, probability mass spreads across positions and
> alleles, and the **returned V/D/J allele set grows** (honest ambiguity) rather than the model
> confidently guessing wrong. In effect: **marginalize over the unobserved DNA** instead of
> translating once and hoping.

Questions we want your expert judgment on:
- Is "carry a distribution over latent frames/indel-paths and marginalize" the right abstraction? Or
  is there a cleaner, better-posed formulation of the same intent?
- Is a **complex/Fourier/phase representation** the right tool for compactly representing the
  many possible alignments/offsets — or is that aesthetically appealing but the wrong hammer? If it
  *is* right, sketch concretely how phase encodes frame/offset and how the allele head reads it out.
- How does the model get a **training signal** for the latent frame-path distribution, given we do
  have the ground-truth DNA/CIGAR at training time (GenAIRR)? (e.g. supervise a frame/indel-position
  head; CTC-like marginalization over frames; a latent-variable/variational objective; distillation
  from the nucleotide teacher.)
- Compare against simpler baselines you should explicitly consider and rank against the fancy idea:
  (a) feed all 3 reading frames as parallel channels and let attention pick/mix; (b) a nucleotide-
  level indel-detection + reframe head *before* translation (the true orientation-head analog, but it
  needs nucleotides); (c) detect-and-mask (keep the first in-frame run); (d) a differentiable
  soft-alignment of the observed protein to germline prototypes that tolerates indels (local/affine-
  gap alignment, or a learned monotonic alignment); (e) train a nucleotide teacher and distill an
  AA student.
- How should **calibrated ambiguity** be represented and trained so the allele *set* grows correctly
  under frameshift (the "return more alleles when less certain" behavior we want)? Note our current
  sigmoid-multilabel heads did NOT learn this from single-allele targets.

## Be honest about scope (do not just accept our premise)

The frameshift problem exists here largely because we translate indel-containing **DNA** reads with a
fixed frame. **Real protein input has no nucleotide frameshift** — a mass-spec/Edman antibody sequence
is whatever residues it is. So part of your job is to judge: **is indel/frameshift robustness even the
right thing to invest in for an amino-acid aligner**, versus (i) fixing the benchmark's AA data
generation and focusing on the provably-83%-fixable V training gap, or (ii) accepting that indel-DNA
reads belong to the nucleotide model? If you think the effort is misdirected, say so and justify it.
Real AA-level corruption (a mis-sequenced single residue) is a much milder, different problem than a
nucleotide frameshift — address that distinction.

## Deliverable

A written architectural review with:
1. **Critical assessment** of the current AA approach and of the frameshift problem framing (including
   the scope question above).
2. **3–5 concrete architectural proposals**, ranked by (impact × feasibility). For each: the
   mechanism, how it handles frameshift, the exact training signal and loss, integration with the
   existing conv-tower/`GeneBranch` architecture (or a justified redesign), compute/parameter cost,
   and its **honest information-theoretic limit** (what it cannot recover).
3. An explicit **verdict on the complex/Fourier superposition idea** — steelman it, then either give a
   concrete design or explain why a cleaner alternative dominates.
4. A **calibrated-ambiguity design**: how to make the returned V/D/J allele set grow correctly under
   uncertainty, and how to train it.
5. **One cheap de-risking experiment** for your top proposal — the smallest thing that would prove or
   kill it before a full build (in the spirit of the oracle/ceiling analyses already in this repo).

## Files and artifacts to read first

- Nucleotide core: `src/alignair/core/model.py`, `core/gene_branch.py`, `core/losses.py`,
  `core/config.py`, `core/layers.py`; orientation head `src/alignair/nn/heads/orientation.py`.
- AA subpackage: `src/alignair/aa/` — especially `gym_bridge.py`, `model.py`, `loss.py`,
  `translation.py` (`project_codons` = the frame/coordinate logic), `forensics.py` (the oracle),
  `benchmark.py`, `eval_gym.py`.
- Design + rationale + all prior verdicts: `docs/superpowers/specs/2026-07-18-aminoacid-alignair-design.md`
  (see especially §14 "Same-Scheme Experiment").
- Frozen results to verify our claims:
  `docs/superpowers/artifacts/2026-07-20-aa-same-scheme/{metrics,forensics,mask_recovery}.json`.
- Analysis scripts (reproduce anything): `scripts/aa/{train_eval_aa_igh,error_forensics,mask_recovery,
  benchmark_aa_igh}.py`.
- GenAIRR simulator (ground-truth DNA/CIGAR available at training time): `/home/thomas/Desktop/GenAIRR`.

Ground every claim in the code or the artifacts. Where you propose something, say what you would need
to verify it and how. Prioritize correctness and honesty over enthusiasm.
