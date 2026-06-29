# Sharper Matcher + Gym Co-Design — Design

**Status:** approved design (agent-reviewed); Layer 1 ready for implementation.
**Date:** 2026-06-29

## Problem (empirically pinned)

The 24h-trained `XAttnAligner` solves clean/moderate reads (V/D/J 0.74–0.99, J robust everywhere)
but fails on **heavy-SHM and near-identical allele discrimination**:
- 2M-sequence benchmark: heavy-SHM-V `top1_in_set` = 0.22–0.32, fragments 0.02–0.17; dominant error
  `same_gene_wrong_allele` (siblings 1–2 SNPs apart collapse to the common sibling).
- Novel alleles: k-mer seed admits them to the pool 100%, but the matcher calls a 3-SNP-novel over
  its parent only 47% of the time (→63% at 10 SNPs).
- Partial genotype (25% donor subset) nearly **doubles** heavy-SHM-V (0.255→0.422) — confusable
  siblings leave the pool.

**Root cause:** the matcher's **mean-MaxSim over learned reps** averages similarity over all ~280
positions, so the 1–2 *diagnostic* SNP columns drown (and learned reps blur the exact nucleotide).
Sibling/novel discrimination needs the **raw bases at the polymorphic columns, SHM-discounted**.

## Agent review of the first idea ("Approach A: pool-aware base-match matcher")

The base-match + SHM-discount intuition is correct, but the first formula was **circular**:
(a) its soft base-match rode the cross-attention alignment computed from the *same blurred reps*
(smears the diagnostic column; worst under heavy SHM); (b) disagreement was defined in read-space
(rides the alignment) when polymorphic columns are **deterministic in germline-space**; (c) a single
`λ` cannot stay commensurate with a **learned, growing** MaxSim scale, so siblings stay decided by
blurred reps. Fix: make the base-match **inform** the alignment, score on the **exact** polymorphic
columns with an **SHM error model**, and **gate** (not λ-blend) the discriminator.

## Design — two layers

### Layer 1 (cheap, NO retrain, de-risk first) — classical raw-base rescore of the neural top-k
For each read, take the trained model's neural top-k V candidate pool, run a **raw-base gapped
alignment** of the read V-segment against each candidate germline (reuse the `align/` package —
`ParasailAligner`/`WFAAligner`, already built for the speed stack), and **re-rank** by raw-base
score / mismatches concentrated at the polymorphic columns. This is the project's own validated lever
(`dnalignair-vs-igblast-reader-finding`: raw gapped-alignment rescore over neural top-k closed the V
gap +0.06, no retrain, no fragment regression). Measure heavy-SHM-V (and fragment, novel) lift on the
current trained checkpoint. **This may resolve most of the sibling problem with no retrain** and sets
the ceiling/baseline for deciding whether Layer 2 is needed.

### Layer 2 (principled neural matcher — only if Layer 1 is insufficient)
Replace mean-MaxSim's sibling decision with a deterministic-polymorphic-column comparator:
1. precompute the **polymorphic-column set P** per candidate pool at `encode_reference` (deterministic
   from the germline DB);
2. align read→germline with a **base-aware DP** (base-match added into the score matrix *before* the
   DP, monotone & gap-aware — reuse the soft-DP/pointer structure), so base evidence informs the
   alignment (kills the circularity);
3. score only on P with a **calibrated SHM error model** (`disc_c = Σ_{p∈P} rel·logL(read|germ_c[p])`),
   not flat ±1;
4. **gate**: `disc` decides among MaxSim-*tied* candidates; MaxSim only sets the gene-level shortlist;
5. keep the **floored raw-base channel** so novel alleles score on real bases (dynamic-genotype
   guarantee; novel-vs-parent differ only at known columns).

## Gym co-design (ships regardless of Layer 1/2 — confirmed correct by review)
- **Embargo**: hold out an allele fraction during training; include at eval — trains+measures the
  dynamic-genotype/novel property (the 24h run had no embargo).
- **Genotype augmentation**: train with random partial genotypes (restrict the pool to a subset incl.
  the true allele) — partial genotype helps 2× at inference; train the model to exploit it.
- **Synthetic novel-vs-parent positives**: `perturb_germline_tokens`/`reader_novel_positive` — teach
  the matcher to separate a novel allele from its parent at the known columns.
- **Heavy-SHM + sibling oversampling** via the StratifiedCurriculum (hard-corner exposure).
- **Sibling-pool fix**: `build_candidates` injects 6 *random* siblings — ensure *the* nearest (1-SNP)
  sibling is reliably in the set-NCE pool, or the matcher never trains on the hard pair.

## Also (separate inference bug found during analysis)
`predict_reads_xattn` reads query coords from the **untrained boundary head** (2M `negative_span` 59%,
`exact_query_span` 0). Fix: derive query coords from the trained `region_logits` via
`decode_boundaries` (segmentation-first). Affects only coord metrics, not allele accuracy.

## Build order & gates
1. **Layer 1** — rescore experiment on the current checkpoint; gate = heavy-SHM-V lift vs the 0.32
   baseline, no fragment/clean regression, novel-vs-parent lift.
2. Decide Layer 2 + gym retrain based on Layer 1.
3. Eval on every step: per-cell (clean/heavy-SHM/indel/fragment), partial-genotype, novel-allele,
   and the 2M assay.
