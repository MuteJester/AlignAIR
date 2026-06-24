# BandedPointerAligner: replacing the soft-DP germline aligner

**Date:** 2026-06-24
**Status:** design approved, pending spec review → implementation plan
**Sub-project:** B (the soft-DP germline-aligner replacement, A/B'd against the hardened gym)

## 1. Problem

The differentiable soft-DP germline aligner (`src/alignair/nn/soft_dp_aligner.py`) is the
runtime bottleneck of DNAlignAIR:

- `soft_dp_end_logits` (`soft_dp_aligner.py:39-76`) is a **sequential** `for i in range(S)`
  recurrence over read positions (S up to ~576). Each step launches tiny
  logsumexp/logcumsumexp kernels → ~184k latency-bound kernel launches. The row recurrence
  cannot be parallelized across positions by construction.
- It runs at **two** sites and dominates both:
  - **inference: 66%** — germline-coordinate decode 47% (the DP runs *twice*, forward for
    `end` and reversed for `start`, `soft_dp_aligner.py:133,139`) + allele rerank 25%.
  - **training: 94% of every step** — 832 ms of an 883 ms step. The neural net is only 18%
    of inference / 6% of a step.

We are retraining DNAlignAIR from scratch (no budget cap) and have a frozen difficulty
lattice + bootstrap-CI competence metric (the "gym") to A/B any replacement. The goal is a
deep-learning germline aligner that is **both far faster and at least as accurate** as the
soft-DP, especially on the two regimes where we currently only tie IgBLAST: **full-length
heavy-SHM V** and **junction boundary jitter (±1-2nt)**.

This design is the product of two independent expert reviews (a computational-biology /
sequence-alignment-optimization expert and a deep-learning architecture / training expert)
that converged on the same conclusions, including two corrections to the initial sketch.

## 2. The reframe: the ±1-2nt jitter is a LOSS bug, not an aligner bug

The germline coordinate loss is plain **hard cross-entropy** over germline columns
(`dnalignair_loss.py:91-92`):

```python
per_row = (F.cross_entropy(sl, gs, reduction="none")
           + F.cross_entropy(el, ge, reduction="none"))
```

Hard CE gives an **identical** penalty to a ±1nt error and a ±50nt error — it only pushes
mass onto the single GT column and applies no pressure to sharpen or center the posterior.
Decoding with `argmax` (`germline_aligner.py:74-76`) over the resulting 1-2nt-wide plateau
then flips under tiny perturbations. **This produces the jitter regardless of which aligner
sits underneath**, and it is invisible to training (argmax is non-differentiable).

Consequence for the plan: the loss fix is **aligner-agnostic** and is the single
highest-ROI change. It is sequenced *first* (ablation #1), applied to the *existing*
soft-DP, to isolate how much of the jitter is loss vs aligner before any rewrite.

## 3. Contract (unchanged — any replacement must satisfy this)

- `germline_coords(seg_reps[B,S,d], seg_mask[B,S], germ_reps[B,Lg,d], germ_mask[B,Lg]) ->
  (start_logits[B,Lg], end_logits[B,Lg])` over the **chosen allele's** germline positions.
  `germ_reps` are encoded **at runtime** from the reference input
  (`dnalignair.py:132-146 encode_reference`) — this is what preserves the **dynamic
  genotype** hard requirement (reference is an input, novel alleles callable, NO allele
  memorization in weights).
- `alignment_score(...) -> [B]` — the allele reader/rerank score.
- `decode_germline_coords(start_logits, end_logits) -> (start, end)` — argmax-based, signature
  unchanged (internally the *loss* uses soft-argmax; see §5).

Call sites that must keep working: `core/dnalignair.py:188-190` (`germline_coords`),
`training/germline_tf.py:35` (`compute_germline_logits`, teacher-forced training/eval),
`inference/dnalignair_infer.py` (decode + rerank), `gym/instrument/evaluator.py`
(lattice eval), `training/reader.py` (reader training).

## 4. Architecture: `BandedPointerAligner`

A fully-parallel, reference-conditioned pointer head. Drop-in for `SoftDPAligner`, selected
by `config.aligner == "pointer"` (alongside the existing `"softdp"` / `"diagonal"`).

### 4.1 Score matrix (port the accuracy-load-bearing channel verbatim)

```
M = scale · ⟨proj_s(seg), proj_g(germ)⟩            # (B,S,Lg) cosine·scale
  + reliability_gated_base_match(seg_tok, germ_tok, seg_reliability)
M = M.masked_fill(~germ_mask, NEG)
```

- `scale = log_scale.clamp(-2.0, 3.0).exp()` — **keep the clamp** (`soft_dp_aligner.py:106`)
  or the downstream LSE/score collapses.
- The base-match channel is **ported verbatim** from `SoftDPAligner._scores`
  (`soft_dp_aligner.py:108-125`): the SNP-sensitive raw-ACGT `+1/-1` term with the
  **novel-allele floor** (`match_floor`, so a never-seen germline still aligns on real
  bases — the dynamic-genotype guarantee) and the **state-conditioned reliability**
  down-weighting (an SHM-substituted position contributes a near-neutral term instead of
  penalizing the true allele).
- M is materialized in full. Worst case B64·S576·Lg350 ≈ **26 MB fp16 / 52 MB fp32** —
  trivial. Compute the cosine/normalize in fp32 even under AMP for stability.

### 4.2 Start / end via single-launch diagonal extraction

The start scores over candidate germline offsets `o` are the **weighted leading diagonal**
of M; end scores are the **reverse diagonal** anchored at the read's last position. Both are
single CUDA launches via `as_strided` (no Python loop — the old `for o in range(Lg)` at
`germline_aligner.py:44` is exactly what we are eliminating):

```python
# leading diagonal  diag[b,i,o] = M[b,i,o+i]   (VERIFIED err 2.4e-7 vs loop)
Mp = F.pad(M, (0, S)); bs, ss, _ = Mp.stride()
diag = Mp.as_strided((B, S, Lg), (bs, ss + 1, 1))
start = (w * diag).sum(dim=1)                       # (B,Lg)

# reverse diagonal  end[b,o] = Σ_i M[b,S-1-i,o-i]   — flip-leading-flip
# (VERIFIED err 4.8e-7; a naive mirror-stride is SILENTLY WRONG, err 10.9)
Mf = torch.flip(M, (1, 2)); Mfp = F.pad(Mf, (0, S)); bs, ss, _ = Mfp.stride()
led = Mfp.as_strided((B, S, Lg), (bs, ss + 1, 1))
end = torch.flip((w * led).sum(dim=1), (1,))        # (B,Lg)
```

> **Correctness note (expert-verified):** the end/reverse diagonal MUST use the
> flip-leading-flip formula above. A plausible-looking "mirror stride" is wrong (err 10.9)
> and would train to the wrong coordinate with no test to catch it. A unit test asserting
> both diagonals match a reference Python-loop implementation to < 1e-5 is **mandatory**.

`start_logits = (temp·start).masked_fill(~germ_mask, NEG)`, likewise for end; `temp` is a
clamped learned temperature (clamp as `germline_aligner.py:55`).

### 4.3 Reliability-gated diagonal weights `w`

```python
w = softmax(diag_bias, dim=0)[None,:,None] * seg_reliability.detach()[:,:,None]   # (B,S,1)
```

- `diag_bias` is a learned per-position bias, **initialized to 0** so `w` starts uniform —
  the untrained head then equals the mean-cosine diagonal (current `GermlineAligner`
  behavior, `germline_aligner.py:56`), giving a localizing signal from step 0 (no dead
  warmup).
- Gating by `seg_reliability` (from `state_head`, **detached**) makes the weighting
  content-dependent: SHM-substituted positions on *this* read contribute ~0, so the
  correlation peak does not flatten under heavy SHM. A free global scalar cannot do this.
- Small entropy regularizer (or weight-decay of `diag_bias` toward 0) to prevent `w`
  collapsing onto a few positions.

### 4.4 Banded offsets (indel-tolerant coordinates)

A single diagonal mis-scores **every position after an indel** (germline pos = read pos +
constant assumption breaks; one 1nt deletion at read pos m corrupts the L−m downstream
positions). To keep *coordinates* (not CIGAR — parasail owns CIGAR, §6) robust on indel'd
reads, score a small offset band `Δ ∈ [−G, +G]` (G ≈ 5-8, covers ~all somatic indels):

```
start_logits[o] = temp · logΣ_Δ exp( diag_Δ(M)·w + γ(|Δ|) )
```

where `diag_Δ` is the Δ-shifted diagonal (same `as_strided`, different offset) and `γ` is a
learned distance penalty on `|Δ|` (a parallel gap prior). This is `(2G+1)` strided reductions
+ one LSE — still ~20-40× cheaper than the S-step DP, fully parallel. **Banded is an ablation
step (#6), gated on the simpler single-diagonal head working first.**

### 4.5 Reader / `alignment_score` (do NOT use length-normalized LSE for sibling resolution)

Both experts reject a length-normalized log-mean reader (`soft_dp_aligner.py:160-161`) as the
**final** sibling discriminator: two alleles differing at 2 of ~290 positions produce scores
differing by `~2·Λ/290` after normalization — drowned by a single SHM event elsewhere. This
is precisely why `v_reader="parasail"` already exists and wins (`dnalignair_infer.py:343-348`:
the learned reader "washes out the 1-3 diagnostic positions").

Policy:
- **Retrieval/seed score:** the windowed max-diagonal sum (length-normalized LSE is fine
  here) — used to seed top-k. Cheap, batched over `B·k` candidates in one pass (kills the
  25% rerank cost and the per-candidate Python loop).
- **Final V sibling discrimination:** **keep `v_reader="parasail"` as the default** until a
  neural reader provably ties it on the heavy-SHM-V sibling stratum in the gym.
- **Optional neural pairwise reader (ablation #7):** a per-position log-odds reader using a
  **learned 4×4/5×5 substitution matrix** `Λ[read_base, germ_base]` (init `+1/-1`, diagonal
  floored = novel-allele guarantee; optionally S5F-5mer-conditioned). Rank top-k by the
  **pairwise difference** `score_c − score_{c'}`, which by construction sums only over
  positions where the two germlines differ — capturing exactly the diagnostic SNPs the
  log-mean drowns. Promote over parasail only on a gym tie/win.

## 5. Loss redesign (replaces hard CE at `dnalignair_loss.py:91-92`)

Per gene, per endpoint. Let `p = softmax(logits)` over germline columns, `pos = arange(Lg)`,
GT coord `y`:

1. **Soft-argmax expected-coordinate, smooth-L1** (the anti-jitter term):
   `ĉ = Σ_o p_o·o`; `L_exp = SmoothL1(ĉ, y)`. Differentiable, continuous in the logits —
   pressures the whole distribution to center on `y`.
2. **Ordinal CDF soft-step** (sharpen, unimodalize): target `T_o = σ((o − y)/τ)`, predicted
   `Ĉ = cumsum(p)`; `L_cdf = Σ_o (Ĉ_o − T_o)²`. Penalizes far mass more than adjacent mass
   (CE does not). `τ` annealed wide→sharp (reuse `cosine_sigma_schedule(3.0→0.75)`).
3. **Minority CE** (0.3×) for exact peak localization on `y`.
4. **Start/end consistency** (single-source the coordinate): `L_cons = SmoothL1(ĉ_end −
   ĉ_start + 1, len_germ)`, applied **only on indel-free rows** (gate by `*_supervise` ∧
   `indel_count≈0`). On indel rows, drop it (parasail owns those). Directly fixes the
   documented corr<1 between the two heads (boundary-jitter memo: .44 V / .69 J).
5. **Decode for reporting/eval** uses soft-argmax (`Σ p·pos`) too, so the gym competence
   metric sees the same sub-integer-stable coordinate the loss optimizes.

### 5.1 Kendall scale hazard (migration risk — must handle)

The new `L_exp`/`L_cons` terms are in **nt units (~0-50)** vs CE in **nats (~0-6)**.
Normalize the L1/consistency terms by `Lg` (or a fixed scale) before they enter the
Kendall-weighted `*_germline` head, or re-tune `protected_max_log_var` — otherwise one term
starves the other. Keep the protected `max_log_var=1.5` on `*_germline` heads
(`dnalignair_loss.py:30`) so the balancer cannot abandon them.

## 6. Indels and CIGAR (no explicit path modeling needed)

Exact indel CIGAR is **already** produced by `parasail` realign (`io/alignment.py`),
*overriding* the neural germline coords on the delivered path when `full_alignment=True`
(`alignment.py:103-104`, `dnalignair_infer.py:431`). The soft-DP's indel-path modeling never
reaches the emitted output. So the fast head only needs **robust start/end coordinates +
reader score**; the banded offsets (§4.4) give net-length tolerance for coordinates, and
parasail keeps exact CIGAR per-read. No affine-gap DP is required anywhere.

## 7. Training supervision

We have exact GT germline coords from the GenAIRR simulator (`*_germline_start/end` in the
batch). Teacher-forced training path is unchanged (`germline_tf.py`):

- Coord heads: the §5 loss against GT coords (soft-Gaussian/CDF + soft-argmax L1 +
  consistency), σ/τ annealed.
- Reader: keep the existing novel-allele SNP-perturbed positive
  (`reader.py` `perturb_germline_tokens`); add **hard-negative siblings** (the true allele's
  nearest 1-2-SNP neighbor as negative) with a margin/contrastive loss on the differing
  positions, so the reader trains on the discrimination that actually matters.
- Dynamic-genotype guarantee verified on the held-out novel-allele lattice stratum (the
  floored substitution channel must keep novel alleles callable).

## 8. Expected results (verified microbenchmarks, RTX 3090 Ti)

| Quantity | soft-DP | pointer head |
|---|---|---|
| start+end forward (B64,S576,Lg350) | **555 ms** | **1.4 ms (`as_strided`)** (~400×) |
| kernel launches | ~184k | ~6 |
| M tensor | — | 26 MB fp16 / 52 MB fp32 |

End-to-end projection: **inference ~2.9× faster** (the net's 18% becomes dominant),
**training step ~10-15× faster** (the 832/883 ms vanishes — this is what unblocks the gym
A/B loop, currently ~2-3h per 3-arm run).

## 9. Ablation ladder (each A/B'd on the frozen gym lattice + bootstrap-CI competence)

1. **soft-argmax + CDF loss on the *existing soft-DP*** — isolate loss vs aligner for the
   jitter (predicted: most of it). Aligner-agnostic; do first.
2. `as_strided` single-diagonal pointer head, **CE-only loss** — isolate the latency win
   (~400×), expect jitter unchanged.
3. **+ soft-argmax L1** — the headline jitter drop.
4. **+ start/end consistency** — heavy-SHM-V coord corr→1, end stops drifting independently.
5. **+ reliability-gated `w`** — heavy-SHM coordinate accuracy.
6. **+ banded (2G+1) diagonals** — indel-read coordinates (watch the indel strata).
7. **+ learned substitution matrix / pairwise reader** — sibling resolution; promote over
   parasail only on a heavy-SHM-V tie/win.

Promotion rule (gym): a change ships only if it climbs ≥ the prior arm on
`heavy_shm_fulllen` + `junction_boundary` (bootstrap-CI lower bound) **without** regressing
the clean / indel / fragment strata.

## 10. Out of scope / non-goals

- No change to the dynamic-genotype contract, the reference encoder, segmentation-first
  design, or the AIRR output schema.
- No replacement of parasail for exact indel CIGAR (it stays; it is cheap per-read).
- GradNorm/PCGrad trunk balancing and the remaining training-side gates are gym Phase 3b/5b,
  tracked separately; this sub-project may surface evidence for them but does not implement
  them.

## 11. Risks

- **Reverse-diagonal correctness** — mandatory unit test vs reference loop (§4.2).
- **Kendall loss-scale mismatch** — normalize nt-unit terms (§5.1).
- **`w` softmax saturation** — uniform init + entropy reg (§4.3).
- **Reader regression on siblings** — gated by keeping parasail until a neural reader ties it
  (§4.5).
- **Banded indel coords** — gated as a late ablation; if it regresses clean strata, drop the
  band and lean on parasail.

## 12. File map (for the implementation plan)

- Create: `src/alignair/nn/pointer_aligner.py` (`BandedPointerAligner`, diagonal extraction
  helpers, reliability-gated weights, banded LSE, reader).
- Create: `tests/alignair/nn/test_pointer_aligner.py` (diagonal-vs-loop correctness incl. the
  reverse-diagonal, masking, shape/contract, novel-allele floor, latency smoke).
- Modify: `src/alignair/losses/dnalignair_loss.py` (soft-argmax + CDF + consistency germline
  loss; scale normalization).
- Modify: `src/alignair/core/dnalignair.py:92` (aligner selection: add `"pointer"`).
- Modify: `src/alignair/config/dnalignair_config.py` (config flag + loss-term weights / τ
  schedule).
- Modify: `src/alignair/nn/germline_aligner.py` (`decode_germline_coords` soft-argmax option)
  and/or the eval/decoders in `gym/instrument/evaluator.py`, `inference/dnalignair_infer.py`.
- Reuse verbatim: `SoftDPAligner._scores` base-match channel (extract to a shared helper so
  both aligners use it).
- A/B harness: extend `scripts/exp_ramp_vs_factored.py` pattern (or a new
  `scripts/exp_aligner_ablation.py`) to run the §9 ladder on the frozen lattice.
