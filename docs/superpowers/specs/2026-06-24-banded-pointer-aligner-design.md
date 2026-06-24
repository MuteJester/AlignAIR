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

## 3. Contract

The **external** contract is unchanged — the aligner still returns germline start/end
logits and a reader score, and the dynamic-genotype guarantee is preserved. But the
**internal signature DOES change**, and the spec is explicit about it (a review caught that
the prior "unchanged" claim was false):

- `germline_coords(seg_reps[B,S,d], seg_mask[B,S], germ_reps[B,Lg,d], germ_mask[B,Lg],
  seg_tok[B,S], germ_tok[B,Lg], seg_reliability[B,S]) -> (start_logits[B,Lg],
  end_logits[B,Lg])`. **The three trailing args are NEW** — the coordinate path currently
  receives only reps+masks (`soft_dp_aligner.py:128`, `core/dnalignair.py:188-190`), and the
  base-match channel + reliability gating that §4.1/§4.3 require for the COORD decode are
  today only wired into `alignment_score`, not `forward`. Threading these through is real new
  plumbing (see call-site edits below). `germ_reps`/`germ_tok` are encoded **at runtime** from
  the reference input (`dnalignair.py:132-146 encode_reference`, which already produces
  `pos_tok`) — preserving the **dynamic genotype** hard requirement (reference is an input,
  novel alleles callable, NO allele memorization in weights).
- `alignment_score(...) -> [B]` — the allele reader/seed score. **This soft-DP site is also
  replaced** (see §4.5 and B3 in §11): it is called every training step by the reader, so
  leaving it on the soft-DP would keep training soft-DP-bound and invalidate §8.
- **Soft-argmax decode is mandatory and consistent across train AND eval.** Add a
  `decode_germline_coords(..., soft=True)` path (`ĉ = Σ p·pos` over valid columns) and switch
  the gym evaluator, `evaluate()`, and inference decode to it — otherwise training optimizes a
  soft coordinate while the competence metric reads argmax over the same plateau, and the A/B
  under-measures the jitter fix (the exact thing we are testing).

**Call sites that MUST be updated to thread tokens + reliability** (the §12 file map covers
these): the model method `core/dnalignair.py:188-190`; the teacher-forced
`training/germline_tf.py:11-36` (it has `tokens` + model in scope but does not yet compute
`seg_tok`/`germ_tok`/`seg_reliability` — lift the gather logic that already exists in the
rerank path at `inference/dnalignair_infer.py:299-323`); `gym/instrument/evaluator.py:68`;
`inference/dnalignair_infer.py:284`. Reader-training call sites: `training/reader.py`,
`training/gym_trainer.py:184-225`.

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
- The base-match channel is **extracted from `SoftDPAligner._scores`
  (`soft_dp_aligner.py:103-126`) into a shared helper** so both aligners use the identical
  code: the SNP-sensitive raw-ACGT `+1/-1` term with the **novel-allele floor**
  (`match_floor`, so a never-seen germline still aligns on real bases — the dynamic-genotype
  guarantee) and the **state-conditioned reliability** down-weighting (an SHM-substituted
  position contributes a near-neutral term instead of penalizing the true allele). NOTE this
  is a NEW capability for the coordinate path (the soft-DP's `forward`/coord decode uses pure
  cosine; only its `alignment_score` uses base-match) — see B2 in §11 for the plumbing.
- M is materialized in full. The cosine M is ~26 MB fp16 / 52 MB fp32 at B64·S576·Lg350, but
  the **real transient peak** is higher: `F.pad(M,(0,S))` allocates `(B,S,Lg+S)` ≈ 136 MB
  fp32 per diagonal call, ×2 for start+end, ×(2G+1) for the band (§4.4). State this in the
  plan; it is still well within budget but is NOT "26 MB". Compute the cosine/normalize **and
  the LSE/softmax reductions** in fp32 even under AMP (`.float()` the LSE input; `NEG=-1e4` is
  fp16-representable but `logsumexp` of it should run in fp32).

### 4.2 Start / end via single-launch diagonal extraction

The start scores over candidate germline offsets `o` are the **weighted leading diagonal**
of M; end scores are the **reverse diagonal** anchored at the read's last position. Both are
single CUDA launches via `as_strided` (no Python loop — the old `for o in range(Lg)` at
`germline_aligner.py:44` is exactly what we are eliminating):

```python
# w is the per-read-position weight (B,S,1), MASKED by seg_mask (§4.3).
# leading diagonal  diag[b,i,o] = M[b,i,o+i]   (VERIFIED err 2.4e-7 vs loop)
Mp = F.pad(M, (0, S)); bs, ss, _ = Mp.stride()
diag = Mp.as_strided((B, S, Lg), (bs, ss + 1, 1))
start = (w * diag).sum(dim=1) / w.sum(dim=1).clamp(min=1e-6)     # (B,Lg), per-row length-norm

# reverse diagonal  end[b,o] = Σ_i w[b,S-1-i]·M[b,S-1-i,o-i]   — flip-leading-flip.
# CRITICAL: w must be flipped INTO THE SAME FRAME as the flipped rows, else w[i] pairs with
# row S-1-i. With non-uniform (reliability-gated) w this is O(1) WRONG (err ~1.9); only the
# w≡1 case is correct without the flip — which is why the earlier "err 4.8e-7" check missed it.
Mf = torch.flip(M, (1, 2)); Mfp = F.pad(Mf, (0, S)); bs, ss, _ = Mfp.stride()
led = Mfp.as_strided((B, S, Lg), (bs, ss + 1, 1))
wf = torch.flip(w, (1,))                                          # flip weights into led's frame
end = torch.flip((wf * led).sum(dim=1), (1,)) / w.sum(dim=1).clamp(min=1e-6)
```

> **Correctness note (the single highest-risk item, review-verified):** the reverse diagonal
> MUST flip `w` into the flipped-row frame (`wf = flip(w)`). The naive `flip((w*led).sum)`
> trains to the wrong coordinate whenever `w` is non-uniform — which it ALWAYS is once §4.3
> gates it by reliability. The **mandatory unit test** must (a) use a **non-uniform random
> `w`**, and (b) compare against a reference Python loop that weights row `S-1-i` by
> `w[:,S-1-i]`, to < 1e-5. A test with `w≡1` gives false confidence and must not be the only
> case. Per-row length normalization (`/ Σw`) mirrors the old aligner's `/seg_len`
> (`germline_aligner.py:54-56`) so short/long segments give comparable logit scales.

`start_logits = (temp·start).masked_fill(~germ_mask, NEG)`, likewise for end; `temp` is a
clamped learned temperature (clamp as `germline_aligner.py:55`).

### 4.3 Reliability-gated diagonal weights `w`

```python
# diag_bias is a fixed-size (max_len,) learned vector, SLICED to S each batch.
w = softmax(diag_bias[:S], dim=0)[None,:,None] * seg_reliability.detach()[:,:,None]  # (B,S,1)
w = w.masked_fill(~seg_mask[:,:,None], 0.0)            # MUST mask padded read positions
```

- `diag_bias` is a learned per-position bias of fixed length `max_len`, sliced to the batch's
  `S`, **initialized to 0** so `w` starts uniform — the untrained head then equals the
  mean-cosine diagonal (current `GermlineAligner` behavior, `germline_aligner.py:56`), giving
  a localizing signal from step 0 (no dead warmup).
- **`w` MUST be masked by `seg_mask`** before the diagonal sum: segments are right-padded
  (`extract_segment`, `dnalignair.py:25-43`) and `state_reliability` on pad positions is
  undefined, so without masking padded read positions leak into the coordinate sum.
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

> The banded **end** head reintroduces the B1 hazard per Δ: it MUST use the §4.2
> flip-leading-flip with `wf = flip(w)` applied for each Δ, with the same `/w.sum` per-row
> length-norm and `seg_mask` masking as the single diagonal. The mandatory non-uniform-`w`
> correctness test must cover at least one `Δ≠0` band case.

### 4.5 Reader / `alignment_score` (do NOT use length-normalized LSE for sibling resolution)

Both experts reject a length-normalized log-mean reader (`soft_dp_aligner.py:160-161`) as the
**final** sibling discriminator: two alleles differing at 2 of ~290 positions produce scores
differing by `~2·Λ/290` after normalization — drowned by a single SHM event elsewhere. This
is precisely why `v_reader="parasail"` already exists and wins (`dnalignair_infer.py:343-348`:
the learned reader "washes out the 1-3 diagnostic positions").

**`alignment_score`'s soft-DP is REQUIRED to be replaced (not optional) — it is the second
every-step training cost.** A review caught that `gym_trainer.py:184-225` calls
`alignment_score → soft_dp_end_logits` ~`(1 + n_sib + n_rand)`×3genes ≈ **39 soft-DP forwards
per training step** (plus the novel-positive). If we leave `alignment_score` on the soft-DP,
training stays soft-DP-bound and the §8 "10-15× faster step" claim is false. So the fast
diagonal-based seed score replaces `alignment_score` for **both training and the inference
top-k seed**. Parasail is layered on top at inference only.

Policy:
- **Fast seed score (replaces `alignment_score`, used in TRAINING + inference seed):** the
  windowed max-diagonal sum over M (length-normalized LSE), batched over `B·k` candidates in
  one pass — kills the 25% inference rerank cost, the per-candidate Python loop, AND the
  every-step training reader cost. This is what makes §8's training number real.
- **Final V sibling discrimination (inference only):** **keep `v_reader="parasail"` as the
  default** until a neural reader provably ties it on the heavy-SHM-V sibling stratum. Parasail
  is NOT on the training path, so keeping it does not slow training.
- **Optional neural pairwise reader (ablation #7, the only path to dropping parasail):** a
  per-position log-odds reader using a **learned 4×4/5×5 substitution matrix**
  `Λ[read_base, germ_base]` (init `+1/-1`, diagonal floored = novel-allele guarantee;
  optionally S5F-5mer-conditioned). Rank top-k by the **pairwise difference**
  `score_c − score_{c'}`, which by construction sums only over positions where the two
  germlines differ — capturing exactly the diagnostic SNPs the log-mean drowns. Promote over
  parasail only on a gym tie/win.

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
   ĉ_start + 1, span)` where **`span = germline_end − germline_start`** (the MATCHED germline
   span, both from GT) — **NOT `Lg`/full germline length**. `ĉ_end − ĉ_start + 1` equals the
   matched span (end-exclusive convention: `ge = germline_end − 1` at `dnalignair_loss.py:90`,
   so `ĉ_end − ĉ_start + 1 = germline_end − germline_start`); targeting full `Lg` is correct
   only for full-length untrimmed reads and wrong for every fragment/trimmed row the gym
   stresses. Applied **only on indel-free rows** (gate by `*_supervise` ∧ `indel_count≈0`).
   On indel rows, drop it (parasail owns those). Directly fixes the documented corr<1 between
   the two heads (boundary-jitter memo: .44 V / .69 J).

**Softmax/soft-argmax over the VALID germline columns only:** `p = softmax(logits)` must
exclude NEG-masked (invalid) columns; `cumsum`/CDF target and `Σ p·pos` run over valid
columns so masked tails don't bias `ĉ`.

### 5.1 Kendall scale strategy (migration risk — one explicit choice)

The four terms are heterogeneous: `L_cdf` (squared, ~0-1), `L_exp`/`L_cons` (nt, ~0-50),
`0.3·CE` (nats, ~0-12). The current loss is start_CE+end_CE meaned over the batch, entering a
**single** `UncertaintyWeight` per `{g}_germline` (`dnalignair_loss.py:34-37,88-99`). One
Kendall precision weight cannot balance nat-scale and nt-scale terms simultaneously, and "÷Lg"
alone is insufficient because the terms scale differently with `Lg`.

**Decision (chosen, not "or"):** combine the four into ONE normalized germline loss term with
**fixed inner weights**, then Kendall-weight that single term (mirrors how segmentation already
mixes `CE + 0.1·Cramér` as one term). Normalize each term to ~unit scale first: divide
`L_exp`/`L_cons` by a fixed constant (e.g. `Lg` or a typical coordinate range ~50) so they land
near the CE/CDF magnitude; the inner weights (`0.3·CE`, `1.0·L_exp`, `0.5·L_cdf`, `0.3·L_cons`)
are then comparable. Keep the protected `max_log_var=1.5` on `*_germline`
(`dnalignair_loss.py:30`) so the balancer cannot abandon the head. (Rejected alternative:
separate Kendall heads per term — more weights to tune, and we WANT the terms moving together.)

## 6. Indels and CIGAR (no explicit path modeling needed)

Exact indel CIGAR is produced by `parasail` realign (`io/alignment.py`), which *overrides* the
neural germline coords **only when `full_alignment=True`** (`alignment.py:103-104`,
`dnalignair_infer.py:429-431`). **`full_alignment` defaults to `False`** (`dnalignair_infer.py:218`),
so on the **default deployed path the neural germline coords ARE delivered** — the fast head's
coordinate quality matters directly for default inference, not just as a parasail seed. The
soft-DP's indel-path modeling never reaches the emitted output either way. So the fast head needs
**robust start/end coordinates + reader score**; the banded offsets (§4.4) give net-length
tolerance, and parasail (opt-in) keeps exact CIGAR per-read. No affine-gap DP is required.

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
| M tensor (cosine) | — | 26 MB fp16 / 52 MB fp32 |
| transient peak (padded ×start/end ×band) | — | ~0.1-0.5 GB fp32 (still fine) |

End-to-end projection: **inference ~2.9× faster** (the net's 18% becomes dominant).
**Training step ~10-15× faster ONLY IF both soft-DP sites are replaced** — the coord decode
(pointer head) AND `alignment_score` (the reader, called ~39×/step, §4.5). Replacing only the
coord head leaves the reader soft-DP-bound and the step barely improves. This conditioning is
the B3 fix; the gym A/B loop (currently ~2-3h per 3-arm run) only speeds up once both are done.

## 9. Ablation ladder (each A/B'd on the frozen gym lattice + bootstrap-CI competence)

1. **soft-argmax + CDF loss on the *existing soft-DP*** — isolate loss vs aligner for the
   jitter (predicted: most of it). Aligner-agnostic, zero pointer-head code; do first.
2. `as_strided` single-diagonal pointer head (coords) **+ fast seed reader replacing
   `alignment_score`** (B3 — both soft-DP sites gone), **CE-only loss** — isolate the latency
   win (the headline ~10-15× training step), expect jitter unchanged.
3. **+ soft-argmax L1** — the headline jitter drop.
4. **+ start/end consistency** — heavy-SHM-V coord corr→1, end stops drifting independently.
5. **+ reliability-gated `w`** (needs the B2 token/reliability plumbing) — heavy-SHM coords.
6. **+ banded (2G+1) diagonals** — indel-read coordinates (watch the indel strata).
7. **+ learned substitution matrix / pairwise reader** — sibling resolution; the ONLY path to
   dropping parasail at inference; promote over parasail only on a heavy-SHM-V tie/win.

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

## 11. Risks (the three review BLOCKERS first — must be handled in the plan)

- **B1 — reverse-diagonal wrong with non-uniform `w`.** The flip-leading-flip MUST flip `w`
  too (`wf = flip(w)`); else the end head trains to the wrong coordinate once `w` is
  reliability-gated. Mandatory test uses **non-uniform random `w`** vs a reference loop (§4.2).
  Highest-risk item: silently-wrong with a green test if the fixture uses `w≡1`.
- **B2 — the internal contract changes.** `germline_coords` must thread
  `seg_tok/germ_tok/seg_reliability`; four call sites + `compute_germline_logits` need the
  gather (lift from `dnalignair_infer.py:299-323`). Not optional plumbing (§3).
- **B3 — `alignment_score` (reader) must also leave the soft-DP**, or training stays
  soft-DP-bound and §8 is false (§4.5).
- **S1 — consistency target is the matched span, not `Lg`** (§5 step 4).
- **S2 — one normalized germline term, fixed inner weights, single Kendall head** (§5.1).
- **S3 — soft-argmax decode mandatory + consistent across train/eval/gym** (§3, §5).
- **S4/S5 — per-row length normalization + `w` masking** (§4.2-4.3).
- `w` softmax saturation — uniform init + entropy reg (§4.3).
- Reader sibling regression — gated by keeping parasail until a neural reader ties it (§4.5).
- Banded indel coords — late, gated ablation; drop the band if it regresses clean strata.
- fp16/AMP — run cosine + all LSE/softmax reductions in fp32 (§4.1).

## 12. File map (for the implementation plan)

- Create: `src/alignair/nn/pointer_aligner.py` (`BandedPointerAligner`: diagonal-extraction
  helpers incl. the flip-`w` reverse diagonal, reliability-gated + masked + length-normalized
  weights, banded LSE, fast seed reader).
- Create: `src/alignair/nn/base_match.py` (or shared helper) — extract `SoftDPAligner._scores`
  base-match channel verbatim so both aligners use identical code (B2 / §4.1).
- Create: `tests/alignair/nn/test_pointer_aligner.py` — diagonal-vs-loop correctness with
  **non-uniform `w`** (incl. reverse diagonal, B1), `seg_mask`/`germ_mask` masking, per-row
  length norm, contract shapes, novel-allele floor, fp32-under-autocast, latency smoke.
- Modify: `src/alignair/losses/dnalignair_loss.py` — soft-argmax L1 + CDF + consistency (span
  target, S1) germline loss as ONE normalized Kendall term (S2); τ schedule.
- Modify: `src/alignair/core/dnalignair.py` — aligner selection add `"pointer"` (~:92); change
  `germline_coords` signature to thread `seg_tok/germ_tok/seg_reliability` (~:188, B2).
- Modify: `src/alignair/training/germline_tf.py` — compute & pass `seg_tok/germ_tok/
  seg_reliability` (gather from the rerank path) into `germline_coords` (B2).
- Modify: `src/alignair/training/gym_trainer.py` + `src/alignair/training/reader.py` — point
  the reader at the fast seed score instead of soft-DP `alignment_score` (B3); add
  hard-negative siblings (§7).
- Modify: decode to soft-argmax consistently — `germline_aligner.py decode_germline_coords`
  (`soft=` option) + switch `gym/instrument/evaluator.py`, `gym_trainer.py:325-330`,
  `inference/dnalignair_infer.py:284-286` (S3).
- Modify: `src/alignair/config/dnalignair_config.py` — `aligner="pointer"`, band G, loss-term
  weights, τ schedule.
- A/B harness: new `scripts/exp_aligner_ablation.py` (extend the `exp_ramp_vs_factored.py`
  pattern) to run the §9 ladder on the frozen lattice. Ablation #1 (loss on existing soft-DP)
  needs **zero pointer-head code** — runnable immediately (review-confirmed, M5).
