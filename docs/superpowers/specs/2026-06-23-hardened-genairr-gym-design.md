# Hardened GenAIRR Gym — Theoretically-Grounded Training Procedure

Design spec, 2026-06-23. Supersedes the curriculum *content* of
`2026-06-23-genairr-gym-curriculum-design.md` (the scaffold) while REUSING its control-flow
mechanics (`GymState`, `GymHUD`, `StruggleReporter`, `PlateauDetector`, the gate/exam wiring
in `src/alignair/gym/control/`).

## Why this redesign

The scaffold gym is mechanically correct but is not a trustworthy *instrument* and its
curriculum is not theoretically grounded. Two independent expert reviews (a curriculum-learning
theorist and a deep-learning-training/optimization expert) converged on the same load-bearing
defects. Capturing their convergent findings as the basis of this design:

1. **Single-scalar all-axes ramp is the central error.** `Curriculum` lerps every GenAIRR knob
   on one scalar `p`. The axes (SHM, trim, indel, crop, orientation…) stress different heads on
   different timescales and are independent at deployment. The project's prior "ramp beat
   stratified mixture" result is explained as *non-stationarity (exposure-concentration) winning,
   not the coupling* — so the synthesis is a **factored per-axis ramp**: keep start-easy
   exposure-concentration, drop the coupling, keep a permanent hard tail.
2. **The coupled ramp structurally excludes the #1 weakness.** Full-length heavy-SHM V lives in
   the anti-correlated corner (high SHM × *low* crop); a coupled ramp at high `p` produces high
   SHM *and* high crop, so it never visits that corner. Decoupling + regret targeting is required
   to even sample it.
3. **Kendall uncertainty weighting fights the curriculum (and likely *causes* heavy-SHM-V).** As
   difficulty rises, a hard head's loss rises → Kendall infers high aleatoric noise → inflates its
   σ² → down-weights the very head we want to push. Both experts flagged this independently.
4. **The exams are a broken ruler.** Fresh-resampled, few-batch exams make promotion a coin flip
   and make "which architecture climbs higher" noise-dominated — invalidating the A/B use case.
5. **The terminal distribution must equal real deployment, incl. the hard tail.** Capping SHM at
   0.15 converges (even perfectly) to the wrong objective; the endpoint must reach the real
   ~0.25–0.30.
6. **Replace arbitrary thresholds + discrete ladder with a competence-driven controller**
   (learning-progress / regret sampling: ALP-GMM, Prioritized Level Replay), which auto-targets
   junction-jitter and heavy-SHM-V.
7. **Anti-forgetting + honest ceiling detection.** The current controller only ever evaluates the
   current floor → silent forgetting. Need whole-lattice eval + easy-replay; declare a ceiling
   only via slope tests + LR-probe + comparison to the simulator's irreducible-error floor.

Key references (named by the experts): Bengio 2009 (curriculum learning / continuation);
Kumar 2010 (self-paced); Weinshall 2018, Hacohen & Weinshall 2019; Wu 2021 "When Do Curricula
Work?"; Platanios 2019 (competence-based pacing); Matiisen 2019 (teacher-student); Portelas 2019
(ALP-GMM) + 2020 survey; Jiang 2021 (Prioritized Level Replay); Kendall 2018 (uncertainty
weighting); Chen 2018 (GradNorm); Yu 2020 (PCGrad); Loshchilov 2017 (SGDR); Pezeshki 2021
(gradient starvation).

## Goal

Turn the gym into a theoretically-grounded training procedure that (a) **measures** model
competence reproducibly and comparably across architectures, and (b) **drives** training so the
AlignAIR model converges toward a SOTA IG/TCR aligner (beating IgBLAST/partis/MiXCR), with
justified confidence that it converged.

## Hard constraints (inherited, non-negotiable)

- **Dynamic genotype** (reference is input; novel alleles callable; no allele memorization) and
  **segmentation-first**. Unchanged.
- Reuse, don't reimplement: `src/alignair/gym/control/` scaffold (`GymState`, `GymHUD`,
  `StruggleReporter`, `PlateauDetector`); `gym/gym.py::AlignAIRGym` and
  `gym/gym.py::build_experiment` (the GenAIRR compile path); `GymTrainer`; the `benchmark/` module
  and `CompetenceMetric` aligning to the canonical IgBLAST `benchmark.cli compare`.

## Architecture & components

### A. The instrument (Phase 1 — built first; everything is judged by it)

- **`TaskSpace`** — the difficulty parameter box `Θ = ∏ᵢ [minᵢ, maxᵢ]`, one axis per GenAIRR knob
  (SHM rate/count, end_loss_5, end_loss_3, indel_count, seq_error_rate, ambiguous_count,
  crop_len, orient_prob, invert_d_prob, contaminate_prob). Each axis's **max set to the real
  deployment 99th percentile** (SHM→~0.30, end-loss→real max, etc.). Axes independently samplable;
  a `sample(rng, density)` draws a θ; `to_genairr_params(θ)` maps to the dict `build_experiment`
  consumes. Replaces the scalar `Curriculum`.
- **`FrozenLattice`** — pre-generated, seeded, **stratified** held-out evaluation cells over Θ,
  with explicit deployment-hard cells (heavy-SHM, **full-length** heavy-SHM, junction-boundary
  band). `N` reads/cell sized for the target CI (≈2k/cell for 2% calls, up to ~14k for 1%).
  Serialized to disk + content-fingerprinted; **never trained on**. The single measurement
  instrument; provides per-cell read sets for evaluation.
- **`CompetenceMetric`** — one pre-registered, **external, non-Kendall** deployment-alignment
  score `S(x) ∈ [0,1]`: a fixed composite of V/D/J allele correctness (multi-allele credit),
  coordinate-MAE→accuracy at a fixed nt tolerance, region per-position accuracy, and junction
  exact-match. Computed per read; aggregated per cell with **bootstrap CIs**. Independent of the
  Kendall-weighted training loss so it is comparable across architectures/runs.
- **`LatticeEvaluator`** — runs a model over the frozen lattice → per-cell `{S mean, bootstrap
  CI, per-sub-metric}`, **teacher-forced AND end-to-end** reported separately (the gap is the
  error-propagation diagnostic). Produces the per-cell competence field consumed by the
  controller, sampler (ALP), HUD, and reporter.
- **Validation gate for Phase 1**: show gym competence `S` correlates with the canonical IgBLAST
  4400-case benchmark on the current `scaled_long` model (the proxy must be real before it drives
  anything).

### B. The curriculum sampler (Phase 2 + Phase 4)

- **`FactoredSampler`** — the time-varying sampling density `p_t(θ)` over Θ, a mixture:
  1. **competence-paced per-axis ramp (~60%)** — per-axis marginals, each admitting harder
     values as measured competence on its current support rises (Platanios competence pacing,
     one clock per axis; spacing in *competence* space, not knob space). [Phase 2]
  2. **ALP-GMM learning-progress targeting (~25%)** — a GMM over Θ sampled ∝ absolute learning
     progress in `S`; auto-concentrates on heavy-SHM-V and the junction-boundary band as those
     become where competence is still moving. [Phase 4]
  3. **anti-forgetting + regret floor (~15%)** — a fixed easy/pristine replay fraction plus
     regret-pinned hard corners (full-length heavy-SHM) that never lose mass. [Phase 2 floor;
     Phase 4 regret weighting]
  - **Terminal `p_T` = the true deployment marginals incl. the hard tail.** Drives
    `AlignAIRGym` (replaces scalar `set_progress`); requires `AlignAIRGym`/`build_experiment` to
    accept per-axis per-read distributions (the stratified form already partly exists).

### C. Loss–curriculum coupling fixes (Phase 3)

- **Kendall σ² floors** (a lower bound on weight, i.e. an upper bound on σ²) on the junction and
  V-call heads so the balancer cannot abandon the head the curriculum is pushing.
- **Trunk gradient balancing** — GradNorm (or PCGrad/CAGrad for conflict) on gradients into the
  shared backbone, so saturating easy heads (region/orientation) don't starve junction/coords.
- **σ² freeze during transients** — hold `log σ²` constant for a short window after each
  difficulty change, then thaw.
- **Schedule staggering** — scheduled-sampling probability and EMA-distillation never co-ramp
  with difficulty; both step back briefly on a difficulty change; a small **SGDR warm-restart**
  on the LR at promotion events absorbs the Adam second-moment transient.

### D. Promotion, ceiling, anti-forgetting (Phase 5)

- **`PromotionController` (hardened)** — promotion requires, for all binding cells/metrics:
  (i) **LCB₉₅(S) ≥ one global τ** (bootstrap CI lower bound, not a point estimate, not a
  per-level relaxing bar), with **BH-FDR** across cells; (ii) training-side gates — per-task loss
  slope settled (**Mann–Kendall** trend test), σ² settled, no pathological trunk gradient
  conflict, and an **LR-probe** that does not unlock further gains; (iii) a **no-regression** gate
  on previously-won cells. Sustained over ≥2 consecutive exams (debounce).
- **`CeilingDetector` (hardened)** — declares a true capacity ceiling only when ALP→0 across cells
  **and** competence ≈ the simulator-estimated **irreducible-error** floor; the LR-probe
  distinguishes a capacity ceiling from an LR-schedule artifact.
- **Anti-forgetting** — whole-lattice eval every checkpoint; a competence *drop* on any won cell
  raises that cell's regret weight in the sampler (closes the loop with B.3).

### E. View + diagnostics (incremental across phases)

- **`GymHUD` (extended)** — rooms = frozen-lattice cells; climbing = competence rising across the
  lattice; renders per-axis difficulty positions, per-cell competence ± CI, an ALP heatmap,
  forgetting alarms, and the training-side gate states. Keeps the 8-bit aesthetic as a VIEW over
  the continuous controller.
- **Convergence dashboard** (via `StruggleReporter` + logging) — per-task loss slopes, σ²
  trajectories, trunk grad-norms/cosines, per-stratum CI metrics (TF + e2e + gap),
  boundary-exactness (J vs V), a named heavy-SHM-V metric, LR-vs-plateau, teacher–student
  divergence, throughput (reads/s).

## Phased build order (each phase independently testable; the gym stays usable)

1. **Phase 1 — Instrument.** `TaskSpace`, `FrozenLattice`, `CompetenceMetric`, `LatticeEvaluator`;
   validate `S` vs the IgBLAST benchmark on `scaled_long`. *Precondition for everything and for any
   A/B.*
2. **Phase 2 — Factored sampler + deployment endpoint.** Per-axis competence-paced ramp + easy
   replay; terminal = deployment distribution incl. hard tail. Re-run the decisive experiment:
   ramp vs mixture vs **factored**.
3. **Phase 3 — Kendall–curriculum coupling fix.** σ² floors + trunk GradNorm + transient freeze +
   schedule staggering. (Likely the direct heavy-SHM-V lever.)
4. **Phase 4 — ALP-GMM / regret targeting.** Automatic concentration on junction + heavy-SHM-V.
5. **Phase 5 — Ceiling / anti-forgetting rigor.** Mann–Kendall slope, irreducible-floor, LR-probe,
   regret-weighted replay, no-regression gate.

## Success criteria

- **Instrument validity (P1):** gym `S` correlates with the canonical IgBLAST benchmark on
  `scaled_long`; per-cell CIs are tight enough (target half-width) that a 2% competence difference
  is resolvable; two evaluations of the same model on the frozen lattice are identical.
- **Curriculum (P2–P4):** the factored sampler matches or beats the prior ramp on held-out
  competence; the heavy-SHM-full-length-V cell is actually sampled and its competence rises; an
  A/B of two architectures on the same frozen lattice produces a CI-separated verdict.
- **Convergence (P3–P5):** σ² no longer abandons junction/V-call under rising difficulty; no
  silent regression on won (easy/pristine) cells; ceiling calls survive an LR-probe and sit at the
  irreducible-error floor.
- **End goal:** a training run driven by this gym produces a model that beats the current
  `scaled_long` on the canonical benchmark, with the known weaknesses (junction `nt_exact`,
  heavy-SHM V) measurably improved and no regression elsewhere.

## Out of scope

- Sub-project B (the learned parallel aligner) — developed later and A/B'd *using this gym*. Some
  loss-side junction fixes (single-source coords, sharpened boundary targets) belong to that work;
  this spec only ensures the gym *measures and targets* those weaknesses.
