# GenAIRR Gym — Competence-Gated Curriculum + Diagnostics

Design spec, 2026-06-23. Sub-project **A** of a two-part DNAlignAIR redesign.

## Context & motivation

DNAlignAIR is broadly more accurate than IgBLAST (wins 21–23 of 24 benchmark metrics)
but ~2.3× slower, and has one real accuracy weakness (junction `nt_exact` 0.53 vs 0.95,
a ±1–2nt boundary jitter). Profiling (2026-06-23, scaled_long, RTX 3090 Ti) shows the
differentiable soft-DP germline aligner is the cost: germline-coordinate decode = 45.9%
of inference, untouched by any rerank change. The intended end-state is a redesign with
two independent sub-projects:

- **(A) THIS SPEC — the GenAIRR Gym:** a competence-gated curriculum trainer with
  per-axis struggle diagnostics. Built FIRST because it is the *instrument* that
  measures whether any architecture change actually helps: "climb until you plateau" is,
  at fixed architecture, a direct measure of model capacity, so it settles the
  architecture bet empirically (Approach 1 plateaus at floor 7 vs Approach 2 at floor 9).
- **(B) LATER — a learned, parallel germline aligner** (replaces the sequential soft-DP
  with a GPU-parallel learned operator; faster AND better on junction/heavy-SHM/indels).
  Specced separately, developed and A/B'd *against the gym from (A)*.

This spec covers **(A) only**.

### Why a curriculum (prior evidence)

A single-scalar difficulty **ramp** (clean→hard, "exposure-concentration") already beat a
decoupled full-range stratified mixture in prior experiments. The gym is the principled
generalization of that winning ramp: it keeps the scalar climb but replaces the fixed
`p = step/total` clock with an **automatic competence gate** + plateau detection +
diagnostics. Multi-axis independent ranks are explicitly rejected — they resemble the
stratified mixture that lost.

## Goal

Replace the fixed-schedule curriculum with a controller that promotes the model up a
ladder of difficulty levels only when it has *mastered* the current level, detects when
it can no longer climb (its capacity ceiling), and reports — per level and at the ceiling —
exactly what difficulty axis and failure mode is blocking it. Surface all of this through
a gamified 8-bit terminal HUD and structured machine-readable reports.

Non-goals (this spec): the aligner redesign (sub-project B); changing the model
architecture, heads, or loss; adaptive/closed-loop hard-example mining (deferred — the
chosen design is "scalar ladder, per-axis diagnostics", report-only at the frontier).

## Hard constraints (inherited, non-negotiable)

- **Dynamic genotype**: reference is an input; novel alleles callable; nothing
  allele-specific memorized in weights (retrieval + reader). The gym must not introduce
  allele memorization.
- **Segmentation-first**: all germline computation stays downstream of region segmentation.
- Build on the **existing** substrate, do not reimplement it:
  - `gym/gym.py::AlignAIRGym` — online GenAIRR generator driven by `set_progress(p)`.
  - `gym/curriculum.py::Curriculum` — the scalar ramp `params(p)` (the winner).
  - `training/gym_trainer.py::GymTrainer` — training loop; `.evaluate(n_batches, p)`
    already returns the per-head metrics the promotion gate needs (region/state/orient
    acc; per-gene call, start_dev, end_dev, gl_*_dev, e2e_gl_*_dev) and already separates
    teacher-forced (aligner-in-isolation) from end-to-end.
  - `benchmark/` (`ErrorLedger`, `BenchmarkReport`) — confusion pairs, error-by-mutation
    -rate, co-occurring errors; the diagnostics substrate.

## Architecture & components

Six single-purpose units. The controller is pure decision logic over metrics; the HUD is
a pure renderer; GenAIRR generation stays in `AlignAIRGym`. No tangling — adding a new
difficulty axis or gate is a config change.

1. **`RankLadder`** — maps integer level `L ∈ [0, N)` → curriculum progress `p` (→ GenAIRR
   params via the existing `Curriculum.params(p)`). Owns named, human-readable level
   descriptions ("L3: moderate SHM, light trim, 20% fragments"). Discretizes the scalar
   that already works; default `N = 10`, `p_L = L / (N-1)`.

2. **`RankExam`** — a frozen, seeded held-out validation pack PER level (default 3–5k
   reads, generated once with fixed seeds → deterministic, comparable across runs and
   architectures). Scores the model via the existing `evaluate()`, extended to **tag each
   eval read with its GenAIRR truth difficulty** (SHM count, indel count, crop length,
   orientation, N count) so metrics bucket per-axis.

3. **`PromotionGate`** — per-level, per-head thresholds that RELAX as levels harden (hard
   levels have lower achievable ceilings). Returns `(promote: bool, blocking_gates: list)`.
   All gates must pass to promote. The blocking gates ARE the struggle signal. Gate heads:
   `v_call`, `d_call`, `j_call` (allele top-1, higher=better), `coords` (boundary +
   germline MAE, lower=better), `junction` (junction exactness, higher=better),
   `productivity`/`orientation` (optional). Thresholds live in config.

4. **`PlateauDetector`** — patience window + improvement-slope on a composite competence
   score. When gates won't all pass AND the composite has stopped improving over the
   patience window → declares "ceiling at level L". Defaults: patience = 8 exams, slope
   epsilon configurable.

5. **`StruggleReporter`** — wraps the Benchmarking module to emit, per exam: the gate
   pass/fail table, per-axis failure attribution (metric vs each difficulty axis → names
   the bottleneck), sibling-allele confusion pairs, error-by-mutation-rate, and a one-line
   headline. Persists `gym_report_floorL_stepN.{json,md}` + appends `climb_curve.jsonl`.

6. **`GymController`** — orchestrator (a training callback / wrapper around `GymTrainer`).
   Owns the level index + patience state; runs exams at a cadence; applies the gate;
   promotes or detects plateau; drives `set_progress`. Replaces the `p = step/total` clock
   in `GymTrainer.fit()`.

A seventh presentation unit:

7. **`GymHUD`** — pure renderer `render(GymState) -> str`. The gamified 8-bit view (see
   below). A VIEW over `GymState`, never a source of truth.

### The single source of truth: `GymState`

Each exam emits one immutable `GymState` snapshot: current floor + name, step, per-gate
progress (value, threshold, open/closed), per-axis breakdown, rooms-cleared count,
patience counter, best-floor, and history pointer. Both the JSON report and the HUD render
from this one object, so the gamified view and the data can never drift.

## Control loop & data flow

```
train K steps at current level L  (GymTrainer.fit inner loop)
        │
        ▼
RankExam(L): model on the frozen seeded exam for L → GymState
        │
        ▼
PromotionGate(L): do ALL head gates pass?
   ├─ yes ─▶ ROOM CLEARED → L+1, set_progress(ladder[L+1]), reset patience
   │         (L was top floor → GYM COMPLETE)
   └─ no ──▶ PlateauDetector: composite still improving within patience?
            ├─ yes ─▶ keep training at L
            └─ no ──▶ CEILING at L → final deep StruggleReport → stop
```

- **Cadence**: run an exam every `exam_every` training steps (default tuned so exams are
  cheap relative to training; configurable).
- **Monotonic climb**: levels only advance; the model never auto-demotes (a transient bad
  exam is absorbed by the patience window, not a demotion).
- **Reproducibility**: exams frozen + seeded → two architectures climb the SAME tower.

## The 8-bit Gym HUD

Curriculum levels are **rooms in a tower**; gates are **locks** that must all open to
clear a room; metrics are **progress bars** toward thresholds. Illustrative:

```
╔═══════════════ A L I G N A I R   G Y M ═══════════════╗
║  FLOOR 7/10  "Heavy-SHM Tower"        step 41,200     ║
║  [✓][✓][✓][✓][✓][✓][▶][ ][ ][ ]                       ║
║                                                       ║
║  LOCKS (all must open to climb):                      ║
║   V-call    ███████░░  0.86 / 0.88  🔒                ║
║   D-call    █████████  0.71 / 0.65  🔓                ║
║   J-call    █████████  0.84 / 0.80  🔓                ║
║   coords    ████████░  2.1 / 2.0px  🔒                ║
║   junction  ██████░░░  0.61 / 0.70  🔒                ║
║                                                       ║
║  ⚠ STUCK ON: junction (J ±2nt jitter),                ║
║              heavy-SHM V sibling → IGHVF3 family       ║
╚═══════════════════════════════════════════════════════╝
   ROOM CLEARED ×6   ·   patience 3/8   ·   best floor 7
```

Event callouts: `★ ROOM CLEARED — LEVEL UP! ★`, `⚑ NEW BEST FLOOR`, `☠ CEILING REACHED`.
ANSI color + box-drawing; **degrades to plain ASCII** on `--no-color` or non-TTY so logs
stay clean. Rendering is snapshot-testable (pure function of `GymState`).

## Professional foundation (initial cut — to be iterated)

The user flagged this foundation as needing to be solid because the gym will be refined
further toward "perfect for training". Initial commitments:

- **Config-driven**: a `GymConfig` dataclass defines `N` levels, per-level gate thresholds,
  exam size, exam cadence, patience, slope epsilon, HUD on/off/color. No magic numbers in
  logic. Adding an axis/gate = editing config + one mapping, not a rewrite.
- **Clean boundaries**: controller decision logic is pure (`(metrics, state) -> decision`)
  and unit-testable without a GPU; HUD is a pure renderer; GenAIRR generation untouched in
  `AlignAIRGym`. `GymState` is the only shared object.
- **Determinism**: exams seeded + frozen; same model + same tower → same climb.
- **Structured output**: JSON is canonical; HUD + markdown are views over it.
- New module: `src/alignair/gym/control/` (ladder, exam, gate, plateau, reporter,
  controller, hud, state) — keeps files focused.

## Testing

- Unit (no GPU): `RankLadder` mapping; `PromotionGate` all-pass/blocking logic across
  relaxing thresholds; `PlateauDetector` (improving vs plateaued vs noisy sequences);
  per-axis bucketing from tagged reads; `GymHUD` snapshot (color + ASCII).
- Integration (small/CPU): a few-step `GymController` run on a tiny reference asserting a
  promotion fires when metrics clear gates and a ceiling fires when they plateau; exam
  determinism (same seed → identical `GymState`).
- Reuse existing Benchmarking tests for the diagnostics primitives.

## Phasing

- **Phase 1**: `GymState`, `RankLadder`, `PromotionGate`, `PlateauDetector`,
  `GymController` wired into `GymTrainer` (replace the clock). Plain-text HUD. Promotion +
  ceiling work end-to-end on a real (short) run.
- **Phase 2**: `RankExam` freezing + per-axis tagging; `StruggleReporter` (Benchmarking
  wiring); JSON/markdown reports + `climb_curve.jsonl`.
- **Phase 3**: full 8-bit `GymHUD` (color, callouts, ASCII fallback).
- **Phase 4 (later, separate)**: use the gym to A/B sub-project B (the learned aligner) —
  whichever architecture climbs higher wins.

## Success criteria

- A real training run climbs multiple floors under competence gating (not a clock) and
  halts at a detected ceiling with a struggle report that names the blocking axis/failure.
- The same model + tower reproduces the same climb (seeded exams).
- The struggle report's named bottleneck matches known weaknesses (junction jitter,
  heavy-SHM V siblings) — i.e. the diagnostics are trustworthy enough to drive sub-project B.
