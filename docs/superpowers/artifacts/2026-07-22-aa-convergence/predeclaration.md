# Frozen predeclared selection rule

Copied verbatim from `docs/superpowers/specs/2026-07-22-aa-convergence-study-design.md` §5
(predeclared and committed BEFORE the +300k continuation training in this study ran). The
executable form of this rule lives in `alignair.aa.convergence.select.select_checkpoint`; this
file is the frozen, human-readable predeclaration copy, not a second implementation.

## §5 FROZEN predeclared selection rule (blinded to Phase 3)
Incumbent = `model_stage1_PASS.pt`. Evaluate every candidate on the §3 frozen cohort. **Phase 3's 16-antibody
cohort is NOT consulted during selection.**
- **GATES (candidate rejected if any fails):**
  - clean V class accuracy ≥ **0.980** (incumbent 0.986)
  - clean J class accuracy ≥ **0.988** (incumbent 0.992)
  - no V or J **macro** accuracy regression > **0.5 pt** vs incumbent
  - per-checkpoint conformal coverage ≥ **0.94** (target 0.95)
- **Objective among gate-passers:** maximize the **count of predeclared weakness metrics improved by ≥ a
  threshold** vs incumbent, over the set {D allele-macro, D gene accuracy, truncation-V accuracy, short-read
  accuracy, rare-class signal (macro−micro gap), calibration mean set-size, calibration p95 set-size}.
  Improvement threshold: ≥ **1.0 pt** for accuracy metrics; ≥ **5%** relative reduction for set-size metrics.
- **Tie-breaks:** (1) larger combined (D allele-macro + truncation-V) gain; (2) **earliest** checkpoint
  (parsimony / lower over-specialization risk).
- **Null result is a valid, honest outcome:** if no gate-passing candidate improves ≥ 1 weakness metric, the
  **incumbent is kept** and the study reports "longer training did not help" -- the snapshot is unchanged.
