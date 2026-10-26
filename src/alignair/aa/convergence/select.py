"""FROZEN blinded checkpoint-selection rule.

Implements §5 of ``Notes and Plans/docs/superpowers/specs/2026-07-22-aa-convergence-study-design.md`` VERBATIM.
This predeclaration was frozen and committed BEFORE any +300k continuation training ran; the
thresholds below MUST NOT be re-tuned after seeing results. Pure function of metric dicts -- no
I/O, no model, no randomness.

Expected metrics-dict shape (incumbent and every candidate; produced by the convergence driver
from ``evaluator.evaluate_checkpoint`` + derived strata/gap fields)::

    {
        "name": str, "step": int,
        "clean_v_acc": float, "clean_j_acc": float,
        "v_macro": float, "j_macro": float,
        "conformal": {"coverage": float},
        "d_allele_macro": float, "d_gene_acc": float,
        "trunc_v_acc": float, "shortread_acc": float, "macro_minus_micro": float,
        "conf_set_mean": float, "conf_set_p95": float,
    }
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# §5 frozen constants -- do not retune.
# ---------------------------------------------------------------------------

GATE_CLEAN_V_ACC_MIN = 0.980
GATE_CLEAN_J_ACC_MIN = 0.988
GATE_MACRO_REGRESSION_MAX = 0.005          # "no V or J macro accuracy regression > 0.5 pt"
GATE_CONFORMAL_COVERAGE_MIN = 0.94

# "§5 Objective ... over the set {D allele-macro, D gene accuracy, truncation-V accuracy,
# short-read accuracy, rare-class signal (macro-micro gap), calibration mean set-size,
# calibration p95 set-size}."
ACCURACY_WEAKNESS_KEYS = (
    "d_allele_macro", "d_gene_acc", "trunc_v_acc", "shortread_acc", "macro_minus_micro",
)
SET_SIZE_WEAKNESS_KEYS = ("conf_set_mean", "conf_set_p95")

ACCURACY_IMPROVEMENT_THRESHOLD = 0.01       # ">= 1.0 pt" absolute
SET_SIZE_RELATIVE_REDUCTION = 0.05          # ">= 5%" relative reduction


def _check_gates(incumbent: dict, candidate: dict) -> list[str]:
    """§5 GATES: candidate rejected if any fails. Returns the list of failure messages (empty
    list == all gates passed)."""
    failures: list[str] = []

    clean_v = candidate["clean_v_acc"]
    if clean_v < GATE_CLEAN_V_ACC_MIN:
        failures.append(f"clean_v_acc {clean_v:.4f} < {GATE_CLEAN_V_ACC_MIN} (gate)")

    clean_j = candidate["clean_j_acc"]
    if clean_j < GATE_CLEAN_J_ACC_MIN:
        failures.append(f"clean_j_acc {clean_j:.4f} < {GATE_CLEAN_J_ACC_MIN} (gate)")

    v_macro, inc_v_macro = candidate["v_macro"], incumbent["v_macro"]
    if v_macro < inc_v_macro - GATE_MACRO_REGRESSION_MAX:
        failures.append(
            f"v_macro regression: {v_macro:.4f} < incumbent {inc_v_macro:.4f} - "
            f"{GATE_MACRO_REGRESSION_MAX} (gate)"
        )

    j_macro, inc_j_macro = candidate["j_macro"], incumbent["j_macro"]
    if j_macro < inc_j_macro - GATE_MACRO_REGRESSION_MAX:
        failures.append(
            f"j_macro regression: {j_macro:.4f} < incumbent {inc_j_macro:.4f} - "
            f"{GATE_MACRO_REGRESSION_MAX} (gate)"
        )

    coverage = (candidate.get("conformal") or {}).get("coverage")
    if coverage is None or coverage < GATE_CONFORMAL_COVERAGE_MIN:
        failures.append(f"conformal.coverage {coverage} < {GATE_CONFORMAL_COVERAGE_MIN} (gate)")

    return failures


def _weakness_objective(incumbent: dict, candidate: dict) -> tuple[list[str], dict[str, float]]:
    """§5 OBJECTIVE: for each predeclared weakness metric, decide whether the candidate improved
    on the incumbent by >= the metric-type threshold. Returns (improved_keys, gain_by_key) where
    gain is always signed so that positive == better (accuracy: candidate - incumbent; set-size:
    incumbent - candidate, i.e. the reduction)."""
    improved: list[str] = []
    gain: dict[str, float] = {}

    for key in ACCURACY_WEAKNESS_KEYS:
        inc_v, cand_v = incumbent.get(key), candidate.get(key)
        if inc_v is None or cand_v is None:
            continue
        gain[key] = cand_v - inc_v
        if cand_v >= inc_v + ACCURACY_IMPROVEMENT_THRESHOLD:
            improved.append(key)

    for key in SET_SIZE_WEAKNESS_KEYS:
        inc_v, cand_v = incumbent.get(key), candidate.get(key)
        if inc_v is None or cand_v is None:
            continue
        gain[key] = inc_v - cand_v
        if cand_v <= inc_v * (1.0 - SET_SIZE_RELATIVE_REDUCTION):
            improved.append(key)

    return improved, gain


def select_checkpoint(incumbent: dict, candidates: list[dict]) -> dict:
    """Apply the FROZEN §5 selection rule.

    Returns ``{"winner": <candidate name or "incumbent">, "rationale": str,
    "per_candidate": [ {name, step, gates_passed, gate_failures, weaknesses_improved,
    objective_count, gain, d_allele_macro_gain, trunc_v_acc_gain, combined_gain}, ... ]}``.
    """
    per_candidate = []
    for cand in candidates:
        gate_failures = _check_gates(incumbent, cand)
        gates_passed = not gate_failures

        weaknesses_improved: list[str] = []
        gain: dict[str, float] = {}
        if gates_passed:
            weaknesses_improved, gain = _weakness_objective(incumbent, cand)

        d_allele_macro_gain = gain.get("d_allele_macro", 0.0)
        trunc_v_acc_gain = gain.get("trunc_v_acc", 0.0)

        per_candidate.append({
            "name": cand["name"],
            "step": cand.get("step"),
            "gates_passed": gates_passed,
            "gate_failures": gate_failures,
            "weaknesses_improved": weaknesses_improved,
            "objective_count": len(weaknesses_improved),
            "gain": gain,
            "d_allele_macro_gain": d_allele_macro_gain,
            "trunc_v_acc_gain": trunc_v_acc_gain,
            "combined_gain": d_allele_macro_gain + trunc_v_acc_gain,
        })

    # Eligible = gates passed AND improves >= 1 predeclared weakness metric.
    eligible = [pc for pc in per_candidate if pc["gates_passed"] and pc["objective_count"] >= 1]

    if not eligible:
        return {
            "winner": "incumbent",
            "rationale": (
                "NULL result: no gate-passing candidate improved >= 1 predeclared weakness "
                "metric vs incumbent -- longer training did not help. Incumbent retained "
                "unchanged (§5)."
            ),
            "per_candidate": per_candidate,
        }

    # §5 winner selection: max objective_count; tie-break (1) larger combined
    # (d_allele_macro + trunc_v_acc) gain; tie-break (2) earliest step.
    eligible_sorted = sorted(
        eligible,
        key=lambda pc: (-pc["objective_count"], -pc["combined_gain"], pc["step"]),
    )
    winner_pc = eligible_sorted[0]

    rationale = (
        f"Selected '{winner_pc['name']}' (step {winner_pc['step']}): passed all gates and "
        f"improved {winner_pc['objective_count']} predeclared weakness metric(s) "
        f"({', '.join(winner_pc['weaknesses_improved'])}) vs incumbent; combined "
        f"D-allele-macro+trunc-V gain={winner_pc['combined_gain']:.4f} (§5 tie-break 1), "
        f"step used as tie-break 2 where applicable."
    )

    return {"winner": winner_pc["name"], "rationale": rationale, "per_candidate": per_candidate}
