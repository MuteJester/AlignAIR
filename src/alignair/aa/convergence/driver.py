"""Convergence-study driver: continue training the frozen Stage-1 incumbent, checkpoint on a
fixed schedule, and evaluate every checkpoint (incumbent + candidates) on the SAME frozen §3
cohort -- producing the per-checkpoint metrics table + provenance artifacts consumed by the
(separate, later) §5 selection step. See
Notes and Plans/docs/superpowers/specs/2026-07-22-aa-convergence-study-design.md §4/§8.

This module does NOT run selection (``alignair.aa.convergence.select.select_checkpoint``) --
that stays a separate, auditable step run against the frozen ``checkpoints_metrics.json`` this
driver produces, so the (frozen, pre-committed) selection rule is never re-run or re-tuned in the
same breath as the numbers it judges.
"""
from __future__ import annotations

import json
import os

from ..resume import load_checkpoint
from ..strategic import strategic_setup
from ..strategic_trunc import stage1_train
from .evaluator import build_cohort, evaluate_checkpoint

DC_NAME = "HUMAN_IGH_OGRDB"
MAX_SEQ_LENGTH = 192
DEFAULT_CKPT_DIR = "experiments/aa/strategic"

# §5 text, copied VERBATIM from the spec (frozen BEFORE this study's training ran) into every
# study's `predeclaration.md` -- see driver docstring above and select.py's own verbatim copy of
# the same section as executable rule.
PREDECLARATION_TEXT = """\
# Frozen predeclared selection rule

Copied verbatim from `Notes and Plans/docs/superpowers/specs/2026-07-22-aa-convergence-study-design.md` §5
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
"""


def flatten_row(name: str, step: int, out: dict) -> dict:
    """Derive the §5 selection-input keys from one ``evaluate_checkpoint`` output and store them
    FLAT alongside ``name``/``step`` (the exact shape ``select.select_checkpoint`` expects -- see
    that module's docstring). A missing stratum (e.g. a tiny/ablated cohort with an empty
    ``aa_truncate_50_80`` or ``short_janchor`` bucket) or a conformal fit that couldn't run
    degrades the corresponding derived metric to ``None`` rather than raising -- ``select.py`` is
    already None-aware (skips a weakness metric it can't compare) by design.
    """
    by_stratum = out.get("by_stratum") or {}
    conformal = out.get("conformal") or {}

    def _stratum_v_acc(stratum_name: str):
        entry = by_stratum.get(stratum_name)
        return entry.get("v_acc") if entry is not None else None

    v_macro, clean_v_acc = out.get("v_macro"), out.get("clean_v_acc")
    macro_minus_micro = (v_macro - clean_v_acc) if v_macro is not None and clean_v_acc is not None else None

    return {
        "name": name,
        "step": step,
        "clean_v_acc": clean_v_acc,
        "clean_j_acc": out.get("clean_j_acc"),
        "v_macro": v_macro,
        "j_macro": out.get("j_macro"),
        "d_allele_macro": out.get("d_allele_macro"),
        "d_gene_acc": out.get("d_gene_acc"),
        "conformal": conformal,
        "trunc_v_acc": _stratum_v_acc("aa_truncate_50_80"),
        "shortread_acc": _stratum_v_acc("short_janchor"),
        "macro_minus_micro": macro_minus_micro,
        "conf_set_mean": conformal.get("set_mean"),
        "conf_set_p95": conformal.get("set_p95"),
    }


def _row_from_evaluator_output(name: str, step: int, out: dict) -> dict:
    """The FULL per-checkpoint metrics row written to ``checkpoints_metrics.json``: the raw
    ``evaluate_checkpoint`` output (every stratum/NLL/Brier/boundary-MAE/freq-bin detail) plus the
    flat §5 selection-input keys layered on top (so the same row is both the audit record and
    directly consumable by ``select.select_checkpoint`` with no further transformation)."""
    return {**out, **flatten_row(name, step, out)}


def _load_fresh_candidate(ckpt_path: str, dc_name: str, max_seq_length: int, device: str):
    """A fresh ``strategic_setup`` model with a saved continuation checkpoint's weights loaded in
    (never reuses the in-training model instance, so a candidate's eval is never contaminated by
    optimizer-adjacent state or a stale ``.eval()``/``.train()`` mode from training)."""
    model, cfg, class_maps, meta = strategic_setup(dc_name, max_seq_length, has_d=True)
    ck = load_checkpoint(ckpt_path)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, cfg, class_maps, meta


def _write_json(path: str, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _render_convergence_md(rows: list, budget: int, ckpt_steps: tuple, cohort_fp: dict) -> str:
    cols = ("name", "step", "clean_v_acc", "clean_j_acc", "v_macro", "j_macro",
           "d_allele_macro", "d_gene_acc", "trunc_v_acc", "shortread_acc",
           "macro_minus_micro", "conf_set_mean", "conf_set_p95")
    lines = [
        "# Convergence study -- per-checkpoint metrics",
        "",
        f"Budget: +{budget} steps. Requested (relative) checkpoint offsets: {list(ckpt_steps)}.",
        f"Cohort fingerprint: `{cohort_fp['fingerprint']}` "
        f"(dc={cohort_fp['dc_name']}, seed={cohort_fp['seed']}, n_per_stratum={cohort_fp['n_per_stratum']}).",
        "",
        "Selection (§5) is a SEPARATE, later step -- this table is descriptive only.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c)) for c in cols) + " |")
    lines.append("")
    return "\n".join(lines)


def run_study(resume_path: str, budget: int, ckpt_steps, device: str, art_dir: str, *,
             cohort_seed: int = 20250722, n_per_stratum: int = 200, dc_name: str = DC_NAME,
             batch_size: int = 64, base_lr: float = 1e-4, lr_min: float = 1e-5,
             val_every: int = 25000, plateau_k: int = 3,
             ckpt_dir: str = DEFAULT_CKPT_DIR) -> dict:
    """Run the §4 convergence driver: incumbent baseline eval -> resumed continuation training
    with checkpoints at ``ckpt_steps`` (RELATIVE offsets from the incumbent's own step, e.g.
    ``(25000, 75000, 150000, 300000)``) -> per-checkpoint eval on the SAME frozen cohort. Writes
    ``{art_dir}/{cohort_fingerprint,checkpoints_metrics}.json``, ``{art_dir}/convergence.md``, and
    ``{art_dir}/predeclaration.md``. Does NOT run §5 selection (a separate, later step).

    ``ckpt_steps`` are relative (as declared on the CLI / in the spec's "+25k/+75k/..."
    phrasing); they are converted to ABSOLUTE step numbers (``start_step + rel``) before being
    passed to ``stage1_train``, since the resumed training loop counts absolute steps starting
    from the incumbent's own checkpointed step (NOT from 0) -- see ``strategic_trunc.stage1_train``
    (`start_step = ck["step"]`, loop range `start_step+1 .. start_step+steps`).
    """
    os.makedirs(art_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_steps = tuple(int(s) for s in ckpt_steps)

    # ---- 1. model/cfg/class_maps/meta ----
    model, cfg, class_maps, meta = strategic_setup(dc_name, MAX_SEQ_LENGTH, has_d=True)

    # ---- 2. frozen evaluator cohort + predeclaration copy ----
    cohort, fingerprint = build_cohort(dc_name, n_per_stratum=n_per_stratum, seed=cohort_seed)
    cohort_fp = {"fingerprint": fingerprint, "seed": cohort_seed,
                "n_per_stratum": n_per_stratum, "dc_name": dc_name}
    _write_json(os.path.join(art_dir, "cohort_fingerprint.json"), cohort_fp)
    with open(os.path.join(art_dir, "predeclaration.md"), "w") as f:
        f.write(PREDECLARATION_TEXT)

    # ---- 3. incumbent baseline ----
    # `start_step` is read from the incumbent checkpoint's OWN `step` field (58000 for the
    # current `model_stage1_PASS.pt`) rather than hardcoded, so a re-run against a different
    # incumbent still computes correct absolute checkpoint/budget numbers.
    ck = load_checkpoint(resume_path)
    model.load_state_dict(ck["model"])
    start_step = int(ck.get("step") or 0)
    model.to(device).eval()
    incumbent_out = evaluate_checkpoint(model, cfg, class_maps, meta, cohort, device=device)
    incumbent_row = _row_from_evaluator_output("incumbent", start_step, incumbent_out)
    print(f"[convergence] incumbent (step {start_step}): clean_v_acc={incumbent_row['clean_v_acc']:.4f} "
         f"clean_j_acc={incumbent_row['clean_j_acc']:.4f} d_allele_macro={_fmt(incumbent_row['d_allele_macro'])}",
         flush=True)

    # ---- 4. resumed continuation training, checkpointing at ABSOLUTE step numbers ----
    abs_ckpt_steps = tuple(start_step + rel for rel in ckpt_steps)
    total_budget = budget  # T_max = number of scheduler steps this run traverses, not absolute step number

    def val_fn(m, step):
        out = evaluate_checkpoint(m, cfg, class_maps, meta, cohort, device=device)
        m.to(device).train()                      # evaluate_checkpoint leaves the model in eval() mode
        row = flatten_row(f"val_step{step}", step, out)
        return {"objective": row["d_allele_macro"], **row}

    stage1_train(model, cfg, class_maps, meta, dc_name, steps=budget, batch_size=batch_size,
                base_lr=base_lr, lr_min=lr_min, total_budget=total_budget, seed=0, device=device,
                resume_path=resume_path, ckpt_dir=ckpt_dir, ckpt_steps=abs_ckpt_steps,
                val_every=val_every, val_fn=val_fn, plateau_k=plateau_k)

    # ---- 5. evaluate every saved candidate checkpoint (fresh model instance each time) ----
    candidate_rows = []
    for abs_step in abs_ckpt_steps:
        ckpt_path = os.path.join(ckpt_dir, f"model_cont_step{abs_step}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[convergence] WARNING: expected checkpoint '{ckpt_path}' was not saved "
                 f"(training may have exhausted the stream before this step) -- skipping.", flush=True)
            continue
        cand_model, cand_cfg, cand_class_maps, cand_meta = _load_fresh_candidate(
            ckpt_path, dc_name, MAX_SEQ_LENGTH, device)
        out = evaluate_checkpoint(cand_model, cand_cfg, cand_class_maps, cand_meta, cohort, device=device)
        row = _row_from_evaluator_output(f"cont_step{abs_step}", abs_step, out)
        candidate_rows.append(row)
        print(f"[convergence] cont_step{abs_step}: clean_v_acc={row['clean_v_acc']:.4f} "
             f"clean_j_acc={row['clean_j_acc']:.4f} d_allele_macro={_fmt(row['d_allele_macro'])}", flush=True)

    # ---- 6. write outputs (NOT running §5 selection here) ----
    rows = [incumbent_row] + candidate_rows
    checkpoints_metrics = {
        "cohort_fingerprint": cohort_fp,
        "rows": rows,
        "budget": budget,
        "ckpt_steps": list(ckpt_steps),
        "provenance": {
            "resume_path": resume_path, "start_step": start_step, "device": device,
            "dc_name": dc_name, "batch_size": batch_size, "base_lr": base_lr, "lr_min": lr_min,
            "total_budget": total_budget, "val_every": val_every, "plateau_k": plateau_k,
            "ckpt_dir": ckpt_dir,
        },
    }
    _write_json(os.path.join(art_dir, "checkpoints_metrics.json"), checkpoints_metrics)
    with open(os.path.join(art_dir, "convergence.md"), "w") as f:
        f.write(_render_convergence_md(rows, budget, ckpt_steps, cohort_fp))

    print(f"\n[convergence] wrote {len(rows)} row(s) ({1} incumbent + {len(candidate_rows)} candidate(s)) "
         f"to {art_dir}. Selection (§5) NOT run here -- see alignair.aa.convergence.select.", flush=True)
    return checkpoints_metrics
