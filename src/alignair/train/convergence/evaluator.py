"""Checkpoint metrics over the declared convergence cohort.

Cohort construction and content identity live in .cohort; selection thresholds remain in .select.
Scoring uses shared inference, categorical evaluation and calibration implementations.
"""
from __future__ import annotations

import numpy as np

from alignair.calibration import Conformal
from alignair.core.runtime import evaluation_mode
from alignair.errors import InferenceError, error_context
from alignair.data.adapters import CategoricalBatchAdapter
from alignair.evaluate import (categorical_nll, evaluate_categorical_cohort, frequency_accuracy_report,
                         macro_accuracy, multiclass_brier_score)
from alignair.predict.probabilities import protein_class_probabilities
from alignair.train.convergence.cohort import _first
from alignair.train.convergence.validation import prepare_evaluation

_DEFAULT_FREQ_EDGES = (0.0, 0.02, 0.05, 1.0)


def _freq_for(rows: list, gene_key: str, allele_class: dict, dc_name: str) -> np.ndarray:
    """Per-row TRUE-allele frequency for freq-binning. Uses ``dc.gene_use_dict`` when the
    dataconfig exposes it; otherwise falls back to a UNIFORM prior over all classifiable alleles
    (``1 / n_alleles``) -- documented here since ``gene_use_dict`` is not guaranteed on every
    GenAIRR dataconfig."""
    import GenAIRR as ga
    dc = getattr(ga, dc_name, None)
    table = getattr(dc, "gene_use_dict", None) if dc is not None else None
    table = (table or {}).get(gene_key)
    names = [_first(r["v_call" if gene_key == "V" else "j_call"]) for r in rows]
    if table:
        return np.array([table.get(n, 0.0) for n in names], dtype=float)
    n_alleles = len(allele_class) or 1
    return np.full(len(names), 1.0 / n_alleles, dtype=float)


def _boundary_mae(model, cfg, samples, device: str) -> dict:
    """Score shared preflight-prepared boundary samples, retaining informational nulls."""
    if not samples:
        return {"v": None, "j": None,
                "note": "deferred: no clean-stratum record survived record_to_sample filtering"}
    res = evaluate_categorical_cohort(
        model, samples, CategoricalBatchAdapter.from_config(cfg, device=device), cfg, genes=("v", "j"))
    b = res["boundaries"]
    return {"v": {"start": b["v_start"]["mae"], "end": b["v_end"]["mae"]},
            "j": {"start": b["j_start"]["mae"], "end": b["j_end"]["mae"]},
            "n": len(samples)}


def _conformal_v(model, cfg, device: str, class_maps, calibration_rows: list, clean_rows: list,
                 alpha: float, seed: int = 0, *, test_probabilities) -> dict:
    """Per-checkpoint APS conformal fit (V gene -- the selection-critical head) on the fixed
    calibration slice, evaluated (coverage + set-size stats) on the clean-stratum test rows."""
    if not calibration_rows or not clean_rows:
        return {"coverage": None, "set_mean": None, "set_p95": None,
                "note": "insufficient rows for conformal fit"}
    vclass = class_maps["V"].allele_class
    y_cal = np.array([vclass[_first(r["v_call"])] for r in calibration_rows])
    p_cal = _class_probabilities(model, [r["aa"] for r in calibration_rows], cfg, device,
                                        ("v_class_logits",))["v_class_logits"]
    conf = Conformal(alpha=alpha, seed=seed).fit(p_cal, y_cal)

    y_tst = np.array([vclass[_first(r["v_call"])] for r in clean_rows])
    p_tst = test_probabilities
    sizes, covered = [], []
    for i in range(len(clean_rows)):
        s = conf.predict_set(p_tst[i])
        sizes.append(len(s))
        covered.append(y_tst[i] in s)
    sizes = np.asarray(sizes, dtype=float)
    return {"gene": "V", "n_cal": len(calibration_rows), "n_test": len(clean_rows),
            "coverage": float(np.mean(covered)),
            "set_mean": float(sizes.mean()), "set_p95": float(np.percentile(sizes, 95))}


def _score_stratum(model, cfg, vclass, rows, device, *, is_clean):
    """Reduce V accuracy locally; only clean heads and truth survive this call."""
    sequences = [row["aa"] for row in rows]
    truth = np.array([vclass[_first(row["v_call"])] for row in rows])
    keys = (("v_class_logits", "j_class_logits") + (("d_class_logits",) if cfg.has_d else ())
            if is_clean else ("v_class_logits",))
    probabilities = _class_probabilities(model, sequences, cfg, device, keys)
    prediction = probabilities["v_class_logits"].argmax(-1)
    summary = {"n": len(rows), "v_acc": float((prediction == truth).mean())}
    return summary, (probabilities, truth) if is_clean else None


def _evaluate_checkpoint(model, cfg, class_maps, meta, cohort, boundary_samples, *, device: str = "cpu",
                        alpha: float = 0.05, freq_edges=_DEFAULT_FREQ_EDGES) -> dict:
    """Evaluate a prepared model and its class maps/D indices over a frozen cohort.

    Results depend on model weights and cohort; the cohort is not mutated.
    """
    vclass, jclass = class_maps["V"].allele_class, class_maps["J"].allele_class
    d_index = meta["d_index"] if cfg.has_d else {}
    d_name = {i: n for n, i in d_index.items()}

    by_stratum_rows = cohort["by_stratum"]
    calibration_rows = cohort["calibration"]

    # ---- per-stratum V accuracy (selection-critical: "clean" feeds clean_v_acc/v_macro below) ----
    by_stratum: dict = {}
    clean_predictions = None
    for name, rows in by_stratum_rows.items():
        if not rows:
            continue
        with error_context(stratum=name):
            summary, retained = _score_stratum(model, cfg, vclass, rows, device, is_clean=name == "clean")
        by_stratum[name] = summary
        if retained is not None:
            clean_predictions = retained

    clean_rows = by_stratum_rows["clean"]
    clean_heads, y_v_clean = clean_predictions
    pv_clean = clean_heads["v_class_logits"]
    pred_v_clean = pv_clean.argmax(-1)

    y_j_clean = np.array([jclass[_first(r["j_call"])] for r in clean_rows])
    pj_clean = clean_heads["j_class_logits"]
    pred_j_clean = pj_clean.argmax(-1)

    clean_v_acc = float((pred_v_clean == y_v_clean).mean())
    clean_j_acc = float((pred_j_clean == y_j_clean).mean())
    v_macro = macro_accuracy(y_v_clean, pred_v_clean)
    j_macro = macro_accuracy(y_j_clean, pred_j_clean)

    # ---- D on clean (allele + gene; gene = allele name before "*") ----
    d_indices = [i for i,r in enumerate(clean_rows)
                 if cfg.has_d and not r["d_inverted"] and _first(r["d_call"]) in d_index]
    d_rows = [clean_rows[i] for i in d_indices]
    if d_rows:
        y_d = np.array([d_index[_first(r["d_call"])] for r in d_rows])
        pd_ = clean_heads["d_class_logits"][d_indices]
        pred_d = pd_.argmax(-1)
        d_allele_acc = float((pred_d == y_d).mean())
        d_allele_macro = macro_accuracy(y_d, pred_d)
        true_gene = np.array([d_name[i].split("*")[0] for i in y_d])
        pred_gene = np.array([d_name[i].split("*")[0] for i in pred_d])
        d_gene_acc = float((pred_gene == true_gene).mean())
        d_nll, d_brier = categorical_nll(pd_, y_d), multiclass_brier_score(pd_, y_d)
    else:
        d_allele_acc = d_allele_macro = d_gene_acc = d_nll = d_brier = None

    nll = {"v": categorical_nll(pv_clean, y_v_clean),
           "j": categorical_nll(pj_clean, y_j_clean), "d": d_nll}
    brier = {"v": multiclass_brier_score(pv_clean, y_v_clean),
             "j": multiclass_brier_score(pj_clean, y_j_clean), "d": d_brier}

    freq_bins = {
        "v": frequency_accuracy_report(
            y_v_clean, pred_v_clean, _freq_for(clean_rows, "V", vclass, cohort["dc_name"]), freq_edges),
        "j": frequency_accuracy_report(
            y_j_clean, pred_j_clean, _freq_for(clean_rows, "J", jclass, cohort["dc_name"]), freq_edges),
    }

    with error_context(stratum="clean"):
        boundary_mae = _boundary_mae(model, cfg, boundary_samples, device)
    with error_context(stratum="__calibration__"):
        conformal = _conformal_v(model, cfg, device, class_maps, calibration_rows, clean_rows, alpha,
                                  test_probabilities=pv_clean)

    return {
        "clean_v_acc": clean_v_acc, "clean_j_acc": clean_j_acc,
        "v_macro": v_macro, "j_macro": j_macro,
        "d_allele_acc": d_allele_acc, "d_allele_macro": d_allele_macro, "d_gene_acc": d_gene_acc,
        "by_stratum": by_stratum,
        "freq_bins": freq_bins,
        "boundary_mae": boundary_mae,
        "nll": nll, "brier": brier,
        "conformal": conformal,
    }


def _class_probabilities(model, sequences, config, device, keys):
    probabilities = protein_class_probabilities(model, sequences, config, device, keys)
    for key, values in probabilities.items():
        gene = key.removesuffix("_class_logits")
        expected = getattr(config, f"{gene}_allele_count")
        if values.shape[1] != expected:
            raise InferenceError("categorical output width differs from its scoring vocabulary",
                                 context={"field": key, "expected_width": expected, "actual_width": values.shape[1]})
    return probabilities


def evaluate_checkpoint(model, cfg, class_maps, meta, cohort, *, device: str = "cpu",
                        alpha: float = 0.05, freq_edges=_DEFAULT_FREQ_EDGES) -> dict:
    """Preflight all consumed inputs, then score while restoring every caller module mode.

    Model placement on the requested device remains explicit and persistent. Modes/gradient state
    are restored on success/failure; weights and cohort records are not updated by scoring.
    """
    samples = prepare_evaluation(model, cfg, class_maps, meta, cohort, device, alpha, freq_edges)
    with error_context(stage="convergence_evaluation"), evaluation_mode(model):
        model.to(device)
        return _evaluate_checkpoint(model, cfg, class_maps, meta, cohort, samples, device=device,
                                    alpha=alpha, freq_edges=freq_edges)
