"""Case metadata and special-condition scoring."""
from __future__ import annotations

from typing import Any

from ..adapters import normalize_call_set
from ...core.schema import BenchmarkCase, GENES
from .primitives import as_bool, as_float, avg, coord, pred_value, string_exact


def score_metadata(pred: dict[str, Any], case: BenchmarkCase) -> dict[str, float]:
    record = case.record
    out: dict[str, float] = {}

    pred_locus = pred_value(pred, "locus", "chain_type")
    truth_locus = record.get("locus")
    if pred_locus is not None and truth_locus:
        out["locus_acc"] = 1.0 if str(pred_locus).upper() == str(truth_locus).upper() else 0.0
        out["chain_type_acc"] = out["locus_acc"]

    truth_has_d = bool(case.genes.get("d") and case.genes["d"].calls)
    if "d_call" in pred or "d_calls" in pred:
        out["has_d_routing_acc"] = 1.0 if bool(normalize_call_set(pred, "d")) == truth_has_d else 0.0

    pred_c = pred_value(pred, "c_call", "constant_call")
    truth_c = record.get("c_call")
    if truth_c:
        out["c_found_rate"] = 1.0 if pred_c is not None else 0.0
        exact = string_exact(pred_c, truth_c)
        if exact is not None:
            out["c_call_acc"] = exact
            out["c_gene_acc"] = 1.0 if str(pred_c).split("*")[0] == str(truth_c).split("*")[0] else 0.0

    pred_d_inv_raw = pred_value(pred, "d_inverted", "d_orientation")
    pred_d_inv = as_bool(pred_d_inv_raw)
    truth_d_inv = as_bool(record.get("d_inverted"))
    if pred_d_inv is not None and truth_d_inv is not None:
        out["d_inversion_acc"] = 1.0 if pred_d_inv == truth_d_inv else 0.0

    for key, metric in (("vj_in_frame", "vj_in_frame_acc"), ("stop_codon", "stop_codon_acc")):
        pred_bool = as_bool(pred_value(pred, key))
        truth_bool = as_bool(record.get(key))
        if pred_bool is not None and truth_bool is not None:
            out[metric] = 1.0 if pred_bool == truth_bool else 0.0

    for key, metric in (
        ("n_fwr1_mutations", "fwr1_mutation_count_mae"),
        ("n_cdr1_mutations", "cdr1_mutation_count_mae"),
        ("n_fwr2_mutations", "fwr2_mutation_count_mae"),
        ("n_cdr2_mutations", "cdr2_mutation_count_mae"),
        ("n_fwr3_mutations", "fwr3_mutation_count_mae"),
    ):
        pred_count = as_float(pred_value(pred, key, key.removeprefix("n_")))
        truth_count = as_float(record.get(key))
        if pred_count is not None and truth_count is not None:
            out[metric] = abs(pred_count - truth_count)

    pred_layout = pred_value(pred, "read_layout")
    truth_layout = record.get("read_layout")
    if pred_layout is not None and truth_layout:
        out["layout_specific_call_acc"] = 1.0 if str(pred_layout) == str(truth_layout) else 0.0

    truth_contaminant = as_bool(record.get("is_contaminant"))
    pred_contaminant = as_bool(pred_value(pred, "is_contaminant", "contaminant"))
    if truth_contaminant:
        any_call = any(normalize_call_set(pred, gene) for gene in GENES)
        flagged = bool(pred_contaminant)
        out["contaminant_no_call_rate"] = 0.0 if any_call else 1.0
        out["contaminant_handled_rate"] = 1.0 if (flagged or not any_call) else 0.0
        out["false_positive_alignment_rate"] = 1.0 if (any_call and not flagged) else 0.0
    if pred_contaminant is not None and truth_contaminant is not None:
        out["contaminant_flag_acc"] = 1.0 if pred_contaminant == truth_contaminant else 0.0

    truth_revision = as_bool(record.get("receptor_revision_applied"))
    pred_revision = as_bool(pred_value(pred, "receptor_revision_applied", "revision_applied"))
    if pred_revision is not None and truth_revision is not None:
        out["revision_flag_acc"] = 1.0 if pred_revision == truth_revision else 0.0
    if truth_revision:
        truth_v = case.genes.get("v")
        if truth_v and truth_v.calls:
            pred_v = normalize_call_set(pred, "v")
            out["revision_case_call_acc"] = 1.0 if pred_v and pred_v[0] in set(truth_v.calls) else 0.0
        boundary_errs = []
        for suffix in ("ss", "se"):
            p = coord(pred, "v", suffix)
            t = truth_v.sequence_start if suffix == "ss" else truth_v.sequence_end
            if p is not None and t is not None:
                boundary_errs.append(abs(p - t))
        if boundary_errs:
            out["revision_case_boundary_mae"] = avg(boundary_errs)

    return out
