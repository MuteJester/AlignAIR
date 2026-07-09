from __future__ import annotations

from typing import Any

from ...core.schema import BenchmarkCase


def case_to_prediction(case: BenchmarkCase, frame: str = "canonical", set_calls: bool = True) -> dict:
    """Return a perfect prediction dict for a benchmark case."""

    pred: dict[str, Any] = {}
    for g, truth in case.truth(frame).items():
        if not truth.present:
            continue
        if set_calls:
            pred[f"{g}_calls"] = list(truth.calls)
        pred[f"{g}_call"] = truth.primary or (truth.calls[0] if truth.calls else None)
        pred[f"{g}_sequence_start"] = truth.sequence_start
        pred[f"{g}_sequence_end"] = truth.sequence_end
        pred[f"{g}_germline_start"] = truth.germline_start
        pred[f"{g}_germline_end"] = truth.germline_end
        for key in (
            f"{g}_cigar",
            f"{g}_identity",
            f"{g}_score",
            f"{g}_support",
            f"{g}_trim_5",
            f"{g}_trim_3",
        ):
            if key in case.record:
                pred[key] = case.record[key]
    pred["orientation_id"] = case.orientation_id
    pred["region_labels"] = case.labels("region", frame)
    pred["state_labels"] = case.labels("state", frame)
    pred.update(case.scalars)
    pred["sequence_id"] = case.case_id
    pred["sequence"] = case.canonical_sequence if frame == "canonical" else case.sequence
    pred["rev_comp"] = case.orientation_id == 1
    for key in (
        "locus",
        "c_call",
        "junction",
        "junction_aa",
        "junction_start",
        "junction_end",
        "junction_length",
        "np1",
        "np2",
        "np1_aa",
        "np2_aa",
        "np1_length",
        "np2_length",
        "p_v_3_length",
        "p_d_5_length",
        "p_d_3_length",
        "p_j_5_length",
        "vj_in_frame",
        "stop_codon",
        "d_inverted",
        "read_layout",
        "is_contaminant",
        "receptor_revision_applied",
        "original_v_call",
        "n_fwr1_mutations",
        "n_cdr1_mutations",
        "n_fwr2_mutations",
        "n_cdr2_mutations",
        "n_fwr3_mutations",
        "n_mutations",
        "n_v_mutations",
        "n_d_mutations",
        "n_j_mutations",
        "n_indels",
        "n_v_indels",
        "n_d_indels",
        "n_j_indels",
    ):
        if key in case.record:
            pred[key] = case.record[key]
    if frame == "presented" and case.orientation_id in (1, 3):
        start = case.record.get("junction_start")
        end = case.record.get("junction_end")
        if start is not None and end is not None:
            length = len(case.canonical_sequence)
            pred["junction_start"] = length - end
            pred["junction_end"] = length - start
    return pred
