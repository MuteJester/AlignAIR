"""Scenario/context labels used for sliced benchmark reporting."""
from __future__ import annotations

from ..core import ORIENTATION_NAMES, BenchmarkCase


def _bin(value: float, edges: tuple[float, ...], labels: tuple[str, ...]) -> str:
    for edge, label in zip(edges, labels):
        if value <= edge:
            return label
    return labels[-1]


def case_contexts(case: BenchmarkCase) -> list[str]:
    """Return stable scenario labels contributed by one benchmark case."""

    tags = case.tags or {}
    record = case.record or {}
    contexts = [f"stratum:{case.stratum}"]
    for tag in tags.get("stratum_tags", ()):
        contexts.append(f"tag:{tag}")
    if record.get("locus"):
        contexts.append(f"locus:{record['locus']}")
    contexts.append("chain:has_d" if case.genes.get("d") and case.genes["d"].calls else "chain:no_d")
    contexts.append(f"orientation:{ORIENTATION_NAMES.get(case.orientation_id, case.orientation_id)}")
    contexts.append(
        "length:"
        + _bin(len(case.sequence), (60, 90, 130, 250), ("<=60", "61-90", "91-130", "131-250", ">250"))
    )
    contexts.append(
        "mutation:"
        + _bin(
            float(tags.get("mutation_rate", 0.0)),
            (0.01, 0.05, 0.12, 0.18),
            ("<=1%", "1-5%", "5-12%", "12-18%", ">18%"),
        )
    )
    contexts.append("indel:" + _bin(float(tags.get("n_indels", 0.0)), (0, 2, 5), ("0", "1-2", "3-5", ">5")))
    noise = float(tags.get("n_quality_errors", 0) or 0) + float(tags.get("n_pcr_errors", 0) or 0)
    contexts.append("noise:" + _bin(noise, (0, 2, 8), ("0", "1-2", "3-8", ">8")))
    contexts.append("productivity:yes" if tags.get("productive") else "productivity:no")
    if case.genes.get("d") and case.genes["d"].calls:
        contexts.append("d_orientation:inverted" if tags.get("d_inverted") else "d_orientation:forward")
    else:
        contexts.append("d_orientation:not_applicable")
    layout = record.get("read_layout") or "single"
    contexts.append(f"read_layout:{layout}")
    contexts.append("contaminant:yes" if record.get("is_contaminant") else "contaminant:no")
    contexts.append("revision:yes" if record.get("receptor_revision_applied") else "revision:no")
    contexts.append("constant_region:present" if record.get("c_call") else "constant_region:absent")
    junction_len = int(record.get("junction_length") or 0)
    contexts.append(
        "junction_length:"
        + _bin(
            junction_len,
            (30, 75, 120),
            ("short_junction", "typical_junction", "long_junction", "very_long_junction"),
        )
    )
    contexts.append("junction_frame:in_frame" if record.get("vj_in_frame") else "junction_frame:out_of_frame")
    if record.get("stop_codon"):
        contexts.append("junction_frame:stop_codon")

    visible = []
    v = case.genes.get("v")
    d = case.genes.get("d")
    j = case.genes.get("j")
    if v and v.sequence_start is not None and v.sequence_end is not None and (v.sequence_end - v.sequence_start) < 80:
        visible.append("short_v_tail")
    if d and d.sequence_start is not None and d.sequence_end is not None and (d.sequence_end - d.sequence_start) < 8:
        visible.append("short_d")
    if j and j.sequence_start is not None and j.sequence_end is not None and (j.sequence_end - j.sequence_start) < 20:
        visible.append("short_j_head")
    if not visible:
        visible.append("all_segments_visible")
    contexts.extend(f"segment_presence:{name}" for name in visible)

    any_multi = False
    for g, truth in case.genes.items():
        if len(truth.calls) > 1:
            any_multi = True
            contexts.append(f"ambiguity:{g}_multi")
        else:
            contexts.append(f"ambiguity:{g}_single")
    contexts.append("ambiguity:any_multi" if any_multi else "ambiguity:all_single")
    return contexts
