"""``show_markdown``: one read side by side, as ``workflow.show`` looked it up (a plain dict), rendered as markdown.

A simulated case: the case and the simulator's calls, its observable truth (0-based, end-exclusive coordinates in the
canonical frame), each tool's AIRR row (as written: 1-based, inclusive), each tool's scores with the metrics on
which they differ marked ``DIFFERS`` (listed first), and each diagnosed model's stored scores the same way. A real
read: the read, its truth, each tool's AIRR row and its per-read classes (the R1 class first).
"""
from __future__ import annotations

import math

from alignair_assay.report.markdown import MISSING, markdown_table

DIFFERS = "≠"
# The per-read class of a real tool's scores the primary R1 reads, shown first.
R1_CLASS = "v"


def show_markdown(found: dict) -> str:
    """``workflow.show``'s ``show_case`` or ``show_read`` result as markdown."""
    lines = _sim(found) if found["kind"] == "sim" else _real(found)
    return "\n".join(lines) + "\n"


def _sim(found: dict) -> list[str]:
    case, truth = found["case"], found["truth"] or {}
    lines = [f"# {case['case_id']}: suite {found['suite']}, stratum {case['stratum']}, {found['arm']} arm", "",
             f"Run directory {found['run_dir']}.", "",
             f"- presented read ({_orientation(case.get('orientation'))}): {case['sequence']}"]
    simulated = case.get("simulated")
    if simulated:
        lines.append("- simulated calls: " + ", ".join(f"{name[0].upper()} {value or MISSING}"
                                                       for name, value in simulated.items()))
    if case.get("labels"):
        lines.append("- labels: " + ", ".join(f"{key} {value}" for key, value in case["labels"].items()))
    lines += ["", "## Observable truth", "", "0-based, end-exclusive, in the canonical (forward) frame; each edge is "
              "the interval of positions equally right.", ""]
    if not truth:
        lines.append("No truth row for this case.")
    else:
        lines += [f"- canonical read: {truth.get('sequence')}",
                  f"- excluded: {truth.get('excluded') or 'no'}; unscorable genes: "
                  f"{', '.join(truth.get('unscorable') or []) or 'none'}"]
        rows = [[gene.upper(), ", ".join((segment.get("alleles") or {}).get("alleles") or []) or MISSING,
                 *(_interval((segment.get("boundaries") or {}).get(edge)) for edge in (
                     "seq_start", "seq_end", "germ_start", "germ_end")),
                 str(segment.get("observed_length", MISSING)), "yes" if segment.get("inverted") else "no"]
                for gene, segment in (truth.get("segments") or {}).items()]
        lines += ["", *markdown_table(["Gene", "Alleles (scored set)", "Read start", "Read end", "Germline start",
                                       "Germline end", "Observed bases", "Inverted"], rows)]
        junction = truth.get("junction")
        lines += ["", "- junction: " + (MISSING if not junction else
                                        f"[{junction['start']}, {junction['end']}) {junction['nt']} "
                                        f"({junction.get('aa') or 'not a whole number of codons'})"),
                  f"- productive: {_text(truth.get('productive'))}; VJ in frame: {_text(truth.get('vj_in_frame'))}; "
                  f"stop codon: {_text(truth.get('stop_codon'))}"]
    lines += _tools(found, "Scores", "No tool ran on this suite and arm in this run directory.")
    diagnosed = found.get("diagnose") or {}
    lines += ["", "## Diagnosed models", ""]
    if not diagnosed:
        lines.append("No model was diagnosed on this suite in this run directory.")
    else:
        stale = [key for key, entry in diagnosed.items() if not entry["current"]]
        lines.append("Each model's stored per-case scores (`diagnose`; head metrics, then the end-to-end baseline), "
                     "read as stored: nothing is replayed."
                     + (f" Not current (the record no longer names the per-case file): {', '.join(stale)}."
                        if stale else ""))
        scores = {key: {**entry["head"], **(entry["baseline"] or {})} for key, entry in diagnosed.items()}
        lines += ["", *_score_table(scores, found.get("diagnose_differ") or [])]
    return lines


def _real(found: dict) -> list[str]:
    read, truth = found["read"], found["truth"]
    lines = [f"# {read['read_id']}: donor {found['donor']} ({read.get('protocol') or MISSING}, "
             f"{read.get('isotype') or MISSING}), {found['arm']} arm", "", f"Run directory {found['run_dir']}.", "",
             f"- read: {read.get('sequence') or MISSING}",
             f"- masked primers: {read.get('primer_masked') or MISSING}; consensus count "
             f"{read.get('consensus_count') or MISSING}, duplicates {read.get('dupcount') or MISSING}",
             "", "## Truth", ""]
    lines += [f"- observed: [{truth['observed'][0]}, {truth['observed'][1]})",
              f"- known: {_alleles(truth.get('known'))} (gene {truth.get('known_gene') or MISSING}; shared "
              f"{_alleles(truth.get('known_shared'))}; novel: {_text(truth.get('novel_known'))})",
              f"- ref_known: {_alleles(truth.get('ref_known'))} (gene {truth.get('ref_known_gene') or MISSING}; "
              f"donor alleles {_alleles(truth.get('ref_donor'))})",
              f"- clonotype: {truth.get('clonotype') or MISSING}; SL15: {truth.get('clonotype_sl15') or MISSING}",
              f"- anchor: {truth.get('anchor') or MISSING}; Cys104 ends at {_text(truth.get('cys_end'))}; naive: "
              f"{_text(truth.get('naive'))}"]
    lines += _tools(found, "Per-read classes", "No tool ran on this donor and arm in this run directory.")
    return lines


def _tools(found: dict, title: str, nothing: str) -> list[str]:
    """Each tool's AIRR row side by side, then its scores (the differing ones marked)."""
    tools = found["tools"]
    lines = ["", "## AIRR rows", ""]
    if not tools:
        return [*lines, nothing]
    names = list(tools)
    fields = list(dict.fromkeys(field for entry in tools.values() for field in (entry["row"] or {})))
    rows = [[field, *(_cell((tools[tool]["row"] or {}).get(field)) for tool in names)] for field in fields]
    missing = [tool for tool in names if tools[tool]["row"] is None]
    lines += ["As each tool wrote them (AIRR: 1-based, inclusive)."
              + (f" No row for this read: {', '.join(missing)}." if missing else ""), ""]
    lines += markdown_table(["Field", *names], rows) if rows else ["No row."]
    lines += ["", f"## {title}", ""]
    scores = {tool: entry["scores"] for tool, entry in tools.items()}
    unscored = [tool for tool, value in scores.items() if value is None]
    if unscored:
        lines += [f"Not scored: {', '.join(unscored)}.", ""]
    return lines + _score_table(scores, found.get("differ") or [], real=found["kind"] == "real")


def _score_table(scores: dict, differ: list[str], *, real: bool = False) -> list[str]:
    """Metric rows, one column per side, the differing metrics first and marked ``DIFFERS``; a real tool's R1
    class leads."""
    names = list(scores)
    metrics = list(dict.fromkeys(metric for side in scores.values() for metric in (side or {})))
    order = sorted(metrics, key=lambda metric: (not (real and metric == R1_CLASS), metric not in differ))
    rows = [[f"R1 class ({metric})" if real and metric == R1_CLASS else metric,
             *(_cell((scores[name] or {}).get(metric)) for name in names), DIFFERS if metric in differ else ""]
            for metric in order]
    return markdown_table(["Metric" if not real else "Field", *names, "Differs"], rows) if rows else ["No score."]


def _cell(value) -> str:
    if value is None or value == "":
        return MISSING
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(map(str, value)) or MISSING
    return str(value)


def _text(value) -> str:
    return MISSING if value is None else ("yes" if value is True else "no" if value is False else str(value))


def _alleles(values) -> str:
    return MISSING if values is None else ", ".join(values) or "none"


def _interval(pair) -> str:
    if not pair:
        return MISSING
    low, high = pair
    return str(low) if low == high else f"{low}..{high}"


def _orientation(value) -> str:
    return {0: "forward", 1: "reverse complement"}.get(value, f"orientation {value}")
