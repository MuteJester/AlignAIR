"""Summarize an AIRR rearrangement TSV: repertoire composition + prediction QC + validation."""
from __future__ import annotations

import csv
import re
import statistics
from collections import Counter

from .validate import validate_airr

_CIG = re.compile(r"\d+([MIDNSX=])")
_LOW_CONF = 0.5


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    return v is not None and str(v).strip().lower() in {"t", "true", "1", "yes", "y"}


def _num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _first(v) -> str:
    return str(v).split(",")[0].strip() if v else ""


def _has_indel(cigar) -> bool:
    return any(op in "ID" for op in _CIG.findall(cigar or ""))


def _stat(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "mean": round(statistics.mean(vals), 3),
            "min": round(min(vals), 3), "max": round(max(vals), 3)}


def _len_summary(vals: list[int]) -> dict:
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "min": min(vals), "max": max(vals),
            "mean": round(statistics.mean(vals), 1), "median": statistics.median(vals),
            "histogram": _histogram(vals)}


def _histogram(vals: list[int], width: int = 6) -> dict:
    c = Counter((v // width) * width for v in vals)
    return {f"{k}-{k + width - 1}": c[k] for k in sorted(c)}


def repertoire(rows: list[dict]) -> dict:
    n = len(rows)
    def pct(x):
        return round(100.0 * x / n, 1) if n else 0.0
    prod = sum(_truthy(r.get("productive")) for r in rows)
    inframe = sum(_truthy(r.get("vj_in_frame")) for r in rows)
    stop = sum(_truthy(r.get("stop_codon")) for r in rows)
    usage = {g: Counter(_first(r.get(f"{g}_call")) for r in rows if _first(r.get(f"{g}_call"))).most_common(10)
             for g in ("v", "d", "j")}
    jlens = [int(x) for x in (_num(r.get("junction_length")) for r in rows) if x]
    junctions_aa = {r.get("junction_aa") for r in rows if r.get("junction_aa")}
    return {"n_reads": n, "productive": {"n": prod, "pct": pct(prod)},
            "vj_in_frame": {"n": inframe, "pct": pct(inframe)},
            "stop_codon": {"n": stop, "pct": pct(stop)},
            "gene_usage": usage, "cdr3_length": _len_summary(jlens),
            "unique_junctions_aa": len(junctions_aa)}


def qc(rows: list[dict]) -> dict:
    n = len(rows)
    orient = Counter("reoriented" if _truthy(r.get("rev_comp")) else "forward" for r in rows)
    indel = sum(1 for r in rows if any(_has_indel(r.get(f"{g}_cigar")) for g in ("v", "d", "j")))
    confs = [x for x in (_num(r.get("v_set_confidence")) for r in rows) if x is not None]
    ident = [x for x in (_num(r.get("v_identity")) for r in rows) if x is not None]
    fields = ["v_call", "j_call", "junction", "cdr3", "sequence_alignment"]
    comp = {f: (round(100.0 * sum(1 for r in rows if r.get(f)) / n, 1) if n else 0.0) for f in fields}
    return {"orientation": dict(orient), "indel_flagged_reads": indel,
            "v_set_confidence": _stat(confs), "v_identity": _stat(ident),
            "low_confidence_reads": {"n": sum(1 for x in confs if x < _LOW_CONF), "threshold": _LOW_CONF},
            "field_completeness_pct": comp}


def analyze_rows(rows: list[dict], columns: list[str] | None = None) -> dict:
    cols = columns or (list(rows[0].keys()) if rows else [])
    return {"repertoire": repertoire(rows), "qc": qc(rows), "validation": validate_airr(rows, cols)}


def analyze_file(path: str) -> dict:
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        cols = reader.fieldnames or []
    return {"input": path, **analyze_rows(rows, cols)}


def format_text(report: dict) -> str:
    rep, q, val = report["repertoire"], report["qc"], report["validation"]
    L = [f"AlignAIR analyze — {report.get('input', '')}", "=" * 60, "REPERTOIRE",
         f"  reads:            {rep['n_reads']}",
         f"  productive:       {rep['productive']['n']} ({rep['productive']['pct']}%)",
         f"  vj_in_frame:      {rep['vj_in_frame']['n']} ({rep['vj_in_frame']['pct']}%)",
         f"  stop_codon:       {rep['stop_codon']['n']} ({rep['stop_codon']['pct']}%)",
         f"  unique junctions: {rep['unique_junctions_aa']}"]
    for g in ("v", "d", "j"):
        top = ", ".join(f"{name} ({c})" for name, c in rep["gene_usage"][g][:5]) or "-"
        L.append(f"  {g.upper()} usage:         {top}")
    cl = rep["cdr3_length"]
    if cl.get("n"):
        L.append(f"  CDR3 length (nt): min {cl['min']} / median {cl['median']} / mean {cl['mean']} / max {cl['max']}")
    L.append("QC")
    L.append("  orientation:      " + (", ".join(f"{k} {v}" for k, v in q["orientation"].items()) or "-"))
    L.append(f"  indel-flagged:    {q['indel_flagged_reads']}")
    conf, idn = q["v_set_confidence"], q["v_identity"]
    L.append("  V confidence:     " + (f"mean {conf['mean']} ({conf['min']}–{conf['max']})" if conf.get("n") else "n/a"))
    L.append("  V identity:       " + (f"mean {idn['mean']}" if idn.get("n") else "n/a"))
    L.append(f"  low-confidence:   {q['low_confidence_reads']['n']} (< {q['low_confidence_reads']['threshold']})")
    L.append("  completeness:     " + ", ".join(f"{k} {v}%" for k, v in q["field_completeness_pct"].items()))
    L.append("VALIDATION")
    L.append("  required columns: " + ("OK" if not val["missing_required_columns"]
                                       else "missing " + ", ".join(val["missing_required_columns"])))
    L.append(f"  coord violations: {val['coord_violations']}")
    L.append(f"  junction-length violations: {val['junction_length_violations']}")
    L.append(f"  => {'VALID' if val['valid'] else 'ISSUES FOUND'}")
    return "\n".join(L)
