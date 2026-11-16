"""Summarize an AIRR rearrangement TSV: repertoire composition + prediction QC + validation."""
from __future__ import annotations

import statistics
from collections import Counter

from ..reference.genes import GENES
from ..predict.germline.cigar import cigar_ops
from .tables import first_call, num, read_airr_table, truthy
from .validate import validate_airr



def _has_indel(cigar) -> bool:
    return any(op in "ID" for _, op in cigar_ops(cigar))


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
    """The repertoire's composition: productive, in-frame and stop-codon shares, the top genes, the CDR3 lengths and
    the unique junctions."""
    n = len(rows)
    def pct(x):
        return round(100.0 * x / n, 1) if n else 0.0
    prod = sum(truthy(r.get("productive")) for r in rows)
    inframe = sum(truthy(r.get("vj_in_frame")) for r in rows)
    stop = sum(truthy(r.get("stop_codon")) for r in rows)
    usage = {g: Counter(first_call(r.get(f"{g}_call")) for r in rows if first_call(r.get(f"{g}_call"))).most_common(10)
             for g in GENES}
    jlens = [int(x) for x in (num(r.get("junction_length")) for r in rows) if x]
    junctions_aa = {r.get("junction_aa") for r in rows if r.get("junction_aa")}
    return {"n_reads": n, "productive": {"n": prod, "pct": pct(prod)},
            "vj_in_frame": {"n": inframe, "pct": pct(inframe)},
            "stop_codon": {"n": stop, "pct": pct(stop)},
            "gene_usage": usage, "cdr3_length": _len_summary(jlens),
            "unique_junctions_aa": len(junctions_aa)}


def _orientation(row: dict) -> str:
    """The transform the read needed: AlignAIR's ``orientation`` column (forward, reverse_complement, complement,
    reverse), else ``rev_comp`` (a table from another tool)."""
    if row.get("orientation"):
        return row["orientation"]
    return "reverse_complement" if truthy(row.get("rev_comp")) else "forward"


def qc(rows: list[dict]) -> dict:
    """Prediction QC: the orientations, the reads with an indel, the V identity and each key field's completeness."""
    n = len(rows)
    orient = Counter(_orientation(r) for r in rows)
    indel = sum(1 for r in rows if any(_has_indel(r.get(f"{g}_cigar")) for g in GENES))
    ident = [x for x in (num(r.get("v_identity")) for r in rows) if x is not None]
    fields = ["v_call", "j_call", "junction", "cdr3", "sequence_alignment"]
    comp = {f: (round(100.0 * sum(1 for r in rows if r.get(f)) / n, 1) if n else 0.0) for f in fields}
    return {"orientation": dict(orient), "indel_flagged_reads": indel,
            "v_identity": _stat(ident),
            "field_completeness_pct": comp}


def analyze_rows(rows: list[dict], columns: list[str] | None = None) -> dict:
    """The report of AIRR ``rows`` (with their ``columns``): repertoire, QC and AIRR validation."""
    cols = columns or (list(rows[0].keys()) if rows else [])
    return {"repertoire": repertoire(rows), "qc": qc(rows), "validation": validate_airr(rows, cols)}


def analyze_file(path: str) -> dict:
    """The report of the AIRR TSV at ``path`` (``analyze_rows``)."""
    rows, cols = read_airr_table(path)
    return {"input": path, **analyze_rows(rows, cols)}


def format_text(report: dict) -> str:
    """A report as plain text."""
    rep, q, val = report["repertoire"], report["qc"], report["validation"]
    L = [f"AlignAIR analyze — {report.get('input', '')}", "=" * 60, "REPERTOIRE",
         f"  reads:            {rep['n_reads']}",
         f"  productive:       {rep['productive']['n']} ({rep['productive']['pct']}%)",
         f"  vj_in_frame:      {rep['vj_in_frame']['n']} ({rep['vj_in_frame']['pct']}%)",
         f"  stop_codon:       {rep['stop_codon']['n']} ({rep['stop_codon']['pct']}%)",
         f"  unique junctions: {rep['unique_junctions_aa']}"]
    for g in GENES:
        top = ", ".join(f"{name} ({c})" for name, c in rep["gene_usage"][g][:5]) or "-"
        L.append(f"  {g.upper()} usage:         {top}")
    cl = rep["cdr3_length"]
    if cl.get("n"):
        L.append(f"  CDR3 length (nt): min {cl['min']} / median {cl['median']} / mean {cl['mean']} / max {cl['max']}")
    L.append("QC")
    L.append("  orientation:      " + (", ".join(f"{k} {v}" for k, v in q["orientation"].items()) or "-"))
    L.append(f"  indel-flagged:    {q['indel_flagged_reads']}")
    idn = q["v_identity"]
    L.append("  V identity:       " + (f"mean {idn['mean']}" if idn.get("n") else "n/a"))
    L.append("  completeness:     " + ", ".join(f"{k} {v}%" for k, v in q["field_completeness_pct"].items()))
    L.append("VALIDATION")
    L.append("  required columns: " + ("OK" if not val["missing_required_columns"]
                                       else "missing " + ", ".join(val["missing_required_columns"])))
    L.append(f"  coord violations: {val['coord_violations']}")
    L.append(f"  junction-length violations: {val['junction_length_violations']}")
    L.append(f"  => {'VALID' if val['valid'] else 'ISSUES FOUND'}")
    return "\n".join(L)
