"""Tool-to-tool AIRR agreement on the USER's own data (no ground truth needed): compare two
rearrangement TSVs (e.g. AlignAIR vs IgBLAST or MiXCR `exportAirr`) row-by-row on `sequence_id`
and report per-gene call agreement, junction/productivity concordance, coverage, and — uniquely —
how often AlignAIR's equivalence SET already contains the other tool's call (so an
allele "disagreement" is really shared ambiguity, not a conflict).

This turns "should I trust AlignAIR over my current tool?" into an evidence report on real data.
"""
from __future__ import annotations

import csv
from typing import Dict, List

GENES = ("v", "d", "j")


def read_airr(path: str) -> Dict[str, dict]:
    """Read an AIRR rearrangement TSV into {sequence_id: row}. Tolerates extra columns and the
    AlignAIR equivalence-set extension columns."""
    with open(path, newline="") as f:
        delim = "\t" if "\t" in (f.readline()) else ","
        f.seek(0)
        return {r["sequence_id"]: r for r in csv.DictReader(f, delimiter=delim) if r.get("sequence_id")}


def _top1(row: dict, gene: str):
    call = (row.get(f"{gene}_call") or "").strip()
    alleles = [c for c in call.split(",") if c]
    return alleles[0] if alleles else None


def _set(row: dict, gene: str) -> set:
    raw = (row.get(f"{gene}_call_set") or row.get(f"{gene}_calls") or row.get(f"{gene}_call") or "")
    return {c for c in raw.replace(";", ",").split(",") if c}


def _gene(call):
    return call.split("*")[0] if call else None


def _norm_bool(v):
    s = str(v).strip().lower()
    return s in ("t", "true", "1", "yes")


def compare_airr(a: Dict[str, dict], b: Dict[str, dict], a_name="model_a", b_name="model_b",
                 n_examples: int = 10) -> dict:
    """Compare two {sequence_id: row} maps. `a` is treated as AlignAIR for set-rescue (the fraction
    of allele disagreements where b's call is inside a's equivalence set)."""
    common = sorted(set(a) & set(b))
    out = {
        "model_a": a_name, "model_b": b_name,
        "coverage": {"a_total": len(a), "b_total": len(b), "matched": len(common),
                     "a_only": len(set(a) - set(b)), "b_only": len(set(b) - set(a))},
        "genes": {}, "examples_disagree": {},
    }
    for g in GENES:
        n = allele_agree = gene_agree = set_rescue = disagree = 0
        examples = []
        for sid in common:
            ta, tb = _top1(a[sid], g), _top1(b[sid], g)
            if not ta or not tb:                      # only score where BOTH tools called this gene
                continue
            n += 1
            if ta == tb:
                allele_agree += 1
            else:
                disagree += 1
                if tb in _set(a[sid], g):             # b's call is inside AlignAIR's equivalence set
                    set_rescue += 1
                if len(examples) < n_examples:
                    examples.append({"sequence_id": sid, a_name: ta, b_name: tb})
            gene_agree += (_gene(ta) == _gene(tb))
        out["genes"][g] = {
            "both_called": n,
            "allele_agreement": round(allele_agree / n, 4) if n else None,
            "gene_agreement": round(gene_agree / n, 4) if n else None,
            "set_rescue_rate": round(set_rescue / disagree, 4) if disagree else None,
        }
        out["examples_disagree"][g] = examples

    # junction nt + productivity concordance (where both present)
    jn = jt = pn = pt = 0
    for sid in common:
        ja, jb = (a[sid].get("junction") or "").upper(), (b[sid].get("junction") or "").upper()
        if ja and jb:
            jt += 1; jn += (ja == jb)
        pa, pb = a[sid].get("productive"), b[sid].get("productive")
        if pa not in (None, "") and pb not in (None, ""):
            pt += 1; pn += (_norm_bool(pa) == _norm_bool(pb))
    out["junction_nt_agreement"] = round(jn / jt, 4) if jt else None
    out["productive_agreement"] = round(pn / pt, 4) if pt else None
    return out


def format_report_md(r: dict) -> str:
    a, b = r["model_a"], r["model_b"]
    c = r["coverage"]
    lines = [f"# AIRR agreement: {a} vs {b}", "",
             f"Matched **{c['matched']}** sequences by `sequence_id` "
             f"({a}: {c['a_total']}, {b}: {c['b_total']}; {a}-only {c['a_only']}, {b}-only {c['b_only']}).",
             "", "## Call agreement (rows where both tools called the gene)", "",
             "| gene | n | allele agreement | gene agreement | set-rescue of disagreements |",
             "| --- | --- | --- | --- | --- |"]
    for g in GENES:
        m = r["genes"][g]
        def pct(x): return f"{x*100:.1f}%" if x is not None else "n/a"
        lines.append(f"| {g.upper()} | {m['both_called']} | {pct(m['allele_agreement'])} | "
                     f"{pct(m['gene_agreement'])} | {pct(m['set_rescue_rate'])} |")
    jn = r["junction_nt_agreement"]; pr = r["productive_agreement"]
    lines += ["", f"- junction (nt) agreement: {jn*100:.1f}%" if jn is not None else "- junction: n/a",
              f"- productivity agreement: {pr*100:.1f}%" if pr is not None else "- productivity: n/a",
              "",
              f"**set-rescue** = of the cases where the top-1 allele differs, how often {b}'s call is "
              f"inside {a}'s equivalence set (i.e. shared ambiguity, not a true conflict).",
              "", "## Example allele disagreements", ""]
    for g in GENES:
        ex = r["examples_disagree"][g]
        if ex:
            lines.append(f"**{g.upper()}** (first {len(ex)}):")
            for e in ex:
                lines.append(f"- `{e['sequence_id']}`: {a}={e[a]}  {b}={e[b]}")
            lines.append("")
    return "\n".join(lines)
