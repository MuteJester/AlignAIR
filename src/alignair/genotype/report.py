"""The explainable genotype report — per-gene calls with counts, evidence, reasons, and novel
candidates (positions + provisional sequence + source/coverage mask). All support/coverage lives here
(the AIRR genotype schema has none). This is the "why/how from the model geometry" artifact."""
from __future__ import annotations


def build_report(gene_calls, *, meta=None) -> dict:
    genes = {}
    for gc in gene_calls:
        genes[gc.gene] = {
            "zygosity": gc.zygosity,
            "deletion_candidate": gc.deletion_candidate,
            "alleles": [dict(a) for a in gc.alleles],
            "novel_candidates": [
                {"near": nv.get("near"), "positions": nv.get("positions", []),
                 "sequence": nv.get("sequence"), "source_mask": nv.get("source_mask", [])}
                for nv in gc.novel],
            "reasons": list(gc.reasons),
        }
    return {"schema": "alignair.genotype.report.v1", "meta": dict(meta or {}), "genes": genes}


def format_report(report: dict) -> str:
    L = ["AlignAIR genotype report", "=" * 60]
    if report.get("meta"):
        L.append("  " + ", ".join(f"{k}={v}" for k, v in report["meta"].items()))
    for gene, g in report["genes"].items():
        tag = f"  [{g['zygosity']}]" + ("  (deletion candidate)" if g["deletion_candidate"] else "")
        L.append(f"\n{gene.upper()}{tag}")
        for a in g["alleles"]:
            extra = []
            if "count" in a:
                extra.append(f"n={a['count']}")
            if "usage_fraction" in a:
                extra.append(f"usage={a['usage_fraction']:.1%}")
            if "mean_confidence" in a:
                extra.append(f"conf={a['mean_confidence']:.2f}")
            extra.append("diagnostic-SNP" if a.get("evidence") else "support-only")
            L.append(f"  + {a['name']}  ({', '.join(extra)})")
        for nv in g["novel_candidates"]:
            L.append(f"  ~ novel candidate near {nv.get('near')}: positions {nv.get('positions')} (provisional)")
        for r in g["reasons"]:
            L.append(f"      · {r}")
    return "\n".join(L)
