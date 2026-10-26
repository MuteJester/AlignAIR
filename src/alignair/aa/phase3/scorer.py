"""Phase-3 scorer. Binding = one rep/lineage; descriptive = all dedup VHs; bootstrap = lineage clusters.
Gold family endpoint never emits 'supported'. Concordance is alignment-derived, not ground truth."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..taxonomy import class_families, class_genes, catalogue_families
from .stats import clopper_pearson, coverage_decision

@dataclass
class Prediction:
    top_class: int
    set_classes: set

def _check(rows, preds):
    if len(rows) != len(preds):
        raise ValueError(f"rows ({len(rows)}) and preds ({len(preds)}) length mismatch")
    if not rows:
        raise ValueError("empty cohort")

def _is_rep(r):
    return str(r.lineage_representative).lower() == "true"

def _set_family(p, cmap):
    return set().union(*[class_families(c, cmap) for c in p.set_classes]) if p.set_classes else set()

def _set_gene(p, cmap):
    return set().union(*[class_genes(c, cmap) for c in p.set_classes]) if p.set_classes else set()

def _sizes(preds, cmap):
    def stat(vals):
        a = np.asarray(vals, float)
        return {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95))}
    classes = [len(p.set_classes) for p in preds]
    fams = [len(_set_family(p, cmap)) for p in preds]
    genes = [len(_set_gene(p, cmap)) for p in preds]
    alleles = [len(set().union(*[cmap.class_members[c] for c in p.set_classes]) if p.set_classes else set())
               for p in preds]
    return {"classes": stat(classes), "families": stat(fams), "genes": stat(genes), "alleles": stat(alleles)}

def score_gold(rows, preds, cmap, alpha, seed, n_boot: int = 2000) -> dict:
    _check(rows, preds)
    cat_fams = catalogue_families(cmap)
    # binding: representatives only
    ridx = [i for i, r in enumerate(rows) if _is_rep(r)]
    if not ridx:
        raise ValueError("cohort has no lineage-representative rows")
    lids = [rows[i].lineage_id for i in ridx]
    top = {rows[i].lineage_id: (rows[i].true_v_family in class_families(preds[i].top_class, cmap)) for i in ridx}
    cov = {rows[i].lineage_id: (rows[i].true_v_family in _set_family(preds[i], cmap)) for i in ridx}
    n = len(ridx); kt = sum(top.values()); kc = sum(cov.values())
    # lineage-cluster bootstrap of the binding top-class accuracy
    rng = np.random.default_rng(seed); uniq = list(dict.fromkeys(lids)); boot = []
    for _ in range(n_boot):
        samp = rng.choice(len(uniq), len(uniq), replace=True)
        boot.append(np.mean([top[uniq[j]] for j in samp]))
    # descriptive: all deduplicated rows
    all_hits = [r.true_v_family in class_families(p.top_class, cmap) for r, p in zip(rows, preds)]
    # oov / in-catalog conditional accuracy
    non_oov = [(r, p) for r, p in zip(rows, preds) if r.true_v_family in cat_fams]
    in_cat_hits = [r.true_v_family in class_families(p.top_class, cmap) for r, p in non_oov]
    return {
        "binding": {
            "n_lineages": n,
            "top_class_family_accuracy": kt / n,
            "top_class_family_ci": list(clopper_pearson(kt, n, alpha)),
            "family_set_coverage": kc / n,
            "family_set_coverage_ci": list(clopper_pearson(kc, n, alpha)),
            "decision": coverage_decision(kc, n, alpha, allow_support=False),
        },
        "descriptive": {
            "n_all_vh": len(rows),
            "top_class_family_accuracy_all": float(np.mean(all_hits)),
            "bootstrap_ci_family_accuracy": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        },
        "oov": {
            "oov_rate": 1 - len(non_oov) / len(rows),
            "in_catalog_accuracy": (float(np.mean(in_cat_hits)) if in_cat_hits else None),
            "end_to_end_accuracy": float(np.mean(all_hits)),
        },
        "set_size": _sizes(preds, cmap),
    }

def score_concordance(rows, preds, cmap, alpha) -> dict:
    _check(rows, preds)
    ridx = [i for i, r in enumerate(rows) if _is_rep(r)]
    if not ridx:
        raise ValueError("cohort has no lineage-representative rows")
    agree = [rows[i].true_v_gene in class_genes(preds[i].top_class, cmap) for i in ridx]
    cov = [rows[i].true_v_gene in _set_gene(preds[i], cmap) for i in ridx]
    n = len(ridx); ka = sum(agree); kc = sum(cov)
    return {
        "label_kind": "alignment-derived (IMGT/ANARCI) — concordance, NOT ground truth",
        "n_lineages": n,
        "top_class_gene_agreement": ka / n if n else 0.0,
        "agreement_ci": list(clopper_pearson(ka, n, alpha)) if n else [0.0, 1.0],
        "gene_set_coverage": kc / n if n else 0.0,
        "gene_set_coverage_ci": list(clopper_pearson(kc, n, alpha)) if n else [0.0, 1.0],
        "set_size": _sizes([preds[i] for i in ridx], cmap),
    }
