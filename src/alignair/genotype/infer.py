"""Orchestrator: repertoire -> genotype, wiring Stages 0-6.

Per gene-type (V/D/J): fit the leakage model (Task 1), aggregate weighted usage (Task 2), get residual
support; then per gene (allele group) resolve each allele's reads by the polymorphism test (Task 3),
call zygosity/CNV with evidence-gated pruning (Task 4), and assemble the report (Task 5) + AIRR
GenotypeSet (Task 6). Thresholds are conservative v1 defaults — calibrated by the benchmark (Task 8).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .aggregate import align_repertoire, read_weights, weighted_usage
from .airr import to_genotype_set
from .geometry import LeakageModel, allele_prototypes, prototype_cosine, residual_support
from .polymorphism import polymorphism_profile, resolve
from .report import build_report
from .zygosity import call_gene

_GENE_RE = re.compile(r"(.+?)\*")


def _gene_of(name: str) -> str:
    m = _GENE_RE.match(name)
    return m.group(1) if m else name


@dataclass
class GenotypeParams:
    """Tunable decision thresholds (calibrated per model/locus; see calibrate.py)."""
    min_support: float = 0.003
    present_thr: float = 0.002
    deletion_floor: float = 0.001
    max_novel_snps: int = 15
    leakage_cap: float = 0.5
    leakage_alpha: float = 1.0
    leakage_cos0: float = 0.5


@dataclass
class GenotypeResult:
    report: dict
    genotype_set: dict
    warnings: list


def _top1(rec, gene):
    return (rec.get(f"{gene}_call") or "").split(",")[0]


def _reads_for(records, gene, name):
    out = []
    for r in records:
        if _top1(r, gene) == name and r.get(f"{gene}_cigar") and r.get(f"{gene}_sequence_start") is not None \
                and r.get(f"{gene}_germline_start") is not None:
            out.append((r["sequence"], r[f"{gene}_cigar"], int(r[f"{gene}_sequence_start"]),
                        int(r[f"{gene}_germline_start"])))
    return out


def _weights_for(records, gene, name, read_w):
    return [float(read_w[i]) for i, r in enumerate(records)
            if _top1(r, gene) == name and r.get(f"{gene}_cigar")
            and r.get(f"{gene}_sequence_start") is not None and r.get(f"{gene}_germline_start") is not None]


def _mean_conf(records, gene, name):
    vals = [r[f"{gene}_likelihoods"][0] for r in records
            if _top1(r, gene) == name and r.get(f"{gene}_likelihoods")]
    return float(np.mean(vals)) if vals else 0.0


def _diagnostic_evidence(profile, idx, group_idxs, cosine, seqs):
    """True if the reads cover a position distinguishing this allele from its nearest sibling."""
    if not profile:
        return False
    sibs = [j for j in group_idxs if j != idx]
    if not sibs:
        return True                                            # single-allele gene: presence suffices
    s = max(sibs, key=lambda j: cosine[idx, j])                # nearest sibling by prototype geometry
    a, b = seqs[idx], seqs[s]
    dist = [g for g in range(min(len(a), len(b))) if a[g] != b[g]]
    return any(g in profile and profile[g]["coverage"] > 0 for g in dist)


def decide_gene_calls(aligned, model, reference, params: GenotypeParams) -> list:
    """Stages 1-6 decision layer over an already-aligned repertoire (no model forward). Sweeping
    ``params`` here is cheap — the expensive Stage-0 alignment is done once (see calibrate.py)."""
    read_w = read_weights(aligned)
    genes = tuple(aligned.gene_names)
    gene_calls = []
    for gtype in genes:
        names = aligned.gene_names[gtype]
        seqs = reference.gene(gtype.upper()).sequences
        W, bias = allele_prototypes(model, gtype)
        cosine = prototype_cosine(W)
        leakage = LeakageModel.fit(W, biases=bias, alpha=params.leakage_alpha,
                                   cos0=params.leakage_cos0, cap=params.leakage_cap)
        usage = weighted_usage(aligned, gtype)
        total = sum(u["mass"] for u in usage.values()) + 1e-9
        mass_by_idx = {i: usage[names[i]]["mass"] for i in range(len(names))}
        present = {i for i, m in mass_by_idx.items() if m > params.present_thr * total}
        residual = residual_support(mass_by_idx, leakage, present=present)

        groups: dict[str, list[int]] = {}
        for i, n in enumerate(names):
            groups.setdefault(_gene_of(n), []).append(i)

        for gene, idxs in groups.items():
            gene_total = sum(usage[names[i]]["mass"] for i in idxs) + 1e-9
            gene_residual, evidence, novels, enrich = {}, {}, [], {}
            for i in idxs:
                name = names[i]
                if usage[name]["count"] == 0 and residual[i] < params.min_support * total * 0.5:
                    continue
                reads_i = _reads_for(aligned.records, gtype, name)
                profile = (polymorphism_profile(reads_i, seqs[i], _weights_for(aligned.records, gtype, name, read_w))
                           if reads_i else {})
                res = resolve(profile, name, reference, gtype) if profile else {"call": "confirm"}
                if res["call"] == "novel" and 0 < len(res.get("positions", [])) <= params.max_novel_snps:
                    novels.append({**res, "near": name,
                                   "promotable": "uncovered" not in res.get("source_mask", [])})
                evidence[name] = _diagnostic_evidence(profile, i, idxs, cosine, seqs)
                gene_residual[name] = max(0.0, residual[i]) / total
                enrich[name] = {"count": usage[name]["count"],
                                "usage_fraction": usage[name]["mass"] / gene_total,
                                "mean_confidence": _mean_conf(aligned.records, gtype, name)}
            if not gene_residual:
                continue
            gc = call_gene(gene, gene_residual, evidence, min_support=params.min_support,
                           deletion_floor=params.deletion_floor, gene_usage=gene_total / total, novel=novels)
            for a in gc.alleles:
                a.update(enrich.get(a["name"], {}))
            gene_calls.append(gc)
    return gene_calls


def infer_genotype(model, reference, reads, *, locus: str = "IGH", germline_set_ref=None,
                   params: GenotypeParams | None = None, device: str = "cpu", batch_size: int = 64,
                   **overrides) -> GenotypeResult:
    aligned = align_repertoire(model, reference, reads, device=device, batch_size=batch_size)
    p = params or GenotypeParams(**{k: v for k, v in overrides.items()
                                    if k in GenotypeParams.__dataclass_fields__})
    gene_calls = decide_gene_calls(aligned, model, reference, p)
    report = build_report(gene_calls, meta={"n_reads": len(aligned.sequences), "locus": locus,
                                            "model_params": sum(pr.numel() for pr in model.parameters())})
    genotype_set, warnings = to_genotype_set(gene_calls, locus, germline_set_ref=germline_set_ref)
    return GenotypeResult(report=report, genotype_set=genotype_set, warnings=warnings)
