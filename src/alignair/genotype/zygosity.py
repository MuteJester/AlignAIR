"""Stage 4-5 — the first true decision point: prune, then call zygosity / CNV (evidence-gated).

Pruning uses residual support (Stage 2, leakage already removed) AND distinguishing-SNP evidence
(Stage 3): an allele is dropped only when its residual is leakage-consistent AND it has no covered
diagnostic SNP. Copy number is never proven from usage — a duplication is a CANDIDATE only when ≥3
alleles each carry independent covered diagnostic evidence; near-zero gene support is a cautious
deletion candidate.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GeneCall:
    gene: str
    alleles: list          # [{name, support, evidence}] kept alleles
    zygosity: str          # homozygous | heterozygous | multi-allelic/duplication-candidate | deletion-candidate | none
    deletion_candidate: bool
    reasons: list          # one string per include/exclude/demote/deletion decision
    novel: list            # novel-allele candidates for this gene (from Stage 3)


def call_gene(gene: str, residual: dict, evidence: dict, *, min_support: float = 0.05,
              deletion_floor: float = 0.005, gene_usage: float | None = None, novel=None) -> GeneCall:
    """``residual``={allele: leakage-removed support}; ``evidence``={allele: covered-diagnostic-SNP?}.
    Deletion is judged on the gene's raw USAGE fraction (``gene_usage``) — leakage-removed residuals
    can be near-zero for a genuinely-present gene, so they must not drive the deletion call."""
    total = gene_usage if gene_usage is not None else sum(residual.values())
    reasons: list[str] = []
    kept: list[dict] = []
    for a, r in sorted(residual.items(), key=lambda kv: -kv[1]):
        has_ev = bool(evidence.get(a, False))
        if r < min_support and not has_ev:                     # leakage-consistent + no evidence -> prune
            reasons.append(f"{a}: excluded (residual {r:.3f} < {min_support}, no covered diagnostic SNP)")
            continue
        kept.append({"name": a, "support": r, "evidence": has_ev})
        reasons.append(f"{a}: included (residual {r:.3f}, "
                       f"{'diagnostic SNP covered' if has_ev else 'above support floor'})")

    deletion = total < deletion_floor
    if deletion:
        reasons.append(f"{gene}: deletion candidate (total support {total:.3f} < {deletion_floor}) — "
                       f"note: 'not found' can mean absent, not-expressed, or low-expression")
        zygosity = "deletion-candidate"
    elif not kept:
        zygosity = "none"
    elif len(kept) == 1:
        zygosity = "homozygous"
    elif len(kept) == 2:
        zygosity = "heterozygous"
    else:                                                       # >=3: duplication candidate needs evidence
        if sum(1 for k in kept if k["evidence"]) >= 3:
            zygosity = "multi-allelic/duplication-candidate"
        else:
            for d in kept[2:]:
                reasons.append(f"{d['name']}: demoted (>2 alleles without independent diagnostic evidence "
                               f"— usage cannot prove copy number)")
            kept = kept[:2]
            zygosity = "heterozygous"

    return GeneCall(gene=gene, alleles=kept, zygosity=zygosity, deletion_candidate=deletion,
                    reasons=reasons, novel=list(novel or []))
