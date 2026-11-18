"""Novel-allele metrics: what a tool makes of the SNPs of a V allele that is not in its reference.

A read from a novel V allele is truthed against the allele's base (``truth.build``), which a perfect fixed-reference
aligner reports, so the novel SNPs are mutations of that truth. ``v_novel.snp_mutations`` counts how many of them
the tool's alignment reports as mismatches. It is informational: the oracle, aligning the base, reports every SNP
the read still shows, and a tool that inferred the novel allele would report none.
"""
from __future__ import annotations

from alignair_assay.core.prediction import PredSegment
from alignair_assay.core.reference import Reference
from alignair_assay.metrics import CaseScores
from alignair_assay.metrics.shm import predicted_germline
from alignair_assay.truth.model import ObservableTruth

NAN = float("nan")
SNP_MUTATIONS = "v_novel.snp_mutations"


def novel_metrics(truth: ObservableTruth, pred: PredSegment, reference: Reference) -> CaseScores:
    """``v_novel.snp_mutations`` for a read whose V is a novel allele with SNPs in the observed V
    (``truth.v_novel_snps``, read positions): the number of those positions the prediction's V alignment (``pred``,
    the V segment of the tool's row; an empty ``PredSegment`` for a missing row) aligns to a germline base that
    differs from the read base. A read ``N`` is never a mismatch, and a position the alignment does not cover does
    not count. ``nan`` when the prediction has no usable V alignment (``metrics.shm.predicted_germline``); ``{}``
    when the read shows no novel SNP.
    """
    if not truth.v_novel_snps:
        return {}
    aligned = predicted_germline("v", truth.sequence, pred, reference)
    if aligned is None:
        return {SNP_MUTATIONS: NAN}
    germline = aligned[0]
    read = truth.sequence
    return {SNP_MUTATIONS: float(sum(position in germline and read[position] != "N"
                                     and read[position] != germline[position]
                                     for position in truth.v_novel_snps))}
