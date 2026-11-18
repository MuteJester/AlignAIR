"""Genotype metrics: whether a tool's calls on a donor's reads stay inside that donor's genotype.

A cohort read belongs to a simulated donor (``groups["subject"]``) whose genotype (``core.genotype.DonorGenotype``)
says which alleles it carries and which genes it lacks. A top-1 call in the read's scored truth set is defensible
whatever the genotype (a simulator re-call or an as-near allele need not be carried), so the per-read metrics count
only calls outside both. ``excess_alleles`` asks the donor-level question: does the tool, over a donor's reads,
report more alleles of a gene than the donor carries? It answers counting all calls and counting only the
avoidable ones, leaving out the defensible calls of non-carried alleles. The oracle calls each read's generative
allele, which its donor carries (or, for a novel allele, the base, which is in the truth set), so it scores 0
throughout.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from alignair_assay.core.genotype import DonorGenotype
from alignair_assay.core.prediction import Prediction
from alignair_assay.core.reference import gene_name
from alignair_assay.metrics import CaseScores
from alignair_assay.metrics.case import MIN_OBSERVED
from alignair_assay.truth.model import ObservableTruth

NAN = float("nan")
# Gene letters scored against the genotype; only V has deletable genes among them (J is never deleted).
GENOTYPE_GENES = ("v", "j")
DELETED_GENE_CALL = "v.deleted_gene_call"


def genotype_metrics(truth: ObservableTruth, pred: Prediction | None, genotype: DonorGenotype) -> CaseScores:
    """Genotype metrics of one read against its donor's ``genotype``, for g in V and J when g is observed with at
    least ``metrics.case.MIN_OBSERVED[g]`` germline bases and is scorable; ``{}`` for an excluded read.

    ``{g}.outside_genotype`` is 1.0 when the top-1 call is neither carried by the donor nor in the read's scored
    truth set, 0.0 otherwise. ``v.deleted_gene_call`` is 1.0 when the top-1 V call's gene is one of the donor's
    deleted genes and the call is not in the scored truth set, 0.0 otherwise. Both are lower-is-better and ``nan``
    when the tool made no top-1 call or emitted no row (``pred`` None).
    """
    if truth.excluded is not None:
        return {}
    scores: CaseScores = {}
    for g in GENOTYPE_GENES:
        if not _applicable(truth, g):
            continue
        names = [f"{g}.outside_genotype"] + ([DELETED_GENE_CALL] if g == "v" else [])
        call = top1_call(pred, g)
        if call is None:
            scores.update(dict.fromkeys(names, NAN))
            continue
        defensible = call in truth.segments[g].alleles.alleles
        scores[f"{g}.outside_genotype"] = float(not defensible and call not in genotype.carried)
        if g == "v":
            scores[DELETED_GENE_CALL] = float(not defensible and gene_name(call) in genotype.deleted)
    return scores


def excess_alleles(truths: Iterable[ObservableTruth], preds: Mapping[str, Prediction | None],
                   genotypes: Mapping[str, DonorGenotype], min_reads: int = 3) -> dict:
    """The donor-level excess-alleles table of one tool: per gene letter in V and J, how often the tool's calls on
    a donor's reads report more alleles of a gene than the donor carries, counting all calls and counting only the
    avoidable ones.

    A read counts for letter g when it is scored (not excluded), has ``groups["subject"]``, g is observed with at
    least ``MIN_OBSERVED[g]`` germline bases and scorable, and the tool's row has a top-1 call for g. Reads are
    grouped into (donor, gene of the top-1 call) pairs. A pair with at least ``min_reads`` reads is considered; it
    is an excess when its distinct top-1 alleles called on at least ``min_reads`` reads each outnumber the alleles
    of that gene the donor carries (none for a deleted gene). It is an avoidable excess when they still do once
    the defensible calls are left out: a call in the read's scored truth set of an allele the donor does not carry
    (one identical to a carried allele over the read, say), which a sequence-only tool cannot avoid. The other
    calls, those of carried alleles or outside the truth set, count as before, so an avoidable excess is always an
    excess; both are counted over the same considered pairs. Returns ``{"min_reads": min_reads, "v": entry, "j":
    entry}``, each entry ``{"n": considered pairs, "excess": excess pairs, "fraction": excess / n,
    "avoidable_excess": avoidable excess pairs, "avoidable_fraction": avoidable_excess / n}``, fractions from 0 to
    1, or None when n is 0. ValueError when ``min_reads`` is not a positive integer or a counted read's subject has
    no genotype in ``genotypes``.
    """
    if isinstance(min_reads, bool) or not isinstance(min_reads, int) or min_reads < 1:
        raise ValueError(f"min_reads must be a positive integer, got {min_reads!r}")
    # Per letter and (donor, gene of the call): the calls per allele, all of them and the avoidable ones.
    calls: dict[str, dict[tuple[str, str], Counter[str]]] = {g: {} for g in GENOTYPE_GENES}
    avoidable: dict[str, dict[tuple[str, str], Counter[str]]] = {g: {} for g in GENOTYPE_GENES}
    donors: dict[str, DonorGenotype] = {}
    for truth in truths:
        subject = truth.groups.get("subject")
        if truth.excluded is not None or subject is None:
            continue
        genotype = donors[subject] = donor_genotype(genotypes, truth)
        pred = preds.get(truth.case_id)
        for g in GENOTYPE_GENES:
            call = top1_call(pred, g)
            if call is None or not _applicable(truth, g):
                continue
            pair = (subject, gene_name(call))
            calls[g].setdefault(pair, Counter())[call] += 1
            if call in genotype.carried or call not in truth.segments[g].alleles.alleles:
                avoidable[g].setdefault(pair, Counter())[call] += 1
    # Carried alleles per gene, by donor; a deleted gene carries none.
    carried = {subject: Counter(gene_name(name) for name in genotype.carried) for subject, genotype in donors.items()}

    def exceeds(alleles: Counter[str], subject: str, gene: str) -> bool:
        return sum(count >= min_reads for count in alleles.values()) > carried[subject][gene]

    table: dict = {"min_reads": min_reads}
    for g in GENOTYPE_GENES:
        pairs = [pair for pair, alleles in calls[g].items() if sum(alleles.values()) >= min_reads]
        excess = sum(exceeds(calls[g][pair], *pair) for pair in pairs)
        avoidable_excess = sum(exceeds(avoidable[g].get(pair, Counter()), *pair) for pair in pairs)
        table[g] = {"n": len(pairs), "excess": excess, "fraction": excess / len(pairs) if pairs else None,
                    "avoidable_excess": avoidable_excess,
                    "avoidable_fraction": avoidable_excess / len(pairs) if pairs else None}
    return table


def donor_genotype(genotypes: Mapping[str, DonorGenotype], truth: ObservableTruth) -> DonorGenotype:
    """The genotype of ``truth``'s donor (``groups["subject"]``); ValueError naming the case and its subject when
    ``genotypes`` has none."""
    subject = truth.groups.get("subject")
    if subject not in genotypes:
        raise ValueError(f"case {truth.case_id}: its subject {subject!r} has no donor genotype (the genotypes are "
                         f"of {sorted(genotypes)}); score with the donors of the suite the truth was built from")
    return genotypes[subject]


def top1_call(pred: Prediction | None, g: str) -> str | None:
    """The prediction's top-1 call for gene letter ``g``; None without a row, a segment or a call."""
    segment = None if pred is None else pred.segments.get(g)
    return None if segment is None else segment.call


def _applicable(truth: ObservableTruth, g: str) -> bool:
    """True when gene ``g`` of the read is scorable and observed with at least ``MIN_OBSERVED[g]`` bases."""
    segment = truth.segments.get(g)
    return g not in truth.unscorable and segment is not None and segment.observed_length >= MIN_OBSERVED[g]
