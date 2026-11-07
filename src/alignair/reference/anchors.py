"""Every missing V and J anchor, and every missing V IMGT-gapped form, filled one way, for training and inference.

``annotate(reference, known)`` takes, for each V and J allele, in order (each step fills only what is still missing):

1. a supplied anchor or gapped form, never changed;
2. the allele of ``known`` (the reference a model came with) with the same sequence, the same name first: its anchor
   and V gapped form, so a sequence the model trained on keeps the model's anchor;
3. GenAIRR's resolver (``alignair.genairr.anchors.resolve_anchor``: a V from its IMGT gaps, a J from a unique
   F/W-G-X-G motif under its locus), whose position counts when it is inferred or verified;
4. for a V still without a gapped form: the gaps of the nearest same-length allele of ``known`` (``gap_from_known``)
   and its anchor. This is the interim until GenAIRR gaps an ungapped V by its nearest template (ask G6), when
   ``gap_from_known`` calls it instead; a V of another length stays without, reported.

Each V and J allele records how its anchor was found (``AlleleInfo.anchor_evidence``: supplied, known_allele,
nearest_known_allele or the resolver's method; ambiguous, unresolved or invalid when none was). A D is left as given.
``anchor_report`` counts the anchored alleles and names those without."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .info import ANCHORED, AlleleInfo, AnchorEvidence
from .reference_set import GeneReference, ReferenceSet, _locus_from_name

MAX_DIFFERENT = 0.1          # the largest share of a V's bases its nearest known allele may differ at
GAP = "."
_GENES = ("V", "J")


def _carry_gaps(gapped: str, sequence: str) -> str:
    """A known allele's gapped form with another allele's bases (the same length) in its base columns."""
    bases = iter(sequence)
    return "".join(GAP if column == GAP else next(bases) for column in gapped)


def gap_from_known(sequence: str, known: GeneReference) -> tuple[str | None, int | None, AnchorEvidence] | None:
    """The gapped form (``sequence``'s bases on its columns) and anchor of the known allele of ``sequence``'s length
    that differs from it at the fewest bases (ties by name), when those are at most ``MAX_DIFFERENT`` of its bases;
    else None."""
    candidates = [(sum(a != b for a, b in zip(other, sequence)), name)
                  for name, other in zip(known.names, known.sequences) if len(other) == len(sequence)]
    if not candidates:
        return None
    different, best = min(candidates)
    if different > MAX_DIFFERENT * len(sequence):
        return None
    gapped = _carry_gaps(known.gapped[best], sequence) if known.gapped and best in known.gapped else None
    anchor = (known.anchors or {}).get(best)
    return gapped, anchor, AnchorEvidence("nearest_known_allele", "inferred" if anchor is not None else "unresolved",
                                          f"{different} of {len(sequence)} bases differ", best)


def _verdict(resolved) -> AnchorEvidence:
    """The evidence of a resolver verdict that gave no anchor: ambiguous (several candidates), unresolved, invalid."""
    status = ("invalid" if resolved.status == "invalid" else
              "ambiguous" if len(resolved.candidates) >= 2 else "unresolved")
    return AnchorEvidence(resolved.method, status, resolved.reason or None, None)


def annotate(reference: ReferenceSet, known: ReferenceSet | None = None) -> ReferenceSet:
    """``reference`` with every V and J allele's missing anchor and every V's missing gapped form filled (the module
    docstring's order) and how each anchor was found recorded; ``reference`` itself is unchanged."""
    from ..genairr.anchors import resolve_anchor
    membership = reference._membership()
    genes = {}
    for gene, alleles in reference.genes.items():
        if gene not in _GENES:
            genes[gene] = alleles
            continue
        target = known.genes.get(gene) if known is not None else None
        by_sequence = {}
        for name, sequence in zip(target.names, target.sequences) if target is not None else ():
            by_sequence.setdefault(sequence, name)
        anchors, gapped, info = dict(alleles.anchors or {}), dict(alleles.gapped or {}), dict(alleles.info or {})
        for i, (name, sequence) in enumerate(zip(alleles.names, alleles.sequences)):
            entry = info.get(name) or AlleleInfo()
            evidence = entry.anchor_evidence if name in anchors else None
            if name in anchors and evidence is None:
                evidence = AnchorEvidence("supplied", "supplied")
            needs_gap = gene == "V" and name not in gapped
            if target is not None and (name not in anchors or needs_gap):
                same = (name if name in target.index and target.sequences[target.index[name]] == sequence
                        else by_sequence.get(sequence))
                if same is not None:
                    if needs_gap and target.gapped and same in target.gapped:
                        gapped[name], needs_gap = target.gapped[same], False
                    if name not in anchors and target.anchors and same in target.anchors:
                        anchors[name] = target.anchors[same]
                        evidence = AnchorEvidence("known_allele", "inferred", None, same)
            if name not in anchors:
                resolved = resolve_anchor(sequence, gene, gapped=gapped.get(name) if gene == "V" else None,
                                          locus=_locus_from_name(name) or membership[gene][i] or "")
                if resolved.status in ("inferred", "verified") and resolved.position is not None:
                    anchors[name] = resolved.position
                    evidence = AnchorEvidence(resolved.method, resolved.status, resolved.reason or None, None)
                else:
                    evidence = _verdict(resolved)
            if needs_gap and target is not None:
                carried = gap_from_known(sequence, target)
                if carried is not None:
                    carried_gaps, carried_anchor, carried_evidence = carried
                    if carried_gaps is not None:
                        gapped[name] = carried_gaps
                    if name not in anchors and carried_anchor is not None:
                        anchors[name], evidence = carried_anchor, carried_evidence
            info[name] = dataclasses.replace(entry, anchor_evidence=evidence)
        genes[gene] = GeneReference(list(alleles.names), list(alleles.sequences), dict(alleles.index),
                                    anchors=anchors or None, gapped=gapped or None, info=info or None)
    return ReferenceSet(genes, reference.has_d, loci=list(reference.loci), provenance=reference.provenance)


@dataclass(frozen=True)
class AnchorReport:
    """V and J: how many alleles have an anchor (``anchored``) of how many (``total``), and the names of those without
    by their status (``missing[gene][status]``: the statuses as they first occur, the names in reference order; no
    evidence counts as unresolved)."""
    anchored: dict
    total: dict
    missing: dict

    def complete(self) -> bool:
        return all(self.anchored[gene] == self.total[gene] for gene in self.total)

    def summary(self) -> str:
        """``"V 2/3, J 1/2 anchored; no anchor: <name> (<status>), ..."``, at most five names, then how many more."""
        counts = ", ".join(f"{gene} {self.anchored[gene]}/{self.total[gene]}" for gene in self.total)
        if self.complete():
            return f"{counts} anchored"
        names = [f"{name} ({status})" for gene in self.missing for status, names in self.missing[gene].items()
                 for name in names]
        more = f", and {len(names) - 5} more" if len(names) > 5 else ""
        return f"{counts} anchored; no anchor: {', '.join(names[:5])}{more}"


def anchor_report(reference: ReferenceSet) -> AnchorReport:
    """How many of ``reference``'s V and J alleles have an anchor, and which have none and why."""
    anchored, total, missing = {}, {}, {}
    for gene in _GENES:
        if gene not in reference.genes:
            continue
        alleles = reference.gene(gene)
        anchors, info = alleles.anchors or {}, alleles.info or {}
        total[gene], anchored[gene] = len(alleles), sum(name in anchors for name in alleles.names)
        by_status = {}
        for name in alleles.names:
            if name not in anchors:
                evidence = info[name].anchor_evidence if name in info else None
                status = evidence.status if evidence is not None and evidence.status not in ANCHORED else "unresolved"
                by_status.setdefault(status, []).append(name)
        if by_status:
            missing[gene] = {status: tuple(names) for status, names in by_status.items()}
    return AnchorReport(anchored, total, missing)
