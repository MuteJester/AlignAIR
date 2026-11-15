"""Every missing V and J anchor, and every missing V IMGT-gapped form, filled one way, for training and inference.

``annotate(reference, known)`` takes, for each V and J allele, in order (each step fills only what is still missing):

1. a supplied anchor or gapped form, never changed;
2. the allele of ``known`` (the reference a model came with) with the same sequence, the same name first: its anchor
   and V gapped form, so a sequence the model trained on keeps the model's anchor;
3. GenAIRR's resolver (``alignair.genairr.anchors.resolve_anchor``: a V from its IMGT gaps, a J from a unique
   F/W-G-X-G motif under its locus), whose position counts when it is inferred or verified;
4. for a V still without a gapped form: the gaps and anchor of the nearest same-length allele of ``known`` (at most
   ``MAX_DIFFERENT`` of its bases differ); a V of another length stays without, reported.

Each V and J allele records how its anchor was found (``AlleleInfo.anchor_evidence``: supplied, known_allele,
nearest_known_allele or the resolver's method; ambiguous, unresolved or invalid when none was). A D is left as given. Given a ``known``
with a locus schema, the result is laid on it (``_on_the_models_loci``): the model's loci, in its order, are the bound
reference's, whatever the alleles' names show. ``anchor_report`` counts the anchored alleles and names those without."""
from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field

import numpy as np

from ..errors import InputError
from .info import ANCHORED, AlleleInfo, AnchorEvidence
from .reference_set import _CHAIN_OF_LOCUS, GeneReference, ReferenceSet, _loci_from_membership, _locus_from_name

class LocusMismatchWarning(UserWarning):
    """A reference of another locus was bound to a model of one locus: allowed, but the model never trained on it."""


MAX_DIFFERENT = 0.1          # the largest share of a V's bases its nearest known allele may differ at
GAP = "."
CYS = ("TGT", "TGC")
_GENES = ("V", "J")


def _carry_gaps(gapped: str, sequence: str) -> str | None:
    """A known allele's gapped form with another allele's bases (the same length) in its base columns; None when the
    form does not hold that many bases."""
    if len(gapped) - gapped.count(GAP) != len(sequence):
        return None
    bases = iter(sequence)
    return "".join(GAP if column == GAP else next(bases) for column in gapped)


def _lends_gaps(known: GeneReference, name: str, sequence: str) -> bool:
    """Whether a known allele's gapped form is its sequence with gaps (a few bundled ones hold other bases)."""
    gapped = (known.gapped or {}).get(name)
    return gapped is not None and gapped.replace(GAP, "").upper() == sequence


class _Donors:
    """The known alleles that can lend their gaps (a gapped form that is the allele's sequence with gaps), grouped by
    length as base arrays, so the nearest of many is found by one vectorised comparison per allele."""

    def __init__(self, known: GeneReference):
        self.known = known
        by_length = {}
        for name, sequence in zip(known.names, known.sequences):
            if _lends_gaps(known, name, sequence):
                by_length.setdefault(len(sequence), []).append((name, sequence))
        self.groups = {length: ([n for n, _ in group],
                                np.frombuffer("".join(s for _, s in group).encode("ascii", "replace"),
                                              dtype=np.uint8).reshape(len(group), length))
                       for length, group in by_length.items()}

    def nearest(self, sequence: str):
        group = self.groups.get(len(sequence))
        if group is None:
            return None
        names, bases = group
        query = np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)
        different = (bases != query).sum(axis=1)
        fewest = int(different.min())
        if fewest > MAX_DIFFERENT * len(sequence):
            return None
        best = min(name for name, d in zip(names, different) if d == fewest)
        gapped = _carry_gaps(self.known.gapped[best], sequence)
        anchor = (self.known.anchors or {}).get(best)
        return gapped, anchor, AnchorEvidence("nearest_known_allele", "inferred" if anchor is not None else "unresolved",
                                              f"{fewest} of {len(sequence)} bases differ", best)


def _verdict(resolved) -> AnchorEvidence:
    """The evidence of a resolver verdict that gave no anchor: ambiguous (several candidates), unresolved, invalid."""
    status = ("invalid" if resolved.status == "invalid" else
              "ambiguous" if len(resolved.candidates) >= 2 else "unresolved")
    return AnchorEvidence(resolved.method, status, resolved.reason or None, None)


def imgt_position(sequence: str, gapped: str | None) -> tuple[int, str] | None:
    """``(anchor, codon)`` of a V whose codon 104 (IMGT-gapped columns 309-311) holds bases that are not Cys, when
    its gapped form is its sequence with gaps on IMGT numbering; else None. The owner's rule (2026-11-07): codon 104
    anchors the V whatever its residue, as IMGT numbers the junction from it. A form whose Cys sits a codon or more
    past column 309 (columns 312-327) is not on IMGT numbering (GenAIRR G2) and gets none."""
    if not gapped or len(gapped) < 312 or GAP in gapped[309:312] or gapped.replace(GAP, "").upper() != sequence:
        return None
    codon = gapped[309:312].upper()
    if codon in CYS or any(gapped[c:c + 3].upper() in CYS for c in range(312, 328, 3)):
        return None
    return len(gapped[:309].replace(GAP, "")), codon


def annotate(reference: ReferenceSet, known: ReferenceSet | None = None) -> ReferenceSet:
    """``reference`` with every V and J allele's missing anchor and every V's missing gapped form filled (the module
    docstring's order) and how each anchor was found recorded; ``reference`` itself is unchanged."""
    from ..genairr.anchors import resolve_anchor
    membership = reference._membership()
    genes = {}
    for gene, alleles in reference.genes.items():
        if gene not in _GENES:
            genes[gene] = _copy(alleles)
            continue
        target = known.genes.get(gene) if known is not None else None
        donors = _Donors(target) if target is not None and gene == "V" else None
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
            carried_evidence = None
            if needs_gap and donors is not None:      # before the resolver: it anchors a V from gaps only
                carried = donors.nearest(sequence)
                if carried is not None:
                    carried_gaps, carried_anchor, carried_evidence = carried
                    if carried_gaps is not None:
                        gapped[name] = carried_gaps
                    if name not in anchors and carried_anchor is not None:
                        anchors[name], evidence = carried_anchor, carried_evidence
            if name not in anchors:
                resolved = resolve_anchor(sequence, gene, gapped=gapped.get(name) if gene == "V" else None,
                                          locus=_locus_from_name(name) or membership[gene][i] or "")
                position = imgt_position(sequence, gapped.get(name)) if gene == "V" and resolved.status == "invalid" \
                    else None
                if resolved.status in ("inferred", "verified") and resolved.position is not None:
                    anchors[name] = resolved.position
                    evidence = AnchorEvidence(resolved.method, resolved.status, resolved.reason or None, None)
                elif position is not None:
                    anchors[name] = position[0]
                    evidence = AnchorEvidence("imgt_position_104", "inferred",
                                              f"codon 104 is {position[1]}, not Cys: anchored at its IMGT position", None)
                else:                                 # the allele that lent gaps says more than "unresolved"
                    evidence = carried_evidence or _verdict(resolved)
            info[name] = dataclasses.replace(entry, anchor_evidence=evidence)
        genes[gene] = GeneReference(list(alleles.names), list(alleles.sequences), dict(alleles.index),
                                    anchors=anchors or None, gapped=gapped or None, info=info or None)
    loci = [dataclasses.replace(info, ranges=dict(info.ranges)) for info in reference.loci]
    annotated = ReferenceSet(genes, reference.has_d, loci=loci, provenance=reference.provenance)
    return _on_the_models_loci(annotated, known) if known is not None and known.loci else annotated


def _copy(alleles: GeneReference) -> GeneReference:
    return GeneReference(list(alleles.names), list(alleles.sequences), dict(alleles.index),
                         anchors=dict(alleles.anchors) if alleles.anchors else None,
                         gapped=dict(alleles.gapped) if alleles.gapped else None,
                         info=dict(alleles.info) if alleles.info else None)


def _on_the_models_loci(reference: ReferenceSet, known: ReferenceSet) -> ReferenceSet:
    """``reference`` laid on ``known``'s locus schema (the chain head's classes, in their order): an allele ``known``
    holds (by name and sequence, else by sequence) takes its locus, a novel one its name's when the model has it, else
    the model's only locus; each gene's alleles grouped by locus, the reference's order within one. InputError for an
    allele no locus of the model can hold."""
    loci = [info.locus for info in known.loci]
    theirs = known._membership()
    genes = {}
    per_index = {}
    for gene, alleles in reference.genes.items():
        target = known.genes.get(gene)
        by_sequence = {}
        for i, sequence in enumerate(target.sequences if target is not None else ()):
            by_sequence.setdefault(sequence, i)
        placed = []
        for name, sequence in zip(alleles.names, alleles.sequences):
            row = (target.index[name] if target is not None and name in target.index
                   and target.sequences[target.index[name]] == sequence else by_sequence.get(sequence))
            locus = theirs[gene][row] if row is not None else None
            if locus is None:
                named = _locus_from_name(name)
                locus = named if named in loci else (loci[0] if len(loci) == 1 else None)
            if locus is None:
                raise InputError(f"the reference's allele {name} belongs to none of the model's loci "
                                 f"({', '.join(loci)})", context={"allele": name, "loci": loci},
                                 hint="Name it after its locus, or bind it to a model of that locus.")
            placed.append((name, locus))
        order = [name for locus in loci for name, x in placed if x == locus]
        source = {name: alleles.index[name] for name in alleles.names}
        pick = lambda values: {n: values[n] for n in order if n in values} if values else None
        genes[gene] = GeneReference(order, [alleles.sequences[source[n]] for n in order],
                                    {n: i for i, n in enumerate(order)}, anchors=pick(alleles.anchors) or None,
                                    gapped=pick(alleles.gapped) or None, info=pick(alleles.info) or None)
        per_index[gene] = [dict(placed)[n] for n in order]
    ranges = _loci_from_membership(per_index, loci)
    schema = [dataclasses.replace(info, ranges=new.ranges) for info, new in zip(known.loci, ranges)]
    other = _other_locus(reference, loci)
    if other is not None:          # one-locus model, a reference wholly of another locus: label its records by it
        warnings.warn(f"the bound reference is {other}, but the model was trained on {loci[0]}: binding is "
                      f"allowed, but its calls on {other} have not been measured; records are labelled {other}",
                      LocusMismatchWarning, stacklevel=4)
        mine = next((info for info in reference.loci if info.locus == other), None)
        schema = [dataclasses.replace(schema[0], locus=other, chain_type=_CHAIN_OF_LOCUS.get(other, other),
                                      species=mine.species if mine is not None else None, shared={})]
    return ReferenceSet(genes, reference.has_d, loci=schema, provenance=reference.provenance)


def _named_loci(name: str) -> set:
    """The loci an allele's name shows: its prefix's (IGKV1-5*01 -> IGK), and TRD too for a TRAV/DV allele, which
    rearranges in TRA and TRD alike; none for a name that shows no locus."""
    named = _locus_from_name(name)
    if named is None:
        return set()
    return {named, "TRD"} if named == "TRA" and "/DV" in str(name).upper() else {named}


def _other_locus(reference: ReferenceSet, loci: list) -> str | None:
    """For a model of one locus: the one other locus every locus-showing allele name of ``reference`` shows, when none
    shows the model's; else None (a reference of the model's locus, of several, or whose names show none)."""
    if len(loci) != 1:
        return None
    shown = [_named_loci(name) for alleles in reference.genes.values() for name in alleles.names]
    shown = [s for s in shown if s]
    if not shown or any(loci[0] in s for s in shown):
        return None
    common = set.intersection(*shown)
    return next(iter(sorted(common))) if len(common) == 1 else None


@dataclass(frozen=True)
class AnchorReport:
    """V and J: how many alleles have an anchor (``anchored``) of how many (``total``), and the names of those without
    by their status (``missing[gene][status]``: the statuses as they first occur, the names in reference order; no
    evidence counts as unresolved), and the V alleles anchored on a codon that is not Cys (``noncanonical``)."""
    anchored: dict
    total: dict
    missing: dict
    noncanonical: dict = field(default_factory=dict)   # V alleles anchored on a codon that is not Cys, by gene

    def complete(self) -> bool:
        return all(self.anchored[gene] == self.total[gene] for gene in self.total)

    def summary(self) -> str:
        """``"V 2/3, J 1/2 anchored; no anchor: <name> (<status>), ..."``, at most five names, then how many more."""
        counts = ", ".join(f"{gene} {self.anchored[gene]}/{self.total[gene]}" for gene in self.total)
        parts = [f"{counts} anchored"]
        if not self.complete():
            names = [f"{name} ({status})" for gene in self.missing for status, names in self.missing[gene].items()
                     for name in names]
            parts.append(f"no anchor: {_first_five(names)}")
        odd = [name for names in self.noncanonical.values() for name in names]
        if odd:
            parts.append(f"codon 104 not Cys: {_first_five(odd)}")
        return "; ".join(parts)


def _first_five(names: list) -> str:
    more = f", and {len(names) - 5} more" if len(names) > 5 else ""
    return ", ".join(names[:5]) + more


def anchor_report(reference: ReferenceSet) -> AnchorReport:
    """How many of ``reference``'s V and J alleles have an anchor, and which have none and why."""
    anchored, total, missing, noncanonical = {}, {}, {}, {}
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
        if gene == "V":
            odd = tuple(name for name, sequence in zip(alleles.names, alleles.sequences) if name in anchors
                        and sequence[anchors[name]:anchors[name] + 3].upper() not in CYS)
            if odd:
                noncanonical[gene] = odd
    return AnchorReport(anchored, total, missing, noncanonical)
