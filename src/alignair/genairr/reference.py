"""A GenAIRR dataconfig's reference, the one AlignAIR trains on, built through ``alignair.reference.annotate``.

``reference_from_dataconfigs`` unions the dataconfigs' alleles in AlignAIR's order (V, J, then D; each dataconfig in
turn; the first allele of a name kept), takes each stored anchor that is not GenAIRR's legacy 0 as supplied, the
IMGT-gapped forms, each allele's functionality, species and source label, each locus's species, and one source record
per dataconfig, then annotates the rest: a stored 0 is resolved by GenAIRR's resolver and its verdict recorded. A
union whose dataconfigs name one allele with two sequences is refused; inside one dataconfig (MOUSE_IGK_IMGT and
MOUSE_IGL_IMGT each name two alleles twice) the first of a name is kept, as AlignAIR always did."""
from __future__ import annotations

import dataclasses
from typing import Dict, List

from ..errors import ConfigurationError
from ..reference import AlleleInfo, AnchorEvidence, SourceProvenance, annotate
from ..reference.reference_set import GeneReference, ReferenceSet, _loci_from_membership, _locus_of
from .catalog import name_of

_FUNCTIONAL = {"functional": "F", "f": "F", "orf": "ORF", "pseudogene": "P", "p": "P"}


def _text(value) -> str | None:
    value = getattr(value, "value", value)
    return None if value is None else str(value)


def _functional(allele) -> str | None:
    status = getattr(allele, "functional_status", None)
    status = getattr(status, "name", status)
    return None if status is None else _FUNCTIONAL.get(str(status).lower())


def reference_from_dataconfigs(*dataconfigs) -> ReferenceSet:
    """The annotated union of ``dataconfigs``' alleles (see the module docstring)."""
    has_d = any(dc.metadata.has_d for dc in dataconfigs)
    wanted = ["v", "j"] + (["d"] if has_d else [])
    locator = {id(dc): name_of(dc) or _text(dc.metadata.reference_set) for dc in dataconfigs}
    species_of = {id(dc): _text(dc.metadata.species) for dc in dataconfigs}
    locus_order: List[str] = []
    locus_chain: Dict[str, str] = {}
    locus_species: Dict[str, set] = {}
    for dc in dataconfigs:
        locus = _locus_of(dc.metadata.chain_type)
        if locus not in locus_order:
            locus_order.append(locus)
            locus_chain[locus] = str(getattr(dc.metadata.chain_type, "value", dc.metadata.chain_type))
        locus_species.setdefault(locus, set()).add(species_of[id(dc)])
    genes: Dict[str, GeneReference] = {}
    per_index: Dict[str, list] = {}
    collisions: List[str] = []
    for g in wanted:
        gene = g.upper()
        names, sequences, index, anchors, gapped, info, membership = [], [], {}, {}, {}, {}, []
        owner: Dict[str, int] = {}                    # name -> the dataconfig that gave it
        for dc in dataconfigs:
            if g == "d" and not dc.metadata.has_d:
                continue
            locus = _locus_of(dc.metadata.chain_type)
            for allele in dc.allele_list(g):
                sequence = allele.ungapped_seq.upper()
                if allele.name in index:              # within one dataconfig the first of a name is kept
                    if (owner[allele.name] != id(dc) and sequences[index[allele.name]] != sequence
                            and allele.name not in collisions):
                        collisions.append(allele.name)
                    continue
                owner[allele.name] = id(dc)
                index[allele.name] = len(names)
                names.append(allele.name)
                membership.append(locus)              # which locus first contributed this allele
                sequences.append(sequence)
                stored = getattr(allele, "anchor", None)
                evidence = None
                if gene in ("V", "J"):
                    if stored:                        # GenAIRR's legacy 0 is no anchor: annotate resolves it
                        anchors[allele.name] = int(stored)
                        evidence = AnchorEvidence("supplied", "supplied", None, locator[id(dc)])
                elif stored is not None:
                    anchors[allele.name] = int(stored)
                gap = getattr(allele, "gapped_seq", None)
                if gap:
                    gapped[allele.name] = gap.upper()
                species = _text(getattr(allele, "species", None)) or (
                    species_of[id(dc)] if len(locus_species[locus]) > 1 else None)
                entry = AlleleInfo(_functional(allele), species, _text(getattr(allele, "source", None)), evidence)
                if entry != AlleleInfo():
                    info[allele.name] = entry
        genes[gene] = GeneReference(names, sequences, index, anchors=anchors or None, gapped=gapped or None,
                                    info=info or None)
        per_index[gene] = membership
    if collisions:
        sources = [locator[id(dc)] for dc in dataconfigs]
        raise ConfigurationError(f"the dataconfigs name {len(collisions)} alleles with two sequences: "
                                 f"{', '.join(collisions[:5])}", context={"alleles": collisions, "dataconfigs": sources},
                                 hint="Use one of these dataconfigs, or rename the alleles of one.")
    loci = [dataclasses.replace(info, species=next(iter(locus_species[info.locus]))
                                if len(locus_species[info.locus]) == 1 else None)
            for info in _loci_from_membership(per_index, locus_order, locus_chain)]
    provenance = list(dict.fromkeys(
        SourceProvenance("genairr_dataconfig", locator=locator[id(dc)], sha256=dc.compute_checksum(),
                         release=_text(dc.metadata.reference_set)) for dc in dataconfigs))
    return annotate(ReferenceSet(genes, has_d, loci=loci, provenance=provenance))
