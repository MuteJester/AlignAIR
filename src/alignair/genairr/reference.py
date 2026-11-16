"""A GenAIRR dataconfig's reference, the one AlignAIR trains on, built through ``alignair.reference.annotate``.

``reference_from_dataconfigs`` unions the dataconfigs' alleles in AlignAIR's order (V, J, then D; each dataconfig in
turn; the first allele of a name kept), takes each stored anchor that is not 0 (GenAIRR's "not found") as supplied, the
IMGT-gapped forms, each allele's functionality, species and source label, each locus's species, and one source record
per dataconfig, then annotates the rest: a stored 0 is resolved by GenAIRR's resolver and its verdict recorded. A
union whose dataconfigs name one allele with two sequences is refused; inside one dataconfig (MOUSE_IGK_IMGT and
MOUSE_IGL_IMGT each name two alleles twice) the first of a name is kept."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from ..errors import ConfigurationError
from ..reference import AlleleInfo, AnchorEvidence, SourceProvenance, annotate
from ..reference.loci import loci_from_membership, locus_of_chain
from ..reference.reference_set import GeneReference, ReferenceSet
from .catalog import name_of

_FUNCTIONAL = {"functional": "F", "f": "F", "orf": "ORF", "pseudogene": "P", "p": "P"}


def _text(value) -> str | None:
    value = getattr(value, "value", value)
    return None if value is None else str(value)


def _functional(allele) -> str | None:
    status = getattr(allele, "functional_status", None)
    status = getattr(status, "name", status)
    return None if status is None else _FUNCTIONAL.get(str(status).lower())


@dataclass
class _GeneUnion:
    """One gene's alleles across the dataconfigs, in the order they come: the first allele of a name is kept, a name
    given another sequence is a collision, and an allele a second locus also lists is recorded as shared with it."""
    gene: str
    names: list[str] = field(default_factory=list)
    sequences: list[str] = field(default_factory=list)
    index: dict[str, int] = field(default_factory=dict)
    anchors: dict[str, int] = field(default_factory=dict)
    gapped: dict[str, str] = field(default_factory=dict)
    info: dict[str, AlleleInfo] = field(default_factory=dict)
    membership: list[str] = field(default_factory=list)      # per allele, the locus that first contributed it
    collisions: list[str] = field(default_factory=list)      # names given two sequences
    shared: dict[str, list[str]] = field(default_factory=dict)   # locus -> alleles of another locus's block it lists

    def add(self, allele, locus: str, source: str, species: str | None) -> None:
        """``allele`` of ``locus``'s dataconfig (``source``, its locator; ``species``, its species when the locus's
        dataconfigs are of several, else None)."""
        sequence = allele.ungapped_seq.upper()
        if allele.name in self.index:
            if self.sequences[self.index[allele.name]] != sequence and allele.name not in self.collisions:
                self.collisions.append(allele.name)
            elif self.membership[self.index[allele.name]] != locus:     # one allele, two loci (TRAV/DV): both
                names = self.shared.setdefault(locus, [])
                if allele.name not in names:
                    names.append(allele.name)
            return
        self.index[allele.name] = len(self.names)
        self.names.append(allele.name)
        self.membership.append(locus)
        self.sequences.append(sequence)
        anchor, evidence = _stored_anchor(allele, self.gene, source)
        if anchor is not None:
            self.anchors[allele.name] = anchor
        gap = getattr(allele, "gapped_seq", None)
        if gap:
            self.gapped[allele.name] = gap.upper()
        entry = AlleleInfo(_functional(allele), _text(getattr(allele, "species", None)) or species,
                           _text(getattr(allele, "source", None)), evidence)
        if entry != AlleleInfo():
            self.info[allele.name] = entry

    def gene_reference(self) -> GeneReference:
        return GeneReference(self.names, self.sequences, self.index, anchors=self.anchors or None,
                             gapped=self.gapped or None, info=self.info or None)


def _stored_anchor(allele, gene: str, source: str) -> tuple[int | None, AnchorEvidence | None]:
    """The anchor the dataconfig stores for ``allele`` and its evidence: a V or J anchor of 0 is GenAIRR's "no anchor"
    (``annotate`` resolves it), any other is supplied by ``source``; a D's is kept as it is."""
    stored = getattr(allele, "anchor", None)
    if gene in ("V", "J"):
        return (int(stored), AnchorEvidence("supplied", "supplied", None, source)) if stored else (None, None)
    return (None if stored is None else int(stored)), None


def reference_from_dataconfigs(*dataconfigs) -> ReferenceSet:
    """The annotated union of ``dataconfigs``' alleles (see the module docstring)."""
    has_d = any(dc.metadata.has_d for dc in dataconfigs)
    wanted = ["v", "j"] + (["d"] if has_d else [])
    locator = {id(dc): name_of(dc) or _text(dc.metadata.reference_set) for dc in dataconfigs}
    species_of = {id(dc): _text(dc.metadata.species) for dc in dataconfigs}
    locus_order: list[str] = []
    locus_chain: dict[str, str] = {}
    locus_species: dict[str, set] = {}
    for dc in dataconfigs:
        locus = locus_of_chain(dc.metadata.chain_type)
        if locus not in locus_order:
            locus_order.append(locus)
            locus_chain[locus] = str(getattr(dc.metadata.chain_type, "value", dc.metadata.chain_type))
        locus_species.setdefault(locus, set()).add(species_of[id(dc)])
    genes: dict[str, GeneReference] = {}
    per_index: dict[str, list] = {}
    shared: dict[str, dict[str, list]] = {}           # locus -> gene -> alleles of another locus's block it lists
    collisions: list[str] = []
    for g in wanted:
        union = _GeneUnion(g.upper())
        for dc in dataconfigs:
            if g == "d" and not dc.metadata.has_d:
                continue
            locus = locus_of_chain(dc.metadata.chain_type)
            species = species_of[id(dc)] if len(locus_species[locus]) > 1 else None
            own = set()                               # within one dataconfig the first of a name is kept
            for allele in dc.allele_list(g):
                if allele.name not in own:
                    own.add(allele.name)
                    union.add(allele, locus, locator[id(dc)], species)
        genes[union.gene], per_index[union.gene] = union.gene_reference(), union.membership
        collisions += [name for name in union.collisions if name not in collisions]
        for locus, names in union.shared.items():
            shared.setdefault(locus, {})[union.gene] = names
    if collisions:
        sources = [locator[id(dc)] for dc in dataconfigs]
        raise ConfigurationError(f"the dataconfigs name {len(collisions)} alleles with two sequences: "
                                 f"{', '.join(collisions[:5])}", context={"alleles": collisions, "dataconfigs": sources},
                                 hint="Use one of these dataconfigs, or rename the alleles of one.")
    loci = [dataclasses.replace(info, species=next(iter(locus_species[info.locus]))
                                if len(locus_species[info.locus]) == 1 else None,
                                shared={g: tuple(names) for g, names in shared.get(info.locus, {}).items()})
            for info in loci_from_membership(per_index, locus_order, locus_chain)]
    provenance = list(dict.fromkeys(
        SourceProvenance("genairr_dataconfig", locator=locator[id(dc)], sha256=dc.compute_checksum(),
                         release=_text(dc.metadata.reference_set)) for dc in dataconfigs))
    return annotate(ReferenceSet(genes, has_d, loci=loci, provenance=provenance))
