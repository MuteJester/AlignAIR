"""``ReferenceSet``: a germline reference, the alleles of each gene (V, D, J) with their anchors, IMGT-gapped forms,
loci and provenance."""
from __future__ import annotations

import dataclasses
import re
from numbers import Integral
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from ..errors import ConfigurationError, DataContractError, InputError
from .info import AlleleInfo
from .loci import (CHAIN_OF_LOCUS, FALLBACK_LOCUS, LocusInfo, loci_from_membership, locus_from_name)
from .provenance import SourceProvenance

@dataclass
class GeneReference:
    """One gene's alleles, in order, with their sequences, anchors, IMGT-gapped forms and source information."""
    names: List[str]          # ordered allele names
    sequences: List[str]      # germline nucleotide seqs (uppercased), aligned with names
    index: Dict[str, int]     # name -> row index
    anchors: Dict[str, int] | None = None  # name -> conserved-anchor germline pos (Cys/Trp-Phe)
    gapped: Dict[str, str] | None = None   # name -> IMGT-gapped germline (AIRR sequence_alignment)
    info: Dict[str, AlleleInfo] | None = None  # name -> what its source told (functionality, species, label, evidence)

    def __len__(self) -> int:
        return len(self.names)


class ReferenceSet:
    """A germline reference: a ``GeneReference`` per gene, its loci and its sources. Build one from GenAIRR
    dataconfigs (``from_dataconfigs``), a FASTA file (``from_fasta``), a genotype mapping (``from_genotype``) or its
    JSON form (``from_serializable``)."""

    def __init__(self, genes: Dict[str, GeneReference], has_d: bool, loci: List[LocusInfo] | None = None,
                 provenance: Iterable[SourceProvenance] = ()):
        self.genes = genes
        self.has_d = has_d
        self.loci = loci or []
        self.provenance = tuple(provenance)      # the sources its alleles came from, in order

    def gene(self, g: str) -> GeneReference:
        """The ``GeneReference`` of gene ``g`` (V, D or J, either case)."""
        return self.genes[g.upper()]

    def validate(self, *, genes=None) -> None:
        """Validate selected ordered vocabularies before model construction."""
        selected = (('V', 'J', 'D') if self.has_d else ('V', 'J')) if genes is None else genes
        for gene in selected:
            gene = gene.upper()
            if gene not in self.genes:
                raise ConfigurationError("selected gene is absent from the reference", context={"gene": gene})
            rows = self.gene(gene)
            if (any(not isinstance(name, str) or not name for name in rows.names)
                    or any(not isinstance(sequence, str) for sequence in rows.sequences)
                    or any(isinstance(value, bool) or not isinstance(value, Integral) for value in rows.index.values())
                    or len(rows.names) != len(rows.sequences) or len(set(rows.names)) != len(rows.names)
                    or rows.index != {name: i for i, name in enumerate(rows.names)}):
                raise DataContractError("reference allele order, sequences, and index disagree", context={"gene": gene})

    def locus_names(self) -> tuple:
        """Ordered AIRR locus codes, in the order the union was built (empty if unknown)."""
        return tuple(l.locus for l in self.loci)

    @property
    def default_locus(self) -> str:
        """The locus a record without its own is labelled with: the reference's one locus, else ``FALLBACK_LOCUS``."""
        loci = self.locus_names()
        return loci[0] if len(loci) == 1 else FALLBACK_LOCUS

    def locus_mask(self, locus: str, gene: str) -> np.ndarray:
        """Boolean mask over the ``gene`` head selecting exactly the alleles that belong to ``locus``.

        No schema (single-chain) -> all-True (unconstrained). With a schema, a locus that *has* no
        alleles for this gene (e.g. D on a light chain) -> **all-False** (an explicit no-call), NOT
        all-True — otherwise a light-chain read could be handed every heavy-chain D allele."""
        n = len(self.gene(gene))
        if not self.loci:                                # no locus schema -> unconstrained
            return np.ones(n, dtype=bool)
        info = next((l for l in self.loci if l.locus == locus), None)
        m = np.zeros(n, dtype=bool)
        if info is not None and gene.upper() in info.ranges:
            s, e = info.ranges[gene.upper()]
            m[s:e] = True                                # else: known schema, locus lacks this gene -> no-call
        if info is not None:                             # alleles of another locus's block this locus lists too
            index = self.gene(gene).index
            for name in info.shared.get(gene.upper(), ()):
                if name in index:
                    m[index[name]] = True
        return m

    @classmethod
    def from_dataconfigs(cls, *dataconfigs) -> "ReferenceSet":
        """The annotated union of GenAIRR dataconfigs' alleles (``alignair.genairr.reference_from_dataconfigs``)."""
        from ..genairr.reference import reference_from_dataconfigs
        return reference_from_dataconfigs(*dataconfigs)

    def to_serializable(self) -> dict:
        """Plain-JSON-safe dict carrying everything inference needs — ordered ``names`` (== model-head
        index), ungapped ``sequences``, IMGT-``gapped`` V, and junction ``anchors`` — with NO pickle.
        This is the safe reference the distributed ``.alignair`` embeds and inference rebuilds from."""
        genes: Dict[str, dict] = {}
        for G in ("V", "D", "J"):
            if G not in self.genes:
                continue
            ref = self.genes[G]
            genes[G] = {"names": list(ref.names), "sequences": list(ref.sequences),
                        "gapped": dict(ref.gapped) if ref.gapped else {},
                        "anchors": dict(ref.anchors) if ref.anchors else {}}
            info = {}
            for name in ref.names:
                entry = (ref.info or {}).get(name)
                if entry is not None and entry.to_dict():
                    info[name] = entry.to_dict()
            if info:                                     # optional in v1: left out when no allele has any
                genes[G]["info"] = info
        loci = []
        for l in self.loci:
            locus = {"locus": l.locus, "chain_type": l.chain_type, "has_d": l.has_d,
                     "ranges": {G: list(se) for G, se in l.ranges.items()}}
            if l.species is not None:
                locus["species"] = l.species
            if l.shared:                                 # additive: written only for loci that share alleles
                locus["shared"] = {G: list(names) for G, names in l.shared.items()}
            loci.append(locus)
        out = {"schema": "alignair.reference.v1", "has_d": self.has_d, "genes": genes, "loci": loci}
        if self.provenance:
            out["provenance"] = [source.to_dict() for source in self.provenance]
        return out

    @classmethod
    def from_serializable(cls, data: dict) -> "ReferenceSet":
        """Rebuild from :meth:`to_serializable` (the safe, no-pickle inference load path)."""
        out: Dict[str, GeneReference] = {}
        for G, gd in data["genes"].items():
            names = list(gd["names"])
            sequences = [str(s).upper() for s in gd["sequences"]]
            anchors = {k: int(v) for k, v in (gd.get("anchors") or {}).items()} or None
            gapped = {k: str(v).upper() for k, v in (gd.get("gapped") or {}).items()} or None
            info, held = {}, set(names)
            for name, entry in (gd.get("info") or {}).items():
                if name not in held:
                    raise DataContractError("reference info names an allele its gene does not hold",
                                            context={"gene": G.upper(), "allele": name})
                info[name] = AlleleInfo.from_dict(entry)
            out[G.upper()] = GeneReference(names, sequences, {n: i for i, n in enumerate(names)},
                                           anchors=anchors, gapped=gapped, info=info or None)
        loci = [LocusInfo(locus=l["locus"], chain_type=l.get("chain_type", l["locus"]),
                          has_d=bool(l.get("has_d")),
                          ranges={G: tuple(se) for G, se in (l.get("ranges") or {}).items()},
                          species=l.get("species"),
                          shared={G: tuple(names) for G, names in (l.get("shared") or {}).items()})
                for l in data.get("loci", [])]
        provenance = [SourceProvenance.from_dict(source) for source in data.get("provenance") or ()]
        return cls(out, bool(data.get("has_d")), loci=loci, provenance=provenance)

    @classmethod
    def from_genotype(cls, genes: Dict[str, Dict[str, str]],
                      anchors: Dict[str, Dict[str, int]] | None = None, *,
                      locus: str | None = None) -> "ReferenceSet":
        """Build a reference from a genotype mapping {gene_type: {allele_name: dna_seq}}.

        gene_type is V/J (+ D for heavy chains; omit D for light). ``anchors`` optionally supplies
        {gene: {allele_name: pos}} so alleles keep their junction anchor.

        The locus schema comes from ``locus`` when given (one locus over every allele), else from the allele names:
        names that show one locus give it to every allele (those whose name shows none join it); names that show
        several, every allele showing one, group each gene's alleles by locus (loci in order of first appearance over
        V, then J, then D; the file's order within a locus); otherwise there is no schema. The alleles otherwise
        keep the file's order. A chip reads its alleles from the bound reference, so the result can be bound to a
        model (``predict --reference``), novel alleles included, or trained against.
        """
        upper = {k.upper(): v for k, v in genes.items()}
        anc_in = {k.upper(): v for k, v in (anchors or {}).items()}
        locus = locus.upper() if locus is not None else None
        has_d = bool(upper.get("D"))
        wanted = ["V", "J"] + (["D"] if has_d else [])
        alleles = {g: dict(upper.get(g, {})) for g in wanted}          # name -> sequence, first of a name kept
        of = {g: [locus or locus_from_name(name) for name in alleles[g]] for g in ("V", "J", "D") if g in alleles}
        shown = list(dict.fromkeys(lc for g in ("V", "J", "D") if g in of for lc in of[g] if lc is not None))
        grouped = len(shown) > 1 and all(lc is not None for lcs in of.values() for lc in lcs)
        if grouped:                                   # several loci: each gene's alleles grouped by locus
            alleles = {g: _grouped_by_locus(alleles[g], of[g], shown) for g in alleles}
        out = {g: _gene_from_alleles(alleles[g], anc_in.get(g, {})) for g in wanted}
        if len(shown) == 1 or locus is not None:      # one locus over every allele
            only = locus or shown[0]
            loci = loci_from_membership({g: [only] * len(out[g]) for g in out}, [only], CHAIN_OF_LOCUS)
        elif grouped:
            loci = loci_from_membership({g: [locus_from_name(n) for n in out[g].names] for g in out}, shown,
                                         CHAIN_OF_LOCUS)
        else:
            loci = []
        return cls(out, has_d, loci=loci)

    def subset(self, allowed: Dict[str, Iterable[str]]) -> "ReferenceSet":
        """Return a new ReferenceSet keeping only the named alleles per gene (a donor's
        reduced genotype), PRESERVING anchors/junction and the IMGT-gapped forms. ``allowed`` = {V|D|J: allele
        names}; a segment it does not name keeps all its alleles. The kept alleles are in this reference's order, each
        once, whatever order (or set) they are named in: the chip's scores depend on its bank's row order."""
        want = {g.upper(): set(v) for g, v in allowed.items()}
        out: Dict[str, GeneReference] = {}
        for G, ref in self.genes.items():
            keep = [n for n in ref.names if n in want[G]] if G in want else list(ref.names)
            seqs = [ref.sequences[ref.index[n]] for n in keep]
            anc = ({n: ref.anchors[n] for n in keep if n in ref.anchors}
                   if ref.anchors else None)
            gapped = ({n: ref.gapped[n] for n in keep if n in ref.gapped} if ref.gapped else None)
            info = {n: ref.info[n] for n in keep if n in ref.info} if ref.info else None
            out[G] = GeneReference(keep, seqs, {n: i for i, n in enumerate(keep)},
                                   anchors=anc or None, gapped=gapped or None, info=info or None)
        membership = self.membership()
        per_index = {G: [membership[G][self.genes[G].index[n]] for n in out[G].names] for G in out}
        return ReferenceSet(out, self.has_d, loci=self._loci_keeping_fields(per_index), provenance=self.provenance)

    def membership(self) -> Dict[str, list]:
        """Each gene's per-index locus (None for an allele no locus range holds), from the locus schema."""
        membership = {G: [None] * len(ref) for G, ref in self.genes.items()}
        for info in self.loci:
            for G, (start, end) in info.ranges.items():
                if G in membership:
                    membership[G][start:end] = [info.locus] * (end - start)
        return membership

    def _loci_keeping_fields(self, per_index: Dict[str, list], extra: List[LocusInfo] = ()) -> List[LocusInfo]:
        """This reference's loci (then ``extra`` ones) with ranges recomputed from ``per_index``, every other field kept;
        none when this reference has no schema."""
        if not self.loci:
            return []
        known = [*self.loci, *extra]
        rebuilt = loci_from_membership(per_index, [l.locus for l in known])
        return [dataclasses.replace(info, ranges=new.ranges, has_d=info.has_d or new.has_d)
                for info, new in zip(known, rebuilt)]

    @staticmethod
    def _insertion(order: list, locus: str, loci: list) -> int:
        """Where an allele of ``locus`` goes in a gene's ``(name, locus)`` order: after its locus's block, else after
        the blocks of the loci before it, else first."""
        for candidate in loci[:loci.index(locus) + 1][::-1]:
            last = max((i for i, (_, x) in enumerate(order) if x == candidate), default=None)
            if last is not None:
                return last + 1
        return 0

    def extend(self, other: "ReferenceSet", *, on_conflict: str = "error") -> "ReferenceSet":
        """This reference with ``other``'s alleles added (a donor's novel or undocumented alleles), each at the end of
        its locus's block in its gene; this reference is unchanged.

        An allele of ``other`` this reference holds with the same sequence is skipped; with another sequence,
        ``on_conflict="error"`` refuses (InputError naming them) and ``"keep"`` keeps this reference's. A new allele's
        locus is ``other``'s, else its name's, else this reference's only locus; one no locus can hold is refused.
        A locus this reference lacks is added after its loci. Anchors, gapped forms and info come along; the
        provenance is this reference's followed by ``other``'s new sources."""
        if on_conflict not in ("error", "keep"):
            raise ValueError(f"on_conflict is 'error' or 'keep', got {on_conflict!r}")
        conflicts = [n for G, ref in other.genes.items() if G in self.genes for n, seq in zip(ref.names, ref.sequences)
                     if n in self.genes[G].index and self.genes[G].sequences[self.genes[G].index[n]] != seq]
        if conflicts and on_conflict == "error":
            raise InputError(f"the added alleles name alleles the reference holds with another sequence: "
                             f"{', '.join(conflicts[:5])}", context={"alleles": conflicts},
                             hint="Rename them, or pass on_conflict='keep' to keep the reference's.")
        added = {G: [n for n in ref.names if n not in (self.genes[G].index if G in self.genes else {})]
                 for G, ref in other.genes.items()}
        loci_of, theirs = {}, other.membership()
        for G, names in added.items():
            for n in names:
                locus = theirs[G][other.genes[G].index[n]] or locus_from_name(n) or (
                    self.loci[0].locus if len(self.loci) == 1 else None)
                if locus is None and self.loci:
                    raise InputError(f"no locus of the reference can hold the added allele {n}",
                                     context={"allele": n, "loci": list(self.locus_names())},
                                     hint="Name it after its locus (e.g. IGKV...), or add it to a one-locus reference.")
                loci_of[(G, n)] = locus
        new_loci = list(dict.fromkeys(loc for loc in loci_of.values() if loc not in self.locus_names()))
        membership = self.membership()
        genes, per_index = {}, {}
        for G in dict.fromkeys([*self.genes, *other.genes]):
            mine = self.genes.get(G) or GeneReference([], [], {})
            order = list(zip(mine.names, membership.get(G, [])))
            all_loci = [*self.locus_names(), *new_loci]
            for n in added.get(G, []):
                locus = loci_of[(G, n)]
                order.insert(self._insertion(order, locus, all_loci) if self.loci else len(order), (n, locus))
            source = {n: (self.genes[G] if n in mine.index else other.genes[G]) for n, _ in order}
            names = [n for n, _ in order]
            genes[G] = GeneReference(names, [source[n].sequences[source[n].index[n]] for n in names],
                                     {n: i for i, n in enumerate(names)},
                                     anchors=_pick(source, names, "anchors") or None,
                                     gapped=_pick(source, names, "gapped") or None,
                                     info=_pick(source, names, "info") or None)
            per_index[G] = [x for _, x in order]
        extra = [next((dataclasses.replace(l, ranges={}) for l in other.loci if l.locus == loc), None)
                 or LocusInfo(loc, CHAIN_OF_LOCUS.get(loc, loc), False) for loc in new_loci]
        provenance = [*self.provenance, *(src for src in other.provenance if src not in self.provenance)]
        return ReferenceSet(genes, self.has_d or other.has_d, loci=self._loci_keeping_fields(per_index, extra),
                            provenance=provenance)

    @staticmethod
    def _infer_segment(name: str) -> str | None:
        """V, D or J for an allele name: the segment after its IG/TR locus code (IGHV1-2*01 -> V, TRBJ2-1 -> J), else
        the one segment letter the name holds (myV1 -> V); None when it holds none of V/D/J, or several."""
        m = _SEGMENT_RE.search(name.upper())
        if m:
            return m.group(1)
        letters = {c for c in name.upper() if c in "VDJ"}
        return letters.pop() if len(letters) == 1 else None

    @classmethod
    def from_fasta(cls, path: str) -> "ReferenceSet":
        """Load a genotype FASTA (``>allele_name`` headers + DNA). Each allele's segment (V/D/J) is read from its name
        (``_infer_segment``); an InputError names the alleles it cannot be read from, and those named twice with two
        sequences.

        Like :meth:`from_genotype`, the result can be bound to a model or trained against."""
        genes: Dict[str, Dict[str, str]] = {"V": {}, "D": {}, "J": {}}
        name = None
        chunks: List[str] = []
        twice: List[str] = []
        unplaced: List[str] = []

        def _flush():
            if name is not None:
                seg = cls._infer_segment(name)
                if seg is None:
                    unplaced.append(name)
                else:
                    sequence = "".join(chunks)
                    if name in genes[seg] and genes[seg][name].upper() != sequence.upper():
                        twice.append(name)
                    genes[seg].setdefault(name, sequence)

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    _flush()
                    name = line[1:].split()[0]
                    chunks = []
                else:
                    chunks.append(line)
        _flush()
        if unplaced:
            raise InputError(f"{path}: the segment (V, D or J) of {len(unplaced)} alleles cannot be read from their "
                             f"names: {', '.join(unplaced[:5])}", context={"path": path, "alleles": unplaced},
                             hint="Name each allele with its locus and segment (IGHV1-2*02, TRBJ2-1*01), or with "
                                  "exactly one of the letters V, D, J.")
        if twice:
            raise InputError(f"{path} names {len(twice)} alleles twice with two sequences: {', '.join(twice[:5])}",
                             context={"path": path, "alleles": twice}, hint="Give each allele one name.")
        return cls.from_genotype({g: m for g, m in genes.items() if m})


# An IG/TR locus code and the segment letter after it, anywhere in an allele name (``_infer_segment``).
_SEGMENT_RE = re.compile(r"(?:IG[HKL]|TR[ABGD])([VDJ])")


def _grouped_by_locus(alleles: dict[str, str], loci: list, order: list) -> dict[str, str]:
    """``alleles`` (name -> sequence, each with its locus in ``loci``) grouped by locus in ``order``, the given order
    kept within a locus."""
    return {n: alleles[n] for lc in order for n, x in zip(list(alleles), loci) if x == lc}


def _gene_from_alleles(alleles: dict[str, str], anchors: dict) -> GeneReference:
    """A gene's reference from its alleles (name -> sequence, ``-`` or ``.`` gaps allowed: an IMGT-gapped sequence
    keeps its gapped form) and the anchors given for them."""
    names: list[str] = []
    sequences: list[str] = []
    index: dict[str, int] = {}
    kept_anchors, gapped = {}, {}
    for name, seq in alleles.items():
        index[name] = len(names)
        names.append(name)
        written = str(seq).upper().replace("-", ".")
        sequences.append(written.replace(".", ""))
        if "." in written:
            gapped[name] = written
        if name in anchors:
            kept_anchors[name] = int(anchors[name])
    return GeneReference(names, sequences, index, anchors=kept_anchors or None, gapped=gapped or None)


def _pick(source: dict, names: list[str], attr: str) -> dict:
    """``attr`` (``anchors``, ``gapped`` or ``info``) of each of ``names`` that its source reference has."""
    return {n: getattr(source[n], attr)[n] for n in names if getattr(source[n], attr) and n in getattr(source[n], attr)}

