"""ReferenceSet: union 1..N GenAIRR dataconfigs into per-gene allele references."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import numpy as np
import torch

# GenAIRR chain-type value -> AIRR locus code
_LOCUS_OF_CHAIN = {
    "BCR_HEAVY": "IGH", "BCR_LIGHT_KAPPA": "IGK", "BCR_LIGHT_LAMBDA": "IGL",
    "TCR_ALPHA": "TRA", "TCR_BETA": "TRB", "TCR_GAMMA": "TRG", "TCR_DELTA": "TRD",
}
_LOCUS_NAME_RE = re.compile(r"^(IG[HKL]|TR[ABGD])")


def _locus_of(chain_type) -> str:
    key = str(getattr(chain_type, "value", chain_type)).upper()
    return _LOCUS_OF_CHAIN.get(key, key)


def _locus_from_name(name: str):
    """The AIRR locus a standard IMGT/AIRR allele name belongs to (IGKV1-5*01 -> IGK), or None."""
    m = _LOCUS_NAME_RE.match(str(name).upper())
    return m.group(1) if m else None


def _infer_loci_from_names(genes: Dict[str, "GeneReference"]) -> List["LocusInfo"]:
    """Reconstruct the per-locus schema from allele names alone — for references serialized before the
    schema was stored. Order follows first appearance in the V head (== the union/chain_type-class-id
    order, since the union appends per-dataconfig), so it aligns with a trained chain_type head."""
    order: List[str] = []
    spans: Dict[tuple, list] = {}
    for G in ("V", "J", "D"):
        ref = genes.get(G)
        if ref is None:
            continue
        for i, name in enumerate(ref.names):
            lc = _locus_from_name(name)
            if lc is None:
                continue
            if lc not in order:
                order.append(lc)
            key = (lc, G)
            if key in spans:
                spans[key] = [min(spans[key][0], i), max(spans[key][1], i)]
            else:
                spans[key] = [i, i]
    loci = []
    for lc in order:
        ranges = {G: (spans[(lc, G)][0], spans[(lc, G)][1] + 1) for G in ("V", "D", "J") if (lc, G) in spans}
        loci.append(LocusInfo(locus=lc, chain_type=lc, has_d=(lc, "D") in spans, ranges=ranges))
    return loci


@dataclass
class GeneReference:
    names: List[str]          # ordered allele names
    sequences: List[str]      # germline nucleotide seqs (uppercased), aligned with names
    index: Dict[str, int]     # name -> row index
    anchors: Dict[str, int] | None = None  # name -> conserved-anchor germline pos (Cys/Trp-Phe)
    gapped: Dict[str, str] | None = None   # name -> IMGT-gapped germline (AIRR sequence_alignment)

    def __len__(self) -> int:
        return len(self.names)


@dataclass
class LocusInfo:
    """One locus in a (possibly multi-chain) union: its AIRR code, GenAIRR chain type, whether it has
    a D gene, and the per-gene ``[start, end)`` index ranges it occupies in the union allele heads.
    The list index equals the model's chain_type class id, so a predicted chain_type maps straight to a
    locus and to the allele-index range that read is allowed to call within (P0-6)."""
    locus: str
    chain_type: str
    has_d: bool
    ranges: Dict[str, tuple] = field(default_factory=dict)   # {"V": (s,e), "J": (s,e), ["D": (s,e)]}


class ReferenceSet:
    """Per-gene union allele references built from one or more GenAIRR DataConfigs."""

    def __init__(self, genes: Dict[str, GeneReference], has_d: bool, loci: List[LocusInfo] | None = None):
        self.genes = genes
        self.has_d = has_d
        self.loci = loci or []

    def gene(self, g: str) -> GeneReference:
        return self.genes[g.upper()]

    def locus_names(self) -> tuple:
        """Ordered AIRR locus codes, aligned to the model's chain_type class id (empty if unknown)."""
        return tuple(l.locus for l in self.loci)

    def locus_mask(self, locus: str, gene: str) -> np.ndarray:
        """Boolean mask over the ``gene`` head selecting exactly the alleles that belong to ``locus``
        (all-True when the locus schema is absent, so single-chain models are unaffected)."""
        n = len(self.gene(gene))
        info = next((l for l in self.loci if l.locus == locus), None)
        if info is None or gene.upper() not in info.ranges:
            return np.ones(n, dtype=bool)
        s, e = info.ranges[gene.upper()]
        m = np.zeros(n, dtype=bool)
        m[s:e] = True
        return m

    @classmethod
    def from_dataconfigs(cls, *dataconfigs) -> "ReferenceSet":
        has_d = any(dc.metadata.has_d for dc in dataconfigs)
        wanted = ["v", "j"] + (["d"] if has_d else [])
        genes: Dict[str, GeneReference] = {}
        locus_order: List[str] = []                       # distinct loci, in chain_type-class-id order
        locus_hasd: Dict[str, bool] = {}
        locus_chain: Dict[str, str] = {}
        spans: Dict[tuple, tuple] = {}                    # (locus, GENE) -> [start, end) in the union head
        for g in wanted:
            names: List[str] = []
            sequences: List[str] = []
            index: Dict[str, int] = {}
            anchors: Dict[str, int] = {}
            gapped: Dict[str, str] = {}
            for dc in dataconfigs:
                locus = _locus_of(dc.metadata.chain_type)
                if locus not in locus_order:
                    locus_order.append(locus)
                    locus_hasd[locus] = bool(dc.metadata.has_d)
                    locus_chain[locus] = str(getattr(dc.metadata.chain_type, "value", dc.metadata.chain_type))
                if g == "d" and not dc.metadata.has_d:
                    continue
                start = len(names)
                for allele in dc.allele_list(g):
                    if allele.name in index:
                        continue
                    index[allele.name] = len(names)
                    names.append(allele.name)
                    sequences.append(allele.ungapped_seq.upper())
                    # conserved junction anchor (Cys-104 for V, Trp/Phe-118 for J); ungapped
                    # germline position. Used to derive the AIRR junction; absent on D.
                    anc = getattr(allele, "anchor", None)
                    if anc is not None:
                        anchors[allele.name] = int(anc)
                    gap = getattr(allele, "gapped_seq", None)
                    if gap:
                        gapped[allele.name] = gap.upper()
                if len(names) > start:                    # this locus contributed a contiguous block
                    key = (locus, g.upper())
                    prev = spans.get(key)
                    spans[key] = (min(prev[0], start), max(prev[1], len(names))) if prev else (start, len(names))
            genes[g.upper()] = GeneReference(names, sequences, index,
                                             anchors=anchors or None, gapped=gapped or None)
        loci = [LocusInfo(locus=lc, chain_type=locus_chain[lc], has_d=locus_hasd[lc],
                          ranges={G: spans[(lc, G)] for G in ("V", "D", "J") if (lc, G) in spans})
                for lc in locus_order]
        return cls(genes, has_d, loci=loci)

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
        loci = [{"locus": l.locus, "chain_type": l.chain_type, "has_d": l.has_d,
                 "ranges": {G: list(se) for G, se in l.ranges.items()}} for l in self.loci]
        return {"schema": "alignair.reference.v1", "has_d": self.has_d, "genes": genes, "loci": loci}

    @classmethod
    def from_serializable(cls, data: dict) -> "ReferenceSet":
        """Rebuild from :meth:`to_serializable` (the safe, no-pickle inference load path)."""
        out: Dict[str, GeneReference] = {}
        for G, gd in data["genes"].items():
            names = list(gd["names"])
            sequences = [str(s).upper() for s in gd["sequences"]]
            anchors = {k: int(v) for k, v in (gd.get("anchors") or {}).items()} or None
            gapped = {k: str(v).upper() for k, v in (gd.get("gapped") or {}).items()} or None
            out[G.upper()] = GeneReference(names, sequences, {n: i for i, n in enumerate(names)},
                                           anchors=anchors, gapped=gapped)
        loci = [LocusInfo(locus=l["locus"], chain_type=l.get("chain_type", l["locus"]),
                          has_d=bool(l.get("has_d")),
                          ranges={G: tuple(se) for G, se in (l.get("ranges") or {}).items()})
                for l in data.get("loci", [])]
        if not loci:                                     # references serialized before the schema existed
            loci = _infer_loci_from_names(out)
        return cls(out, bool(data.get("has_d")), loci=loci)

    @classmethod
    def from_genotype(cls, genes: Dict[str, Dict[str, str]],
                      anchors: Dict[str, Dict[str, int]] | None = None) -> "ReferenceSet":
        """Build a reference from a genotype mapping {gene_type: {allele_name: dna_seq}}.

        gene_type is V/J (+ D for heavy chains; omit D for light). ``anchors`` optionally supplies
        {gene: {allele_name: pos}} so alleles keep their junction anchor.

        NOTE (P0-1): the *fixed-head* production model's V/D/J classification indices are tied to the
        allele order it was trained on. This builder produces a reference in whatever order the file
        provides, which will **not** match a trained head — so it is for reference-conditioned/retrieval
        code and for constructing a reference to (re)train against, NOT for relabeling a fixed-head
        model's outputs. Novel alleles are not callable by a fixed-head model (see
        :class:`alignair.genotype.NovelAlleleUnsupportedError`).
        """
        upper = {k.upper(): v for k, v in genes.items()}
        anc_in = {k.upper(): v for k, v in (anchors or {}).items()}
        has_d = bool(upper.get("D"))
        wanted = ["V", "J"] + (["D"] if has_d else [])
        out: Dict[str, GeneReference] = {}
        for g in wanted:
            names: List[str] = []
            sequences: List[str] = []
            index: Dict[str, int] = {}
            ganc = {}
            for name, seq in upper.get(g, {}).items():
                if name in index:
                    continue
                index[name] = len(names)
                names.append(name)
                sequences.append(str(seq).upper().replace("-", "").replace(".", ""))
                if name in anc_in.get(g, {}):
                    ganc[name] = int(anc_in[g][name])
            out[g] = GeneReference(names, sequences, index, anchors=ganc or None)
        return cls(out, has_d)

    def subset(self, allowed: Dict[str, Iterable[str]]) -> "ReferenceSet":
        """Return a new ReferenceSet keeping only the named alleles per gene (a donor's
        reduced genotype), PRESERVING anchors/junction. ``allowed`` = {gene: [names]}."""
        want = {g.upper(): list(v) for g, v in allowed.items()}
        out: Dict[str, GeneReference] = {}
        for G, ref in self.genes.items():
            keep = [n for n in want.get(G, ref.names) if n in ref.index]
            seqs = [ref.sequences[ref.index[n]] for n in keep]
            anc = ({n: ref.anchors[n] for n in keep if n in ref.anchors}
                   if ref.anchors else None)
            out[G] = GeneReference(keep, seqs, {n: i for i, n in enumerate(keep)},
                                   anchors=anc or None)
        return ReferenceSet(out, self.has_d)

    @classmethod
    def from_yaml(cls, path: str) -> "ReferenceSet":
        """Load a genotype YAML: top-level keys v/d/j, each {allele_name: dna_seq}."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_genotype(data)

    @staticmethod
    def _infer_segment(name: str) -> str | None:
        """Infer V/D/J from an AIRR/IMGT allele name (IGHV1-2*01 -> V, TRBJ2-1 -> J)."""
        import re
        m = re.search(r"(?:IG[HKL]|TR[ABGD])([VDJ])", name.upper())
        if m:
            return m.group(1)
        return next((c for c in name.upper() if c in "VDJ"), None)

    @classmethod
    def from_fasta(cls, path: str) -> "ReferenceSet":
        """Load a genotype FASTA (``>allele_name`` headers + DNA). Gene type (V/D/J) is
        inferred from each allele name; alleles whose segment can't be inferred are skipped.

        See :meth:`from_genotype` re: the fixed-head contract — the resulting allele order will not
        match a trained head, so this is for (re)training / reference-conditioned code, not for
        relabeling a fixed-head model's output columns (P0-1)."""
        genes: Dict[str, Dict[str, str]] = {"V": {}, "D": {}, "J": {}}
        name = None
        chunks: List[str] = []

        def _flush():
            if name is not None:
                seg = cls._infer_segment(name)
                if seg is not None:
                    genes[seg][name] = "".join(chunks)

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
        return cls.from_genotype({g: m for g, m in genes.items() if m})

    def to_yaml(self, path: str) -> None:
        import yaml
        data = {g.lower(): dict(zip(ref.names, ref.sequences))
                for g, ref in self.genes.items()}
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    def infer_locus(self) -> str | None:
        """Infer the locus (IGH/IGK/IGL/TRA/TRB/TRD/TRG) from the V allele names, or None."""
        import re
        from collections import Counter
        loci = Counter()
        for n in self.gene("V").names:
            m = re.match(r"(IG[HKL]|TR[ABGD])", n.upper())
            if m:
                loci[m.group(1)] += 1
        return loci.most_common(1)[0][0] if loci else None

    def genotype_mask(self, gene: str, allowed_names: Iterable[str]) -> torch.Tensor:
        ref = self.gene(gene)
        allowed = set(allowed_names)
        return torch.tensor([n in allowed for n in ref.names], dtype=torch.bool)
