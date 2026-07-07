"""ReferenceSet: union 1..N GenAIRR dataconfigs into per-gene allele references."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch


@dataclass
class GeneReference:
    names: List[str]          # ordered allele names
    sequences: List[str]      # germline nucleotide seqs (uppercased), aligned with names
    index: Dict[str, int]     # name -> row index
    anchors: Dict[str, int] | None = None  # name -> conserved-anchor germline pos (Cys/Trp-Phe)
    gapped: Dict[str, str] | None = None   # name -> IMGT-gapped germline (AIRR sequence_alignment)

    def __len__(self) -> int:
        return len(self.names)


class ReferenceSet:
    """Per-gene union allele references built from one or more GenAIRR DataConfigs."""

    def __init__(self, genes: Dict[str, GeneReference], has_d: bool):
        self.genes = genes
        self.has_d = has_d

    def gene(self, g: str) -> GeneReference:
        return self.genes[g.upper()]

    @classmethod
    def from_dataconfigs(cls, *dataconfigs) -> "ReferenceSet":
        has_d = any(dc.metadata.has_d for dc in dataconfigs)
        wanted = ["v", "j"] + (["d"] if has_d else [])
        genes: Dict[str, GeneReference] = {}
        for g in wanted:
            names: List[str] = []
            sequences: List[str] = []
            index: Dict[str, int] = {}
            anchors: Dict[str, int] = {}
            gapped: Dict[str, str] = {}
            for dc in dataconfigs:
                if g == "d" and not dc.metadata.has_d:
                    continue
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
            genes[g.upper()] = GeneReference(names, sequences, index,
                                             anchors=anchors or None, gapped=gapped or None)
        return cls(genes, has_d)

    @classmethod
    def from_genotype(cls, genes: Dict[str, Dict[str, str]],
                      anchors: Dict[str, Dict[str, int]] | None = None) -> "ReferenceSet":
        """Build a reference from a genotype mapping {gene_type: {allele_name: dna_seq}}.

        gene_type is V/J (+ D for heavy chains; omit D for light). Allele names need
        NOT be from the training reference — NOVEL alleles are just rows the encoder
        will embed at predict time, so the model conditions on whatever it is handed.
        ``anchors`` optionally supplies {gene: {allele_name: pos}} so KNOWN alleles keep
        their junction anchor (novel alleles simply omit it -> no junction emitted).
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
        Subset OR novel alleles both work — every record is a row the encoder embeds at
        predict time, so the model conditions on exactly what the file provides."""
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
