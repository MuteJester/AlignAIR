"""Germline reference used by the assay: allele sequences per gene plus conserved anchors.

With the ``diagnostics`` layer, this is the only assay code that imports ``alignair`` (``Reference.from_dataconfig``).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace

from alignair_assay.core.cases import read_fasta

# The GenAIRR dataconfig every command uses unless ``--reference`` names another.
DEFAULT_REFERENCE = "HUMAN_IGH_OGRDB"


def dataconfig(name: str):
    """The GenAIRR DataConfig ``name`` (an attribute of ``GenAIRR.data``, e.g. ``HUMAN_IGH_OGRDB``); ValueError for
    an unknown name. Suites simulate from this object and the truth reference is built from it."""
    from alignair.errors import ConfigurationError
    from alignair.genairr import resolve_dataconfig
    try:
        return resolve_dataconfig(name)
    except ConfigurationError as error:
        raise ValueError(f"unknown GenAIRR dataconfig {name!r}") from error


def is_tcr(config) -> bool:
    """Whether the GenAIRR DataConfig ``config`` is a T-cell receptor's (no somatic hypermutation)."""
    from alignair.reference.reference_set import is_tcr as declares_tcr
    return declares_tcr(config)


def gene_name(allele: str) -> str:
    """Gene part of an allele name (text before ``*``)."""
    return allele.split("*", 1)[0]


@dataclass(frozen=True)
class Reference:
    """The germline alleles every truth and score is built against, keyed by gene letter and allele name, with the
    conserved junction anchors (0-based germline positions)."""

    # gene letter ("v" | "d" | "j") -> allele name -> uppercase germline sequence
    alleles: dict[str, dict[str, str]]
    # allele name -> 0-based germline anchor (V Cys codon start, J Trp/Phe codon start)
    anchors: dict[str, int]
    # V allele name -> uppercase IMGT-gapped sequence ("." for a gap column), where the source has one (a GenAIRR
    # dataconfig does; a FASTA file and a ReferenceSet do not). Not part of equality or of the digest.
    gapped_v: dict[str, str] = field(default_factory=dict, compare=False)

    def seq(self, name: str) -> str:
        """Germline sequence of allele ``name`` (KeyError when it is not in the reference)."""
        return self.alleles[self.gene_letter(name)][name]

    def gene_letter(self, name: str) -> str:
        """``"v"``, ``"d"`` or ``"j"``: the gene table holding allele ``name`` (KeyError when none does)."""
        for letter, table in self.alleles.items():
            if name in table:
                return letter
        raise KeyError(name)

    def gapped(self, name: str) -> str | None:
        """IMGT-gapped sequence of allele ``name`` (uppercase, ``.`` for a gap column, so IMGT position ``p`` is
        column ``p - 1``), or None when the reference holds none for it (D and J alleles, and a reference read from a
        FASTA file). KeyError when the allele is not in the reference."""
        self.gene_letter(name)
        return self.gapped_v.get(name)

    def has(self, name: str) -> bool:
        """True when allele ``name`` is in the reference."""
        return any(name in table for table in self.alleles.values())

    def digest(self) -> str:
        """SHA-256 over every allele's gene letter, name and sequence, in sorted order (anchors excluded)."""
        h = hashlib.sha256()
        for letter in sorted(self.alleles):
            for name in sorted(self.alleles[letter]):
                h.update(f"{letter}\t{name}\t{self.alleles[letter][name]}\n".encode())
        return h.hexdigest()

    def gapped_digest(self) -> str:
        """SHA-256 over every IMGT-gapped V form's name and sequence, in sorted order. ``digest`` leaves the gapped
        forms out, so the caches keyed on it stay valid; what reads the gapped forms (Cys104, the 318 rule) is keyed
        on this too."""
        h = hashlib.sha256()
        for name in sorted(self.gapped_v):
            h.update(f"{name}\t{self.gapped_v[name]}\n".encode())
        return h.hexdigest()

    def sequences(self) -> dict[str, str]:
        """Every allele's sequence by name, across the gene letters."""
        return {name: seq for table in self.alleles.values() for name, seq in table.items()}

    def counts(self) -> dict[str, int]:
        """Allele count per gene letter, in v, d, j order; a letter the reference lacks is left out."""
        return {g: len(self.alleles[g]) for g in ("v", "d", "j") if g in self.alleles}

    def counts_text(self) -> str:
        """The allele counts as text, e.g. ``V=198 D=33 J=7``."""
        return " ".join(f"{g.upper()}={n}" for g, n in self.counts().items())

    def first_difference(self, truth: "Reference", label: str) -> str:
        """The first allele, in name order, that this reference (called ``label``, e.g. ``"tool"``) lacks, adds or
        holds with another sequence than ``truth``, as text naming the allele."""
        mine, theirs = self.sequences(), truth.sequences()
        for name in sorted(mine.keys() | theirs.keys()):
            if name not in mine:
                return f"{name} is missing from the {label} reference"
            if name not in theirs:
                return f"{name} is extra in the {label} reference"
            if mine[name] != theirs[name]:
                return f"{name} has a different sequence in the {label} reference"
        return "the same alleles under other gene letters"

    def first_missing(self, truth: "Reference", label: str) -> str | None:
        """The first allele of ``truth``, in name order, that this reference (called ``label``) lacks or holds with
        another sequence, as text naming the allele; None when it holds every truth allele with its sequence (it
        may hold more: a reference of several loci holds a one-locus truth reference)."""
        mine, theirs = self.sequences(), truth.sequences()
        for name in sorted(theirs):
            if name not in mine:
                return f"{name} is missing from the {label} reference"
            if mine[name] != theirs[name]:
                return f"{name} has a different sequence in the {label} reference"
        return None

    @classmethod
    def from_reference_set(cls, rs) -> "Reference":
        """Copy an ``alignair`` ReferenceSet: uppercase sequences per gene (D only when the locus has one) and the
        V/J anchors."""
        alleles: dict[str, dict[str, str]] = {}
        anchors: dict[str, int] = {}
        for letter in ("V", "D", "J"):
            if letter == "D" and not rs.has_d:
                continue
            gene = rs.gene(letter)
            alleles[letter.lower()] = dict(zip(gene.names, (s.upper() for s in gene.sequences)))
            if gene.anchors:
                anchors.update({k: int(v) for k, v in gene.anchors.items()})
        return cls(alleles=alleles, anchors=anchors)

    @classmethod
    def from_fasta(cls, path) -> "Reference":
        """The reference a FASTA file holds (for example one a tool exported): one allele per record, named by
        the header's first word, with its sequence uppercased. The gene letter is the lower-cased fourth
        character of the name (``IGHV1-2*02`` -> ``v``). Anchors are not in a FASTA file, so they are ``{}``.
        A name whose letter is not ``v``, ``d`` or ``j``, or that appears twice, raises ValueError."""
        alleles: dict[str, dict[str, str]] = {}
        for name, sequence in read_fasta(path):
            letter = name[3:4].lower()
            if letter not in ("v", "d", "j"):
                raise ValueError(f"{path}: allele {name!r} has no V, D or J gene letter as its fourth character")
            if any(name in table for table in alleles.values()):
                raise ValueError(f"{path}: allele {name!r} appears twice")
            alleles.setdefault(letter, {})[name] = sequence.upper()
        return cls(alleles=alleles, anchors={})

    @classmethod
    def from_dataconfig(cls, name: str) -> "Reference":
        """The reference of GenAIRR dataconfig ``name`` (see ``dataconfig``), with the dataconfig's IMGT-gapped V
        sequences (``gapped``; one whose gaps removed is not the allele's sequence is left out); ValueError for an
        unknown name."""
        from alignair.reference.reference_set import ReferenceSet

        config = dataconfig(name)
        reference = cls.from_reference_set(ReferenceSet.from_dataconfigs(config))
        v = reference.alleles.get("v", {})
        gapped = {allele.name: allele.gapped_seq.upper() for alleles in (config.v_alleles or {}).values()
                  for allele in alleles if allele.gapped_seq}
        return replace(reference, gapped_v={allele: seq for allele, seq in gapped.items()
                                            if v.get(allele) == seq.replace(".", "")})
