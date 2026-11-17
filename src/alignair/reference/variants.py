"""Synthetic novel alleles for training banks: seeded SNP variants of a reference's V (or J) alleles.

``snp_variant`` changes ``substitutions`` bases of an eligible base allele (``eligible_bases``: its own supplied anchor
on its conserved codon, its gapped form its sequence with gaps, no stop codon in its coding frame), never inside the
anchor codon, and keeps a variant only when its coding frame has no stop codon, it equals no allele of the reference and
its base is its one nearest allele (every other allele of its length differs from it at more bases). A variant is
named by its content (``novel_name``), so two draws of one variant name it alike. The draw order is fixed: a seeded
``random.Random`` gives the same variant in any process.

A read simulated from a novel allele carries its definition under ``NOVEL_FIELD`` (``{gene: NovelAllele.to_dict()}``),
so a batch can put the allele in its bank."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from .codons import CONSERVED_CODONS, STOP_CODONS

NOVEL_FIELD = "novel_alleles"
MAX_ATTEMPTS = 50
_GENES = ("V", "D", "J")


@dataclass(frozen=True)
class NovelAllele:
    """A novel allele: its name, the reference allele it derives from (``base``), its sequence and its gene."""
    name: str
    base: str
    sequence: str
    gene: str = "V"

    def to_dict(self) -> dict:
        """The fields as a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> NovelAllele:
        """From ``to_dict``'s form: a ValueError for a missing name, base or sequence or a gene not V, D or J."""
        data = dict(data)
        missing = [key for key in ("name", "base", "sequence") if key not in data]
        if missing:
            raise ValueError(f"a novel allele needs {', '.join(missing)}")
        gene = data.get("gene", "V")
        if gene not in _GENES:
            raise ValueError(f"a novel allele's gene is V, D or J, got {gene!r}")
        return cls(str(data["name"]), str(data["base"]), str(data["sequence"]).upper(), gene)


def novel_name(base: str, sequence: str) -> str:
    """``<base>_nv<12 hex of the sequence's SHA-256>``: the text before ``*`` stays the base's gene (GenAIRR takes the
    gene from it)."""
    return f"{base}_nv{hashlib.sha256(sequence.encode('ascii')).hexdigest()[:12]}"


def _stop_in_frame(sequence: str, anchor: int, gene: str) -> bool:
    """A stop codon in the coding frame GenAIRR checks: a V's codons before its anchor, a J's from its anchor on."""
    start, end = (0, anchor) if gene == "V" else (anchor, len(sequence))
    return any(sequence[i:i + 3] in STOP_CODONS for i in range(start, end - 2, 3))


def eligible_bases(reference, gene: str = "V") -> dict[str, tuple[str, ...]]:
    """Each gene name (the text before ``*``) with its alleles a variant may be drawn from, in reference order: the
    allele's anchor is its source's own (evidence ``supplied``, the anchor GenAIRR validates a variant on), on its
    conserved codon; its gapped form is its sequence with gaps; its coding frame has no stop codon."""
    alleles = reference.gene(gene)
    anchors, gapped, info = alleles.anchors or {}, alleles.gapped or {}, alleles.info or {}
    out: dict[str, list[str]] = {}
    for name, sequence in zip(alleles.names, alleles.sequences):
        anchor = anchors.get(name)
        evidence = info[name].anchor_evidence if name in info else None
        if (anchor is None or evidence is None or evidence.status != "supplied"
                or sequence[anchor:anchor + 3] not in CONSERVED_CODONS[gene]
                or gapped.get(name, "").replace(".", "").upper() != sequence
                or _stop_in_frame(sequence, anchor, gene)):
            continue
        out.setdefault(name.split("*")[0], []).append(name)
    return {key: tuple(names) for key, names in out.items()}


def snp_variant(reference, base: str, substitutions: int, rng) -> NovelAllele | None:
    """A variant of ``base`` at ``substitutions`` positions outside its anchor codon (see the module docstring), or
    None when ``MAX_ATTEMPTS`` draws give none."""
    gene = "V" if base in reference.gene("V").index else "J"
    alleles = reference.gene(gene)
    sequence, anchor = alleles.sequences[alleles.index[base]], alleles.anchors[base]
    positions = [p for p in range(len(sequence)) if not anchor <= p < anchor + 3]
    known = set(alleles.sequences)
    others = [other for name, other in zip(alleles.names, alleles.sequences)
              if name != base and len(other) == len(sequence)]
    for _ in range(MAX_ATTEMPTS):
        variant = list(sequence)
        for p in sorted(rng.sample(positions, substitutions)):
            variant[p] = rng.choice([b for b in "ACGT" if b != sequence[p]])
        drawn = "".join(variant)
        if (_stop_in_frame(drawn, anchor, gene) or drawn in known
                or any(sum(a != b for a, b in zip(other, drawn)) <= substitutions for other in others)):
            continue
        return NovelAllele(novel_name(base, drawn), base, drawn, gene)
    return None
