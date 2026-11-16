"""A training batch's bank: the reference extended with the novel alleles its reads were simulated from.

A read from a novel allele carries the allele's definition (``alignair.reference.variants.NOVEL_FIELD``).
``batch_bank`` collects a batch's, annotates them against the training reference (each takes its base's anchor and
gap columns, as inference anchors a novel allele: ``alignair.reference.annotate``) and extends the reference with
them; a novel allele whose sequence is a reference allele's is that allele (an alias, no row). The batch's targets are
built against the extended reference; a read that cannot tell a novel allele from its base over its germline span
targets both (``BatchBank.ties``); each novel allele's base joins the bank mask (``paired``), so the chip must tell
them apart; the extension's rows ship in the batch's inputs (``network.bank.extension_inputs``)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from ..errors import TrainingDataError
from ..reference import annotate
from ..reference.reference_set import ReferenceSet
from ..reference.variants import NOVEL_FIELD, NovelAllele

_GENES = ("V", "J")


@dataclass(frozen=True)
class BatchBank:
    """A batch's bank when its reads include novel alleles: the training reference plus a row per novel allele."""
    reference: ReferenceSet                              # the training reference extended with the batch's novel rows
    base: ReferenceSet                                   # the training reference
    novel: Mapping[str, tuple]                           # gene -> the novel alleles given a row
    aliases: Mapping[str, Mapping[str, str]]             # gene -> novel name -> the reference allele it equals

    def resolve(self, record):
        """``record``, or a copy whose call of a novel allele equal to a reference allele names that allele."""
        changed = {f"{gene.lower()}_call": names[record[f"{gene.lower()}_call"]]
                   for gene, names in self.aliases.items() if record.get(f"{gene.lower()}_call") in names}
        return {**record, **changed} if changed else record

    def ties(self, gene: str, call: str, span: tuple[int, int]) -> frozenset:
        """The other bank alleles of ``call``'s family (its base and the base's novel alleles in this batch; for a
        reference allele, its novel alleles) of ``call``'s length whose bases equal ``call``'s over ``span``."""
        gene = gene.upper()
        alleles = self.reference.gene(gene)
        if call not in alleles.index:
            return frozenset()
        novel = self.novel.get(gene, ())
        base = next((a.base for a in novel if a.name == call), call)
        family = {base, *(a.name for a in novel if a.base == base)} - {call}
        start, end = span
        sequence = alleles.sequences[alleles.index[call]]
        return frozenset(name for name in family
                         if len(other := alleles.sequences[alleles.index[name]]) == len(sequence)
                         and other[start:end] == sequence[start:end])

    def paired(self) -> dict:
        """Per gene (lower case): True over the bank's alleles at each novel allele's base."""
        out = {}
        for gene, novel in self.novel.items():
            alleles = self.reference.gene(gene)
            mask = torch.zeros(len(alleles), dtype=torch.bool)
            for allele in novel:
                mask[alleles.index[allele.base]] = True
            out[gene.lower()] = mask
        return out

    def inputs(self, frames) -> dict:
        """The extension's rows, for each gene with novel rows."""
        from ..network.bank import extension_inputs
        out = {}
        for gene in self.novel:
            out.update(extension_inputs(self.base, self.reference, frames, gene.lower()))
        return out


def _columns(gapped: str | None) -> list:
    return [i for i, column in enumerate(gapped or "") if column == "."]


def _novel_alleles(records) -> dict[str, dict[str, NovelAllele]]:
    """The batch's novel alleles, per gene by name (each record's ``NOVEL_FIELD``); TrainingDataError for a gene other
    than V or J, or one name with two sequences."""
    found: dict[str, dict[str, NovelAllele]] = {}
    for record in records:
        if not isinstance(record, Mapping):          # the adapter refuses it, with its record's context
            continue
        for gene, definition in (record.get(NOVEL_FIELD) or {}).items():
            allele = NovelAllele.from_dict(definition)
            if gene not in _GENES or allele.gene != gene:
                raise TrainingDataError(f"a novel allele of gene {gene} cannot enter a batch's bank (V and J only)",
                                        context={"alleles": [allele.name]})
            held = found.setdefault(gene, {})
            if allele.name in held and held[allele.name].sequence != allele.sequence:
                raise TrainingDataError(f"the batch names the novel allele {allele.name} with two sequences",
                                        context={"alleles": [allele.name]})
            held[allele.name] = allele
    return found


def _known_or_novel(reference: ReferenceSet, gene: str, alleles: dict[str, NovelAllele]
                    ) -> tuple[dict[str, str], tuple[NovelAllele, ...]]:
    """``(aliases, rows)`` of a gene's novel alleles: each one whose sequence a reference allele has, by name to that
    allele's (the first of that sequence), and the others, the bank's new rows. TrainingDataError for an allele whose
    base the reference lacks."""
    known = reference.gene(gene)
    lost = [a.name for a in alleles.values() if a.base not in known.index]
    if lost:
        raise TrainingDataError("novel alleles of a base the training reference lacks", context={"alleles": lost})
    by_sequence = {}
    for name, sequence in zip(known.names, known.sequences):
        by_sequence.setdefault(sequence, name)
    aliases = {a.name: by_sequence[a.sequence] for a in alleles.values() if a.sequence in by_sequence}
    return aliases, tuple(a for a in alleles.values() if a.name not in aliases)


def batch_bank(reference: ReferenceSet, records) -> BatchBank | None:
    """The bank of a batch whose reads carry novel alleles; None when none does. TrainingDataError for one name with
    two sequences, a base the reference lacks, a gene other than V or J, or an allele whose annotated anchor or gap
    columns are not its base's (GenAIRR simulates its reads on its base's)."""
    found = _novel_alleles(records)
    if not found:
        return None
    novel, aliases, genotype = {}, {}, {}
    for gene, alleles in found.items():
        aliases[gene], rows = _known_or_novel(reference, gene, alleles)
        if rows:
            novel[gene] = rows
            genotype[gene] = {a.name: a.sequence for a in rows}
    aliases = {gene: names for gene, names in aliases.items() if names}
    if not novel:
        return BatchBank(reference, reference, {}, aliases)
    annotated = annotate(ReferenceSet.from_genotype({"V": {}, "J": {}, **genotype}), known=reference)
    wrong = [a.name for gene, rows in novel.items() for a in rows
             if (annotated.gene(gene).anchors or {}).get(a.name) != (reference.gene(gene).anchors or {}).get(a.base)
             or _columns((annotated.gene(gene).gapped or {}).get(a.name))
             != _columns((reference.gene(gene).gapped or {}).get(a.base))]
    if wrong:
        raise TrainingDataError("novel alleles whose annotated anchor or gap columns are not their base's",
                                context={"alleles": wrong})
    return BatchBank(reference.extend(annotated), reference, novel, aliases)
