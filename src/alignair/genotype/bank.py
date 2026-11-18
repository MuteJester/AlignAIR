"""A donor's genotype as the chip's bank: the subset of a reference the chip reads its calls from."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from ..errors import InputError, UnknownAlleleError
from ..reference.reference_set import ReferenceSet
from .files import GenesOnly, check_segment_keys, check_segment_values
from .infer import gene_of


def genotype_bank(reference: ReferenceSet, genotype: Mapping[str, Iterable[str]]) -> ReferenceSet:
    """``reference``'s subset of the alleles ``genotype`` names (``{gene letter: allele names}``, any case), in the
    reference's order and with its locus schema (every locus stays, its ranges over the kept alleles); a segment it
    does not name keeps all its alleles, and so, for a ``GenesOnly`` genotype (an AIRR GenotypeSet), does every gene
    it does not name. ConfigurationError for a key that is not a segment (V, D, J), gene names
    included, or an entry that is not a list of names (one string, say); UnknownAlleleError when it names alleles
    the reference does not hold (give their sequences in the reference); InputError when it leaves a gene with no
    allele."""
    check_segment_keys(genotype)
    check_segment_values(genotype)
    allowed = {gene.upper(): list(names) for gene, names in genotype.items()}
    missing = sorted(name for gene, names in allowed.items() for name in names
                     if gene not in reference.genes or name not in reference.gene(gene).index)
    if missing:
        raise UnknownAlleleError(f"the genotype names alleles the reference does not hold: {', '.join(missing[:5])}",
                                 context={"missing": missing},
                                 hint="Give their sequences in the reference (alignair predict --reference; in "
                                      "Python, Aligner.from_pretrained(reference=) or bound_to).")
    empty = sorted(gene for gene, names in allowed.items() if not names)
    if empty:
        raise InputError(f"the genotype leaves gene {', '.join(empty)} with no allele", context={"genes": empty})
    if isinstance(genotype, GenesOnly):                 # restrict only the genes it names
        named = {segment: {gene_of(name) for name in names} for segment, names in allowed.items()}
        allowed = {segment: [name for name in reference.gene(segment).names
                             if gene_of(name) not in named[segment] or name in names]
                   for segment, names in allowed.items()}
    return reference.subset(allowed)
