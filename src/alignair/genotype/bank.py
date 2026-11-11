"""A donor's genotype as the chip's bank: the subset of a reference the chip reads its calls from."""
from __future__ import annotations

from typing import Iterable, Mapping

from ..errors import InputError
from ..reference.reference_set import ReferenceSet


def genotype_bank(reference: ReferenceSet, genotype: Mapping[str, Iterable[str]]) -> ReferenceSet:
    """``reference``'s subset of the alleles ``genotype`` names (``{gene letter: allele names}``, any case), in the
    reference's order and with its locus schema (every locus stays, its ranges over the kept alleles); a gene it
    does not name keeps all its alleles. ConfigurationError for a key that is not a segment (V, D, J), gene names
    included; InputError when it names alleles the reference does not hold (give their sequences in the reference) or
    leaves a gene with no allele."""
    from .files import check_segment_keys
    check_segment_keys(genotype)
    allowed = {gene.upper(): list(names) for gene, names in genotype.items()}
    missing = sorted(name for gene, names in allowed.items() for name in names
                     if gene not in reference.genes or name not in reference.gene(gene).index)
    if missing:
        raise InputError(f"the genotype names alleles the reference does not hold: {', '.join(missing[:5])}",
                         context={"missing": missing},
                         hint="Give their sequences in the reference (alignair predict --reference).")
    empty = sorted(gene for gene, names in allowed.items() if not names)
    if empty:
        raise InputError(f"the genotype leaves gene {', '.join(empty)} with no allele", context={"genes": empty})
    return reference.subset(allowed)
