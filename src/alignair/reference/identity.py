"""Reference identity for fixed output labels, germline biology and ordered locus mappings."""
from __future__ import annotations

from numbers import Integral

from ..errors import ConfigurationError, DataContractError, ModelError, error_context
from .reference_set import ReferenceSet


def _gene_identity(reference, gene):
    if gene not in reference.genes:
        return None
    rows = reference.gene(gene)
    return {'allele_order': tuple(rows.names),
            'sequences': tuple(sequence.upper() for sequence in rows.sequences),
            'gapped': tuple(sorted((rows.gapped or {}).items())),
            'anchors': tuple(sorted((rows.anchors or {}).items()))}


def _locus_identity(reference):
    try:
        rows, names = [], set()
        for row in reference.loci:
            if (not isinstance(row.locus, str) or not row.locus or row.locus in names
                    or not isinstance(row.chain_type, str) or not row.chain_type or type(row.has_d) is not bool):
                raise ValueError('invalid or duplicate locus label')
            names.add(row.locus)
            ranges = []
            for gene, bounds in row.ranges.items():
                start, end = bounds
                if (gene not in reference.genes
                        or any(isinstance(value, bool) or not isinstance(value, Integral) for value in (start, end))
                        or not 0 <= start <= end <= len(reference.gene(gene))):
                    raise ValueError('invalid locus allele range')
                ranges.append((gene, (int(start), int(end))))
            rows.append((row.locus, row.chain_type, row.has_d, tuple(sorted(ranges))))
        return tuple(rows)
    except (AttributeError, TypeError, ValueError) as exc:
        raise DataContractError('reference locus schema is malformed', context={'field': 'loci'}) from exc


def _mismatch(field, gene=None):
    descriptions = {'genes': 'gene set differs', 'has_d': 'D availability differs',
                    'allele_order': 'allele set/order differs', 'sequences': 'germline sequences differ',
                    'gapped': 'gapped germline sequences differ', 'anchors': 'junction anchors differ',
                    'loci': 'ordered locus schema differs'}
    context = {'field': field}
    suffix = ''
    if gene is not None:
        context['gene'] = gene
        suffix = f' for gene {gene}'
    raise ModelError(f'reference does not match expected reference{suffix}: {descriptions[field]}',
                     context=context,
                     hint='Use the expected reference; changed labels or biology require a compatible model.')


def validate_reference_identity(candidate: ReferenceSet, expected: ReferenceSet) -> None:
    """Require the same ordered genes, biology, D selection and ordered locus schema.

    Gene indices must agree with their names before comparison. Ungapped DNA retains the existing
    case-insensitive comparison; gapped strings, anchors and declared locus metadata must match.
    Mapping insertion order is irrelevant, while allele and locus row order defines model labels.
    This compares declared identity without guessing missing provenance or modifying either input.
    """
    with error_context(stage='reference_identity'):
        if not isinstance(candidate, ReferenceSet) or not isinstance(expected, ReferenceSet):
            raise ConfigurationError('reference identity requires two ReferenceSet objects',
                                     context={'field': 'reference'})
        candidate.validate(genes=tuple(candidate.genes))
        expected.validate(genes=tuple(expected.genes))
        if candidate.has_d != expected.has_d:
            _mismatch('has_d')
        for gene in sorted(set(candidate.genes) | set(expected.genes)):
            actual, target = _gene_identity(candidate, gene), _gene_identity(expected, gene)
            if actual == target:
                continue
            if actual is None or target is None:
                _mismatch('genes', gene)
            for field in target:
                if actual[field] != target[field]:
                    _mismatch(field, gene)
        if _locus_identity(candidate) != _locus_identity(expected):
            _mismatch('loci')
