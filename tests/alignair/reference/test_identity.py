"""Reference identity includes biology, model-column order and locus interpretation."""
from copy import deepcopy

import pytest

from alignair.errors import ConfigurationError, DataContractError, ModelError
from alignair.reference import ReferenceSet, validate_reference_identity
from alignair.reference.reference_set import LocusInfo

pytestmark = pytest.mark.component


@pytest.fixture
def reference():
    result = ReferenceSet.from_genotype({
        'V': {'IGHV1*01': 'ACGT', 'TRBV1*01': 'GGGG'},
        'D': {'IGHD1*01': 'AAAA', 'TRBD1*01': 'CCCC'},
        'J': {'IGHJ1*01': 'TTTT', 'TRBJ1*01': 'ACAC'},
    }, anchors={'V': {'IGHV1*01': 1, 'TRBV1*01': 2}, 'J': {'IGHJ1*01': 0, 'TRBJ1*01': 0}})
    result.gene('V').gapped = {'IGHV1*01': 'AC.GT', 'TRBV1*01': 'GG.GG'}
    result.loci = [LocusInfo('IGH', 'BCR_HEAVY', True, {'V': (0, 1), 'D': (0, 1), 'J': (0, 1)}),
                   LocusInfo('TRB', 'TCR_BETA', True, {'V': (1, 2), 'D': (1, 2), 'J': (1, 2)})]
    return result


def test_identical_references_match_without_changing_either_input(reference):
    candidate = deepcopy(reference)
    before = candidate.to_serializable()
    validate_reference_identity(candidate, reference)
    assert candidate.to_serializable() == before == reference.to_serializable()


def test_mapping_insertion_order_and_ungapped_case_do_not_change_identity(reference):
    candidate = deepcopy(reference)
    candidate.genes = dict(reversed(list(candidate.genes.items())))
    candidate.gene('V').gapped = dict(reversed(list(candidate.gene('V').gapped.items())))
    candidate.gene('V').anchors = dict(reversed(list(candidate.gene('V').anchors.items())))
    candidate.gene('V').sequences = [sequence.lower() for sequence in candidate.gene('V').sequences]
    for row in candidate.loci:
        row.ranges = {gene: list(bounds) for gene, bounds in reversed(list(row.ranges.items()))}
    validate_reference_identity(candidate, reference)


@pytest.mark.parametrize('field', ['allele_order', 'sequences', 'gapped', 'anchors'])
def test_changed_gene_identity_is_rejected_with_precise_context(reference, field):
    candidate = deepcopy(reference)
    rows = candidate.gene('V')
    if field == 'allele_order':
        rows.names.reverse()
        rows.sequences.reverse()
        rows.index = {name: i for i, name in enumerate(rows.names)}
    elif field == 'sequences':
        rows.sequences[0] = 'AAAA'
    elif field == 'gapped':
        rows.gapped['IGHV1*01'] = 'A.CGT'
    else:
        rows.anchors['IGHV1*01'] = 3
    with pytest.raises(ModelError, match='does not match') as caught:
        validate_reference_identity(candidate, reference)
    assert caught.value.context == {'stage': 'reference_identity', 'field': field, 'gene': 'V'}
    assert caught.value.hint


@pytest.mark.parametrize('missing_from', ['candidate', 'expected'])
def test_missing_gene_is_not_hidden_by_matching_other_heads(reference, missing_from):
    candidate = deepcopy(reference)
    del (candidate if missing_from == 'candidate' else reference).genes['D']
    with pytest.raises(ModelError) as caught:
        validate_reference_identity(candidate, reference)
    assert caught.value.context['field'] == 'genes'
    assert caught.value.context['gene'] == 'D'


def test_d_availability_is_part_of_identity_even_if_gene_rows_match(reference):
    candidate = deepcopy(reference)
    candidate.has_d = False
    with pytest.raises(ModelError) as caught:
        validate_reference_identity(candidate, reference)
    assert caught.value.context['field'] == 'has_d'


def test_extra_allele_cannot_silently_extend_a_fixed_output_panel(reference):
    candidate = deepcopy(reference)
    rows = candidate.gene('V')
    rows.index['extra'] = len(rows.names)
    rows.names.append('extra')
    rows.sequences.append('AAAA')
    with pytest.raises(ModelError) as caught:
        validate_reference_identity(candidate, reference)
    assert caught.value.context['field'] == 'allele_order'


@pytest.mark.parametrize('change', ['order', 'name', 'chain_type', 'has_d', 'ranges', 'absent'])
def test_changed_locus_interpretation_is_rejected(reference, change):
    candidate = deepcopy(reference)
    if change == 'order':
        candidate.loci.reverse()
    elif change == 'name':
        candidate.loci[0].locus = 'other'
    elif change == 'chain_type':
        candidate.loci[0].chain_type = 'other'
    elif change == 'has_d':
        candidate.loci[0].has_d = False
    elif change == 'ranges':
        candidate.loci[0].ranges['V'] = (0, 2)
    else:
        candidate.loci = []
    with pytest.raises(ModelError) as caught:
        validate_reference_identity(candidate, reference)
    assert caught.value.context == {'stage': 'reference_identity', 'field': 'loci'}


@pytest.mark.parametrize('problem', ['none', 'duplicate', 'label', 'chain_type', 'd_type',
    'float_range', 'bool_range', 'range_size', 'range_bounds', 'range_order', 'foreign_gene'])
def test_malformed_locus_metadata_is_a_data_contract_error(reference, problem):
    candidate = deepcopy(reference)
    if problem == 'none':
        candidate.loci = None
    elif problem == 'duplicate':
        candidate.loci.append(deepcopy(candidate.loci[0]))
    elif problem == 'label':
        candidate.loci[0].locus = ''
    elif problem == 'chain_type':
        candidate.loci[0].chain_type = None
    elif problem == 'd_type':
        candidate.loci[0].has_d = 'yes'
    else:
        gene, bounds = {'float_range': ('V', (0., 1.)), 'bool_range': ('V', (False, True)),
            'range_size': ('V', (0,)), 'range_bounds': ('V', (0, 99)),
            'range_order': ('V', (1, 0)), 'foreign_gene': ('C', (0, 1))}[problem]
        candidate.loci[0].ranges[gene] = bounds
    with pytest.raises(DataContractError) as caught:
        validate_reference_identity(candidate, reference)
    assert caught.value.context['field'] == 'loci'
    assert caught.value.__cause__ is not None


def test_invalid_gene_index_cannot_pass_an_identity_check(reference):
    candidate = deepcopy(reference)
    candidate.gene('V').index['IGHV1*01'] = 1
    with pytest.raises(DataContractError, match='index disagree'):
        validate_reference_identity(candidate, reference)


def test_identity_requires_reference_objects(reference):
    with pytest.raises(ConfigurationError, match='two ReferenceSet'):
        validate_reference_identity({}, reference)


def test_matching_absent_gaps_and_anchors_are_supported():
    reference = ReferenceSet.from_genotype({'V': {'V1': 'ACGT'}, 'J': {'J1': 'AAAA'}})
    validate_reference_identity(deepcopy(reference), reference)
