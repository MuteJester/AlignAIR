"""Shared content normalization distinguishes file quality gates from prediction preparation."""
from dataclasses import replace

import pytest

from alignair.data.inputs import (SequenceValidationPolicy, normalize_sequence_inputs,
                                  nucleotide_file_policy, nucleotide_prediction_policy,
                                  prepare_nucleotide_inputs, validate_nucleotide_sequence)
from alignair.errors import ConfigurationError, InputError, current_error_context

pytestmark = pytest.mark.component


def test_prediction_crops_over_length_and_flags_exact_short_and_long_reads():
    sequences, cropped = prepare_nucleotide_inputs(['ac', 'ACGT', 'ACGTAC'], max_length=4)
    assert sequences == ['AC', 'ACGT', 'ACGT']
    assert cropped == [False, False, True]


def test_file_and_prediction_policies_share_cleaning_but_keep_their_rejection_and_window_rules():
    file_policy = nucleotide_file_policy(4)
    prediction_policy = nucleotide_prediction_policy(4)
    assert file_policy.normalize(' acgr ').sequence == prediction_policy.normalize(' acgr ').sequence == 'ACGN'
    assert file_policy.normalize('???').error == 'ambiguous'
    assert prediction_policy.normalize('???').sequence == 'NNN'
    assert file_policy.normalize('ACGTN').error == 'overlength'
    result = prediction_policy.normalize('ACGTN')
    assert result.sequence == 'ACGT' and result.was_cropped


def test_all_iupac_codes_are_tolerated_ambiguities_and_not_counted_as_unusable():
    assert nucleotide_file_policy().normalize(' ryswkmbdhvn ').sequence == 'N' * 11
    assert nucleotide_file_policy().normalize('ACGT?').sequence == 'ACGTN'
    assert nucleotide_file_policy().normalize('ACG??').error == 'ambiguous'
    assert nucleotide_file_policy(2).normalize('ACG??').error == 'ambiguous'
    assert validate_nucleotide_sequence('ACG??', max_len=2) == (None, 'ambiguous')
    assert validate_nucleotide_sequence('ACGT', max_len=2) == (None, 'too_long')


def test_surrounding_whitespace_is_stripped_and_interior_whitespace_obeys_content_policy():
    assert prepare_nucleotide_inputs(['  ac\tgt  '], 8) == (['ACNGT'], [False])
    assert nucleotide_file_policy().normalize('AC GT').sequence == 'ACNGT'
    assert nucleotide_file_policy().normalize('A C').error == 'ambiguous'
    assert SequenceValidationPolicy(frozenset('ACGTN'), 8).normalize('AC GT').error == 'interior_whitespace'


def test_crop_flags_keep_original_positions_when_invalid_rows_are_skipped():
    batch = normalize_sequence_inputs(['', 'acgt', 'n', None, 'gtgtg'], nucleotide_prediction_policy(3), on_invalid='skip')
    assert batch.valid_indices == (1, 2, 4)
    assert batch.cropped_indices == (1, 4) and batch.crop_flags == (True, False, True)
    assert [record.sequence for record in batch.records] == ['ACG', 'N', 'GTG']
    assert batch.restore_order(['first', 'second', 'third']) == [batch.errors[0], 'first', 'second', batch.errors[1], 'third']


def test_case_expansion_and_cropping_use_normalized_length():
    policy = SequenceValidationPolicy(frozenset('S'), 1, overflow='prefix')
    result = policy.normalize('ß')
    assert result.sequence == 'S' and result.was_cropped
    assert policy.normalize('Z').error == 'invalid_character'


def test_strict_unbounded_policy_is_explicit_and_does_not_crop():
    result = SequenceValidationPolicy(frozenset('AC'), None).normalize('a' * 1000)
    assert result.sequence == 'A' * 1000 and not result.was_cropped


def test_replacement_can_reject_an_explicit_stop_symbol():
    policy = SequenceValidationPolicy(frozenset('ACX'), 4, stop_symbol='*', unknown_symbol='X')
    assert policy.normalize('Z').sequence == 'X'
    assert policy.normalize('*').error == 'invalid_character'


@pytest.mark.parametrize('value', [None, 1, False, b'AC', {'sequence': 'AC'}, ['AC']])
def test_prediction_does_not_coerce_non_string_values_into_reads(value):
    with pytest.raises(InputError) as caught:
        prepare_nucleotide_inputs(['AC', value], 8)
    assert caught.value.context['reason'] == 'invalid_type'
    assert caught.value.context['record_index'] == 1
    assert current_error_context() == {}


@pytest.mark.parametrize('batch', ['AC', b'AC', {'sequence': 'AC'}, None, 1])
def test_prediction_requires_a_batch_without_splitting_bare_values(batch):
    with pytest.raises(InputError):
        prepare_nucleotide_inputs(batch, 8)


@pytest.mark.parametrize('batch', [[], iter([])])
def test_empty_prediction_batches_are_typed_errors(batch):
    with pytest.raises(InputError) as caught:
        prepare_nucleotide_inputs(batch, 8)
    assert caught.value.context['reason'] == 'empty_batch'


def test_prediction_fail_fast_preserves_generator_position():
    def source():
        yield 'AC'
        yield ' '
        pytest.fail('consumed past the first invalid sequence')
    with pytest.raises(InputError) as caught:
        prepare_nucleotide_inputs(source(), 8)
    assert caught.value.context['record_index'] == 1 and caught.value.context['reason'] == 'empty'


@pytest.mark.parametrize('maximum', [None, 0, -1, True, 1.5])
def test_prediction_window_is_required_before_source_consumption(maximum):
    def source(): pytest.fail('consumed invalid request'); yield
    with pytest.raises(ConfigurationError):
        prepare_nucleotide_inputs(source(), maximum)


@pytest.mark.parametrize('options', [
    {'unknown_symbol': []}, {'unknown_symbol': 'Z'}, {'unknown_symbol': ''},
    {'ambiguity_symbols': None}, {'ambiguity_symbols': ('aa',)}, {'ambiguity_symbols': ('a',)},
    {'ambiguity_symbols': (' ',)}, {'ambiguity_symbols': ('A',)}, {'ambiguity_symbols': (1,)},
    {'ambiguity_symbols': ('R',)}, {'max_invalid_fraction': .2},
    {'unknown_symbol': 'N', 'max_invalid_fraction': -1},
    {'unknown_symbol': 'N', 'max_invalid_fraction': 2},
    {'unknown_symbol': 'N', 'max_invalid_fraction': 10**1000},
    {'unknown_symbol': 'N', 'max_invalid_fraction': float('nan')},
    {'unknown_symbol': 'N', 'max_invalid_fraction': float('inf')},
    {'unknown_symbol': 'N', 'max_invalid_fraction': True},
    {'unknown_symbol': 'N', 'max_invalid_fraction': '0.2'},
    {'overflow': 'crop'}, {'overflow': []}, {'max_length': None, 'overflow': 'prefix'},
    {'reject_interior_whitespace': 1},
])
def test_invalid_content_policies_fail_before_use(options):
    kwargs = {'alphabet': frozenset('ACGTN'), 'max_length': 8}
    kwargs.update(options)
    with pytest.raises(ConfigurationError):
        SequenceValidationPolicy(**kwargs)


def test_policy_ambiguity_symbols_are_snapshots_and_fraction_endpoints_are_supported():
    symbols = {'R'}
    policy = SequenceValidationPolicy(frozenset('ACGTN'), 8, unknown_symbol='N', ambiguity_symbols=symbols,
                                      max_invalid_fraction=0)
    symbols.add('Z')
    assert policy.normalize('R').sequence == 'N' and policy.normalize('Z').error == 'ambiguous'
    assert replace(policy, max_invalid_fraction=1).normalize('Z').sequence == 'N'


def test_record_acceptance_is_explicit_and_invalid_option_is_eager():
    policy = nucleotide_prediction_policy(8)
    assert normalize_sequence_inputs([{'sequence': 'AC'}], policy).records[0].sequence == 'AC'
    with pytest.raises(InputError) as caught:
        normalize_sequence_inputs([{'sequence': 'AC'}], policy, allow_records=False)
    assert caught.value.context['reason'] == 'invalid_type'
    with pytest.raises(ConfigurationError):
        normalize_sequence_inputs([], policy, allow_records=None)


def test_io_compatibility_interfaces_delegate_to_shared_data_owners():
    from alignair.io.sequence_reader import validate_sequence, validate
    from alignair.data.inputs import validate_nucleotide
    assert validate_sequence is validate_nucleotide_sequence and validate is validate_nucleotide
    assert prepare_nucleotide_inputs(['acgtac'], 4) == (['ACGT'], [True])
