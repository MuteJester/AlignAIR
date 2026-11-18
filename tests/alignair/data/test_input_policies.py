"""One nucleotide normaliser: a file read is dropped when mostly unusable; a prediction read is cut to the window."""
import pytest

from alignair.data.inputs import normalize_file_read, normalize_nucleotide, prepare_nucleotide_inputs
from alignair.errors import ConfigurationError, InputError, current_error_context

pytestmark = pytest.mark.component


def test_prediction_crops_over_length_and_flags_exact_short_and_long_reads():
    sequences, cropped = prepare_nucleotide_inputs(['ac', 'ACGT', 'ACGTAC'], max_length=4)
    assert sequences == ['AC', 'ACGT', 'ACGT']
    assert cropped == [False, False, True]


def test_file_and_prediction_reads_share_cleaning_but_keep_their_rejection_and_window_rules():
    assert normalize_file_read(' acgr ').sequence == prepare_nucleotide_inputs([' acgr '], 4)[0][0] == 'ACGN'
    assert normalize_file_read('???').error == 'ambiguous'
    assert prepare_nucleotide_inputs(['???'], 4)[0] == ['NNN']
    assert normalize_nucleotide('ACGTN', max_length=4).error == 'overlength'
    result = normalize_nucleotide('ACGTN', max_length=4, crop=True)
    assert result.sequence == 'ACGT' and result.was_cropped


def test_all_iupac_codes_are_tolerated_ambiguities_and_not_counted_as_unusable():
    assert normalize_file_read(' ryswkmbdhvn ').sequence == 'N' * 11
    assert normalize_file_read('ACGT?').sequence == 'ACGTN'
    assert normalize_file_read('ACG??').error == 'ambiguous'


def test_surrounding_whitespace_is_stripped_and_interior_whitespace_is_an_unusable_character():
    assert prepare_nucleotide_inputs(['  ac\tgt  '], 8) == (['ACNGT'], [False])
    assert normalize_file_read('AC GT').sequence == 'ACNGT'
    assert normalize_file_read('A C').error == 'ambiguous'


@pytest.mark.parametrize('value', ['', '   ', None, 1])
def test_a_file_read_that_is_empty_or_not_a_string_is_dropped_with_its_reason(value):
    result = normalize_file_read(value)
    assert result.sequence is None and result.error == ('empty' if isinstance(value, str) else 'invalid_type')


@pytest.mark.parametrize('value', [None, 1, False, b'AC', {'sequence': 'AC'}, ['AC']])
def test_prediction_does_not_coerce_non_string_values_into_reads(value):
    with pytest.raises(InputError) as caught:
        prepare_nucleotide_inputs(['AC', value], 8)
    assert caught.value.context['reason'] == 'invalid_type'
    assert caught.value.context['record_index'] == 1 and caught.value.context['record_id'] == 'seq_1'
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
    with pytest.raises(InputError, match="seq_1: Empty sequence after strip") as caught:
        prepare_nucleotide_inputs(source(), 8)
    assert caught.value.context['record_index'] == 1 and caught.value.context['reason'] == 'empty'


@pytest.mark.parametrize('maximum', [None, 0, -1, True, 1.5])
def test_prediction_window_is_required_before_source_consumption(maximum):
    def source(): pytest.fail('consumed invalid request'); yield
    with pytest.raises(ConfigurationError):
        prepare_nucleotide_inputs(source(), maximum)
