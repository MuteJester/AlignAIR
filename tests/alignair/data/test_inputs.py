"""One normalization pass preserves source identity independently of alphabet and validation outcome."""
from collections import UserDict

import pytest

from alignair.data import (NormalizedSequenceBatch, SequenceInputFailure, SequenceRecord,
                           SequenceValidationPolicy, normalize_sequence_inputs)
from alignair.errors import ConfigurationError, DataContractError, InputError, current_error_context

pytestmark = pytest.mark.component


def policy(alphabet='ACGTN', max_length=8):
    return SequenceValidationPolicy(frozenset(alphabet),max_length)


@pytest.mark.parametrize('alphabet, sequences, expected', [('ACGTN',['  acgt ','nN'],['ACGT','NN']),
                                                         ('ACDEFGHIKLMNPQRSTVWYX',['  acde ','xX'],['ACDE','XX'])])
def test_one_normalizer_supports_both_sequence_alphabets(alphabet,sequences,expected):
    result = normalize_sequence_inputs(sequences,policy(alphabet))
    assert [record.sequence for record in result.records] == expected
    assert result.valid_indices == (0,1) and not result.errors


def test_mixed_records_generator_and_duplicate_ids_keep_source_positions():
    seen = []
    items = ['',{'sequence_id':'same','sequence':'ac'},UserDict(sequence_id='same',sequence='A T'),
             UserDict(sequence_id='same',sequence='gt'),{'sequence_id':None,'sequence':'N'}]
    def source():
        for index,item in enumerate(items):
            seen.append(index)
            yield item
    result = normalize_sequence_inputs(source(),policy(),on_invalid='skip')
    assert seen == [0,1,2,3,4]
    assert result.valid_indices == (1,3,4)
    assert [record.sequence_id for record in result.records] == ['same','same','seq_4']
    assert [failure.sequence_id for failure in result.errors] == ['seq_0','same']
    restored = result.restore_order(iter(['first','second','third']))
    assert restored == [result.errors[0],'first',result.errors[1],'second','third']
    assert current_error_context() == {}


@pytest.mark.parametrize('provided, expected', [('', 'seq_0'), (None,'seq_0'), (False,'seq_0'),
                                             (12,'seq_0'), ('0','0'), ('  id  ','  id  ')])
def test_id_fallback_preserves_nonempty_strings_without_stripping_them(provided,expected):
    result = normalize_sequence_inputs([{'sequence_id':provided,'sequence':'AC'}],policy())
    assert result.records[0].sequence_id == expected


@pytest.mark.parametrize('value', [None,1,False,b'AC',['AC'],object()])
def test_non_string_sequence_values_are_structured_failures(value):
    result = normalize_sequence_inputs([value],policy(),on_invalid='skip')
    assert result.valid_indices == ()
    assert result.errors[0].to_dict() == {'sequence_id':'seq_0','error':'invalid_type','detail':'Sequence must be a string'}


def test_fail_fast_stops_iteration_and_reports_original_id_index_and_reason():
    calls = []
    def source():
        calls.append(0)
        yield {'sequence_id':'first','sequence':'AC'}
        calls.append(1)
        yield {'sequence_id':'bad','sequence':None}
        pytest.fail('consumed records beyond the first invalid input')
    with pytest.raises(InputError) as caught:
        normalize_sequence_inputs(source(),policy())
    assert calls == [0,1]
    assert caught.value.context == {'stage':'sequence_input_validation','record_index':1,
                                   'record_id':'bad','reason':'invalid_type'}
    assert current_error_context() == {}


@pytest.mark.parametrize('items', [{},UserDict(sequence='AC'),b'AC'])
def test_bare_mappings_and_bytes_are_format_errors(items):
    with pytest.raises(TypeError):
        normalize_sequence_inputs(items,policy(),on_invalid='skip')


def test_bare_string_is_one_read_and_missing_sequence_remains_an_empty_failure():
    result = normalize_sequence_inputs(' ac ',policy())
    assert result.outcomes == (SequenceRecord('seq_0','AC'),)
    missing = normalize_sequence_inputs([{}],policy(),on_invalid='skip')
    assert missing.errors[0].error == 'empty'


def test_case_expansion_is_checked_against_the_normalized_window():
    result = normalize_sequence_inputs(['ß'],policy('S',1),on_invalid='skip')
    assert result.errors[0].error == 'overlength'
    assert result.errors[0].detail == 'Sequence length 2 exceeds max_len 1'
    assert normalize_sequence_inputs(['ß'],policy('S',2)).records[0].sequence == 'SS'


def test_validation_priority_and_special_stop_reason_are_preserved():
    protein = SequenceValidationPolicy(frozenset('AC'),2,stop_symbol='*')
    result = normalize_sequence_inputs([' ','A C','AA*','*','Z'],protein,on_invalid='skip')
    assert [failure.error for failure in result.errors] == ['empty','interior_whitespace','overlength','invalid_character','invalid_character']
    assert 'stop character' in result.errors[3].detail
    assert "valid: ['A', 'C']" in result.errors[4].detail


@pytest.mark.parametrize('alphabet', [None,1,(),('aa',),('a',),(1,),(' ',)])
def test_invalid_alphabet_policy_is_rejected(alphabet):
    with pytest.raises(ConfigurationError):
        SequenceValidationPolicy(alphabet,8)


@pytest.mark.parametrize('stop', ['',1,'**','A','a',' '])
def test_invalid_stop_policy_is_rejected(stop):
    with pytest.raises(ConfigurationError):
        SequenceValidationPolicy(frozenset('AC'),8,stop_symbol=stop)


@pytest.mark.parametrize('length', [0,-1,True,1.5])
def test_invalid_window_is_rejected_before_input_consumption(length):
    with pytest.raises(ConfigurationError) as caught:
        SequenceValidationPolicy(frozenset('AC'),length)
    assert caught.value.context['field'] == 'max_length'


def test_policy_alphabet_is_a_snapshot_of_caller_configuration():
    alphabet = {'A','C'}
    configured = SequenceValidationPolicy(alphabet,8)
    alphabet.add('Z')
    result = normalize_sequence_inputs('Z',configured,on_invalid='skip')
    assert result.errors[0].error == 'invalid_character'


@pytest.mark.parametrize('on_invalid', ['Skip','bad',None,1,[]])
def test_invalid_error_disposition_is_eager(on_invalid):
    def source(): pytest.fail('started input source'); yield
    with pytest.raises(ConfigurationError):
        normalize_sequence_inputs(source(),policy(),on_invalid=on_invalid)


def test_invalid_policy_and_prediction_cardinality_are_classified():
    with pytest.raises(ConfigurationError):
        normalize_sequence_inputs([],None)
    result = normalize_sequence_inputs(['AC','','GT'],policy(),on_invalid='skip')
    for predictions in ([],['one'],['one','two','three'],None):
        with pytest.raises(DataContractError):
            result.restore_order(predictions)
    assert result.restore_order(['one','two']) == ['one',result.errors[0],'two']


def test_empty_and_all_invalid_batches_need_no_predictions():
    empty = normalize_sequence_inputs([],policy())
    assert empty == NormalizedSequenceBatch(()) and empty.restore_order(()) == []
    invalid = normalize_sequence_inputs(['','Z'],policy(),on_invalid='skip')
    assert invalid.restore_order(()) == list(invalid.errors)


def test_legacy_product_types_are_aliases_of_shared_record_types():
    from alignair.aa.product.validate import Record,ErrorResult,normalize_inputs,normalize_input_batch
    assert Record is SequenceRecord and ErrorResult is SequenceInputFailure
    records,errors = normalize_inputs([' ac ','*'],on_invalid='skip')
    batch = normalize_input_batch([' ac ','*'],on_invalid='skip')
    assert records == list(batch.records) and errors == list(batch.errors)


def test_accepted_values_are_normalized_once_instead_of_revalidating_each_item():
    calls = []
    class Traced(str):
        def strip(self):
            calls.append(self)
            return super().strip()
    from alignair.aa.product.validate import normalize_input_batch
    result = normalize_input_batch([Traced(' ac '),Traced(' de ')])
    assert [record.sequence for record in result.records] == ['AC','DE']
    assert calls == [' ac ',' de ']
