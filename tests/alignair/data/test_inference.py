"""Inference tokenization policies, positional integrity, and preflight failures."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from alignair.data import SequenceInputAdapter
from alignair.data.tokenizer import CenterPaddedTokenizer
from alignair.errors import ConfigurationError, DataContractError, InputError, current_error_context

pytestmark = pytest.mark.component


def test_the_adapter_preserves_nucleotide_tokens_with_explicit_padding():
    nucleotide = SequenceInputAdapter(CenterPaddedTokenizer(), 6, vocab_size=6)
    assert nucleotide(['ATGC','N'])['tokenized_sequence'].tolist() == [[1,2,3,4,0,0],[5,0,0,0,0,0]]


def test_supplied_tokenizer_is_used_instead_of_replaced_and_prefix_policy_is_explicit():
    calls = []
    def encode(sequence):
        calls.append(sequence)
        return np.full(len(sequence), 7, dtype=np.int64)
    adapter = SequenceInputAdapter(SimpleNamespace(encode=encode), 3, overflow='prefix', vocab_size=8)
    assert adapter(['ABCDE'])['tokenized_sequence'].tolist() == [[7,7,7]]
    assert calls == ['ABC']
    with pytest.raises(InputError):
        SequenceInputAdapter(adapter.tokenizer,3)(['ABCDE'])
    assert calls == ['ABC']


def test_legacy_nucleotide_policy_can_retain_empty_and_unknown_inputs():
    adapter = SequenceInputAdapter(CenterPaddedTokenizer(),4,allow_empty=True)
    assert adapter(['','?'])['tokenized_sequence'].tolist() == [[0,0,0,0],[5,0,0,0]]


@pytest.mark.parametrize('options', [{'max_length':0}, {'max_length':True}, {'vocab_size':0},
                                    {'tokenizer':object()}, {'overflow':'crop'}, {'allow_empty':1},
                                    {'device':'bad_device'}, {'alphabet':1}, {'alphabet':()},
                                    {'alphabet':('AA',)}, {'alphabet':(1,)}])
def test_invalid_adapter_configuration_is_classified(options):
    args = {'tokenizer':CenterPaddedTokenizer(), 'max_length':8}
    args.update(options)
    with pytest.raises(ConfigurationError):
        SequenceInputAdapter(**args)


@pytest.mark.parametrize('sequences', ['AC',None,[],[''],[1],['ACZ'],['AAAAA']])
def test_invalid_batches_fail_with_input_errors_and_no_scope_leak(sequences):
    adapter = SequenceInputAdapter(CenterPaddedTokenizer(),4,alphabet=frozenset('AC'))
    with pytest.raises(InputError):
        adapter(sequences)
    assert current_error_context() == {}


@pytest.mark.parametrize('encoded', [None, ['bad'], np.ones(2), np.ones((2,1),dtype=np.int64),
                                    [True,False], np.array([1j,2j]), [1], [-1,1], [1,9], [0,0],
                                    torch.ones(2,dtype=torch.long,device='meta')])
def test_tokenizer_contract_violations_are_not_silently_coerced(encoded):
    adapter = SequenceInputAdapter(SimpleNamespace(encode=lambda _: encoded),4,vocab_size=9)
    with pytest.raises(DataContractError) as caught:
        adapter(['AC'])
    assert caught.value.context['stage'] == 'inference_tokenization'
    assert caught.value.context['record_index'] == 0
    assert current_error_context() == {}
