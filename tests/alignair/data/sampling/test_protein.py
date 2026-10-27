from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from alignair.data.sampling import ProteinSampleBuilder, ProteinSamplePolicy
from alignair.data.projection import translated_record_usable
from alignair.errors import ConfigurationError, TrainingDataError
from alignair.data.sampling import prepare_protein_sample as record_to_sample
from alignair.reference.labels import build_equivalence_classes, db_hash
from alignair.data.tokenizer import AAProteinTokenizer

pytestmark = pytest.mark.component


def _cmaps():
    v = build_equivalence_classes("V", {"IGHV1-1*01": "QVQLC", "IGHV2-2*01": "EVQLC"}, db_hash([("x", "a")]))
    j = build_equivalence_classes("J", {"IGHJ1*01": "WGQGT", "IGHJ2*01": "WGKGT"}, db_hash([("y", "b")]))
    return {"V": v, "J": j}


def _rec(**over):
    # 24 nt at frame 0 -> exactly 8 residues; sequence_aa MUST be 8 chars to stay in one coord system.
    base = {"sequence": "CAGGTGCAGCTGTGCTGGGGCCAG", "sequence_aa": "QVQLCWGQ", "junction_start": 0, "productive": True,
            "v_call": "IGHV1-1*01", "j_call": "IGHJ1*01",
            "v_sequence_start": 0, "v_sequence_end": 12,
            "d_sequence_start": 12, "d_sequence_end": 12,
            "j_sequence_start": 12, "j_sequence_end": 24}
    base.update(over)
    return base


def test_projects_boundaries_and_maps_classes():
    s, why = record_to_sample(_rec(), _cmaps(), AAProteinTokenizer(), 192)
    assert why is None and s is not None
    assert (s["v_start"], s["v_end"]) == (0, 4)      # 4 fully-owned V residues
    assert (s["j_start"], s["j_end"]) == (4, 8)
    assert isinstance(s["tokens"], np.ndarray)


def test_nonzero_frame_offset_shifts_the_residue_grid():
    # junction_start=1 -> frame 1; V=[1,13) J=[13,25); sequence 25nt -> (25-1)//3 = 8 residues.
    s, why = record_to_sample(
        _rec(sequence="A" * 25, junction_start=1, v_sequence_start=1, v_sequence_end=13,
             d_sequence_start=13, d_sequence_end=13, j_sequence_start=13, j_sequence_end=25),
        _cmaps(), AAProteinTokenizer(), 192)
    assert why is None
    assert (s["v_start"], s["v_end"]) == (0, 4) and (s["j_start"], s["j_end"]) == (4, 8)


def test_translation_length_mismatch_is_rejected():
    """A sequence_aa whose length disagrees with (len(sequence)-frame)//3 would put boundaries and
    tokens in different coordinate systems."""
    s, why = record_to_sample(_rec(sequence_aa="QVQLCWGQG"),      # 9 residues, but 24nt -> 8
                              _cmaps(), AAProteinTokenizer(), 192)
    assert s is None and why == "translation_length_mismatch"


def test_straddling_codon_is_masked_out_of_owned_span():
    # V=[0,11) straddles: residue r3 (nt 9..12) is not fully owned -> V owns only r0..r2.
    s, why = record_to_sample(_rec(v_sequence_end=11), _cmaps(), AAProteinTokenizer(), 192)
    assert why is None and (s["v_start"], s["v_end"]) == (0, 3)


def test_terminal_exclusive_endpoint_equals_residue_count():
    """The genuine terminal case: the exclusive end lands exactly ON the token count, i.e. the
    end slot n that rev-1 could never decode."""
    s, _ = record_to_sample(_rec(), _cmaps(), AAProteinTokenizer(), 192)
    assert len(s["tokens"]) == 8
    assert s["j_end"] == 8 == len(s["tokens"])        # exclusive end == n, not n-1


@pytest.mark.parametrize("over,reason", [
    ({"productive": False}, "not_productive"),
    ({"sequence_aa": ""}, "no_aa"),
    ({"junction_start": None}, "no_frame"),
    ({"sequence_aa": "QVQ*CWGQ"}, "stop_codon"),
    ({"v_call": "IGHV9-9*99"}, "unknown_allele"),
])
def test_rejections_are_reported_by_reason(over, reason):
    s, why = record_to_sample(_rec(**over), _cmaps(), AAProteinTokenizer(), 192)
    assert s is None and why == reason


def test_exact_length_is_rejected_to_reserve_the_endpoint_slot():
    # len(tokens) must be < max_len so slot n exists for an exclusive end.
    s, why = record_to_sample(_rec(sequence="A" * 48, sequence_aa="Q" * 16,
                                   v_sequence_end=24, d_sequence_start=24, d_sequence_end=24,
                                   j_sequence_start=24, j_sequence_end=48),
                              _cmaps(), AAProteinTokenizer(), max_len=16)
    assert s is None and why == "too_long"



def test_acceptance_policies_share_targets_and_make_stop_productivity_choices_explicit():
    record = _rec(productive=False, sequence_aa='QVQ*CWGQ')
    strict = ProteinSampleBuilder(_cmaps(), AAProteinTokenizer(), 192)
    permissive = ProteinSampleBuilder(_cmaps(), AAProteinTokenizer(), 192,
                                      policy=ProteinSamplePolicy(require_productive=False, allow_stops=True))
    assert strict(record).reason == 'not_productive'
    assert strict(_rec(sequence_aa='QVQ*CWGQ')).reason == 'stop_codon'
    assert permissive(record).sample is not None
    normal = _rec()
    first, second = strict(normal).sample, permissive(normal).sample
    assert first.keys() == second.keys()
    for key in first: np.testing.assert_array_equal(first[key], second[key])


@pytest.mark.parametrize('change', [{'d_inverted': True}, {'d_call': 'UNKNOWN'},
                                    {'d_sequence_start': 12, 'd_sequence_end': 12}])
def test_d_alleles_remain_explicitly_masked_when_unobservable(change):
    record = _rec(d_call='D*01', d_sequence_start=12, d_sequence_end=15, j_sequence_start=15)
    builder = ProteinSampleBuilder(_cmaps(), AAProteinTokenizer(), 192, d_index={'D*01': 0})
    valid = builder(record).sample
    assert valid['d_class'] == 0 and valid['d_valid'] == 1.
    assert (valid['d_start'], valid['d_end']) == (4, 5)
    invalid = builder({**record, **change}).sample
    assert invalid['d_class'] == -1 and invalid['d_valid'] == 0.
    assert invalid['d_start'] == invalid['d_end'] == 0


def test_absent_required_bounds_and_empty_encoding_have_distinct_reasons():
    normal = ProteinSampleBuilder(_cmaps(), AAProteinTokenizer(), 192)
    assert normal(_rec(v_sequence_end=0)).reason == 'no_bounds'
    empty = ProteinSampleBuilder(_cmaps(), SimpleNamespace(encode=lambda _: np.array([], dtype=np.int64)), 192)
    assert empty(_rec()).reason == 'empty'


@pytest.mark.parametrize('change,field', [({'junction_start': True}, 'junction_start'),
                                        ({'junction_start': .5}, 'junction_start'),
                                        ({'sequence_aa': 123}, 'sequence_aa'),
                                        ({'sequence_aa': 0}, 'sequence_aa'),
                                        ({'sequence_aa': False}, 'sequence_aa'),
                                        ({'sequence': None}, 'sequence')])
def test_malformed_source_annotations_raise_located_errors(change, field):
    record = _rec(sequence_id='sample-17', **change)
    with pytest.raises(TrainingDataError) as caught:
        ProteinSampleBuilder(_cmaps(), AAProteinTokenizer(), 192)(record)
    assert caught.value.context['record_id'] == 'sample-17'
    assert caught.value.context['field'] == field
    with pytest.raises(TrainingDataError): translated_record_usable(record)


@pytest.mark.parametrize('option', [dict(max_length=0), dict(max_length=1), dict(max_length=True),
                                   dict(max_length=2.5), dict(class_maps={}), dict(tokenizer=None),
                                   dict(policy=None), dict(d_index={'D': 1}), dict(d_index={'D': True}),
                                   dict(d_index={'': 0}), dict(d_index=[]), dict(d_index={'D': .0})])
def test_invalid_sample_configuration_fails_before_read_processing(option):
    kwargs = dict(class_maps=_cmaps(), tokenizer=AAProteinTokenizer(), max_length=192)
    kwargs.update(option)
    with pytest.raises(ConfigurationError): ProteinSampleBuilder(**kwargs)


@pytest.mark.parametrize('options', [dict(require_productive=1), dict(allow_stops='yes')])
def test_policy_flags_are_boolean_configuration(options):
    with pytest.raises(ConfigurationError): ProteinSamplePolicy(**options)


def test_preparation_is_repeatable_and_does_not_mutate_the_record():
    record = _rec()
    before = deepcopy(record)
    builder = ProteinSampleBuilder(_cmaps(), AAProteinTokenizer(), 192)
    np.testing.assert_array_equal(builder(record).sample['tokens'], builder(record).sample['tokens'])
    assert record == before
    assert builder(_rec(v_call=None)).reason == 'unknown_allele'
    with pytest.raises(TrainingDataError, match='mappings'): builder(None)
    with pytest.raises(TrainingDataError, match='mappings'): translated_record_usable(None)


def test_historical_entry_point_delegates_to_the_shared_builder_function():
    from alignair.aa.data import record_to_sample as old
    from alignair.data.sampling import prepare_protein_sample
    assert old is prepare_protein_sample


def test_a_length_changing_tokenizer_cannot_silently_desynchronize_boundary_labels():
    tokenizer = SimpleNamespace(encode=lambda _: np.array([1], dtype=np.int64))
    with pytest.raises(TrainingDataError, match='preserve residue positions') as caught:
        ProteinSampleBuilder(_cmaps(), tokenizer, 192)(_rec())
    assert caught.value.context['field'] == 'tokens'
