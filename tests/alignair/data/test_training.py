"""Reference-bound adapter selection and explicit record preparation share one training path."""
from copy import deepcopy
from dataclasses import replace

import pytest

from alignair.core import ModelSetup, prepare_model
from alignair.core.setup import validate_label_schema
from alignair.data.training import prepare_training_data
from alignair.errors import ConfigurationError, TrainingDataError
from alignair.reference import GeneReference, ReferenceSet

pytestmark = pytest.mark.component


@pytest.fixture
def reference():
    return ReferenceSet({
        'V': GeneReference(['V1', 'V2'], ['ATGTGT', 'ATGTGC'], {'V1': 0, 'V2': 1}, {'V1': 3, 'V2': 3}),
        'J': GeneReference(['J1'], ['TGG'], {'J1': 0}, {'J1': 0}),
    }, False)


@pytest.mark.parametrize('space,labels', [('nucleotide', 'alleles'), ('protein', 'alleles'), ('protein', 'protein_classes')])
def test_training_and_validation_adapters_follow_the_same_model_setup(reference, space, labels):
    setup = prepare_model(reference, input_space=space, label_space=labels)
    data = prepare_training_data(setup)
    record = {'sequence': 'ATGTGTTGG', 'sequence_aa': 'MCW', 'junction_start': 3,
              'v_sequence_start': 0, 'v_sequence_end': 6, 'j_sequence_start': 6, 'j_sequence_end': 9,
              'v_call': 'V1', 'j_call': 'J1', 'productive': False, 'mutation_rate': 0.}
    rows = list(data.prepare_records([record]))
    assert len(rows) == 1
    if labels == 'protein_classes':
        assert rows[0]['v_class'] == 0 and data.translated_only is False
        assert data.batch_adapter(rows).inputs['tokenized_sequence'][0, :3].tolist() == [11, 2, 19]
    else:
        assert rows[0] is record
    schema = setup.label_schema()
    validate_label_schema(schema, reference, setup.config)
    assert schema['genes']['v'] == ([['V1', 'V2']] if labels == 'protein_classes' else ['V1', 'V2'])


def test_unimplemented_record_head_combinations_fail_before_training(reference):
    for setup in (prepare_model(reference, input_space='nucleotide', label_space='protein_classes'),
                  prepare_model(reference, classification_mode='categorical'),
                  prepare_model(reference, input_space='protein', state_head=True),
                  prepare_model(reference, input_space='protein', label_space='protein_classes', use_meta_heads=True)):
        with pytest.raises(ConfigurationError):
            prepare_training_data(setup)


def test_finite_rejected_sources_end_but_prolonged_class_rejection_fails(reference):
    data = prepare_training_data(prepare_model(reference, input_space='protein', label_space='protein_classes'))
    invalid = {'sequence': 'ACGT', 'sequence_aa': None}
    assert list(data.prepare_records([])) == []
    assert list(data.prepare_records([invalid] * 3)) == []
    with pytest.raises(TrainingDataError) as caught:
        list(data.prepare_records([invalid] * 1000))
    assert caught.value.context['reason'] == 'no_aa'


def test_model_setup_rejects_config_space_relabeling(reference):
    setup = prepare_model(reference)
    with pytest.raises(ConfigurationError, match='input/label spaces'):
        ModelSetup(reference, replace(setup.config, input_space='protein'), 'nucleotide', 'alleles')


@pytest.mark.parametrize('corruption', ['version', 'input', 'genes', 'length', 'empty_class', 'bad_class_type',
                                      'unknown_member', 'duplicate_member'])
def test_invalid_label_schemas_cannot_change_meaning_of_weight_columns(reference, corruption):
    setup = prepare_model(reference, input_space='protein', label_space='protein_classes')
    schema = deepcopy(setup.label_schema())
    if corruption == 'version': schema['schema_version'] = 2
    elif corruption == 'input': schema['input_space'] = 'nucleotide'
    elif corruption == 'genes': schema['genes']['d'] = []
    elif corruption == 'length': schema['genes']['v'] = []
    elif corruption == 'empty_class': schema['genes']['v'] = [[]]
    elif corruption == 'bad_class_type': schema['genes']['v'] = ['V1']
    elif corruption == 'unknown_member': schema['genes']['v'] = [['V3']]
    else: schema['genes']['v'] = [['V1', 'V1']]
    with pytest.raises(ConfigurationError):
        validate_label_schema(schema, reference, setup.config)


def test_allele_schema_requires_original_order(reference):
    setup = prepare_model(reference)
    schema = setup.label_schema()
    schema['genes']['v'].reverse()
    with pytest.raises(ConfigurationError, match='order'):
        validate_label_schema(schema, reference, setup.config)
