"""Reference-bound adapter selection and explicit record preparation share one training path."""
from copy import deepcopy

import pytest

from alignair.core import ModelSetup, prepare_model
from alignair.core.setup import validate_label_schema
from alignair.data.training import prepare_training_data
from alignair.errors import ConfigurationError
from alignair.reference import GeneReference, ReferenceSet

pytestmark = pytest.mark.component


@pytest.fixture
def reference():
    return ReferenceSet({
        'V': GeneReference(['V1', 'V2'], ['ATGTGT', 'ATGTGC'], {'V1': 0, 'V2': 1}, {'V1': 3, 'V2': 3}),
        'J': GeneReference(['J1'], ['TGG'], {'J1': 0}, {'J1': 0}),
    }, False)


def test_training_and_validation_adapters_follow_the_same_model_setup(reference):
    setup = prepare_model(reference)
    data = prepare_training_data(setup)
    assert data.batch_adapter.config is data.validation_adapter.config is setup.config
    schema = setup.label_schema()
    validate_label_schema(schema, reference, setup.config)
    assert schema['genes']['v'] == ['V1', 'V2']


@pytest.mark.parametrize('spaces', [{'input_space': 'protein'}, {'label_space': 'protein_classes'}])
def test_only_nucleotide_reads_with_allele_labels_are_prepared(reference, spaces):
    with pytest.raises(ConfigurationError):
        prepare_model(reference, **spaces)


def test_model_setup_rejects_config_space_relabeling(reference):
    setup = prepare_model(reference)
    with pytest.raises(ConfigurationError, match='nucleotide'):
        ModelSetup(reference, setup.config, 'protein', 'alleles')


@pytest.mark.parametrize('corruption', ['version', 'input', 'genes', 'length', 'unknown_member'])
def test_invalid_label_schemas_cannot_change_meaning_of_weight_columns(reference, corruption):
    setup = prepare_model(reference)
    schema = deepcopy(setup.label_schema())
    if corruption == 'version': schema['schema_version'] = 2
    elif corruption == 'input': schema['input_space'] = 'protein'
    elif corruption == 'genes': schema['genes']['d'] = []
    elif corruption == 'length': schema['genes']['v'] = []
    else: schema['genes']['v'] = ['V1', 'V3']
    with pytest.raises(ConfigurationError):
        validate_label_schema(schema, reference, setup.config)


def test_allele_schema_requires_original_order(reference):
    setup = prepare_model(reference)
    schema = setup.label_schema()
    schema['genes']['v'].reverse()
    with pytest.raises(ConfigurationError, match='order'):
        validate_label_schema(schema, reference, setup.config)
