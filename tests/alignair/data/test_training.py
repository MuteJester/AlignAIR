"""Reference-bound adapter selection and explicit record preparation share one training path."""
import pytest

from alignair.core import prepare_model
from alignair.data.training import prepare_training_data
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
    assert data.batch_adapter.reference is setup.reference
