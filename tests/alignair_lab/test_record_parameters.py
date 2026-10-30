"""A record cache stores each stream's parameters losslessly: values that compare equal in Python, or collide in plain
JSON (1 and 1.0, True and 1, a tuple and a list), stay distinct, through the manifest's JSON and back."""
import json
import math

import pytest

from alignair.errors import ConfigurationError
from alignair_lab.records import canonical_parameters, encode_parameters

pytestmark = pytest.mark.component

STREAM = {'indel_count': ((0, 0.9), (1, 0.075), (2, 0.02)), 'end_loss_5': (0, 15), 'mutation_rate': 0.25,
          'productive_only': True, 'genotype': None, 'restrict_alleles': {'v': ['IGHV1-2*02'], 'd': 2}}


@pytest.mark.parametrize('left,right', [({'a': 1}, {'a': 1.0}), ({'a': (0, 15)}, {'a': [0, 15]}),
                                        ({'a': True}, {'a': 1}), ({'a': ((1, 0.75),)}, {'a': [[1, 0.75]]}),
                                        ({'a': {'b': (1,)}}, {'a': {'b': [1]}}), ({'a': 0.0}, {'a': -0.0})])
def test_values_python_or_plain_json_confuses_are_told_apart(left, right):
    assert left == right or json.loads(json.dumps(left)) == json.loads(json.dumps(right))
    assert canonical_parameters(left) != canonical_parameters(right)


def test_parameters_survive_the_manifests_json():
    stored = json.loads(json.dumps(encode_parameters(STREAM)))
    assert canonical_parameters(stored, encoded=True) == canonical_parameters(dict(STREAM))
    assert canonical_parameters(stored, encoded=True) != canonical_parameters({**STREAM, 'mutation_rate': 0.3})
    assert canonical_parameters({'b': 1, 'a': 2}) == canonical_parameters({'a': 2, 'b': 1})


@pytest.mark.parametrize('value', [{'a': math.nan}, {'a': math.inf}, {'a': object()}, {1: 'key'},
                                   {'a': {'__tuple__': [1]}}])
def test_a_value_a_manifest_cannot_hold_losslessly_is_refused(value):
    with pytest.raises(ConfigurationError, match='parameters') as caught:
        encode_parameters(value)
    assert caught.value.context['field'] == 'parameters'
