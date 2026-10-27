from copy import deepcopy

import pytest
from alignair.errors import ModelError
from alignair.reference.labels import (
    db_hash, build_equivalence_classes, EquivalenceClassMap,
)

pytestmark = pytest.mark.component


def _toy():
    # a*01 and a*02 translate identically (synonymous) ; b*01 differs.
    aa = {"a*01": "MKV", "a*02": "MKV", "b*01": "MKL"}
    return build_equivalence_classes("V", aa, db_hash([("a*01", "atg"), ("b*01", "atc")]))


def test_identical_translations_share_a_class():
    cmap = _toy()
    assert cmap.n_classes() == 2
    assert cmap.class_of("a*01") == cmap.class_of("a*02")
    assert cmap.class_of("b*01") != cmap.class_of("a*01")


def test_ambiguity_set_returns_all_alleles_in_class():
    cmap = _toy()
    cid = cmap.class_of("a*01")
    assert cmap.ambiguity_set(cid) == ["a*01", "a*02"]


def test_json_round_trip_is_identity():
    cmap = _toy()
    back = EquivalenceClassMap.from_json(cmap.to_json())
    assert back.gene == cmap.gene
    assert back.db_hash == cmap.db_hash
    assert back.allele_class == cmap.allele_class
    assert back.class_members == cmap.class_members
    legacy = cmap.to_json()
    del legacy["schema"]
    assert EquivalenceClassMap.from_json(legacy).to_json() == cmap.to_json()


def test_db_hash_is_deterministic_and_sensitive():
    h1 = db_hash([("a*01", "atg"), ("b*01", "atc")])
    h2 = db_hash([("b*01", "atc"), ("a*01", "atg")])   # order-independent
    h3 = db_hash([("a*01", "atg"), ("b*01", "atg")])   # a seq changed
    assert h1 == h2
    assert h1 != h3 and len(h1) == 16


@pytest.mark.parametrize('defect', ['gap', 'inverse', 'duplicate', 'empty_members', 'bool_label',
                                    'bad_member', 'string_members', 'fields', 'schema', 'class_key',
                                    'fractional_key', 'boolean_key', 'duplicate_key', 'empty_gene', 'not_mapping'])
def test_malformed_label_maps_fail_before_prediction_or_calibration(defect):
    data = deepcopy(_toy().to_json())
    if defect == 'gap': data['class_members']['2'] = data['class_members'].pop('1')
    elif defect == 'inverse': data['allele_class']['a*01'] = 1
    elif defect == 'duplicate': data['class_members']['1'].append('a*01')
    elif defect == 'empty_members': data['class_members']['1'] = []
    elif defect == 'bool_label': data['allele_class']['b*01'] = True
    elif defect == 'bad_member': data['class_members']['1'] = [3]
    elif defect == 'string_members': data['class_members']['1'] = 'b*01'
    elif defect == 'fields': del data['allele_class']
    elif defect == 'schema': data['schema'] = 'unknown.v3'
    elif defect == 'class_key': data['class_members']['oops'] = []
    elif defect == 'fractional_key': data['class_members'][1.5] = data['class_members'].pop('1')
    elif defect == 'boolean_key': data['class_members'][True] = data['class_members'].pop('1')
    elif defect == 'duplicate_key': data['class_members'][0] = data['class_members']['0']
    elif defect == 'empty_gene': data['gene'] = ''
    else: data = []
    with pytest.raises(ModelError) as caught:
        EquivalenceClassMap.from_json(data)
    assert caught.value.context['stage'] == 'reference_labels'


def test_class_id_assignment_is_independent_of_input_order_and_members_are_copied():
    first = build_equivalence_classes('V', {'z': 'MK', 'a': 'MK', 'b': 'ML'}, 'fixture')
    second = build_equivalence_classes('V', {'b': 'ML', 'a': 'MK', 'z': 'MK'}, 'fixture')
    assert first.to_json() == second.to_json()
    assert first.allele_class == {'a': 0, 'b': 1, 'z': 0}
    data = first.to_json()
    loaded = EquivalenceClassMap.from_json(data)
    data['class_members']['0'].clear()
    data['allele_class']['a'] = 7
    assert loaded.ambiguity_set(0) == ['a', 'z'] and loaded.class_of('a') == 0
