"""Bound cache reuse, validated feature vectors and publication failure boundaries."""
from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace

import numpy as np
import pytest

from alignair.aa.guard import feature_cache as features
from alignair.aa.guard.features import FEATURE_NAMES
from alignair.errors import ArtifactIntegrityError, ConfigurationError, DataContractError, InputError, OperationIOError
from alignair.io import atomic

pytestmark = pytest.mark.component


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    bank = SimpleNamespace(checksum=lambda: 'bank')
    calls = []
    def extract(bank, sequence, *, mask_groups):
        calls.append((sequence, mask_groups))
        vector = np.full(len(FEATURE_NAMES), len(sequence) + len(mask_groups), dtype=float)
        return SimpleNamespace(to_vector=lambda: vector)
    monkeypatch.setattr(features, 'hmm_features', extract)
    return dict(bank=bank, bank_checksum='bank', cache_path=tmp_path/'cache.json'), calls


def test_matching_v1_cache_reuses_vectors_and_only_computes_new_unique_sequences(preparation, monkeypatch):
    options, calls = preparation
    first = features.precompute_features([{'aa': 'QVQL'}, {'aa': 'QVQL'}], **options)
    before = options['cache_path'].read_bytes()
    def forbidden(*args, **kwargs): pytest.fail('a warm cache must not be rewritten')
    with monkeypatch.context() as patch:
        patch.setattr(features, 'atomic_write_json', forbidden)
        again = features.precompute_features([{'aa': 'QVQL'}], **options)
    assert dict(first) == dict(again) and len(calls) == 1 and options['cache_path'].read_bytes() == before
    extended = features.precompute_features([{'aa': 'QVQL'}, {'aa': 'EVQL'}, {'aa': 'EVQL'}], **options)
    assert len(extended) == 2 and len(calls) == 2
    assert json.loads(options['cache_path'].read_text())['schema_version'] == 1


def test_bank_mask_and_feature_order_mismatch_rebuilds_without_reusing_stale_vectors(preparation):
    options, calls = preparation
    first = features.precompute_features([{'aa': 'QVQL'}], **options)
    masked = features.precompute_features([{'aa': 'QVQL'}], **options, mask_groups={'human_igk'})
    assert dict(masked) != dict(first) and calls[-1][1] == frozenset({'human_igk'})
    new_bank = SimpleNamespace(checksum=lambda: 'different')
    options.update(bank=new_bank, bank_checksum='different')
    features.precompute_features([{'aa': 'QVQL'}], **options)
    stored = json.loads(options['cache_path'].read_text()); stored['feature_names'].reverse()
    options['cache_path'].write_text(json.dumps(stored))
    features.precompute_features([{'aa': 'QVQL'}], **options)
    assert len(calls) == 4


def test_legacy_unbound_cache_is_rebuilt(preparation):
    options, calls = preparation
    key = features.sequence_key('QVQL')
    options['cache_path'].write_text(json.dumps({key: [999] * len(FEATURE_NAMES)}))
    result = features.precompute_features([{'aa': 'QVQL'}], **options)
    assert result[key] == [4.] * len(FEATURE_NAMES) and len(calls) == 1


@pytest.mark.parametrize('content', [b'broken JSON', b'\xff', b'[]'])
def test_malformed_file_is_preserved_without_extraction(preparation, content):
    options, calls = preparation; options['cache_path'].write_bytes(content)
    with pytest.raises(DataContractError): features.precompute_features([], **options)
    assert options['cache_path'].read_bytes() == content and not calls


@pytest.mark.parametrize('bad', [None, [], {'bad-key': [1] * len(FEATURE_NAMES)},
    {'a'*64: []}, {'a'*64: [float('nan')] * len(FEATURE_NAMES)}, {'a'*64: ['1'] * len(FEATURE_NAMES)}])
def test_matching_but_corrupt_vectors_are_refused_including_unrequested_entries(preparation, bad):
    options, calls = preparation
    features.precompute_features([], **options)
    data = json.loads(options['cache_path'].read_text()); data['vectors'] = bad
    options['cache_path'].write_text(json.dumps(data)); before = options['cache_path'].read_bytes()
    with pytest.raises(DataContractError): features.precompute_features([{'aa': 'QVQL'}], **options)
    assert options['cache_path'].read_bytes() == before and not calls


def test_missing_vectors_field_is_not_silently_repaired(preparation):
    options, calls = preparation
    features.precompute_features([], **options)
    data = json.loads(options['cache_path'].read_text()); del data['vectors']
    options['cache_path'].write_text(json.dumps(data)); before = options['cache_path'].read_bytes()
    with pytest.raises(DataContractError): features.precompute_features([], **options)
    assert options['cache_path'].read_bytes() == before and not calls


def test_boolean_schema_version_is_not_accepted_as_integer_version_one(preparation):
    options, calls = preparation
    features.precompute_features([{'aa': 'QVQL'}], **options)
    data = json.loads(options['cache_path'].read_text()); data['schema_version'] = True
    options['cache_path'].write_text(json.dumps(data))
    features.precompute_features([{'aa': 'QVQL'}], **options)
    assert len(calls) == 2 and type(json.loads(options['cache_path'].read_text())['schema_version']) is int


def test_bank_mismatch_is_refused_even_for_a_warm_cache(preparation):
    options, calls = preparation
    features.precompute_features([{'aa': 'QVQL'}], **options)
    before = options['cache_path'].read_bytes()
    with pytest.raises(ArtifactIntegrityError):
        features.precompute_features([{'aa': 'QVQL'}], **{**options, 'bank_checksum': 'wrong'})
    assert len(calls) == 1 and options['cache_path'].read_bytes() == before


@pytest.mark.parametrize('changes', [{'bank': None}, {'bank_checksum': ''}, {'mask_groups': 'human_igk'},
    {'mask_groups': [None]}, {'mask_groups': [[]]}])
def test_invalid_cache_request_cannot_extract_or_write(preparation, changes):
    options, calls = preparation
    with pytest.raises(ConfigurationError): features.precompute_features([], **{**options, **changes})
    assert not options['cache_path'].exists() and not calls


@pytest.mark.parametrize('record', [None, {}, {'aa': None}, {'aa': ''}])
def test_all_records_are_preflighted_before_extraction(preparation, record):
    options, calls = preparation
    with pytest.raises(InputError) as caught: features.precompute_features([{'aa': 'QVQL'}, record], **options)
    assert caught.value.context['record_index'] == 1 and not calls and not options['cache_path'].exists()


@pytest.mark.parametrize('value', [True, '1', float('inf'), 2**10000])
def test_bound_vectors_cannot_coerce_invalid_scalar_values(value):
    with pytest.raises(DataContractError):
        features.FeatureCache({'a'*64: [value] * len(FEATURE_NAMES)}, bank_checksum='bank')


def test_cache_is_read_only_but_exposes_json_compatible_list_copies():
    key = features.sequence_key('QVQL'); vector = [1] * len(FEATURE_NAMES)
    cache = features.FeatureCache({key: vector}, bank_checksum='bank', mask_groups={'human_igk'})
    vector[0] = 99; exposed = cache[key]; exposed[0] = 99
    assert cache[key][0] == 1 and json.loads(json.dumps(dict(cache)))[key][0] == 1
    with pytest.raises(FrozenInstanceError): cache.bank_checksum = 'other'
    with pytest.raises(TypeError): cache.vectors[key] = ()
    assert features.bind_feature_cache(cache, bank_checksum='bank', mask_groups={'human_igk'}) is cache
    with pytest.raises(ArtifactIntegrityError): features.bind_feature_cache(cache, bank_checksum='bank')
    with pytest.raises(ConfigurationError): features.bind_feature_cache(cache, bank_checksum='')
    with pytest.raises(DataContractError) as caught: features.feature_vector(cache, 'EVQL')
    assert caught.value.context['sequence_hash'] == features.sequence_key('EVQL')


@pytest.mark.parametrize('changes', [{'bank_checksum': ''}, {'feature_names': None}, {'feature_names': ('wrong',)}])
def test_invalid_constructor_identity_is_classified(changes):
    with pytest.raises((ConfigurationError, ArtifactIntegrityError)):
        features.FeatureCache({}, **{'bank_checksum': 'bank', **changes})


def test_cache_read_io_failure_is_chained(preparation, tmp_path):
    options, calls = preparation
    with pytest.raises(OperationIOError) as caught:
        features.precompute_features([], **{**options, 'cache_path': tmp_path})
    assert isinstance(caught.value.__cause__, OSError) and not calls


def test_extractor_failure_does_not_publish_any_partial_extension(preparation, monkeypatch):
    options, calls = preparation
    features.precompute_features([{'aa': 'QVQL'}], **options); before = options['cache_path'].read_bytes()
    def fail(*args, **kwargs): raise RuntimeError('HMM search failed')
    monkeypatch.setattr(features, 'hmm_features', fail)
    with pytest.raises(RuntimeError, match='HMM search failed'):
        features.precompute_features([{'aa': 'EVQL'}], **options)
    assert options['cache_path'].read_bytes() == before


def test_invalid_extractor_output_does_not_replace_a_good_cache(preparation, monkeypatch):
    options, calls = preparation
    features.precompute_features([{'aa': 'QVQL'}], **options); before = options['cache_path'].read_bytes()
    monkeypatch.setattr(features, 'hmm_features', lambda *a, **k: SimpleNamespace(to_vector=lambda: np.zeros(1)))
    with pytest.raises(DataContractError): features.precompute_features([{'aa': 'EVQL'}], **options)
    assert options['cache_path'].read_bytes() == before


def test_atomic_replacement_failure_preserves_the_last_complete_cache(preparation, monkeypatch):
    options, calls = preparation
    features.precompute_features([{'aa': 'QVQL'}], **options); before = options['cache_path'].read_bytes()
    def fail(*args): raise OSError('replacement failed')
    monkeypatch.setattr(atomic.os, 'replace', fail)
    with pytest.raises(OperationIOError): features.precompute_features([{'aa': 'EVQL'}], **options)
    assert options['cache_path'].read_bytes() == before and list(options['cache_path'].parent.iterdir()) == [options['cache_path']]
