import numpy as np
import pytest

from alignair.applicability.features import FEATURE_NAMES
from alignair.applicability.feature_cache import FeatureCache, sequence_key
from alignair.applicability.fusion import fit_fusion
from alignair.applicability.tokenizer import AaTokenizer
from alignair.applicability.training import data
from alignair.errors import ArtifactIntegrityError, DataContractError

pytestmark = pytest.mark.component


def test_targets_preserve_masks_and_outliers_receive_no_domain_labels():
    vector = np.arange(len(FEATURE_NAMES), dtype=float)
    fusion = fit_fusion(FEATURE_NAMES, [vector, vector + 1], "bank")
    records = [
        {"aa": "QVQLVQSG", "group": "positive_real", "category": "real", "expected_state": "supported"},
        {"aa": "VVVVVVVV", "group": "structural", "category": "cdr3_only"},
    ]
    cache = {sequence_key(record['aa']): vector.tolist() for record in records}
    tensors = data.build_tensors(records, cache, fusion, AaTokenizer(), "bank", {}, set(),
                                 mask_groups=frozenset({"human_igk"}))
    assert tensors["type_m"].tolist() == [1, 0]
    assert tensors["comp_m"].tolist() == [1, 1]
    assert tensors["energy_m"].tolist() == [1, 0]
    assert tensors["applicability"].tolist() == [1, 0]
    oe = data.build_tensors(records, cache, fusion, AaTokenizer(), "bank", {}, set(), is_oe=True)
    for field in ("type_m", "comp_m", "source_m", "contrast_m"):
        assert oe[field].sum() == 0
    assert oe["energy_out"].tolist() == [1, 1]


def test_all_overlength_records_raise_instead_of_silently_truncating():
    fusion = fit_fusion(FEATURE_NAMES, [np.zeros(len(FEATURE_NAMES))], "bank")
    with pytest.raises(DataContractError, match="length contract"):
        data.build_tensors([{"aa": "A" * 193}], {}, fusion, AaTokenizer(), "bank", {}, set())


def test_source_vocabulary_is_ordered_and_eligibility_requires_multiple_categories():
    records = [
        {"group": "positive_real", "category": "study-b"},
        {"group": "positive_real", "category": "study-a"},
        {"group": "igk_igl", "category": "igk"},
        {"group": "structural", "category": "reversed_vh"},
    ]
    vocabulary, eligible, count = data.source_setup(records)
    assert vocabulary == {"igk": 0, "study-a": 1, "study-b": 2}
    assert count == 3
    assert data.domain_type(records[0]) in eligible
    assert data.domain_type(records[2]) not in eligible


def test_tensor_preparation_requires_features_instead_of_running_another_hmm_search(monkeypatch):
    from alignair.applicability import feature_cache
    def forbidden(*args, **kwargs): pytest.fail('tensor assembly must never run HMM extraction')
    monkeypatch.setattr(feature_cache, 'hmm_features', forbidden)
    fusion = fit_fusion(FEATURE_NAMES, [np.zeros(len(FEATURE_NAMES))], 'bank')
    with pytest.raises(DataContractError, match='not prepared') as caught:
        data.build_tensors([{'aa': 'QVQL'}], {}, fusion, AaTokenizer(), 'bank', {}, set())
    assert caught.value.context['sequence_hash'] == sequence_key('QVQL')


def test_tensor_preparation_rejects_features_bound_to_another_fold():
    cache = FeatureCache({sequence_key('QVQL'): [0.] * len(FEATURE_NAMES)}, bank_checksum='bank', mask_groups={'human_igk'})
    fusion = fit_fusion(FEATURE_NAMES, [np.zeros(len(FEATURE_NAMES))], 'bank')
    with pytest.raises(ArtifactIntegrityError) as caught:
        data.build_tensors([{'aa': 'QVQL'}], cache, fusion, AaTokenizer(), 'bank', {}, set())
    assert caught.value.context['fields'] == ['mask_groups']
