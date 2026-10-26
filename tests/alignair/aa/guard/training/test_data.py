import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from alignair.aa.guard.features import FEATURE_NAMES
from alignair.aa.guard.fusion import fit_fusion
from alignair.aa.guard.tokenizer import AaTokenizer
from alignair.aa.guard.training import data
from alignair.errors import DataContractError

pytestmark = pytest.mark.component


def test_feature_cache_reuses_only_matching_bank_and_mask(tmp_path, monkeypatch):
    calls = []

    def features(bank, sequence, *, mask_groups):
        calls.append((bank, sequence, mask_groups))
        vector = np.full(len(FEATURE_NAMES), len(mask_groups) + len(calls), dtype=float)
        return SimpleNamespace(to_vector=lambda: vector)

    monkeypatch.setattr(data, "hmm_features", features)
    records = [{"aa": "QVQLVQSG"}]
    path = tmp_path / "feature cache.json"
    first = data.precompute_features(records, "bank", bank_checksum="bank-1", cache_path=path)
    assert data.precompute_features(records, "bank", bank_checksum="bank-1", cache_path=path) == first
    assert len(calls) == 1
    masked = data.precompute_features(records, "bank", bank_checksum="bank-1", cache_path=path,
                                     mask_groups=frozenset({"human_igk"}))
    assert masked != first
    assert calls[-1][2] == frozenset({"human_igk"})
    changed_bank = data.precompute_features(records, "bank", bank_checksum="bank-2", cache_path=path)
    assert len(calls) == 3 and changed_bank != first


def test_legacy_unbound_cache_is_rebuilt(tmp_path, monkeypatch):
    sequence = "QVQLVQSG"
    key = hashlib.sha256(sequence.encode()).hexdigest()
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({key: [999] * len(FEATURE_NAMES)}))
    monkeypatch.setattr(data, "hmm_features", lambda *a, **k: SimpleNamespace(
        to_vector=lambda: np.zeros(len(FEATURE_NAMES))))
    result = data.precompute_features([{"aa": sequence}], None, bank_checksum="bank", cache_path=path)
    assert result[key] == [0] * len(FEATURE_NAMES)
    assert json.loads(path.read_text())["bank_checksum"] == "bank"


def test_malformed_feature_cache_is_preserved(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("unfinished")
    with pytest.raises(DataContractError, match="JSON"):
        data.precompute_features([], None, bank_checksum="bank", cache_path=path)
    assert path.read_text() == "unfinished"


def test_targets_preserve_masks_and_outliers_receive_no_domain_labels(monkeypatch):
    vector = np.arange(len(FEATURE_NAMES), dtype=float)
    seen = []

    def features(bank, sequence, *, mask_groups):
        seen.append((bank, mask_groups))
        return SimpleNamespace(to_vector=lambda: vector)

    monkeypatch.setattr(data, "hmm_features", features)
    fusion = fit_fusion(FEATURE_NAMES, [vector, vector + 1], "bank")
    records = [
        {"aa": "QVQLVQSG", "group": "positive_real", "category": "real", "expected_state": "supported"},
        {"aa": "VVVVVVVV", "group": "structural", "category": "cdr3_only"},
    ]
    tensors = data.build_tensors(records, {}, fusion, AaTokenizer(), "bank", {}, set(),
                                 bank="explicit bank", mask_groups=frozenset({"human_igk"}))
    assert tensors["type_m"].tolist() == [1, 0]
    assert tensors["comp_m"].tolist() == [1, 1]
    assert tensors["energy_m"].tolist() == [1, 0]
    assert tensors["applicability"].tolist() == [1, 0]
    assert seen == [("explicit bank", frozenset({"human_igk"}))] * 2
    oe = data.build_tensors(records, {}, fusion, AaTokenizer(), "bank", {}, set(), bank=None, is_oe=True)
    for field in ("type_m", "comp_m", "source_m", "contrast_m"):
        assert oe[field].sum() == 0
    assert oe["energy_out"].tolist() == [1, 1]


def test_all_overlength_records_raise_instead_of_silently_truncating():
    fusion = fit_fusion(FEATURE_NAMES, [np.zeros(len(FEATURE_NAMES))], "bank")
    with pytest.raises(DataContractError, match="length contract"):
        data.build_tensors([{"aa": "A" * 193}], {}, fusion, AaTokenizer(), "bank", {}, set(), bank=None)


def test_source_vocabulary_is_ordered_and_eligibility_requires_multiple_categories():
    records = [
        {"group": "positive_real", "category": "study-b"},
        {"group": "positive_real", "category": "study-a"},
        {"group": "igk_igl", "category": "igk"},
    ]
    vocabulary, eligible, count = data.source_setup(records)
    assert vocabulary == {"igk": 0, "study-a": 1, "study-b": 2}
    assert count == 3
    assert data.domain_type(records[0]) in eligible
    assert data.domain_type(records[2]) not in eligible
