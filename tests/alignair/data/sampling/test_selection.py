"""Stateless class selection, caller-owned quotas and bounded collection accounting."""
from collections import Counter

import numpy as np
import pytest

from alignair.data.sampling import (
    ClassDiversity, SampleDiversityError, SampleResult, assert_batch_diverse,
    check_collection_counts, collect_samples, diverse_selector,
)
from alignair.errors import ConfigurationError, TrainingDataError

pytestmark = pytest.mark.component


def test_selector_prefers_new_classes_then_fills_after_all_quotas():
    select = ClassDiversity({"v_class": 2, "j_class": 2})
    chosen = [{"v_class": 1, "j_class": 0}]
    assert select({"v_class": 2, "j_class": 0}, chosen)
    assert select({"v_class": 1, "j_class": 5}, chosen)
    assert not select({"v_class": 1, "j_class": 0}, chosen)
    # Preserve the original OR rule even when only J's quota is still unmet.
    chosen += [{"v_class": 2, "j_class": 0}]
    assert select({"v_class": 3, "j_class": 0}, chosen)
    assert not select({"v_class": 1, "j_class": 0}, chosen)
    chosen += [{"v_class": 1, "j_class": 5}]
    assert select({"v_class": 1, "j_class": 0}, chosen)
    assert select.validate(chosen) is None


def test_policy_is_field_based_immutable_and_has_no_cross_collection_memory():
    minimums = {"d_allele_id": np.int64(2)}
    policy = ClassDiversity(minimums)
    minimums["d_allele_id"] = 100
    assert policy.minimums == {"d_allele_id": 2}
    with pytest.raises(TypeError):
        policy.minimums["d_allele_id"] = 1
    assert policy({"d_allele_id": 0}, [])
    assert not policy({"d_allele_id": 0}, [{"d_allele_id": 0}])
    assert policy({"d_allele_id": 0}, [])
    policy.validate([{"d_allele_id": np.int64(0)}, {"d_allele_id": 1}])


def test_insufficient_diversity_reports_observed_and_required_counts():
    policy = ClassDiversity({"v_class": 3, "j_class": 2})
    with pytest.raises(SampleDiversityError) as caught:
        policy.validate({"v_class": i, "j_class": 0} for i in range(2))
    assert isinstance(caught.value, TrainingDataError)
    assert isinstance(caught.value, ValueError)  # historical assertion compatibility
    assert caught.value.context == {"stage": "sample_diversity",
                                    "observed": {"v_class": 2, "j_class": 1},
                                    "required": {"v_class": 3, "j_class": 2}}
    with pytest.raises(SampleDiversityError):
        policy.validate([])


@pytest.mark.parametrize("minimums", [None, [], {}, {"": 2}, {1: 2}, {"v": True},
                                      {"v": 0}, {"v": -1}, {"v": 1.5}, {"v": "2"}])
def test_invalid_quotas_fail_at_policy_construction(minimums):
    with pytest.raises(ConfigurationError):
        ClassDiversity(minimums)


@pytest.mark.parametrize("sample", [{}, {"v_class": -1}, {"v_class": True},
                                   {"v_class": 1.5}, {"v_class": "1"},
                                   {"v_class": None}, {"v_class": []}, None, []])
def test_invalid_class_ids_never_count_towards_diversity(sample):
    policy = ClassDiversity({"v_class": 1})
    with pytest.raises(TrainingDataError) as caught:
        policy(sample, [])
    assert caught.value.context["field"] == "v_class"
    with pytest.raises(TrainingDataError):
        policy.validate([sample])
    with pytest.raises(TrainingDataError):
        policy({"v_class": 1}, [sample])


def test_collection_preserves_selection_census_and_requires_explicit_final_validation():
    records = [{"v_class": v, "j_class": j} for v, j in [(0,0), (0,0), (1,0), (1,1), (0,0)]]
    policy = ClassDiversity({"v_class": 2, "j_class": 2})
    counts = Counter()
    result = collect_samples(records, lambda r: SampleResult(r), want=4, budget=5,
                             select=policy, counter=counts)
    assert result.samples == [records[i] for i in (0,2,3,4)]
    assert counts == {"generated": 5, "usable": 5, "selected": 4, "selection_skipped": 1}
    check_collection_counts(counts)
    policy.validate(result.samples)
    # Filling a small batch with unseen V classes can still leave J's quota unmet.
    result = collect_samples(records, lambda r: SampleResult(r), want=2, budget=5, select=policy)
    with pytest.raises(SampleDiversityError):
        policy.validate(result.samples)


def test_historical_vj_helpers_delegate_to_shared_policy():
    assert isinstance(diverse_selector(), ClassDiversity)
    same = [{"v_class": 1, "j_class": 0} for _ in range(8)]
    with pytest.raises(ValueError, match="diver"):
        assert_batch_diverse(same)
    mixed = [{"v_class": i % 4, "j_class": i % 2} for i in range(8)]
    assert_batch_diverse(mixed)
