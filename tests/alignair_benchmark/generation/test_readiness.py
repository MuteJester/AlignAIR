"""Preflight readiness thresholds and reference-allele coverage."""
import pytest
pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair_benchmark import (
    ReadinessThresholds,
    assess_benchmark_readiness,
    generate_benchmark,
    readiness_thresholds,
)
from alignair.reference.reference_set import ReferenceSet
from tests.alignair_benchmark._support import (_FakeReferenceSet, _manual_case, _small_spec)

pytestmark = pytest.mark.integration


def test_readiness_profiles_distinguish_smoke_from_assay_scale():
    cases = generate_benchmark(_small_spec())
    reference_set = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    smoke = assess_benchmark_readiness(cases, reference_set=reference_set, profile="smoke")
    assay = assess_benchmark_readiness(cases, reference_set=reference_set, profile="assay")
    assert readiness_thresholds("assay").min_cases > readiness_thresholds("smoke").min_cases
    assert readiness_thresholds("allele_complete").min_per_reference_allele > 0
    assert readiness_thresholds("allele_stratified").min_per_allele_context > 0
    assert smoke["grade"] == "pass"
    assert smoke["truth_source"] == "GenAIRR benchmark cases"
    assert assay["grade"] == "fail"
    assert any(check["name"] == "case_count" and check["grade"] == "fail" for check in assay["checks"])
    assert any(check["name"] == "reference_allele_fraction" for check in assay["checks"])
    allele_check = next(check for check in assay["checks"] if check["name"] == "observed_allele_counts")
    assert allele_check["observed"]["j"]["threshold"] == len(reference_set.gene("J").names)
    assert allele_check["observed"]["j"]["threshold"] < assay["thresholds"]["min_observed_alleles_per_gene"]


def test_readiness_can_require_minimum_count_for_every_reference_allele():
    thresholds = ReadinessThresholds(
        profile="unit_allele_complete",
        min_cases=1,
        min_per_stratum=1,
        min_observed_alleles_per_gene=0,
        min_reference_allele_fraction=1.0,
        min_per_reference_allele=2,
    )
    reference_set = _FakeReferenceSet()
    low_cases = [
        _manual_case("c1", ("IGHV1*01",)),
        _manual_case("c2", ("IGHV1*01",)),
        _manual_case("c3", ("IGHV2*01",)),
    ]
    low_report = assess_benchmark_readiness(
        low_cases,
        reference_set=reference_set,
        thresholds=thresholds,
    )
    min_check = next(
        check for check in low_report["checks"] if check["name"] == "reference_allele_min_counts"
    )
    assert low_report["grade"] == "fail"
    assert min_check["grade"] == "fail"
    assert min_check["observed"]["v"]["n_low"] == 1
    assert min_check["details"]["low_allele_examples"]["v"] == {"IGHV2*01": 1}

    passing_cases = low_cases + [_manual_case("c4", ("IGHV2*01",))]
    passing_report = assess_benchmark_readiness(
        passing_cases,
        reference_set=reference_set,
        thresholds=thresholds,
    )
    passing_min_check = next(
        check for check in passing_report["checks"] if check["name"] == "reference_allele_min_counts"
    )
    assert passing_report["grade"] == "pass"
    assert passing_min_check["grade"] == "pass"
    assert passing_min_check["observed"]["v"]["min_count"] == 2


def test_readiness_can_require_reference_alleles_across_contexts():
    thresholds = ReadinessThresholds(
        profile="unit_allele_stratified",
        min_cases=1,
        min_per_stratum=1,
        min_observed_alleles_per_gene=0,
        min_reference_allele_fraction=1.0,
        min_per_reference_allele=1,
        min_per_allele_context=1,
        allele_contexts=("stratum:easy", "orientation:reverse_complement"),
    )
    reference_set = _FakeReferenceSet()
    low_cases = [
        _manual_case("c1", ("IGHV1*01",), stratum="easy", orientation_id=0),
        _manual_case("c2", ("IGHV2*01",), stratum="easy", orientation_id=0),
        _manual_case("c3", ("IGHV1*01",), stratum="hard", orientation_id=1),
    ]
    low_report = assess_benchmark_readiness(
        low_cases,
        reference_set=reference_set,
        thresholds=thresholds,
    )
    context_check = next(
        check for check in low_report["checks"] if check["name"] == "reference_allele_context_min_counts"
    )
    assert low_report["grade"] == "fail"
    assert context_check["grade"] == "fail"
    assert context_check["observed"]["v"]["n_low"] == 1
    assert context_check["details"]["low_cell_examples"]["v"] == [
        {"allele": "IGHV2*01", "context": "orientation:reverse_complement", "count": 0}
    ]

    passing_cases = low_cases + [_manual_case("c4", ("IGHV2*01",), stratum="hard", orientation_id=1)]
    passing_report = assess_benchmark_readiness(
        passing_cases,
        reference_set=reference_set,
        thresholds=thresholds,
    )
    passing_context_check = next(
        check for check in passing_report["checks"] if check["name"] == "reference_allele_context_min_counts"
    )
    assert passing_report["grade"] == "pass"
    assert passing_context_check["grade"] == "pass"
    assert passing_context_check["observed"]["v"]["min_count"] == 1
