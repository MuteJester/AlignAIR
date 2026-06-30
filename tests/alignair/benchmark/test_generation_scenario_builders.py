import pytest

genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata

from alignair.benchmark import (
    BenchmarkSpec,
    StratumSpec,
    case_measurement_scenarios,
    generate_benchmark,
    genotype_subject_spec,
    measurement_coverage_summary,
    multi_locus_specs,
    restricted_allele_panel_spec,
    select_sibling_allele_panels,
    validate_stratum_records,
)
from alignair.benchmark.generation.strata import isolated_params
from alignair.reference.reference_set import ReferenceSet


def test_restricted_allele_panel_spec_uses_genairr_restrict_alleles():
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    panel = select_sibling_allele_panels(rs, segment="V", n_panels=1, alleles_per_panel=2)[0]
    spec = restricted_allele_panel_spec(
        n_panels=1,
        alleles_per_panel=2,
        n_per_panel=4,
        seed=31,
    )

    case = generate_benchmark(spec, gdata.HUMAN_IGH_OGRDB, rs)[0]

    assert case.genes["v"].primary in set(panel.alleles)
    assert case.record["benchmark_measurement"] == "allele_coverage_and_candidates"
    assert "allele_coverage_and_candidates" in case_measurement_scenarios(case)
    assert case.record["restricted_gene"] == panel.gene
    assert set(case.record["restricted_alleles"].split(",")) == set(panel.alleles)


def test_genotype_subject_spec_uses_genairr_genotype_and_validation_gate():
    spec = genotype_subject_spec(n_subjects=1, n_per_subject=2, genotype_seed=41, validate_records=True)

    cases = generate_benchmark(spec, gdata.HUMAN_IGH_OGRDB)

    assert len(cases) == 2
    for case in cases:
        assert case.record["benchmark_measurement"] == "genotype_masked_inference"
        assert "genotype_masked_inference" in case_measurement_scenarios(case)
        assert case.record["benchmark_subject"] == "subject_000"
        assert case.record["subject_id"] == "subject_000"
        assert case.record["haplotype"] in (0, 1)


def test_multi_locus_specs_create_per_config_probe_specs():
    specs = multi_locus_specs(
        dataconfig_names=("HUMAN_IGH_OGRDB", "HUMAN_IGK_OGRDB", "HUMAN_IGL_OGRDB"),
        n_per_locus=1,
        seed=53,
    )

    assert [spec.dataconfig_name for spec in specs] == [
        "HUMAN_IGH_OGRDB",
        "HUMAN_IGK_OGRDB",
        "HUMAN_IGL_OGRDB",
    ]
    for spec in specs:
        dataconfig = getattr(gdata, spec.dataconfig_name)
        case = generate_benchmark(spec, dataconfig)[0]
        assert case.record["benchmark_measurement"] == "multi_locus_chain"
        assert "multi_locus_chain" in case_measurement_scenarios(case)
        assert case.record["benchmark_locus"].startswith("IG")
        assert case.genes["v"].calls
        assert case.genes["j"].calls


def test_measurement_coverage_summary_counts_dynamic_builder_cases():
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    spec = restricted_allele_panel_spec(n_panels=1, alleles_per_panel=2, n_per_panel=2, seed=71)

    cases = generate_benchmark(spec, gdata.HUMAN_IGH_OGRDB, rs)
    summary = measurement_coverage_summary(cases)

    assert summary["explicit_by_measurement"]["allele_coverage_and_candidates"] == 2
    assert summary["by_measurement"]["allele_coverage_and_candidates"] == 2
    assert summary["n_unmapped_cases"] == 0


def test_validate_stratum_records_runs_genairr_record_validator():
    stratum = StratumSpec(
        name="validated_clean",
        n=1,
        progress=0.0,
        param_overrides=isolated_params(),
    )

    report = validate_stratum_records(gdata.HUMAN_IGH_OGRDB, stratum, n=1, seed=61)

    assert report["valid"]
    assert report["n_records"] == 1
    assert report["error"] is None
