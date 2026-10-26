"""Benchmark generation, deterministic ordering, and coordinate transforms."""
import pytest
pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair_benchmark import (
    BenchmarkSpec,
    StratumSpec,
    coverage_summary,
    default_igh_assay_spec,
    focused_igh_spec,
    generate_benchmark,
    generate_benchmark_with_report,
    stream_benchmark,
)
from alignair.reference.reference_set import ReferenceSet
from tests.alignair_benchmark._support import _small_spec

pytestmark = pytest.mark.integration


def test_generate_benchmark_cases_and_coverage():
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    cases = generate_benchmark(_small_spec(), gdata.HUMAN_IGH_OGRDB, rs)
    assert len(cases) == 7
    assert {c.stratum for c in cases} == {"clean", "oriented"}
    assert {c.orientation_id for c in cases if c.stratum == "oriented"} == {0, 1, 2, 3}
    for case in cases:
        assert case.sequence
        assert case.genes["v"].calls
        assert case.presented_genes["v"].calls == case.genes["v"].calls
        assert len(case.region_labels) == len(case.canonical_sequence)
        assert len(case.presented_region_labels) == len(case.sequence)
    cov = coverage_summary(cases)
    assert cov["n_cases"] == 7
    assert cov["by_stratum"]["clean"] == 3


@pytest.mark.filterwarnings("error:.*multi-threaded.*fork.*:DeprecationWarning")
def test_parallel_generation_preserves_case_order_and_sequences():
    spec = BenchmarkSpec(
        name="parallel",
        dataconfig_name="HUMAN_IGH_OGRDB",
        seed=23,
        strata=(
            StratumSpec(name="clean", n=1, progress=0.0),
            StratumSpec(name="hard", n=1, progress=0.4),
        ),
    )

    serial = generate_benchmark(spec, workers=1)
    parallel = generate_benchmark(spec, workers=2)
    assert [case.case_id for case in parallel] == [case.case_id for case in serial]
    assert [case.sequence for case in parallel] == [case.sequence for case in serial]
    assert [case.orientation_id for case in parallel] == [case.orientation_id for case in serial]

    result = generate_benchmark_with_report(spec, workers=2)
    assert [case.case_id for case in result.cases] == [case.case_id for case in serial]
    assert result.report["mode"] == "fixed_size"
    assert result.report["workers"] == 2
    assert result.report["parallel"] is True
    assert result.report["n_cases"] == len(serial)
    assert result.report["generated_candidates"] == len(serial)
    assert result.report["cases_per_second"] is not None


def test_focused_igh_spec_generates_hard_to_hit_scenarios():
    cases = generate_benchmark(focused_igh_spec(n_per_scenario=1, seed=23))
    by_stratum = {case.stratum: case for case in cases}
    assert {
        "productive_only_clean",
        "forced_d_inversion",
        "receptor_revision",
        "contaminant",
        "paired_end",
        "ultra_short_fragment_40",
    }.issubset(by_stratum)
    assert by_stratum["productive_only_clean"].record["productive"] is True
    assert by_stratum["forced_d_inversion"].record["d_inverted"] is True
    assert by_stratum["receptor_revision"].record["receptor_revision_applied"] is True
    assert by_stratum["contaminant"].record["is_contaminant"] is True
    assert by_stratum["paired_end"].record["read_layout"] == "paired_end"
    assert by_stratum["paired_end"].record["r1_sequence"]
    assert by_stratum["paired_end"].record["r2_sequence"]
    assert len(by_stratum["ultra_short_fragment_40"].sequence) == 40


def test_assay_spec_combines_broad_and_focused_strata():
    spec = default_igh_assay_spec(n_per_stratum=1, n_per_focus=1, seed=29)
    names = {stratum.name for stratum in spec.strata}
    assert "clean_full" in names
    assert "forced_d_inversion" in names
    assert "contaminant" in names
    assert spec.n_cases > focused_igh_spec(n_per_scenario=1).n_cases


def test_reverse_orientation_maps_presented_coordinates():
    cases = generate_benchmark(
        BenchmarkSpec(
            name="rev",
            dataconfig_name="HUMAN_IGH_OGRDB",
            seed=9,
            strata=(StratumSpec(name="rev", n=1, progress=0.1, orientation_ids=(3,)),),
        )
    )
    case = cases[0]
    L = len(case.canonical_sequence)
    v = case.genes["v"]
    pv = case.presented_genes["v"]
    assert pv.sequence_start == L - v.sequence_end
    assert pv.sequence_end == L - v.sequence_start
    assert case.presented_region_labels == list(reversed(case.region_labels))


def test_stream_benchmark_yields_without_materializing():
    stream = stream_benchmark(_small_spec())
    first = next(stream)
    second = next(stream)
    assert first.case_id != second.case_id
    assert first.sequence and second.sequence
