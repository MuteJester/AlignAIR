import pytest
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair_benchmark.core.schema import StratumSpec, BenchmarkSpec
from alignair_benchmark.generation import (
    CoveragePlan,
    default_igh_assay_spec,
    generate_coverage_benchmark,
    genairr_feature_catalog,
    validate_genairr_feature_catalog,
)
from alignair_benchmark.generation.strata import adaptive_igh_strata
from alignair_benchmark.generation.generate import generate_benchmark
from alignair.reference.reference_set import ReferenceSet


def test_stratum_anchor_field_defaults_none():
    s = StratumSpec(name="x", n=1, progress=1.0)
    assert s.anchor is None


def test_adaptive_strata_generate_short_reads_with_dropped_5prime_v():
    strata = adaptive_igh_strata(n_per_scenario=6)
    names = {s.name for s in strata}
    assert {"adaptive_fr3", "adaptive_fr2", "adaptive_janchor"} <= names
    spec = BenchmarkSpec(name="adp", dataconfig_name="HUMAN_IGH_OGRDB", seed=1,
                         strata=tuple(s for s in strata if s.name == "adaptive_fr3"))
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    cases = generate_benchmark(spec, reference_set=rs)
    assert cases
    checked = 0
    for c in cases:
        assert len(c.sequence) < 240
        v = c.genes.get("v")                     # BenchmarkCase.genes is lowercase-keyed
        if v is not None and v.germline_start is not None:
            assert v.germline_start >= 150       # FR3-anchored: 5' V truncated near the primer site
            checked += 1
    assert checked > 0                           # the assertion above actually ran


def test_default_assay_spec_includes_adaptive_amplicon_strata():
    spec = default_igh_assay_spec(n_per_stratum=1, n_per_focus=1)
    names = {stratum.name for stratum in spec.strata}
    assert {"adaptive_fr1", "adaptive_fr2", "adaptive_fr3", "adaptive_janchor"} <= names


def test_genairr_feature_catalog_is_valid_and_tracks_future_work():
    validation = validate_genairr_feature_catalog()
    assert validation["valid"], validation["errors"]
    statuses = {feature["status"] for feature in genairr_feature_catalog()}
    assert {"integrated", "partial", "planned"} <= statuses


def test_generation_builder_passes_through_genairr_stratum_options():
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    locked_v = rs.gene("V").names[0]
    stratum = StratumSpec(
        name="genairr_passthrough",
        n=1,
        progress=0.0,
        param_overrides={
            "restrict_alleles": {"v": locked_v},
            "metadata": {"sample_id": "benchmark_subject"},
            "mutation_rate": 0.0,
            "end_loss_5": (0, 0),
            "end_loss_3": (0, 0),
            "indel_count": (0, 0),
            "pcr_error_count": 1,
            "seq_error_count": 1,
            "ambiguous_count": (0, 0),
            "crop_prob": 0.0,
        },
    )
    spec = BenchmarkSpec(
        name="genairr_passthrough",
        dataconfig_name="HUMAN_IGH_OGRDB",
        seed=7,
        strata=(stratum,),
    )

    case = generate_benchmark(spec, gdata.HUMAN_IGH_OGRDB, rs)[0]

    assert case.genes["v"].primary == locked_v
    assert case.record["sample_id"] == "benchmark_subject"
    assert case.record["n_pcr_errors"] == 1
    assert case.record["n_quality_errors"] == 1


def test_coverage_planned_generation_applies_adaptive_anchor_in_serial_path():
    stratum = next(s for s in adaptive_igh_strata(n_per_scenario=3) if s.name == "adaptive_fr3")
    spec = BenchmarkSpec(name="adp_planned", dataconfig_name="HUMAN_IGH_OGRDB", seed=5, strata=(stratum,))
    plan = CoveragePlan(name="min_adaptive", min_cases=3, max_candidates=3)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)

    result = generate_coverage_benchmark(spec, gdata.HUMAN_IGH_OGRDB, rs, plan=plan, workers=1)

    assert len(result.cases) == 3
    assert result.report["generation_profile"]["workers"] == 1
    for case in result.cases:
        assert len(case.sequence) < 240
        assert tuple(case.tags["anchor"]) == ("v_germline", 200)
        if case.genes["v"].germline_start is not None:
            assert case.genes["v"].germline_start >= 150
