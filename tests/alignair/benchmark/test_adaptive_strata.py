import pytest
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.benchmark.core.schema import StratumSpec, BenchmarkSpec
from alignair.benchmark.generation.strata import adaptive_igh_strata
from alignair.benchmark.generation.generate import generate_benchmark
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
    for c in cases:
        assert len(c.sequence) < 200
        v = c.genes.get("V")
        if v is not None and v.germline_start is not None:
            assert v.germline_start >= 150
