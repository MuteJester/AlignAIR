import hashlib
import math
import re

import pytest

pytest.importorskip("GenAIRR")

from GenAIRR import Experiment  # noqa: E402

from alignair_assay.core.reference import dataconfig  # noqa: E402
from alignair_assay.core.seqops import revcomp  # noqa: E402
from alignair_assay.suites import SUITES, get_suite  # noqa: E402
from alignair_assay.suites.base import Stratum, Suite  # noqa: E402
from alignair_assay.suites.donors import DonorSpec, sample_donors  # noqa: E402
from alignair_assay.suites.engine import SPLITS, derive_seed, generate_suite, requested_count  # noqa: E402

REFERENCE = "HUMAN_IGH_OGRDB"


def _generate(suite="S0", **overrides):
    options = dict(reference_name=REFERENCE, split="dev", seed=7, scale=1.0, workers=1) | overrides
    return generate_suite(get_suite(suite) if isinstance(suite, str) else suite, **options)


def _by_stratum(cases):
    out = {}
    for case in cases:
        out.setdefault(case["stratum"], []).append(case)
    return out


def _igh(ctx):
    return Experiment.on(ctx.config).recombine()


def _uses_j6(record):
    return record["j_call"].startswith("IGHJ6")


def _v_family(record, ctx):
    return {"v_family": record["v_call"].split("-")[0]}


def _lineages(ctx):
    return Experiment.on(ctx.config).recombine().clonal_lineage(n_clones=ctx.n, n_sample=4, rate=0.05)


def _one_lineage_short(ctx):
    return Experiment.on(ctx.config).recombine().clonal_lineage(n_clones=ctx.n - 1, n_sample=4, rate=0.05)


def _suite(*strata, name="T"):
    return Suite(name=name, slug="test", title="Test suite", purpose="Exercises the engine.", primary=(),
                 strata=tuple(strata))


def test_derive_seed_formula():
    expected = int.from_bytes(hashlib.sha256(b"20261019:dev:S0:plain:0").digest()[:4], "big") % 2**31
    assert derive_seed(20261019, "dev", "S0", "plain", 0) == expected
    assert derive_seed(1, "dev", "S0", "plain", 0) != derive_seed(1, "test", "S0", "plain", 0)
    assert 0 <= derive_seed(5, "orientation") < 2**31


def test_unknown_split_raises():
    assert SPLITS == ("dev", "test")
    with pytest.raises(ValueError, match=r"unknown split 'train'.*dev, test"):
        _generate(split="train")


def test_unknown_suite_lists_known():
    assert {"S0", "S0_EXHAUST", "S0_COHORT"} <= set(SUITES)
    assert all(suite.name == name for name, suite in SUITES.items())
    with pytest.raises(ValueError, match=r"unknown suite 'S99'; known suites: S0, S0_EXHAUST, S0_COHORT"):
        get_suite("S99")


def test_case_ids_unique_and_formatted():
    cases = _generate().cases
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids)) == 50
    assert all(re.fullmatch(r"S0:(plain|with_d):\d{6}", i) for i in ids)
    strata = _by_stratum(cases)
    assert list(strata) == ["plain", "with_d"]
    for name, group in strata.items():
        assert [c["case_id"] for c in group] == [f"S0:{name}:{i:06d}" for i in range(len(group))]
        assert all(c["suite"] == "S0" and c["labels"] == {} and c["groups"] == {} for c in group)


def test_rc_fraction():
    strata = _by_stratum(_generate(scale=10).cases)
    plain = strata["plain"]
    assert len(plain) == 300
    assert 0.35 <= sum(c["orientation"] for c in plain) / len(plain) <= 0.65
    assert {c["orientation"] for c in strata["with_d"]} == {0}  # rc_fraction 0
    for case in plain[:20]:
        forward = case["record"]["sequence"].upper()
        assert case["sequence"] == (revcomp(forward) if case["orientation"] else forward)


def test_accept_filter_applied():
    generated = _generate()
    with_d = _by_stratum(generated.cases)["with_d"]
    assert len(with_d) == 20 and all(c["record"]["d_call"] for c in with_d)
    stats = generated.stats["with_d"]
    assert stats["requested"] == 20 and stats["rounds"] >= 1 and stats["generated"] >= 40  # factor 2 first

    # A selective filter keeps the first accepted records in generation order, round 0 first.
    suite = _suite(Stratum("j6", 40, _igh, accept=_uses_j6))
    generated = _generate(suite)
    assert len(generated.cases) == 40 and all(_uses_j6(c["record"]) for c in generated.cases)
    stats = generated.stats["j6"]
    assert stats["accepted"] >= 40 and stats["generated"] > stats["accepted"]
    first_round = (Experiment.on(dataconfig(REFERENCE)).recombine()
                   .run_records(n=80, seed=derive_seed(7, "dev", "T", "j6", 0)).records)
    prefix = [r for r in first_round if _uses_j6(r)][:40]
    assert prefix and [c["record"] for c in generated.cases[:len(prefix)]] == prefix
    # Round 0 over-asks by 2; round 1 by 1 / acceptance so far.
    k = len(prefix)
    assert stats["rounds"] == 2 and k < 40
    assert stats["generated"] == 80 + math.ceil((40 - k) * min(20, 1 / max(k / 80, 0.05)))


def test_scale():
    assert requested_count(Stratum("x", 30, _igh), 0.5) == 15
    assert requested_count(Stratum("x", 30, _igh), 0.001) == 1
    counts = {name: len(group) for name, group in _by_stratum(_generate(scale=0.5).cases).items()}
    assert counts == {"plain": 15, "with_d": 10}
    with pytest.raises(ValueError, match="scale"):
        _generate(scale=0)


def test_exhaustion_raises_with_names():
    with pytest.raises(RuntimeError, match=re.escape("S0_EXHAUST/never: accepted 0 of 5 after 50 rounds")):
        _generate("S0_EXHAUST")


def test_splits_differ():
    dev = _generate(split="dev").cases
    test = _generate(split="test").cases
    assert [c["case_id"] for c in dev] == [c["case_id"] for c in test]
    assert not {c["sequence"] for c in dev} & {c["sequence"] for c in test}


def test_workers_do_not_change_output():
    one = _generate(workers=1)
    two = _generate(workers=2)
    assert one.cases == two.cases and one.stats == two.stats and one.donors == two.donors == ()


def test_parallel_generation_needs_a_registered_suite():
    suite = _suite(Stratum("a", 2, _igh), Stratum("b", 2, _igh))
    with pytest.raises(ValueError, match="registered"):
        _generate(suite, workers=2)
    assert len(_generate(suite, workers=1).cases) == 4


def test_pairs_labels_and_clones():
    suite = _suite(Stratum("pairs", 3, _igh, labels={"design": "full"}, both_orientations=True, rc_fraction=0.5,
                           case_labels=_v_family),
                   Stratum("lineages", 2, _lineages, clonal=True, rc_fraction=0.5))
    generated = _generate(suite)
    strata = _by_stratum(generated.cases)

    pairs = strata["pairs"]
    assert [c["case_id"] for c in pairs] == [f"T:pairs:{i:06d}:{side}" for i in range(3) for side in ("fwd", "rc")]
    assert [c["orientation"] for c in pairs] == [0, 1] * 3
    for fwd, rc in zip(pairs[0::2], pairs[1::2]):
        assert fwd["record"] == rc["record"] and rc["sequence"] == revcomp(fwd["sequence"])
        assert fwd["groups"] == rc["groups"] == {"pair": fwd["case_id"].removesuffix(":fwd")}
        assert fwd["labels"] == {"design": "full", "v_family": fwd["record"]["v_call"].split("-")[0]}
    assert generated.stats["pairs"]["cases"] == 6

    lineages = strata["lineages"]
    assert [c["case_id"] for c in lineages] == [f"T:lineages:{i:06d}" for i in range(len(lineages))]
    assert {c["groups"]["clone"] for c in lineages} == {"0", "1"}
    assert all(c["groups"] == {"clone": str(c["record"]["clone_id"])} for c in lineages)
    stats = generated.stats["lineages"]
    assert stats["requested"] == 2 and stats["clones"] == 2 and stats["rounds"] == 1
    assert stats["cases"] == stats["generated"] == len(lineages)


def test_short_clonal_stratum_raises():
    suite = _suite(Stratum("lineages", 3, _one_lineage_short, clonal=True))
    with pytest.raises(RuntimeError, match=re.escape("T/lineages: generated 2 of 3 clones")):
        _generate(suite)


def test_cohort_stratum_needs_donors():
    suite = _suite(Stratum("cohort", 4, _igh, cohort=True))
    with pytest.raises(ValueError, match="T/cohort: a cohort stratum needs donors"):
        _generate(suite)


def test_cohort_strata(reference):
    generated = _generate("S0_COHORT")
    spec = get_suite("S0_COHORT").donors
    assert spec == DonorSpec(n=3, tag="S0")
    donors = sample_donors(reference, 3, derive_seed(7, "dev", "S0_COHORT", "donors", "S0"))
    assert generated.donors == tuple(donors)
    strata = _by_stratum(generated.cases)

    # 10 cases over 3 donors: ceil(10 / 3) = 4 records per donor, truncated to the first 10 in record order.
    cohort = strata["cohort"]
    assert [c["case_id"] for c in cohort] == [f"S0_COHORT:cohort:{i:06d}" for i in range(10)]
    subjects = ["donor00"] * 4 + ["donor01"] * 4 + ["donor02"] * 2
    assert [c["groups"] for c in cohort] == [{"subject": s} for s in subjects]
    assert all(c["record"]["subject_id"] == c["groups"]["subject"] for c in cohort)
    stats = generated.stats["cohort"]
    assert stats == {"requested": 10, "accepted": 12, "generated": 12, "rounds": 1, "cases": 10}
    cfg = dataconfig(REFERENCE)
    expected = (Experiment.on(cfg).recombine().mutate(rate=0.03)
                .run_cohort([d.genotype(cfg) for d in donors], n_per_subject=4,
                            seed=derive_seed(7, "dev", "S0_COHORT", "cohort", 0)).records)
    assert [c["record"] for c in cohort] == expected[:10]

    # Fewer cases than donors: one record each, so the last donor has no case.
    few = strata["few"]
    assert [c["groups"]["subject"] for c in few] == ["donor00", "donor01"]
    assert generated.stats["few"] == {"requested": 2, "accepted": 3, "generated": 3, "rounds": 1, "cases": 2}


def test_cohort_workers_do_not_change_output():
    one = _generate("S0_COHORT", workers=1)
    two = _generate("S0_COHORT", workers=2)
    assert one.cases == two.cases and one.stats == two.stats and one.donors == two.donors
    assert len(one.donors) == 3
    test = _generate("S0_COHORT", split="test")
    assert test.donors != one.donors


@pytest.mark.parametrize("options, message", [
    (dict(rc_fraction=1.5), "rc_fraction"),
    (dict(n=0), "n must be"),
    (dict(cohort=True, clonal=True), "cohort and clonal"),
    (dict(clonal=True, accept=_uses_j6), "accept"),
    (dict(cohort=True, accept=_uses_j6), "accept"),
])
def test_invalid_stratum(options, message):
    with pytest.raises(ValueError, match=message):
        Stratum(**({"name": "x", "n": 5, "recipe": _igh} | options))


def test_invalid_suite():
    with pytest.raises(ValueError, match="duplicate stratum"):
        _suite(Stratum("a", 1, _igh), Stratum("a", 1, _igh))
    with pytest.raises(ValueError, match="no strata"):
        _suite()
    with pytest.raises(ValueError, match="':'"):
        _suite(Stratum("a:b", 1, _igh))
