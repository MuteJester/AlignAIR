import json
import statistics

import pytest

pytest.importorskip("GenAIRR")

from GenAIRR import Experiment  # noqa: E402

from alignair_assay.core.cases import read_cases  # noqa: E402
from alignair_assay.core.headline import HEADLINE  # noqa: E402
from alignair_assay.core.layout import RunLayout  # noqa: E402
from alignair_assay.core.reference import dataconfig  # noqa: E402
from alignair_assay.core.seqops import revcomp  # noqa: E402
from alignair_assay.suites import get_suite  # noqa: E402
from alignair_assay.suites.base import RecipeContext  # noqa: E402
from alignair_assay.suites.donors import DonorSpec, donors_from_json  # noqa: E402
from alignair_assay.suites.engine import generate_suite  # noqa: E402
from alignair_assay.suites.recipes import (  # noqa: E402
    READ_DESIGNS,
    SHM_CLASSES,
    pooled_trim_marginals,
    repertoire,
)
from alignair_assay.testing import assert_oracle_perfect  # noqa: E402
from alignair_assay.truth.io import read_truth_jsonl  # noqa: E402
from alignair_assay.workflow.build import build_suite  # noqa: E402

REFERENCE = "HUMAN_IGH_OGRDB"
SEED = 31
SCALE = 0.02
SUITES = ("S1", "S10", "S11", "S12")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Each repertoire suite built once at ``SCALE`` into one run directory: ``(layout, {suite: build record})``."""
    layout = RunLayout(tmp_path_factory.mktemp("run"))
    builds = {name: build_suite(name, layout, reference_name=REFERENCE, split="dev", seed=SEED, scale=SCALE,
                                workers=1) for name in SUITES}
    return layout, builds


@pytest.fixture(scope="module")
def cases(built):
    layout, _ = built
    return {name: read_cases(layout.cases(name)) for name in SUITES}


@pytest.fixture(scope="module")
def ctx():
    return RecipeContext(config=dataconfig(REFERENCE), reference=None)


def _by_stratum(suite_cases):
    out = {}
    for case in suite_cases:
        out.setdefault(case["stratum"], []).append(case)
    return out


def _records(experiment, n=200, seed=3):
    return experiment.run_records(n=n, seed=seed).records


def test_designs_and_shm_classes():
    assert READ_DESIGNS == {"full": ((0, 0), (0, 0)), "fr1": ((0, 25), (0, 5)), "fr2": ((105, 135), (0, 5)),
                            "fr3": ((225, 255), (0, 5))}
    assert SHM_CLASSES == {"igm": 0.03, "igg": 0.075, "high": 0.20}


@pytest.mark.parametrize("design", list(READ_DESIGNS))
def test_repertoire_end_loss_per_design(ctx, design):
    (lo5, hi5), (lo3, hi3) = READ_DESIGNS[design]
    records = _records(repertoire(ctx, design=design), n=60)
    assert all(lo5 <= r["end_loss_5_length"] <= hi5 for r in records)
    assert all(lo3 <= r["end_loss_3_length"] <= hi3 for r in records)
    if design != "full":
        assert max(r["end_loss_5_length"] for r in records) > lo5


def test_repertoire_rate_kernel_and_noise(ctx):
    assert all(r["n_mutations"] == 0 for r in _records(repertoire(ctx, rate=0), n=40))
    igm = statistics.fmean(r["n_mutations"] for r in _records(repertoire(ctx, rate=0.03)))
    high = statistics.fmean(r["n_mutations"] for r in _records(repertoire(ctx, rate=0.20)))
    assert 0 < igm < high
    # The kernel reaches GenAIRR: the same seed gives different mutations under another model.
    s5f = [r["sequence"] for r in _records(repertoire(ctx), n=20)]
    assert s5f != [r["sequence"] for r in _records(repertoire(ctx, model="uniform"), n=20)]
    assert s5f != [r["sequence"] for r in _records(repertoire(ctx, s5f_model="hkl_s5f"), n=20)]
    # The noise step is applied verbatim.
    noisy = _records(repertoire(ctx, noise=("polymerase_indels", {"count": 2})), n=40)
    assert all(r["n_indels"] == 2 for r in noisy)
    with pytest.raises(ValueError, match="unknown read design 'fr4'.*full, fr1, fr2, fr3"):
        repertoire(ctx, design="fr4")


def test_pooled_trim_marginals_are_genairr_defaults(ctx):
    config = ctx.config
    marginals = pooled_trim_marginals(config)
    assert list(marginals) == ["v_3", "d_5", "d_3", "j_5"]
    for side, pairs in marginals.items():
        lengths = [length for length, _ in pairs]
        assert lengths == sorted(set(lengths)) and lengths[0] == 0, side
        assert abs(sum(weight for _, weight in pairs) - 1) < 1e-9 and all(w > 0 for _, w in pairs), side
    # GenAIRR caps D trims so the shortest D allele (11 nt) keeps its middle: at most (11 - 3) // 2 per side.
    assert max(length for length, _ in marginals["d_5"]) == 4
    default = _records(Experiment.on(config).recombine(), n=300, seed=5)
    assert _records(Experiment.on(config).recombine().trim(**marginals), n=300, seed=5) == default


def test_np_and_trim_shifts(ctx):
    config = ctx.config
    default = _records(Experiment.on(config).recombine(), n=400)
    np_shifted = _records(repertoire(ctx, rate=0, np_shift=4), n=400)
    # A record's NP length is GenAIRR's annotation, which can be a base shorter than the sampled length, so the
    # shift is checked on the mean.
    for field in ("np1_length", "np2_length"):
        shift = statistics.fmean(r[field] for r in np_shifted) - statistics.fmean(r[field] for r in default)
        assert 3 < shift < 5, field
    trim_shifted = _records(repertoire(ctx, rate=0, trim_shift=3), n=400)
    for field in ("v_trim_3", "j_trim_5"):
        assert min(r[field] for r in trim_shifted) >= 3
        shift = statistics.fmean(r[field] for r in trim_shifted) - statistics.fmean(r[field] for r in default)
        assert 2.5 < shift < 3.5, field
    # D trims are clamped to GenAIRR's cap (4), so the shortest D allele is never over-trimmed.
    for field in ("d_trim_5", "d_trim_3"):
        assert {r[field] for r in trim_shifted} == {3, 4}, field


def test_suite_definitions():
    s1, s10, s11, s12 = (get_suite(name) for name in SUITES)
    assert (s1.slug, s10.slug, s11.slug, s12.slug) == ("core", "heldout", "orientation", "noise")
    assert s1.primary == s10.primary == s12.primary == HEADLINE
    assert s11.primary == ("orientation.correct", *HEADLINE)
    assert s1.donors == DonorSpec(20, "S1") and s10.donors == DonorSpec(20, "S10")
    assert s11.donors is None and s12.donors is None
    assert {s.name: s.n for s in s1.strata} == {f"{d}_{shm}": n for d in ("full", "fr1", "fr2", "fr3")
                                                for shm, n in (("igm", 500), ("igg", 500), ("high", 250))}
    assert sum(s.n for s in s1.strata) == 5000
    assert {s.name: s.n for s in s10.strata} == {"uniform": 600, "hkl_s5f": 600, "np_shift": 600,
                                                 "trim_shift": 600}
    assert {s.name: s.n for s in s11.strata} == {"full_igg": 500, "fr3_igg": 500}
    assert {s.name: s.n for s in s12.strata} == {name: 300 for name in (
        "seq_err_0.005", "seq_err_0.01", "pcr_0.002", "n_bases_3", "n_bases_8", "pol_indel_1", "pol_indel_2")}
    for suite in (s1, s10, s12):
        assert all(s.rc_fraction == 0.5 and not s.both_orientations for s in suite.strata)
    assert all(s.both_orientations for s in s11.strata)
    assert all(s.cohort for s in s1.strata + s10.strata)
    assert not any(s.cohort for s in s11.strata + s12.strata)
    assert all(suite.purpose.endswith(".") and suite.purpose.count(". ") == 0 for suite in (s1, s10, s11, s12))


def test_s1_designs_end_loss(cases):
    strata = _by_stratum(cases["S1"])
    assert list(strata) == [f"{d}_{shm}" for d in ("full", "fr1", "fr2", "fr3") for shm in ("igm", "igg", "high")]
    assert {name: len(group) for name, group in strata.items()} == {
        f"{d}_{shm}": n for d in ("full", "fr1", "fr2", "fr3") for shm, n in (("igm", 10), ("igg", 10), ("high", 5))}
    for name, group in strata.items():
        design, shm = name.split("_")
        (lo5, hi5), (lo3, hi3) = READ_DESIGNS[design]
        assert all(c["labels"] == {"design": design, "shm": shm} for c in group)
        assert all(lo5 <= c["record"]["end_loss_5_length"] <= hi5 for c in group)
        assert all(lo3 <= c["record"]["end_loss_3_length"] <= hi3 for c in group)
    assert all(105 <= c["record"]["end_loss_5_length"] <= 135 for c in strata["fr2_igg"])
    assert all(c["record"]["end_loss_5_length"] == 0 == c["record"]["end_loss_3_length"]
               for name in ("full_igm", "full_igg", "full_high") for c in strata[name])
    assert {c["orientation"] for c in cases["S1"]} == {0, 1}


def test_s1_cases_have_subjects(built, cases):
    layout, builds = built
    donors = donors_from_json(json.loads(layout.donors("S1").read_text()))
    ids = {donor.subject_id for donor in donors}
    assert len(donors) == 20 and builds["S1"]["donors_summary"]["donors"] == 20
    assert all(c["groups"]["subject"] in ids and c["record"]["subject_id"] == c["groups"]["subject"]
               for c in cases["S1"])
    assert len({c["groups"]["subject"] for c in cases["S1"]}) == 10  # ten cases per stratum, one per donor


def test_s10_strata_and_labels(built, cases):
    layout, _ = built
    strata = _by_stratum(cases["S10"])
    assert {name: len(group) for name, group in strata.items()} == {
        "uniform": 12, "hkl_s5f": 12, "np_shift": 12, "trim_shift": 12}
    for name, group in strata.items():
        assert all(c["labels"] == {"heldout": name} and "subject" in c["groups"] for c in group)
        assert all(c["record"]["end_loss_5_length"] == 0 for c in group)
    recipes = {s.name: (s.recipe.func, s.recipe.keywords) for s in get_suite("S10").strata}
    assert recipes == {"uniform": (repertoire, {"model": "uniform", "rate": 0.075}),
                       "hkl_s5f": (repertoire, {"s5f_model": "hkl_s5f", "rate": 0.075}),
                       "np_shift": (repertoire, {"rate": 0.075, "np_shift": 4}),
                       "trim_shift": (repertoire, {"rate": 0.075, "trim_shift": 3})}
    assert all(c["record"]["v_trim_3"] >= 3 and c["record"]["j_trim_5"] >= 3 for c in strata["trim_shift"])
    assert layout.donors("S10").is_file()


def test_s11_pairs_are_reverse_complements(cases):
    strata = _by_stratum(cases["S11"])
    assert {name: len(group) for name, group in strata.items()} == {"full_igg": 20, "fr3_igg": 20}
    for name, group in strata.items():
        assert [c["orientation"] for c in group] == [0, 1] * 10
        for fwd, rc in zip(group[0::2], group[1::2]):
            assert fwd["record"] == rc["record"] and rc["sequence"] == revcomp(fwd["sequence"])
            assert fwd["groups"] == rc["groups"] == {"pair": fwd["case_id"].removesuffix(":fwd")}
            assert fwd["labels"] == rc["labels"] == {"design": name.split("_")[0]}
    assert all(225 <= c["record"]["end_loss_5_length"] <= 255 for c in strata["fr3_igg"])


def test_s12_noise_present(cases):
    strata = _by_stratum(cases["S12"])
    assert {name: len(group) for name, group in strata.items()} == {name: 6 for name in (
        "seq_err_0.005", "seq_err_0.01", "pcr_0.002", "n_bases_3", "n_bases_8", "pol_indel_1", "pol_indel_2")}
    for name, group in strata.items():
        assert all(c["labels"] == {"noise": name} for c in group)

    def errors(name):
        return statistics.fmean(c["record"]["n_quality_errors"] + c["record"]["n_pcr_errors"] for c in strata[name])

    assert errors("seq_err_0.005") > 0 and errors("seq_err_0.01") > 0 and errors("pcr_0.002") > 0
    assert all(c["record"]["n_pcr_errors"] == 0 for c in strata["seq_err_0.01"])
    assert all("N" in c["sequence"] for c in strata["n_bases_8"])
    assert all(c["record"]["n_indels"] == 1 for c in strata["pol_indel_1"])
    assert all(c["record"]["n_indels"] == 2 for c in strata["pol_indel_2"])


def test_oracle_perfect_on_repertoire_suites(built, reference):
    layout, builds = built
    for name in SUITES:
        truths = read_truth_jsonl(layout.truth(name))
        assert len(truths) == builds[name]["n_cases"] and builds[name]["truth"]["excluded"] == {}
        scores = assert_oracle_perfect(truths, reference)
        # Every primary metric is one the pipeline actually emits on this suite.
        emitted = {metric for case in scores.values() for metric in case}
        assert set(get_suite(name).primary) <= emitted, name


def test_spawned_workers_match_serial():
    # Recipes are module-level functions or partials, so spawned workers resolve them from the registry.
    options = dict(reference_name=REFERENCE, split="dev", seed=SEED, scale=SCALE)
    serial = generate_suite(get_suite("S11"), workers=1, **options)
    parallel = generate_suite(get_suite("S11"), workers=2, **options)
    assert serial.cases == parallel.cases and serial.stats == parallel.stats
