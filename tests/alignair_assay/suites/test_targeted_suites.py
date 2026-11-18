"""The targeted suites S2 (siblings), S3 (junction), S4 (truncation) and S6 (D observability)."""
import collections

import pytest

pytest.importorskip("GenAIRR")

from GenAIRR import Experiment  # noqa: E402

from alignair_assay.core.cases import read_cases  # noqa: E402
from alignair_assay.core.layout import RunLayout  # noqa: E402
from alignair_assay.core.reference import Reference, dataconfig  # noqa: E402
from alignair_assay.scoring.slices import slice_labels, v_len_bin  # noqa: E402
from alignair_assay.suites import get_suite  # noqa: E402
from alignair_assay.suites.base import RecipeContext  # noqa: E402
from alignair_assay.suites.catalog.s02_siblings import sibling_alleles, sibling_distance_bin  # noqa: E402
from alignair_assay.suites.recipes import d_alleles_for_trim, pooled_trim_marginals  # noqa: E402
from alignair_assay.testing import assert_oracle_perfect  # noqa: E402
from alignair_assay.truth.io import read_truth_jsonl  # noqa: E402
from alignair_assay.workflow.build import build_suite  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.genairr_build]  # these tests build GenAIRR suites
REFERENCE = "HUMAN_IGH_OGRDB"
SEED = 41
SCALE = 0.02
SUITES = ("S2", "S3", "S4", "S6")
NP_STRATA = [f"{sweep}_{value}" for sweep in ("np1", "np2") for value in (0, 3, 6, 12, 20)]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Each targeted suite built once at ``SCALE`` into one run directory: ``(layout, {suite: build record})``."""
    layout = RunLayout(tmp_path_factory.mktemp("run"))
    builds = {name: build_suite(name, layout, reference_name=REFERENCE, split="dev", seed=SEED, scale=SCALE,
                                workers=1) for name in SUITES}
    return layout, builds


@pytest.fixture(scope="module")
def cases(built):
    layout, _ = built
    return {name: _by_stratum(read_cases(layout.cases(name))) for name in SUITES}


@pytest.fixture(scope="module")
def truths(built):
    layout, _ = built
    return {name: read_truth_jsonl(layout.truth(name)) for name in SUITES}


@pytest.fixture(scope="module")
def ctx(reference):
    return RecipeContext(config=dataconfig(REFERENCE), reference=reference)


def _by_stratum(suite_cases):
    out = {}
    for case in suite_cases:
        out.setdefault(case["stratum"], []).append(case)
    return out


def _d_lengths(config):
    return {allele.name: len(allele.ungapped_seq) for alleles in config.d_alleles.values() for allele in alleles}


def test_targeted_suite_definitions():
    s2, s3, s4, s6 = (get_suite(name) for name in SUITES)
    assert (s2.slug, s3.slug, s4.slug, s6.slug) == ("siblings", "junction", "truncation", "d_observability")
    assert s2.primary == ("v.top1_in_set", "v.set_precision", "v.gene_top1", "j.top1_in_set")
    assert s3.primary == ("junction.nt_exact", "junction.length_exact", "clone_key.exact", "junction.start_abs_err",
                          "junction.end_abs_err", "v.seq_end_exact", "j.seq_start_exact")
    assert s4.primary == ("v.top1_in_set", "v_short.abstain_ok", "v.germ_start_err", "j.top1_in_set",
                          "j.germ_end_err", "v.spurious_call", "junction.nt_exact")
    assert s6.primary == ("d.top1_in_set", "d_short.abstain_ok", "d.seq_start_exact", "d.seq_end_exact")
    assert {s.name: s.n for s in s2.strata} == {f"shm_{rate}": 600 for rate in ("0", "0.03", "0.08", "0.15", "0.25")}
    sweeps = {"np1": (0, 3, 6, 12, 20), "np2": (0, 3, 6, 12, 20), "v3": (0, 2, 5, 10), "j5": (0, 4, 8, 14),
              "d_both": (0, 4, 8)}
    assert {s.name: s.n for s in s3.strata} == {f"{sweep}_{v}": 150 for sweep, values in sweeps.items()
                                                for v in values}
    assert {s.name: dict(s.labels) for s in s3.strata} == {f"{sweep}_{v}": {"sweep": sweep, "value": str(v)}
                                                           for sweep, values in sweeps.items() for v in values}
    expected_s4 = {**{f"5p_{k}": {"sweep": "5p", "loss": str(k)} for k in range(0, 301, 25)},
                   **{f"3p_{k}": {"sweep": "3p", "loss": str(k)} for k in (0, 10, 20, 30, 40)}}
    assert {s.name: dict(s.labels) for s in s4.strata} == expected_s4 and all(s.n == 150 for s in s4.strata)
    assert {s.name: (s.n, dict(s.labels)) for s in s6.strata} == {
        **{f"trim_{k}": (400, {"d_trim": str(k), "inverted": "no"}) for k in (0, 3, 6, 9, 12)},
        "inverted": (500, {"d_trim": "default", "inverted": "yes"})}
    for suite in (s2, s3, s4, s6):
        assert suite.donors is None and suite.arms == ("shared",)
        assert all(s.rc_fraction == 0 and not s.both_orientations and not s.cohort and not s.clonal
                   for s in suite.strata)
        assert suite.purpose.endswith(".") and suite.purpose.count(". ") == 0


def test_sibling_alleles_hand_reference():
    reference = Reference(alleles={
        "v": {"IGHVX*01": "AAAAAAAAAA", "IGHVX*02": "AAAAAAAAAC",  # 1 apart
              "IGHVY*01": "CCCAAAAAAA",  # 3 from X*01, 4 from X*02
              "IGHVZ*01": "GGGGGGGGGG",  # 10 from every other 10-mer
              "IGHVW*01": "AAAAAAAAA"},  # alone at 9 nt: no allele of equal length
        "d": {"IGHDX*01": "AAAAAAAAAA", "IGHDY*01": "GGGGGGGGGG"},
    }, anchors={})
    assert sibling_alleles(reference) == {"IGHVX*01": 1, "IGHVX*02": 1, "IGHVY*01": 3}
    assert sibling_alleles(reference, max_distance=2) == {"IGHVX*01": 1, "IGHVX*02": 1}
    assert sibling_alleles(reference, letter="d") == {}  # segments are never compared with each other
    assert sibling_alleles(reference, letter="d", max_distance=10) == {"IGHDX*01": 10, "IGHDY*01": 10}
    assert sibling_alleles(reference, letter="j") == {}
    sibling_alleles(reference).clear()  # callers get a copy
    assert len(sibling_alleles(reference)) == 3
    assert [sibling_distance_bin(d) for d in (0, 1, 2, 3, 4, 5)] == ["0", "1", "2", "3", "4-5", "4-5"]
    for distance in (-1, 6):
        with pytest.raises(ValueError, match="sibling distance"):
            sibling_distance_bin(distance)


def test_s2_calls_restricted(reference, cases, truths):
    siblings = sibling_alleles(reference)
    # Most human IGH V alleles have a sibling within 5 bases, but not all of them.
    assert 100 < len(siblings) < len(reference.alleles["v"])
    strata = cases["S2"]
    assert {name: len(group) for name, group in strata.items()} == {
        f"shm_{rate}": 12 for rate in ("0", "0.03", "0.08", "0.15", "0.25")}
    for group in strata.values():
        for case in group:
            generative = case["record"]["truth_v_call"]
            assert generative in siblings and case["orientation"] == 0
            assert case["labels"] == {"sibling_distance": sibling_distance_bin(siblings[generative])}
    assert all(c["record"]["n_mutations"] == 0 for c in strata["shm_0"])
    assert sum(c["record"]["n_mutations"] for c in strata["shm_0.25"]) > sum(
        c["record"]["n_mutations"] for c in strata["shm_0.03"])
    assert len({c["labels"]["sibling_distance"] for group in strata.values() for c in group}) >= 2
    assert all(t.labels["sibling_distance"] for t in truths["S2"])


def test_s3_sweep_values(cases):
    strata = cases["S3"]
    assert all(len(group) == 3 for group in strata.values()) and len(strata) == 21
    for name, group in strata.items():
        sweep, value = name.rsplit("_", 1)
        value = int(value)
        for case in group:
            record = case["record"]
            p_nt = any(record[f"p_{side}_length"] for side in ("v_3", "d_5", "d_3", "j_5"))
            assert case["labels"] == {"sweep": sweep, "value": str(value), "p_nt": "yes" if p_nt else "no"}
            assert case["orientation"] == 0 and record["end_loss_5_length"] == 0 == record["end_loss_3_length"]
            if sweep == "v3":
                assert record["v_trim_3"] == value
            elif sweep == "j5":
                assert record["j_trim_5"] == value
            elif sweep == "d_both":
                assert record["d_trim_5"] == value == record["d_trim_3"]
            else:
                # GenAIRR's record can annotate an NP a base shorter than sampled (see test_s3_np_lengths, which
                # holds the requested length on at least 95% of records).
                assert record[f"{sweep}_length"] in (value, max(value - 1, 0))
    assert all(c["record"]["v_trim_3"] == 5 for c in strata["v3_5"])


@pytest.mark.parametrize("name", NP_STRATA)
def test_s3_np_lengths(ctx, name):
    """A swept NP is sampled at exactly the requested length. GenAIRR's record annotates one base fewer when its
    junction annotation claims an NP base for the neighbouring segment; that happens on under 5% of records."""
    sweep, value = name.rsplit("_", 1)
    value = int(value)
    stratum = next(s for s in get_suite("S3").strata if s.name == name)
    records = stratum.recipe(ctx).run_records(n=400, seed=5, expose_provenance=True).records
    lengths = collections.Counter(r[f"{sweep}_length"] for r in records)
    assert set(lengths) <= {value, value - 1}
    assert lengths[value] >= 0.95 * len(records)


def test_s3_unswept_sides_keep_default_trims(ctx):
    stratum = next(s for s in get_suite("S3").strata if s.name == "v3_10")
    records = stratum.recipe(ctx).run_records(n=200, seed=5, expose_provenance=True).records
    assert {r["v_trim_3"] for r in records} == {10}
    for field in ("j_trim_5", "d_trim_5", "d_trim_3"):
        assert len({r[field] for r in records}) > 1, field
    # trim() keeps GenAIRR's default for every side it is not given, which is exactly the pooled marginals.
    explicit = {**pooled_trim_marginals(ctx.config), "v_3": [(10, 1.0)]}
    spelled_out = Experiment.on(ctx.config).recombine().trim(**explicit).mutate(rate=0.02)
    assert spelled_out.run_records(n=200, seed=5, expose_provenance=True).records == records


def test_s4_exact_end_loss(cases):
    strata = cases["S4"]
    assert len(strata) == 18 and all(len(group) == 3 for group in strata.values())
    for name, group in strata.items():
        sweep, loss = name.split("_")
        five, three = (int(loss), 0) if sweep == "5p" else (0, int(loss))
        for case in group:
            assert case["labels"] == {"sweep": sweep, "loss": loss} and case["orientation"] == 0
            assert (case["record"]["end_loss_5_length"], case["record"]["end_loss_3_length"]) == (five, three)
    assert all(c["record"]["end_loss_5_length"] == 125 for c in strata["5p_125"])


def test_d_alleles_for_trim(ctx):
    lengths = _d_lengths(ctx.config)
    assert min(lengths.values()) == 11
    # GenAIRR's own cap is 4 per side, so up to 4 every D allele keeps at least 2 bases.
    assert d_alleles_for_trim(ctx.config, 0) is None and d_alleles_for_trim(ctx.config, 4) is None
    for per_side in (5, 6, 9, 12):
        alleles = d_alleles_for_trim(ctx.config, per_side)
        assert alleles == sorted(name for name, n in lengths.items() if n >= 2 * per_side + 2), per_side
    assert len(d_alleles_for_trim(ctx.config, 12)) < len(d_alleles_for_trim(ctx.config, 6)) < len(lengths)


def test_s6_trims_and_inversion(ctx, cases, truths):
    lengths = _d_lengths(ctx.config)
    strata = cases["S6"]
    assert {name: len(group) for name, group in strata.items()} == {
        **{f"trim_{k}": 8 for k in (0, 3, 6, 9, 12)}, "inverted": 10}
    for k in (0, 3, 6, 9, 12):
        for case in strata[f"trim_{k}"]:
            record = case["record"]
            # The label is the real trim: every record is trimmed exactly k at both D ends.
            assert case["labels"] == {"d_trim": str(k), "inverted": "no"}
            assert record["d_trim_5"] == k == record["d_trim_3"] and not record["d_inverted"]
            assert lengths[record["truth_d_call"]] >= 2 * k + 2
    assert all(c["record"]["d_inverted"] and c["labels"] == {"d_trim": "default", "inverted": "yes"}
               for c in strata["inverted"])
    inverted = [t for t in truths["S6"] if t.stratum == "inverted"]
    assert len(inverted) == 10 and all(t.segments["d"].inverted for t in inverted)
    for truth in truths["S6"]:
        assert not truth.warnings and not truth.unscorable, (truth.case_id, truth.warnings, truth.unscorable)
    # Deeper trims leave less D in the read.
    observed = {k: [t.segments["d"].observed_length for t in truths["S6"]
                    if t.stratum == f"trim_{k}" and "d" in t.segments] for k in (0, 12)}
    assert sum(observed[12]) / len(observed[12]) < sum(observed[0]) / len(observed[0])


def test_v_len_bin_slice(truths):
    seen = collections.Counter()
    for truth in truths["S4"]:
        v = truth.segments.get("v")
        label = slice_labels(truth)["v_len_bin"]
        assert label == v_len_bin(v.observed_length if v is not None else None)
        seen[label] += 1
    # The 5' sweep runs from full-length V to reads that start past it.
    assert seen[">=250"] and seen["absent"] and len(seen) >= 4


def test_oracle_perfect_on_targeted_suites(built, truths, reference):
    _, builds = built
    for name in SUITES:
        assert len(truths[name]) == builds[name]["n_cases"] and builds[name]["truth"]["excluded"] == {}
        scores = assert_oracle_perfect(truths[name], reference)
        emitted = {metric for case in scores.values() for metric in case}
        primary = set(get_suite(name).primary)
        assert primary <= emitted, (name, primary - emitted)
