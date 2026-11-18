"""S5 (V indels) and S9 (clonal lineages), and within-clone agreement scored through the pipeline."""
import collections
from dataclasses import replace

import pytest

pytest.importorskip("GenAIRR")

from alignair_assay.core.cases import read_cases  # noqa: E402
from alignair_assay.core.layout import RunLayout  # noqa: E402
from alignair_assay.core.reference import gene_name  # noqa: E402
from alignair_assay.scoring.pipeline import score_predictions  # noqa: E402
from alignair_assay.suites import get_suite  # noqa: E402
from alignair_assay.testing import assert_oracle_perfect  # noqa: E402
from alignair_assay.truth.io import read_truth_jsonl  # noqa: E402
from alignair_assay.truth.oracle import oracle_prediction  # noqa: E402
from alignair_assay.workflow.build import build_suite  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.genairr_build]  # these tests build GenAIRR suites
REFERENCE = "HUMAN_IGH_OGRDB"
SEED = 43
# S5 at 6 cases per stratum; S9 at 6 clones.
SCALES = {"S5": 0.02, "S9": 0.1}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """S5 and S9 built once into one run directory: ``(layout, {suite: build record})``."""
    layout = RunLayout(tmp_path_factory.mktemp("run"))
    builds = {name: build_suite(name, layout, reference_name=REFERENCE, split="dev", seed=SEED, scale=scale,
                                workers=1) for name, scale in SCALES.items()}
    return layout, builds


@pytest.fixture(scope="module")
def cases(built):
    layout, _ = built
    return {name: read_cases(layout.cases(name)) for name in SCALES}


@pytest.fixture(scope="module")
def truths(built):
    layout, _ = built
    return {name: read_truth_jsonl(layout.truth(name)) for name in SCALES}


def test_suite_definitions():
    s5, s9 = get_suite("S5"), get_suite("S9")
    assert (s5.slug, s9.slug) == ("indels", "clonal")
    assert s5.primary == ("v.indel_found", "v.indel_false", "v.seq_end_exact", "junction.nt_exact",
                          "v.top1_in_set")
    assert s9.primary == ("clone_key.exact", "clone.agree_modal", "junction.nt_exact", "v.top1_in_set")
    names = [*(f"v_{k}" for k in range(1, 7)), "j_1"]
    assert [s.name for s in s5.strata] == names
    assert {s.name: (s.n, dict(s.labels)) for s in s5.strata} == {name: (300, {"indels": name}) for name in names}
    assert all(s.rc_fraction == 0 and s.accept is not None and not s.clonal and not s.cohort for s in s5.strata)
    (lineages,) = s9.strata
    assert (lineages.name, lineages.n, lineages.clonal, lineages.rc_fraction) == ("lineages", 60, True, 0.5)
    for suite in (s5, s9):
        assert suite.donors is None and suite.arms == ("shared",)
        assert suite.purpose.endswith(".") and suite.purpose.count(". ") == 0


def test_s5_indel_counts(built, cases):
    _, builds = built
    strata = collections.defaultdict(list)
    for case in cases["S5"]:
        strata[case["stratum"]].append(case)
    assert {name: len(group) for name, group in strata.items()} == {
        **{f"v_{k}": 6 for k in range(1, 7)}, "j_1": 6}
    for name, group in strata.items():
        k = int(name.split("_")[1])
        for case in group:
            record = case["record"]
            assert case["labels"] == {"indels": name} and case["orientation"] == 0 and case["groups"] == {}
            assert record["n_indels"] == k
            assert (record["n_v_indels"], record["n_j_indels"]) == ((k, 0) if name.startswith("v_") else (0, 1))
            assert record["end_loss_5_length"] == 0 == record["end_loss_3_length"]
    # Rejection sampling: every stratum asked GenAIRR for more records than it kept.
    stats = builds["S5"]["strata"]
    assert all(s["requested_unit"] == "records" and s["generated"] > s["requested"] == 6 for s in stats.values())


def test_s9_clone_groups(built, cases):
    _, builds = built
    lineages = cases["S9"]
    assert {c["stratum"] for c in lineages} == {"lineages"}
    assert all(c["groups"] == {"clone": str(c["record"]["clone_id"])} for c in lineages)
    # Every clonal record names its generative alleles (the engine simulates with provenance).
    assert all(c["record"]["truth_v_call"] and c["record"]["truth_j_call"] for c in lineages)
    sizes = collections.Counter(c["groups"]["clone"] for c in lineages)
    assert len(sizes) == 6
    assert sum(n for n in sizes.values() if n >= 2) >= 0.8 * len(lineages)
    assert {c["orientation"] for c in lineages} == {0, 1}
    stats = builds["S9"]["strata"]["lineages"]
    assert stats["requested_unit"] == "clones" and stats["requested"] == 6 == stats["clones"]
    assert stats["cases"] == stats["accepted"] == len(lineages)
    # Members of one clone share their recombination: generative V and J and the junction length.
    by_clone = collections.defaultdict(set)
    for case in lineages:
        record = case["record"]
        by_clone[case["groups"]["clone"]].add((record["truth_v_call"], record["truth_j_call"],
                                               len(record["junction"])))
    assert all(len(keys) == 1 for keys in by_clone.values())


def test_oracle_perfect_on_s5_s9(built, truths, reference):
    _, builds = built
    scores = {}
    for name in SCALES:
        assert len(truths[name]) == builds[name]["n_cases"] and builds[name]["truth"]["excluded"] == {}
        scores[name] = assert_oracle_perfect(truths[name], reference)
        emitted = {metric for case in scores[name].values() for metric in case}
        assert set(get_suite(name).primary) <= emitted, (name, set(get_suite(name).primary) - emitted)
    # The v_k reads show their V indels (``v.indel_found``) and the j_1 reads none in V. An indel at a segment
    # edge cannot be told from trimming, so a few reads show fewer indels than GenAIRR placed (2 of 60 j_1 reads
    # at scale 0.2).
    by_stratum = collections.defaultdict(list)
    for case_id, case in scores["S5"].items():
        by_stratum[case_id.split(":")[1]].append(case)
    v_cases = [case for name, group in by_stratum.items() if name.startswith("v_") for case in group]
    assert sum("v.indel_found" in case for case in v_cases) >= 0.8 * len(v_cases)
    assert all("v.indel_false" in case for case in by_stratum["j_1"])
    assert sum("j.indel_found" in case for case in by_stratum["j_1"]) >= 0.8 * len(by_stratum["j_1"])
    agreeing = [case for case in scores["S9"].values() if "clone.agree_modal" in case]
    assert len(agreeing) >= 0.8 * len(truths["S9"])


def test_clone_agreement_through_the_pipeline(truths, reference):
    s9 = truths["S9"]
    preds = {t.case_id: oracle_prediction(t) for t in s9}
    clones = collections.Counter(t.groups["clone"] for t in s9)
    clone = max(clones, key=clones.get)
    member = next(t for t in s9 if t.groups["clone"] == clone)
    # Call that member's V as an allele of another gene: only its key leaves the clone's mode.
    v = preds[member.case_id].segments["v"]
    other = next(a for a in sorted(reference.alleles["v"]) if gene_name(a) != gene_name(v.call))
    preds[member.case_id] = replace(preds[member.case_id],
                                    segments={**preds[member.case_id].segments,
                                              "v": replace(v, call=other, call_set=(other,))})
    scores = score_predictions(s9, preds, reference)
    assert scores[member.case_id]["clone.agree_modal"] == 0.0
    assert all(scores[t.case_id]["clone.agree_modal"] == 1.0 for t in s9
               if t.groups["clone"] == clone and t.case_id != member.case_id)
    # A missing row fails its own agreement and leaves the rest of its clone in agreement.
    del preds[member.case_id]
    scores = score_predictions(s9, preds, reference)
    assert scores[member.case_id]["clone.agree_modal"] == 0.0
