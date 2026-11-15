"""The novel-allele screen (``alignair_lab.novel``): chip-v1 trained from a novel record cache, scored on the
held-out S7 donors' novel alleles in two banks (the model's reference plus every donor's novel alleles; each donor's
genotype) against the baseline seeds, and gated by the owner's pass rule (2026-11-07)."""
import json
from types import SimpleNamespace

import GenAIRR.data as gd
import pytest

from alignair.reference.reference_set import ReferenceSet



def _scores(novel, known, genotype_novel=None, genotype_known=None, reads=1000):
    def block(n, k):
        return {"novel_reads": reads, "novel_exact": round(n * reads), "known_reads": reads,
                "known_exact": round(k * reads), "novel_called_base": 0, "by_snps": {}}
    return {"extended": block(novel, known),
            "genotype": block(genotype_novel if genotype_novel is not None else novel,
                              genotype_known if genotype_known is not None else known)}


def _decisions(decision="within noise", regressions=()):
    from alignair_lab.novel import GATED
    return {metric: {"decision": decision, "regressions": list(regressions)} for metric in GATED}


BASELINES = [_scores(0.80, 0.90), _scores(0.81, 0.90), _scores(0.82, 0.90)]


@pytest.mark.component
def test_two_donors_novel_alleles_of_one_name_get_a_row_each():
    from alignair_lab.novel import heldout_reference
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    v = reference.gene("V")
    base = "IGHVF1-G1*01"
    s = v.sequences[v.index[base]]
    one, two = s[:50] + ("A" if s[50] != "A" else "C") + s[51:], s[:60] + ("A" if s[60] != "A" else "C") + s[61:]
    g8 = "IGHVF3-G8*01" if "IGHVF3-G8*01" in v.index else next(n for n in v.names if n.startswith("IGHVF3-G8"))
    s8 = v.sequences[v.index[g8]]
    three = s8[:40] + ("A" if s8[40] != "A" else "C") + s8[41:]
    allele = lambda name, b, seq: SimpleNamespace(name=name, base=b, sequence=seq)
    donors = [SimpleNamespace(subject_id="d0", novel={"IGHVF1-G1*900": allele("IGHVF1-G1*900", base, one),
                                                      "IGHVF3-G8*901": allele("IGHVF3-G8*901", g8, three)}),
              SimpleNamespace(subject_id="d1", novel={"IGHVF1-G1*900": allele("IGHVF1-G1*900", base, two)})]
    extended, rows = heldout_reference(reference, donors)
    assert len(extended.gene("V")) == 198 + 3
    assert {"IGHVF1-G1*900_d0", "IGHVF1-G1*900_d1", "IGHVF3-G8*901"} <= set(extended.gene("V").names)
    assert rows[("d1", "IGHVF1-G1*900")] == "IGHVF1-G1*900_d1" and rows[("d0", "IGHVF3-G8*901")] == "IGHVF3-G8*901"


@pytest.mark.component
@pytest.mark.parametrize("candidate, decisions, baselines, verdict, reason", [
    (_scores(0.85, 0.90), _decisions(), BASELINES, "pass", None),
    (_scores(0.829, 0.90), _decisions(), BASELINES, "fail", "novel gain"),
    (_scores(0.815, 0.90), _decisions(), [_scores(0.70, 0.9), _scores(0.70, 0.9), _scores(0.83, 0.9)], "fail",
     "within the baseline seeds"),
    (_scores(0.85, 0.894), _decisions(), BASELINES, "fail", "known"),
    (_scores(0.85, 0.90), _decisions("beyond noise (worse)"), BASELINES, "fail", "dev"),
    (_scores(0.85, 0.90), _decisions(regressions=["S6"]), BASELINES, "fail", "S6"),
    (_scores(0.85, 0.90), _decisions(), BASELINES[:2], "uncalibrated", None),
])
def test_the_verdict(candidate, decisions, baselines, verdict, reason):
    from alignair_lab.novel import novel_verdict
    found = novel_verdict(candidate, baselines, decisions)
    assert found["verdict"] == verdict
    if reason:
        assert any(reason in line for line in found["reasons"]), found["reasons"]


@pytest.mark.component
def test_training_overlap_names_a_held_out_allele_a_training_epoch_drew(tmp_path):
    from alignair.train.records.novel import NovelAlleles, epoch_alleles
    from alignair_assay.core.genotype import NovelAllele as HeldOut
    from alignair_assay.suites.donors import Donor, donors_to_json
    from alignair_lab.novel import training_overlap
    spec = NovelAlleles(epoch_records=8)
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    drawn = epoch_alleles(reference, spec, 2 ** 62 // 8)[0]
    base = reference.gene("V").sequences[reference.gene("V").index[drawn.base]]
    snps = tuple(i for i, (a, b) in enumerate(zip(base, drawn.sequence)) if a != b)
    gene = drawn.base.split("*")[0]
    name = f"{gene}*999"
    donor = Donor("d0", {gene: (name,)}, frozenset(), {name: HeldOut(name, drawn.base, drawn.sequence, snps)})
    (tmp_path / "suite").mkdir()
    (tmp_path / "suite" / "donors.json").write_text(json.dumps(donors_to_json([donor])))
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "manifest.json").write_text(json.dumps({"components": [
        {"seed": 2 ** 62, "records": 10, "reference": "HUMAN_IGH_OGRDB",
         "parameters": {"novel_alleles": spec.to_dict()}}]}))
    assert training_overlap(tmp_path / "cache", tmp_path / "suite") == frozenset({drawn.sequence})


@pytest.mark.component
def test_the_novel_recipe_is_the_recipe_with_novel_alleles():
    import dataclasses

    from alignair_lab.recipe import NOVEL_ALLELES, NOVEL_POLICY, POLICY, training_config
    assert training_config(steps=10).novel_alleles is None
    assert training_config(steps=10, novel_alleles=NOVEL_ALLELES).novel_alleles == NOVEL_ALLELES
    assert NOVEL_POLICY == dataclasses.replace(POLICY, novel_alleles=NOVEL_ALLELES)


@pytest.mark.integration
def test_heldout_scores_count_each_donors_reads_against_its_own_alleles(tmp_path):
    from alignair_assay.core.cases import read_cases
    from alignair_assay.core.layout import RunLayout
    from alignair_assay.workflow.build import build_suite
    from alignair_lab.novel import heldout_scores
    from tests.alignair.predict.golden_rows import TEST_CHIP
    layout = RunLayout(tmp_path)
    build_suite("S7", layout, reference_name="HUMAN_IGH_OGRDB", split="dev", seed=5, scale=0.02, workers=1)
    suite = layout.suite_dir("S7")
    cases = read_cases(suite / "cases.jsonl.gz")
    scores = heldout_scores(str(TEST_CHIP), suite, device="cpu", batch_size=16)
    novel = sum(case["labels"]["novel_snps"] != "none" for case in cases)
    for condition in ("extended", "genotype"):
        found = scores[condition]
        assert found["novel_reads"] + found["known_reads"] == len(cases) and found["novel_reads"] == novel
        assert 0 <= found["novel_exact"] <= found["novel_reads"] and 0 <= found["known_exact"] <= found["known_reads"]
    assert scores["rows"] > 0


@pytest.mark.integration
def test_a_novel_screen_trains_from_the_novel_cache_and_writes_its_verdict(rig, baseline):
    from alignair_lab.cli import main
    from tests.alignair_lab.rig import common, options
    assert main(["novel-screen", "--name", "novel", "--steps", "2", *options("--model-option"),
                 *options("--baseline-model-option"), *common(rig),
                 "--novel-cache", str(rig.novel_cache), "--heldout", str(rig.heldout)]) == 0
    screen = rig.screens / "novel"
    record = json.loads((screen / "screen.json").read_text())
    assert record["training_cache"]["path"] == str(rig.novel_cache) and record["cache"]["path"] == str(rig.cache)
    assert record["novel_alleles"]["generator"] == 1
    assert json.loads((screen / "novel.json").read_text())["verdict"] == "uncalibrated"
    assert any(line.startswith("Verdict: uncalibrated") for line in (screen / "headline.txt").read_text().splitlines())


@pytest.mark.integration
def test_a_novel_screen_without_a_novel_cache_is_refused_before_anything_runs(rig, tmp_path):
    from alignair_lab.cli import main
    from tests.alignair_lab.rig import common, options
    assert main(["novel-screen", "--name", "nocache", "--steps", "2", *options("--model-option"),
                 *common(rig, tmp_path / "screens"), "--heldout", str(rig.heldout)]) == 2
    assert not (tmp_path / "screens" / "nocache").exists()


@pytest.mark.integration
def test_the_cache_command_writes_the_novel_recipes_cache(tmp_path):
    from alignair_lab.cli import main
    from alignair_lab.recipe import NOVEL_ALLELES
    assert main(["cache", "--novel", "--steps", "1", "--out", str(tmp_path / "novel"), "--workers", "2"]) == 0
    assert json.loads((tmp_path / "novel" / "manifest.json").read_text())["policy"]["novel_alleles"] == \
        NOVEL_ALLELES.to_dict()


@pytest.mark.component
def test_the_verdict_states_that_its_band_omits_the_data_draw():
    """The candidate trains on other records than the baseline seeds (the novel cache draws its own reads): the band
    and the seeds measure init-seed noise on fixed batches, which the verdict and the headline say."""
    from alignair_lab.novel import _block, novel_verdict
    found = novel_verdict(_scores(0.85, 0.90), BASELINES, _decisions())
    assert "data draw" in found["caveat"]
    novel = {"heldout": "S7", "overlap": [], "scores": {"candidate": _scores(0.85, 0.90), "baselines": {}}, **found}
    lines = _block(novel)
    assert any("data draw" in line for line in lines) and lines[-1].startswith("Verdict: pass")
