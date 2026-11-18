"""S13 ``flanks``: reads with the non-germline sequence real libraries carry around V(D)J (a 5' UTR and leader, a primer
masked to N, a 3' constant region), each read coordinate of the GenAIRR record moved with its read."""
import random

import pytest

from alignair_assay.core.headline import HEADLINE
from alignair_assay.suites.flanks import READ_COORDINATES, Flanks, flanked, masked

component = pytest.mark.component

RECORD = {"sequence": "ACGTACGTAC", "sequence_length": 10, "v_sequence_start": 0, "v_sequence_end": 6,
          "v_alignment_start": 0, "v_alignment_end": 6, "v_germline_start": 3, "v_germline_end": 9,
          "d_sequence_start": None, "d_sequence_end": None, "d_alignment_start": None, "d_alignment_end": None,
          "j_sequence_start": 7, "j_sequence_end": 10, "j_alignment_start": 7, "j_alignment_end": 10,
          "j_germline_start": 0, "j_germline_end": 3, "junction_start": 4, "junction_end": 10}


@component
def test_flanked_moves_every_read_coordinate_and_no_germline_one():
    out = flanked(RECORD, "GGG", "TT")
    assert out["sequence"] == "GGGACGTACGTACTT" and out["sequence_length"] == 15
    assert out["flank_5_length"] == 3 and out["flank_3_length"] == 2
    for key in READ_COORDINATES:
        assert out[key] == (None if RECORD[key] is None else RECORD[key] + 3), key
    assert (out["v_germline_start"], out["j_germline_end"]) == (3, 3)
    assert RECORD["sequence"] == "ACGTACGTAC"                       # the input record is left as it was


@component
def test_masked_turns_the_first_bases_to_n_and_keeps_coordinates():
    out = masked(RECORD, 4)
    assert out["sequence"] == "NNNNACGTAC" and out["mask_5_length"] == 4
    assert all(out[key] == RECORD[key] for key in READ_COORDINATES)


@component
def test_flanks_draw_lengths_in_their_ranges():
    rng = random.Random(7)
    for _ in range(50):
        out = Flanks(prefix=(2, 4), suffix=(1, 3), mask=(0, 2))(RECORD, rng)
        prefix, suffix = out["flank_5_length"], out["flank_3_length"]
        assert 2 <= prefix <= 4 and 1 <= suffix <= 3 and prefix <= out["mask_5_length"] <= prefix + 2
        assert out["sequence"][:out["mask_5_length"]] == "N" * out["mask_5_length"]
        assert set(out["sequence"][out["mask_5_length"]:prefix]) <= set("ACGT")
        assert out["v_sequence_start"] == prefix


@component
def test_flanks_without_a_side_leave_it():
    out = Flanks(suffix=(2, 2))(RECORD, random.Random(1))
    assert out["flank_5_length"] == 0 and out["mask_5_length"] == 0 and out["sequence"].startswith("ACGTACGTAC")


@component
def test_s13_definition():
    from alignair_assay.suites import get_suite

    s13 = get_suite("S13")
    assert s13.slug == "flanks" and s13.donors is None
    assert s13.primary == (*HEADLINE, "v.seq_start_exact", "j.seq_end_exact")
    assert {s.name: s.n for s in s13.strata} == {"leader_5p": 500, "primer_n_5p": 500, "c_region_3p": 500,
                                                "leader_c_region": 500}
    assert all(s.flank is not None and s.rc_fraction == 0.5 for s in s13.strata)
    assert s13.purpose.endswith(".") and s13.purpose.count(". ") == 0


@pytest.mark.integration
@pytest.mark.genairr_build
def test_s13_builds_flanked_reads_whose_truth_the_oracle_scores_perfectly(tmp_path):
    pytest.importorskip("GenAIRR")
    from alignair_assay.core.cases import read_cases
    from alignair_assay.core.layout import RunLayout
    from alignair_assay.core.reference import Reference
    from alignair_assay.testing import assert_oracle_perfect
    from alignair_assay.truth.io import read_truth_jsonl
    from alignair_assay.workflow.build import build_suite

    layout = RunLayout(tmp_path)
    build = build_suite("S13", layout, reference_name="HUMAN_IGH_OGRDB", split="dev", seed=31, scale=0.02,
                        workers=1)
    cases = read_cases(layout.cases("S13"))
    by_stratum = {}
    for case in cases:
        by_stratum.setdefault(case["stratum"], []).append(case["record"])
    assert {name: len(records) for name, records in by_stratum.items()} == {
        "leader_5p": 10, "primer_n_5p": 10, "c_region_3p": 10, "leader_c_region": 10}
    assert all(60 <= r["flank_5_length"] <= 130 and r["flank_3_length"] == 0 for r in by_stratum["leader_5p"])
    assert all(r["mask_5_length"] > r["flank_5_length"] > 0 for r in by_stratum["primer_n_5p"])
    assert all(r["flank_5_length"] == 0 and 20 <= r["flank_3_length"] <= 80 for r in by_stratum["c_region_3p"])
    assert all(r["flank_5_length"] and r["flank_3_length"] for r in by_stratum["leader_c_region"])
    truths = read_truth_jsonl(layout.truth("S13"))
    assert len(truths) == build["n_cases"] and build["truth"]["excluded"] == {}
    scores = assert_oracle_perfect(truths, Reference.from_dataconfig("HUMAN_IGH_OGRDB"))
    emitted = {metric for case in scores.values() for metric in case}
    from alignair_assay.suites import get_suite
    assert set(get_suite("S13").primary) <= emitted
