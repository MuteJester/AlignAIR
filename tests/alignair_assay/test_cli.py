import gzip
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from alignair_assay.cigar import format_cigar, parse_cigar
from alignair_assay.cli import d_len_bin, main, slice_labels
from alignair_assay.ingest.airr import PredSegment, Prediction, load_airr_tsv
from alignair_assay.metrics import is_informational, lower_is_better
from alignair_assay.metrics.case import score_case
from alignair_assay.reference import Reference, gene_name
from alignair_assay.testing import oracle_prediction, write_airr_tsv
from alignair_assay.truth.build import truth_from_bench_case
from alignair_assay.truth.io import read_truth_jsonl

ga = pytest.importorskip("GenAIRR")

REF = Reference.from_dataconfig("HUMAN_IGH_OGRDB")
FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_NAMES = ("amplicon_janchor_000000.json", "orientation_000001.json", "orientation_000002.json")
D_BINS = {"absent", "<5", "5-7", "8-9", "10-11", "12-14", ">=15"}


def _fixture(name):
    return json.loads((FIXTURES / name).read_text())


def _bench_cases():
    """40 forward GenAIRR records (one with an unknown J allele, ten with an inverted D) plus the three fixtures."""
    recs = list(ga.Experiment.on("human_igh").recombine().mutate(rate=0.05).polymerase_indels(count=(0, 2))
                .end_loss_5prime(length=(0, 120)).end_loss_3prime(length=(0, 30)).run_records(n=30, seed=17))
    recs += list(ga.Experiment.on("human_igh").recombine().invert_d(prob=1.0).run_records(n=10, seed=17))
    recs[0] = dict(recs[0], j_call="IGHJ99*01")
    cases = [{"case_id": f"sim:{i:03d}", "stratum": "inverted_d" if rec["d_inverted"] else "sim",
              "orientation_id": 0, "sequence": rec["sequence"], "record": rec} for i, rec in enumerate(recs)]
    return cases + [_fixture(name) for name in FIXTURE_NAMES]


def _write_jsonl(rows, path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def _build_truth(tmp_path, cases, name="cases.jsonl.gz"):
    cases_path = _write_jsonl(cases, tmp_path / name)
    truth_path = tmp_path / "truth" / "truth.jsonl.gz"
    assert main(["truth", "--bench-cases", str(cases_path), "--reference", "HUMAN_IGH_OGRDB",
                 "--out", str(truth_path)]) == 0
    return truth_path, read_truth_jsonl(truth_path)


def _read_scores(path):
    with gzip.open(path, "rt") as fh:
        return {row["case_id"]: row["scores"] for row in map(json.loads, fh)}


def _jsonable(scores):
    return {k: None if math.isnan(v) else v for k, v in scores.items()}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _without_empty_segments(pred):
    return replace(pred, segments={g: s for g, s in pred.segments.items() if s != PredSegment()})


def test_cli_end_to_end_small(tmp_path, capsys):
    truth_path, truths = _build_truth(tmp_path, _bench_cases())
    printed = capsys.readouterr().out
    unscorable = {g: sum(g in t.unscorable for t in truths) for g in "vdj"}
    assert unscorable["j"] >= 1
    assert "cases: 43\n" in printed
    assert "excluded: 1 (non_biological_orientation: 1)\n" in printed
    assert f"cases with warnings: {sum(bool(t.warnings) for t in truths)}\n" in printed
    assert f"unscorable genes: v={unscorable['v']} d={unscorable['d']} j={unscorable['j']}\n" in printed
    assert "IGHJ99*01 not in reference" in printed
    disagreements = {g: sum(t.segments[g].alleles.disagrees_with_simulator for t in truths
                            if t.excluded is None and g in t.segments) for g in "vdj"}
    assert sum(disagreements.values()) > 0
    assert (f"simulator allele-set disagreements (non-excluded cases): v={disagreements['v']} "
            f"d={disagreements['d']} j={disagreements['j']}\n") in printed

    scored = [t for t in truths if t.excluded is None]
    assert [t.case_id for t in truths] == [c["case_id"] for c in _bench_cases()]
    assert len(scored) == 42
    assert {t.orientation for t in scored} == {0, 1}
    assert any("d" in t.segments and t.segments["d"].inverted for t in scored)

    # The oracle survives the AIRR TSV round trip unchanged.
    oracles = [oracle_prediction(t) for t in scored]
    tsv = tmp_path / "oracle.tsv"
    write_airr_tsv(oracles, tsv)
    loaded, stats = load_airr_tsv(tsv)
    assert stats == {"rows": 42, "duplicates": 0, "invalid_rows": 0}
    for p in oracles:
        assert _without_empty_segments(loaded[p.sequence_id]) == p

    out_dir = tmp_path / "scores"
    assert main(["score", "--truth", str(truth_path), "--predictions", str(tsv), "--tool", "oracle",
                 "--out-dir", str(out_dir)]) == 0
    printed = capsys.readouterr().out
    assert "missing 0" in printed and "extra 0" in printed

    # Every oracle metric is perfect through the TSV, and equals the in-memory score.
    cli_scores = _read_scores(out_dir / "oracle.scores.jsonl.gz")
    assert list(cli_scores) == [t.case_id for t in scored]
    checked = 0
    for t, p in zip(scored, oracles):
        assert cli_scores[t.case_id] == _jsonable(score_case(t, p, REF))
        for name, value in cli_scores[t.case_id].items():
            if not is_informational(name):
                assert value == (0.0 if lower_is_better(name) else 1.0), (t.case_id, name, value)
                checked += 1
    assert checked > 1000

    report = json.loads((out_dir / "oracle.report.json").read_text())
    clone_key = report["overall"]["clone_key.exact"]
    assert clone_key["mean"] == 1.0 and clone_key["coverage"] == 1.0 and clone_key["n"] >= 30
    slices = report["by_slice"]
    assert set(slices) == {"stratum", "orientation", "d_len_bin", "d_inverted"}
    assert set(slices["d_inverted"]) == {"yes", "no"}
    assert slices["d_inverted"]["yes"]["clone_key.exact"]["n"] >= 5
    assert set(slices["stratum"]) == {"sim", "inverted_d", "amplicon_janchor", "orientation"}
    assert set(slices["orientation"]) == {"0", "1"}
    assert set(slices["d_len_bin"]) <= D_BINS and len(slices["d_len_bin"]) >= 3
    prov = report["provenance"]
    assert prov["command"] == "score" and prov["tool"] == "oracle" and prov["reference"] == "HUMAN_IGH_OGRDB"
    assert prov["min_observed"] == {"v": 20, "d": 8, "j": 10}
    assert prov["simulator_disagreements"] == disagreements
    assert prov["inputs"]["truth"] == {"path": str(truth_path), "sha256": _sha256(truth_path)}
    assert prov["inputs"]["predictions"] == {"path": str(tsv), "sha256": _sha256(tsv)}
    assert "git_commit" in prov and prov["timestamp_utc"].endswith("+00:00")
    assert prov["matching"] == {"truth_cases": 43, "excluded": 1, "scored": 42, "rows": 42, "predictions": 42,
                                "duplicates": 0, "invalid_rows": 0, "missing": 0, "extra": 0,
                                "excluded_ignored": 0}
    md = (out_dir / "oracle.report.md").read_text()
    assert "| clone_key.exact | 100.0% | " in md
    assert "## By d_len_bin" in md


def test_compare_scores_missing_cases_and_drops_informational(tmp_path, capsys):
    truth_path, truths = _build_truth(tmp_path, _bench_cases())
    scored = [t for t in truths if t.excluded is None]
    excluded = next(t for t in truths if t.excluded is not None)
    oracles = [oracle_prediction(t) for t in scored]
    a_tsv = tmp_path / "a.tsv"
    write_airr_tsv(oracles, a_tsv)

    # b: the first two cases missing, every V call a wrong gene, every junction placed 3 bases early (late could
    # run past a 3'-trimmed read), plus a row for an unknown id and one for the excluded case.
    broken = []
    for t, p in zip(scored[2:], oracles[2:]):
        segments = dict(p.segments)
        if "v" in segments:
            v = segments["v"]
            other = next(a for a in sorted(REF.alleles["v"]) if gene_name(a) != gene_name(v.call))
            segments["v"] = replace(v, call=other, call_set=(other,))
        if p.junction_start is not None:
            start, end = p.junction_start - 3, p.junction_end - 3
            p = replace(p, junction_start=start, junction_end=end, junction_nt=t.sequence[start:end])
        broken.append(replace(p, segments=segments))
    b_tsv = tmp_path / "b.tsv"
    write_airr_tsv(broken + [Prediction(sequence_id="ghost"), Prediction(sequence_id=excluded.case_id)], b_tsv)

    out_dir = tmp_path / "cmp"
    assert main(["compare", "--truth", str(truth_path), "--a", str(a_tsv), "--a-name", "oracle",
                 "--b", str(b_tsv), "--b-name", "broken", "--out-dir", str(out_dir)]) == 0
    printed = capsys.readouterr().out
    assert "missing 2" in printed and "extra 1" in printed
    # The headline table is printed, not a tally of verdicts.
    assert "| Metric | oracle | broken | n paired | oracle (paired) | broken (paired) |" in printed
    assert "| v.top1_in_set |" in printed and "| v.seq_start_exact |" not in printed
    assert "better" not in printed.replace("a_better", "").replace("b_better", "")

    rep = json.loads((out_dir / "comparison.json").read_text())
    assert rep["kind"] == "alignair_assay.comparison_report" and (rep["a"], rep["b"]) == ("oracle", "broken")
    prov = rep["provenance"]
    assert prov["command"] == "compare"
    assert prov["min_observed"] == {"v": 20, "d": 8, "j": 10}
    assert prov["alpha"] == 0.05 and "McNemar" in prov["tests"]
    assert set(prov["simulator_disagreements"]) == {"v", "d", "j"}
    assert "n_boot" not in prov and "seed" not in prov  # analytic tests: nothing random
    assert set(prov["inputs"]) == {"truth", "a", "b"}
    assert prov["inputs"]["b"]["sha256"] == _sha256(b_tsv)
    match_a, match_b = prov["matching"]["a"], prov["matching"]["b"]
    assert (match_a["missing"], match_a["extra"], match_a["excluded_ignored"]) == (0, 0, 0)
    assert (match_b["missing"], match_b["extra"], match_b["excluded_ignored"]) == (2, 1, 1)
    assert match_b["scored"] == 42 and match_b["rows"] == 42

    # Family: every name either tool produced, minus the informational ones.
    preds_b, _ = load_airr_tsv(b_tsv)
    names = set()
    for t, p in zip(scored, oracles):
        names |= set(score_case(t, p, REF)) | set(score_case(t, preds_b.get(t.case_id), REF))
    comp = rep["comparison"]
    assert set(comp) == {n for n in names if not is_informational(n)}
    assert prov["family_size"] == len(comp)
    assert "v.set_size" in names and "v.set_size" not in comp
    assert "junction.start_err" in names and "junction.start_err" not in comp

    # Missing cases are scored (not dropped): validity is paired over all 42 cases.
    assert comp["airr.valid"]["n_paired"] == 42
    assert comp["airr.valid"]["b"]["mean"] == pytest.approx(40 / 42)
    assert comp["v.top1_in_set"]["verdict"] == "a_better"
    # Lower-is-better metrics: b's larger junction error is a win for a.
    assert comp["junction.start_abs_err"]["delta"] == 3.0
    assert comp["junction.start_abs_err"]["verdict"] == "a_better"

    md = (out_dir / "comparison.md").read_text()
    assert "# AlignAIR assay: oracle vs broken" in md
    assert "| Metric | oracle | broken | n paired | oracle (paired) | broken (paired) | Δ (b−a) |" in md


def _igblast_cigar(seg, read_len):
    """IgBLAST pads CIGARs with read soft clips (S) and skipped germline (N) on both sides."""
    allele_len = len(REF.seq(seg.call))
    ops = [(seg.seq_start, "S"), (seg.germ_start, "N"), *parse_cigar(seg.cigar),
           (read_len - seg.seq_end, "S"), (allele_len - seg.germ_end, "N")]
    return format_cigar(ops)


def test_rev_comp_row_scores_like_forward(tmp_path):
    case = _fixture("orientation_000001.json")
    assert case["orientation_id"] == 1
    forward = truth_from_bench_case(dict(case, orientation_id=0, sequence=case["record"]["sequence"]), REF)
    expected = _jsonable(score_case(forward, oracle_prediction(forward), REF))

    truth_path, (truth,) = _build_truth(tmp_path, [case], name="cases.jsonl")
    assert truth.orientation == 1
    oracle = oracle_prediction(truth)
    assert oracle.rev_comp is True
    # As IgBLAST writes it: rev_comp=T, the canonical read in `sequence` with coordinates on it, padded CIGARs,
    # no junction coordinates.
    n = len(truth.sequence)
    row = replace(oracle, sequence=truth.sequence, junction_start=None, junction_end=None,
                  segments={g: replace(s, cigar=_igblast_cigar(s, n)) for g, s in oracle.segments.items()})
    assert all("S" in s.cigar and "N" in s.cigar for s in row.segments.values())
    tsv = tmp_path / "igblast.tsv"
    write_airr_tsv([row], tsv)

    out_dir = tmp_path / "out"
    assert main(["score", "--truth", str(truth_path), "--predictions", str(tsv), "--tool", "igblast",
                 "--out-dir", str(out_dir)]) == 0
    scores = _read_scores(out_dir / "igblast.scores.jsonl.gz")[truth.case_id]
    assert scores == expected
    assert scores["orientation.correct"] == 1.0 and scores["junction.start_abs_err"] == 0.0

    write_airr_tsv([replace(row, rev_comp=False)], tsv)
    assert main(["score", "--truth", str(truth_path), "--predictions", str(tsv), "--tool", "igblast",
                 "--out-dir", str(out_dir)]) == 0
    flipped = _read_scores(out_dir / "igblast.scores.jsonl.gz")[truth.case_id]
    # rev_comp=F with the reverse-complemented read in `sequence`: wrong orientation and an invalid row.
    assert flipped == dict(expected, **{"orientation.correct": 0.0, "airr.sequence_ok": 0.0, "airr.valid": 0.0})


@pytest.mark.parametrize("length,label", [
    (None, "absent"), (0, "<5"), (4, "<5"), (5, "5-7"), (7, "5-7"), (8, "8-9"), (9, "8-9"), (10, "10-11"),
    (11, "10-11"), (12, "12-14"), (14, "12-14"), (15, ">=15"), (40, ">=15"),
])
def test_d_len_bin(length, label):
    assert d_len_bin(length) == label


def test_d_inverted_slice():
    truth = truth_from_bench_case(_fixture("amplicon_janchor_000000.json"), REF)
    d = truth.segments["d"]
    assert slice_labels(truth)["d_inverted"] == "no"
    assert slice_labels(replace(truth, segments={**truth.segments, "d": replace(d, inverted=True)}))[
        "d_inverted"] == "yes"
    no_d = replace(truth, segments={g: s for g, s in truth.segments.items() if g != "d"})
    assert slice_labels(no_d)["d_inverted"] == "absent"


def test_is_informational():
    for name in ("v.set_size", "d.set_size", "junction.start_err", "junction.end_err"):
        assert is_informational(name), name
    for name in ("junction.start_abs_err", "junction.end_abs_err", "xjunction.start_err", "v.top1_in_set",
                 "v.seq_start_err"):
        assert not is_informational(name), name
