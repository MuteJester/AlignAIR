"""The diagnostics report: markdown rendered from the plain dict ``diagnostics.evaluate`` builds."""
import copy
import math

from alignair_assay.report.diagnostics import (
    DIAGNOSTICS_KIND,
    SECTIONS,
    build_diagnostics_report,
    diagnostics_markdown,
)

KEY = "0123456789ab-fedcba987654"


def _cell(mean, n=10, coverage=1.0):
    return {"mean": mean, "n": n, "n_applicable": n, "coverage": coverage}


def _entry(own_from, own_to, paired_from, paired_to, verdict, *, n=10):
    delta = paired_to - paired_from
    return {"mean_from": own_from, "n_from": n, "coverage_from": 1.0, "mean_to": own_to, "n_to": n - 1,
            "coverage_to": 0.9, "mean_from_paired": paired_from, "mean_to_paired": paired_to, "delta": delta,
            "n_paired": n - 1, "ci95": [delta - 0.05, delta + 0.05], "test": "mcnemar_exact", "p_value": 0.0123,
            "verdict": verdict}


def _bins(filled):
    """Ten calibration bins, the given ones filled: ``{index: (n, confidence, accuracy)}``."""
    return [{"lo": k / 10, "hi": (k + 1) / 10, "n": filled[k][0] if k in filled else 0,
             "confidence": filled[k][1] if k in filled else None, "accuracy": filled[k][2] if k in filled else None}
            for k in range(10)]


def _suite(name="S1", *, reused=False) -> dict:
    heads = {
        "head.v.start_exact.soft": _cell(0.907), "head.v.start_exact.argmax": _cell(1.0),
        "head.v.start_abs_err.soft": _cell(0.093), "head.v.start_abs_err.argmax": _cell(0.0),
        "head.v.start_err.soft": _cell(-0.5), "head.v.start_err.argmax": _cell(0.0),
        "head.j.end_exact.soft": _cell(0.003), "head.j.end_exact.argmax": _cell(0.843),
        "head.j.end_abs_err.soft": _cell(2.02), "head.j.end_abs_err.argmax": _cell(0.2),
        "head.j.end_err.soft": _cell(-2.02), "head.j.end_err.argmax": _cell(-0.1),
        "head.v.allele_top1": _cell(0.927), "head.v.allele_top3": _cell(0.963), "head.v.allele_top5": _cell(0.977),
        "head.v.allele_top1_generative": _cell(0.8), "head.v.n_above_threshold": _cell(1.25),
        "head.mutation_rate_abs_err.read": _cell(0.083), "head.productive.correct": _cell(0.906),
        "head.orientation.correct": _cell(0.997), "head.cropped": _cell(0.06),
    }
    calibration = {"v": {"gene": "v", "n": 10, "bins": _bins({8: (2, 0.85, 0.5), 9: (8, 0.95, 1.0)}),
                         "ece": 0.059, "confidence": 0.93, "accuracy": 0.9},
                   "j": {"gene": "j", "n": 0, "bins": _bins({}), "ece": None, "confidence": None, "accuracy": None}}
    variants = {
        "baseline": {"scores": {"junction.nt_exact": _cell(0.6)}, "fallbacks": {}, "seconds": {"forward": 0.5},
                     "substituted": []},
        "true_spans": {"scores": {"junction.nt_exact": _cell(0.8)}, "fallbacks": {"spans.d.unobserved": 5},
                       "seconds": {"forward": 0.5}, "substituted": ["orientation", "spans", "classifier_spans"]},
        "argmax_decoder": {"scores": {"junction.nt_exact": _cell(0.75)}, "fallbacks": {},
                           "seconds": {"forward": 0.5}, "substituted": ["decoder"]},
    }
    contributions = [{"from": "baseline", "to": "true_spans", "stages": ["orientation", "segmentation"],
                      "metrics": {"junction.nt_exact": _entry(0.6, 0.8, 0.602, 0.8, "to_better"),
                                  "v.top1_in_set": _entry(0.9, 0.9, 0.9, 0.9, "tie_or_inconclusive")}}]
    swaps = [{"from": "baseline", "to": "argmax_decoder",
              "metrics": {"junction.nt_exact": _entry(0.6, 0.75, 0.6, 0.75, "to_better"),
                          "v.top1_in_set": _entry(0.9, 0.85, 0.9, 0.85, "from_better")}}]
    return {"suite": name, "key": {"version": "1"},
            "reads": {"built": 100, "diagnosed": 50, "scored": 48, "excluded": 2, "cropped": 3},
            "heads": heads, "calibration": calibration, "metrics": ["junction.nt_exact", "v.top1_in_set"],
            "variants": variants, "contributions": contributions, "swaps": swaps, "seconds": 12.3,
            "reused": reused}


PROFILE = {"device": "cuda", "batch_size": 256, "reads": 1000, "repeats": 3, "reads_per_second": 1103.4,
           "seconds_per_run": 0.906,
           "modules": {"embedding": {"ms_per_read": 0.01, "share": 0.05}, "v.cls_tower": {"ms_per_read": 0.03,
                                                                                         "share": 0.15},
                       "other": {"ms_per_read": 0.16, "share": 0.80}},
           "stages": {"forward": {"ms_per_read": 0.225, "share": 0.249}, "germline": {"ms_per_read": 0.53,
                                                                                     "share": 0.585},
                      "airr": {"ms_per_read": 0.15, "share": 0.166}}}


def _report(suites=None, profile=PROFILE) -> dict:
    suites = suites if suites is not None else {"S1": _suite(), "S3": _suite("S3", reused=True)}
    model = {"model_key": KEY, "source": "/models/m.alignair", "weights_sha256": "0123456789ab" + "0" * 52,
             "code_sha256": "fedcba987654" + "1" * 52, "alignair_source": "/src/alignair", "torch_version": "2.4.0",
             "assay_code_sha256": "2" * 64}
    config = {"run_dir": "/runs/iter", "suites": list(suites), "device": "cuda", "batch_size": 256,
              "variants": ["baseline", "true_spans", "argmax_decoder"], "max_cases": 500, "profile": profile is not None}
    return build_diagnostics_report(model=model, config=config, suites=suites, profile=profile)


def _section(md: str, title: str) -> str:
    """The text of section ``## title`` up to the next ``## `` heading."""
    start = md.index(f"\n## {title}\n")
    end = md.find("\n## ", start + 1)
    return md[start:end if end != -1 else len(md)]


def test_report_dict_is_strict_json():
    suite = _suite()
    suite["heads"]["head.v.start_err.soft"]["mean"] = math.nan
    report = _report({"S1": suite})
    assert report["kind"] == DIAGNOSTICS_KIND and report["version"] == "1"
    assert report["model"]["model_key"] == KEY and list(report["suites"]) == ["S1"]
    assert report["suites"]["S1"]["heads"]["head.v.start_err.soft"]["mean"] is None
    assert suite["heads"]["head.v.start_err.soft"]["mean"] != suite["heads"]["head.v.start_err.soft"]["mean"]


def test_every_section_renders_in_order():
    md = diagnostics_markdown(_report())
    assert md.startswith(f"# AlignAIR diagnostics: model {KEY}\n")
    assert SECTIONS == ("Head metrics", "Stage contributions", "Calibration", "Single-swap variants", "Profile",
                        "Fallbacks and crops")
    positions = [md.index(f"\n## {title}\n") for title in SECTIONS]
    assert positions == sorted(positions)
    # The per-suite sections each have one subsection per suite.
    assert md.count("\n### S1\n") == 4 and md.count("\n### S3\n") == 4
    # The overview names what was computed and what was reused.
    assert "| S1 | 50 of 100 | 3 | computed in 12.3 s |" in md
    assert "| S3 | 50 of 100 | 3 | reused (computed in 12.3 s) |" in md
    assert "`0123456789ab`" in md and "torch 2.4.0" in md and "max cases 500" in md


def test_heads_show_soft_and_argmax_side_by_side():
    heads = _section(diagnostics_markdown(_report()), "Head metrics")
    assert ("| Edge | Exact (soft) | Exact (argmax) | Abs. error, nt (soft) | Abs. error, nt (argmax) | "
            "Signed error, nt (soft) | Signed error, nt (argmax) |") in heads
    assert ("| V start | 90.7% (n=10, 100%) | 100.0% (n=10, 100%) | 0.093 (n=10, 100%) | 0.000 (n=10, 100%) | "
            "-0.500 (n=10, 100%) | 0.000 (n=10, 100%) |") in heads
    assert "| J end | 0.3% (n=10, 100%) | 84.3% (n=10, 100%) |" in heads
    assert "| V end |" not in heads  # no row without data
    # Allele ranks; the number of alleles over the threshold is a count, not a rate.
    assert ("| V | 92.7% (n=10, 100%) | 96.3% (n=10, 100%) | 97.7% (n=10, 100%) | 80.0% (n=10, 100%) | "
            "1.250 (n=10, 100%) |") in heads
    assert "| head.mutation_rate_abs_err.read | 0.083 (n=10, 100%) |" in heads
    assert "| head.productive.correct | 90.6% (n=10, 100%) |" in heads


def test_stage_contributions_show_paired_means_beside_the_delta():
    section = _section(diagnostics_markdown(_report()), "Stage contributions")
    # The cumulative overview: one column per step, Δ and verdict.
    assert "| Metric | Δ orientation, segmentation |" in section
    assert "| junction.nt_exact | +19.8 pp (better) |" in section
    assert "| v.top1_in_set | +0.0 pp (tie) |" in section
    # The step: own means with coverage, then the paired means the Δ is their difference of.
    assert "#### baseline → true_spans (orientation, segmentation)" in section
    assert ("| Metric | baseline | true_spans | n paired | baseline (paired) | true_spans (paired) | Δ | 95% CI | p | "
            "Verdict |") in section
    assert ("| junction.nt_exact | 60.0% (n=10, 100%) | 80.0% (n=9, 90%) | 9 | 60.2% | 80.0% | +19.8 pp | "
            "[+14.8 pp, +24.8 pp] | 0.0123 | better |") in section


def test_calibration_single_swaps_profile_and_fallbacks():
    md = diagnostics_markdown(_report())
    calibration = _section(md, "Calibration")
    assert "| Gene | n | Mean confidence | Accuracy | ECE |" in calibration
    assert "| V | 10 | 93.0% | 90.0% | 0.059 |" in calibration
    assert "| J | 0 | — | — | — |" in calibration
    assert "| 0.8–0.9 | 2: 50.0% / 85.0% | — |" in calibration  # reads: accuracy / mean confidence
    assert "| 0.9–1.0 | 8: 100.0% / 95.0% | — |" in calibration
    assert "| 0.0–0.1 |" not in calibration  # empty bins are left out

    swaps = _section(md, "Single-swap variants")
    assert "#### baseline → argmax_decoder" in swaps
    assert "| v.top1_in_set | 90.0% (n=10, 100%) | 85.0% (n=9, 90%) | 9 | 90.0% | 85.0% | -5.0 pp |" in swaps
    assert "| worse |" in swaps

    profile = _section(md, "Profile")
    assert "Device cuda, batch size 256, 1000 reads: 1103.4 reads/s" in profile
    assert "| germline | 0.530 | 58.5% |" in profile
    modules = profile[profile.index("| Module |"):]
    assert modules.index("| other |") < modules.index("| v.cls_tower |") < modules.index("| embedding |")

    fallbacks = _section(md, "Fallbacks and crops")
    assert "| Suite | Built | Diagnosed | Scored | Excluded | Cropped |" in fallbacks
    assert "| S1 | 100 | 50 | 48 | 2 | 3 |" in fallbacks
    assert "| S1 | true_spans | spans.d.unobserved | 5 |" in fallbacks


def test_sections_without_data_say_so():
    suite = _suite()
    suite.update(variants={}, contributions=[], swaps=[], calibration={})
    md = diagnostics_markdown(_report({"S1": suite}, profile=None))
    assert "No cumulative-chain variant was run" in _section(md, "Stage contributions")
    assert "No single-swap variant was run" in _section(md, "Single-swap variants")
    assert "Not measured" in _section(md, "Profile")
    assert "No variant fell back" in _section(md, "Fallbacks and crops")
    assert "No allele head" in _section(md, "Calibration")
    for title in SECTIONS:
        assert f"\n## {title}\n" in md


def test_rendering_does_not_change_the_report():
    report = _report()
    before = copy.deepcopy(report)
    diagnostics_markdown(report)
    assert report == before
