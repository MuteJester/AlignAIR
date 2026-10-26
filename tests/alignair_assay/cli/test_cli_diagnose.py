"""``diagnose --against``'s printed comparison: each significant metric with its direction."""
from types import SimpleNamespace

from alignair_assay.cli.diagnose import _print_against
from alignair_assay.core.layout import RunLayout
from alignair_assay.report.diagnostics import against_summary

THIS, BASELINE = "0123456789ab-fedcba987654", "aaaaaaaaaaaa-fedcba987654"


def _entry(a, b, verdict):
    """A ``stats.compare`` entry (a: the baseline, b: this model)."""
    return {"n_paired": 100, "n_clusters": 100, "mean_a_paired": a, "mean_b_paired": b, "delta": b - a,
            "ci95": [b - a - 0.01, b - a + 0.01], "test": "mcnemar_exact", "p_value": 0.001, "verdict": verdict}


def _family(comparison: dict) -> dict:
    return {"metrics": list(comparison), "comparison": comparison,
            "changes": {"metrics": {}, "reads": {"fixed": 1, "broken": 2, "fixed_examples": [], "broken_examples": []},
                        "a_only": 0, "b_only": 0, "a_only_examples": [], "b_only_examples": []}}


def test_the_significant_metrics_are_printed_with_their_direction(tmp_path, capsys):
    against = {"baseline": BASELINE, "source": "/models/b.alignair", "same_model": False, "replay_verified": True,
               "suites": {"S1": {"cases": 100, "clusters": {"clusters": 0},
                                 "baseline": _family({"junction.nt_exact": _entry(0.6, 0.55, "a_better"),
                                                      "v.top1_in_set": _entry(0.9, 0.9, "tie_or_inconclusive")}),
                                 "heads": _family({"head.v.start_exact.soft": _entry(0.9, 0.95, "b_better")})},
                          "S3": {"cases": 50, "clusters": {"clusters": 0},
                                 "baseline": _family({"junction.nt_exact": _entry(0.7, 0.7, "tie_or_inconclusive")}),
                                 "heads": None}}}
    result = SimpleNamespace(model_key=THIS, against={**against, "summary": against_summary(against)})
    _print_against(result, RunLayout(tmp_path))
    out = capsys.readouterr().out
    assert ("significant: junction.nt_exact (baseline better), head.v.start_exact.soft (this better);"
            in out.splitlines()[1]), out
    assert "S3 against" in out.splitlines()[2] and "significant: none;" in out.splitlines()[2]
