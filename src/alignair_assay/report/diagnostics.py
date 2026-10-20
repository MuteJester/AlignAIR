"""The diagnostics report of one model: the plain dict ``diagnostics.evaluate`` builds, and its markdown.

``build_diagnostics_report`` assembles the dict (strict JSON: NaN and infinity become None) and
``diagnostics_markdown`` renders it: the model and the suites overview, then ``SECTIONS`` in order, each always
present (a section without data says so):

1. **Head metrics**, per suite: the segment boundaries under the soft-argmax and the argmax decoders side by side,
   the allele ranks and the metadata heads.
2. **Stage contributions**, per suite: the cumulative chain's Δ per step with its verdict, then each step's
   comparison, with the paired means beside the Δ.
3. **Calibration**, per suite: the reliability of each gene's top-1 allele probability.
4. **Single-swap variants**, per suite: each swap against the baseline.
5. **Profile**: time per stage and per network module.
6. **Fallbacks and crops**: the reads of each suite and the substitutions that kept the model's output.
"""
from __future__ import annotations

from alignair_assay.report.build import _clean
from alignair_assay.report.markdown import MISSING, format_cell, format_value, markdown_table

DIAGNOSTICS_KIND = "alignair_assay.diagnostics_report"
DIAGNOSTICS_VERSION = "1"
SECTIONS = ("Head metrics", "Stage contributions", "Calibration", "Single-swap variants", "Profile",
            "Fallbacks and crops")
# A comparison's verdict, from the "to" variant's side (a step adds truth; a swap changes the baseline).
VERDICTS = {"to_better": "better", "from_better": "worse", "tie_or_inconclusive": "tie"}
_GENES = ("v", "d", "j")
_EDGES = ("start", "end")
_BOUNDARY_COLUMNS = (("exact", "soft"), ("exact", "argmax"), ("abs_err", "soft"), ("abs_err", "argmax"),
                     ("err", "soft"), ("err", "argmax"))
_ALLELE_METRICS = ("allele_top1", "allele_top3", "allele_top5", "allele_top1_generative", "n_above_threshold")
_META_METRICS = ("head.mutation_rate_abs_err.read", "head.mutation_rate_abs_err.record", "head.indel_count_abs_err",
                 "head.productive.correct", "head.orientation.correct", "head.d.short_d_prob_ok", "head.cropped")

_OVERVIEW_NOTE = ("Each suite's diagnostics are cached under the model key and reused only while the model's "
                  "weights, the AlignAIR code, torch, the assay's scoring code, the suite's build and the "
                  "configuration (variants, max cases, device, batch size) are all unchanged.")
_HEADS_NOTE = ("The raw head outputs of the product pipeline (the baseline replay), scored against observable "
               "truth before any post-processing. Boundaries are 0-based and end-exclusive. soft: the floor of the "
               "soft-argmax expectation the product decodes with; argmax: the position logits' argmax, whose end "
               "can reach the read end (the soft-argmax cannot). Exact: the edge is inside the truth interval; "
               "errors are in nucleotides, a signed one negative before the interval. Columns marked "
               "informational are not tested.")
_CONTRIBUTIONS_NOTE = ("The truth of one more stage is fed in at each step (baseline → true_orientation → "
                       "true_spans → true_calls → true_germline); each variant is scored like a tool and compared "
                       "with the previous one on the cases both cover, Holm-corrected over the metrics. A step's Δ "
                       "is the error that stage contributes given perfect upstream stages, not a share of the "
                       "baseline's error, so the steps need not add up. Verdicts: better (the truth improves the "
                       "metric significantly), worse, tie (no significant difference). Each step's table gives each "
                       "variant's own mean over its own cases (n, coverage) and the paired means, whose difference "
                       "is Δ; rates are in percent, their differences in percentage points.")
_CALIBRATION_NOTE = ("Reliability of each gene's top-1 calibrated allele probability against the top-1 being in the "
                     "scored truth set, in equal-width bins. ECE: the bins' read-weighted |accuracy − confidence|. "
                     "A bin's cell is reads: accuracy / mean confidence; empty bins are left out.")
_SWAPS_NOTE = ("Each swap changes one thing on the baseline (argmax_decoder: the segments from the position "
               "logits' argmax; true_classifier_spans: the truth spans into the classifier only, the segments "
               "unchanged) and is compared with the baseline like a step above. better: the swap improves the "
               "metric significantly.")
_PROFILE_NOTE = ("Stages: the replay's stages, with their share of the whole run. Modules: the network's leaf "
                 "modules, with their share of the forward stage, from separate runs with timing hooks; other is "
                 "the forward time spent outside them.")
_FALLBACKS_NOTE = ("Cropped: reads longer than AlignAIR's window, which it crops; their edges past the cut are not "
                   "scored. A fallback is a read whose truth for a segment could not be substituted in a variant "
                   "(unscorable, unobserved, a wrong model orientation, cropped, an allele the model cannot call, an "
                   "inverted D): that segment kept the model's output.")


def build_diagnostics_report(*, model: dict, config: dict, suites: dict, profile: dict | None) -> dict:
    """The diagnostics report: ``kind``, ``version``, the ``model`` (``model_key``, ``source``,
    ``weights_sha256``, ``code_sha256``, ``alignair_source``, ``torch_version``, ``assay_code_sha256``), the run
    ``config`` (``run_dir``, ``suites``, ``device``, ``batch_size``, ``variants``, ``max_cases``, ``profile``), the
    per-suite results by suite name and the ``profile`` (None when not measured), all strictly JSON-serializable.
    The inputs are not changed."""
    return {"kind": DIAGNOSTICS_KIND, "version": DIAGNOSTICS_VERSION, "model": _clean(model),
            "config": _clean(config), "suites": _clean(suites), "profile": _clean(profile)}


def diagnostics_markdown(report: dict) -> str:
    """The report as markdown (see the module docstring for its sections)."""
    model, config = report["model"], report["config"]
    lines = [f"# AlignAIR diagnostics: model {model['model_key']}", "", _model_text(model, config), "",
             _OVERVIEW_NOTE, "", *_overview(report["suites"])]
    for title, render in zip(SECTIONS, (_heads, _contributions, _calibration, _swaps, _profile, _fallbacks)):
        lines += ["", f"## {title}", "", *render(report)]
    return "\n".join(lines) + "\n"


def _model_text(model: dict, config: dict) -> str:
    max_cases = config.get("max_cases")
    return (f"Model key `{model['model_key']}`: weights SHA-256 `{model['weights_sha256'][:12]}` "
            f"({model['source']}), AlignAIR code SHA-256 `{model['code_sha256'][:12]}` ({model['alignair_source']}), "
            f"torch {model['torch_version']}. Run on {config['device']} with batch size {config['batch_size']}, "
            f"max cases {max_cases if max_cases else 'all'} per suite, variants "
            f"{', '.join(config['variants']) or 'none'}; run directory {config['run_dir']}. Every suite is on the "
            "dev split.")


def _overview(suites: dict) -> list[str]:
    rows = []
    for name, suite in suites.items():
        reads = suite["reads"]
        seconds = f"computed in {suite['seconds']:.1f} s"
        rows.append([name, f"{reads['diagnosed']} of {reads['built']}", str(len(suite.get("variants") or {})),
                     f"reused ({seconds})" if suite.get("reused") else seconds])
    return markdown_table(["Suite", "Reads", "Variants", "Result"], rows)


def _per_suite(report: dict, note: str, render) -> list[str]:
    lines = [note]
    for name, suite in report["suites"].items():
        lines += ["", f"### {name}", "", *render(suite)]
    return lines


def _heads(report: dict) -> list[str]:
    return _per_suite(report, _HEADS_NOTE, _suite_heads)


def _suite_heads(suite: dict) -> list[str]:
    heads = suite.get("heads") or {}
    lines: list[str] = []
    rows = []
    for g in _GENES:
        for edge in _EDGES:
            names = [f"head.{g}.{edge}_{kind}.{decoder}" for kind, decoder in _BOUNDARY_COLUMNS]
            if any(name in heads for name in names):
                rows.append([f"{g.upper()} {edge}", *(format_cell(name, heads.get(name)) for name in names)])
    if rows:
        lines += ["Boundaries:", "", *markdown_table(
            ["Edge", "Exact (soft)", "Exact (argmax)", "Abs. error, nt (soft)", "Abs. error, nt (argmax)",
             "Signed error, nt (soft)", "Signed error, nt (argmax)"], rows), ""]
    rows = []
    for g in _GENES:
        names = [f"head.{g}.{metric}" for metric in _ALLELE_METRICS]
        if any(name in heads for name in names):
            rows.append([g.upper(), *(format_cell(name, heads.get(name)) for name in names)])
    if rows:
        lines += ["Alleles (ranked by calibrated probability):", "", *markdown_table(
            ["Gene", "Top-1 in set", "Top-3 in set", "Top-5 in set", "Top-1 generative (informational)",
             "Alleles at p ≥ 0.5 (informational)"], rows), ""]
    rows = [[name, format_cell(name, heads[name])] for name in _META_METRICS if name in heads]
    if rows:
        lines += ["Metadata heads:", "", *markdown_table(["Head metric", "Value"], rows)]
    return lines or ["No head metric applies to these reads."]


def _contributions(report: dict) -> list[str]:
    return _per_suite(report, _CONTRIBUTIONS_NOTE, _suite_contributions)


def _suite_contributions(suite: dict) -> list[str]:
    steps = suite.get("contributions") or []
    if not steps:
        return ["No cumulative-chain variant was run (beyond the baseline), so there is no step to show."]
    metrics = _metrics(suite, steps)
    rows = [[name, *(_delta_cell(name, step["metrics"].get(name)) for step in steps)] for name in metrics]
    lines = markdown_table(["Metric", *(f"Δ {', '.join(step['stages'])}" for step in steps)], rows)
    for step in steps:
        lines += ["", f"#### {step['from']} → {step['to']} ({', '.join(step['stages'])})", "",
                  *_comparison_table(step, metrics)]
    return lines


def _swaps(report: dict) -> list[str]:
    return _per_suite(report, _SWAPS_NOTE, _suite_swaps)


def _suite_swaps(suite: dict) -> list[str]:
    swaps = suite.get("swaps") or []
    if not swaps:
        return ["No single-swap variant was run."]
    metrics = _metrics(suite, swaps)
    lines: list[str] = []
    for swap in swaps:
        lines += [*([""] if lines else []), f"#### {swap['from']} → {swap['to']}", "",
                  *_comparison_table(swap, metrics)]
    return lines


def _metrics(suite: dict, comparisons: list[dict]) -> list[str]:
    """The compared metrics in the suite's family order, then any other a comparison has."""
    present = dict.fromkeys(name for comparison in comparisons for name in comparison["metrics"])
    return [name for name in suite.get("metrics") or [] if name in present] + [
        name for name in present if name not in (suite.get("metrics") or [])]


def _delta_cell(name: str, entry: dict | None) -> str:
    if entry is None or entry.get("delta") is None:
        return MISSING
    return f"{format_value(name, entry['delta'], delta=True)} ({VERDICTS.get(entry['verdict'], entry['verdict'])})"


def _comparison_table(comparison: dict, metrics: list[str]) -> list[str]:
    a, b = comparison["from"], comparison["to"]
    rows = []
    for name in metrics:
        e = comparison["metrics"].get(name)
        if e is None:
            continue
        own_a = {"mean": e["mean_from"], "n": e["n_from"], "coverage": e["coverage_from"]}
        own_b = {"mean": e["mean_to"], "n": e["n_to"], "coverage": e["coverage_to"]}
        p = e.get("p_value")
        rows.append([name, format_cell(name, own_a), format_cell(name, own_b), str(e["n_paired"]),
                     format_value(name, e.get("mean_from_paired")), format_value(name, e.get("mean_to_paired")),
                     format_value(name, e.get("delta"), delta=True), _ci(name, e.get("ci95")),
                     MISSING if p is None else f"{p:.3g}", VERDICTS.get(e["verdict"], e["verdict"])])
    return markdown_table(["Metric", a, b, "n paired", f"{a} (paired)", f"{b} (paired)", "Δ", "95% CI", "p",
                           "Verdict"], rows)


def _ci(name: str, ci) -> str:
    if not ci:
        return MISSING
    return f"[{format_value(name, ci[0], delta=True)}, {format_value(name, ci[1], delta=True)}]"


def _calibration(report: dict) -> list[str]:
    return _per_suite(report, _CALIBRATION_NOTE, _suite_calibration)


def _suite_calibration(suite: dict) -> list[str]:
    tables = suite.get("calibration") or {}
    genes = [g for g in _GENES if g in tables]
    if not genes:
        return ["No allele head was calibrated on these reads."]
    rows = [[g.upper(), str(tables[g]["n"]), _percent(tables[g]["confidence"]), _percent(tables[g]["accuracy"]),
             MISSING if tables[g]["ece"] is None else f"{tables[g]['ece']:.3f}"] for g in genes]
    lines = markdown_table(["Gene", "n", "Mean confidence", "Accuracy", "ECE"], rows)
    edges = tables[genes[0]]["bins"]
    rows = []
    for k, edge in enumerate(edges):
        cells = [tables[g]["bins"][k] for g in genes]
        if any(cell["n"] for cell in cells):
            rows.append([f"{edge['lo']:.1f}–{edge['hi']:.1f}",
                         *(f"{cell['n']}: {_percent(cell['accuracy'])} / {_percent(cell['confidence'])}"
                           if cell["n"] else MISSING for cell in cells)])
    if rows:
        lines += ["", *markdown_table(["Top-1 probability", *(g.upper() for g in genes)], rows)]
    return lines


def _profile(report: dict) -> list[str]:
    profile = report.get("profile")
    if not profile:
        return ["Not measured: the profile was turned off (`--no-profile`, or `profile=False`)."]
    reused = "; reused from the cache" if profile.get("reused") else ""
    lines = [f"Device {profile['device']}, batch size {profile['batch_size']}, {profile['reads']} reads: "
             f"{profile['reads_per_second']:.1f} reads/s ({profile['seconds_per_run']:.2f} s per run, the median of "
             f"{profile['repeats']} runs{reused}).", "", _PROFILE_NOTE, ""]
    stages = [[name, f"{entry['ms_per_read']:.3f}", _percent(entry["share"])]
              for name, entry in profile["stages"].items()]
    lines += markdown_table(["Stage", "ms/read", "Share"], stages)
    modules = sorted(profile["modules"].items(), key=lambda item: (-item[1]["share"], item[0]))
    lines += ["", *markdown_table(["Module", "ms/read", "Share of forward"],
                                  [[name, f"{entry['ms_per_read']:.3f}", _percent(entry["share"])]
                                   for name, entry in modules])]
    return lines


def _fallbacks(report: dict) -> list[str]:
    suites = report["suites"]
    reads = [[name, *(str(suite["reads"][key]) for key in ("built", "diagnosed", "scored", "excluded", "cropped"))]
             for name, suite in suites.items()]
    lines = [_FALLBACKS_NOTE, "", *markdown_table(["Suite", "Built", "Diagnosed", "Scored", "Excluded", "Cropped"],
                                                  reads), ""]
    rows = [[name, variant, reason, str(count)] for name, suite in suites.items()
            for variant, result in (suite.get("variants") or {}).items()
            for reason, count in (result.get("fallbacks") or {}).items()]
    if rows:
        lines += markdown_table(["Suite", "Variant", "Fallback", "Reads"], rows)
    else:
        lines.append("No variant fell back to the model's output.")
    return lines


def _percent(value) -> str:
    return MISSING if value is None else f"{100 * value:.1f}%"
