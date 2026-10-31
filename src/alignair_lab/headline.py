"""A screen's headline: the candidate against the baseline at the same step, per suite, on one screen.

``render_headline`` takes the assay's comparison summary (``evaluate(..., against=...)``'s
``result.against["summary"]``, ``report.diagnostics_against.against_summary``) and renders three tables, one row
per suite and a tally row (suites where the candidate is better / worse / tied):

1. ``HEADLINE``: ``clone_key.exact``, ``junction.nt_exact`` and V / D / J top-1 in set, each as the Δ (candidate −
   baseline, percentage points), its 95% CI, the verdict and the reads fixed / broken; then four rows that weigh
   the metric against run-to-run training noise (``noise.decide``): ``pooled``, its pooled Δ (read-weighted over the
   suites) / its null band, printed as a reference; ``seeds``, the calibration's baseline seeds' Δs against the same
   baseline screen (mean±SD n); ``diff``, the pooled Δ less their mean, Welch's z with the candidate's spread taken as
   the baseline's, and the suites whose diff has its sign, of those with enough clusters for a verdict (marked ``*``
   when none has, so every suite counts; information only); and ``noise``, the decision: beyond noise (better or
   worse), mixed (a gain with a suite regressing, the suites named), within noise, or uncalibrated without three
   baseline seeds; then the ``REGRESSIONS`` line, every regression of the five metrics (Holm over each metric's
   suites, falls of 1 pp or more; ``noise.suite_tests``) whatever their verdicts;
2. and 3. ``BOUNDARIES``: each boundary's exact rate under the soft-argmax decoder, then under the
   position logits' argmax (the product's decoder), as the Δ, the verdict and the reads fixed / broken. Their CIs are in the assay's
   per-suite detail (``<S>/against_<baseline key>.md``), as on the assay's own summary page.

Verdicts are the assay's (Holm within each suite's family, cluster-aware where a suite has clusters): ``+`` the
candidate better, ``-`` the baseline better, ``=`` a tie, ``?`` a tie with too few clusters to test. ``not paired``:
no read pairs; ``n/a``: the suite does not compare the metric. Plain text, no IO.

The verdicts treat both checkpoints as fixed and resample reads only. Training is not reproducible on the GPU (cuDNN's
default kernels are not deterministic), and at a constant learning rate two runs of one architecture on the same
records drift apart: a 20k-step screen of ``lab-null-v1`` against the conv-v1 milestone showed junction Δs of +6 and
+13 pp judged significant, and boundary Δs up to 23 pp. The legend says so, and the decision rows measure a
candidate's Δs against the null band of baseline screens (``calibrate``).
"""
from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence

from alignair_lab.noise import MIN_SEEDS, MIXED, UNCALIBRATED, pooled

HEADLINE = (("clone_key.exact", "clone_key.exact"), ("junction.nt_exact", "junction.nt_exact"), ("v.top1_in_set", "V top-1"),
            ("d.top1_in_set", "D top-1"), ("j.top1_in_set", "J top-1"))
# Metrics the headline metrics cannot see, weighed by the same rule from the assay's cached scores (``secondary``):
# a D presence gate's benefit is abstention on a D too short to call, which can only lower D top-1 there.
SECONDARY = (("d_short.abstain_ok", "D short abstain"),)
BOUNDARIES = tuple((f"head.{gene}.{edge}_exact", f"{gene.upper()} {edge}") for gene in "vdj" for edge in ("start", "end"))
DECODERS = (("soft", "Boundary exact rates, the soft-argmax decoder:"),
            ("argmax", "Boundary exact rates, the position logits' argmax decoder (the product's):"))
MARKS = {"b_better": "+", "a_better": "-", "tie_or_inconclusive": "="}
LEGEND = ("Δ = candidate − baseline (pp) [95% CI], verdict (+ better, - worse, = tie, ? few clusters), reads fixed/broken. "
          "These omit",
          "run-to-run training noise, weighed below: pooled/band (lab-null-v1), seeds mean±SD n, diff z agree. Better: "
          "t p<=.025, none regress.",
          "One seed: its spread is assumed the baseline seeds' (sc = sb), so a noisier candidate makes z "
          "anti-conservative (too large).",
          "Regressions: Holm per metric over its suites (family α 0.05), falls of 1 pp or more, for all 5 metrics "
          "whatever their verdicts.")
POOLED_LEGEND = (
    "candidate: each init seed's Δ against the baseline screen (read-weighted over the suites), mean±SD k; "
    "baseline seeds: the",
    "calibration's baseline screens' Δs against that screen, mean±SD n. diff: their difference; z: diff / "
    "sqrt(sc²/k + sb²/n), with",
    "sc = sb for one seed; agree: suites whose diff has its sign (*: none has enough clusters, so all; information); "
    "band: the",
    "largest Δ between baseline screens, a reference. p: two-sided, Student's t at the Welch df. Better beyond noise: "
    "a gain's",
    "one-sided p <= .025 and no regression (below), else mixed, naming the suites; worse: a loss's <= .025. A suite "
    "with too few",
    "pairs or clusters is reported (~) but never counted. Uncalibrated without 3 seeds. Regressions: Holm per metric "
    "over its suites",
    "(family α 0.05), falls of 1 pp or more, listed for all 5 metrics whatever their verdicts. With one seed (k1) its "
    "spread is assumed",
    "the baseline seeds' (sc = sb): a noisier candidate makes z anti-conservative (too large).")
REGRESSIONS = "Regressions (Holm over each metric's suites, family α 0.05, falls of 1 pp or more):"
POOLED_HEADING = "The candidate's init seeds against the calibration's baseline seeds (pp):"
SEEDS_HEADING = "Each init seed's pooled Δ against the baseline screen (pp):"
SUITES_HEADING = ("Per-suite z, the candidate's seeds against the baseline seeds in each suite (! a regression above; ~ "
                  "too few pairs or clusters):")
WIDTH = 132                               # the columns a headline line fits in
TALLY = "+/-/="
POOLED, SEEDS, DIFF, NOISE = "pooled", "seeds", "diff", "noise"
_SEPARATOR = "  "


def _mark(entry: dict) -> str:
    mark = MARKS.get(entry.get("verdict"), "?")
    return "?" if entry.get("test") == "too_few_clusters" and mark == "=" else mark


def _cell(entry: dict | None, *, ci: bool) -> str:
    if not entry or entry.get("delta") is None:
        return "n/a"
    if entry.get("n_paired") == 0:
        return "not paired"
    text = f"{100 * entry['delta']:+.1f}"
    if ci and entry.get("ci95"):
        low, high = entry["ci95"]
        text += f" [{100 * low:.1f},{100 * high:.1f}]"
    text += f" {_mark(entry)}"
    if entry.get("fixed") is not None:
        text += f" {entry['fixed']}/{entry['broken']}"
    return text


def _tally(entries: Sequence[dict | None]) -> str:
    counts = {"+": 0, "-": 0, "=": 0}
    for entry in entries:
        if entry and entry.get("delta") is not None and entry.get("n_paired") != 0:
            counts["=" if _mark(entry) == "?" else _mark(entry)] += 1
    return f"{counts['+']}/{counts['-']}/{counts['=']}"


def seeds_cell(found: Mapping) -> str:
    """The baseline seeds of a decision (``noise.decide``): mean±SD n in pp, ``n<k> < 3`` with too few, or none."""
    if found["baseline_mean"] is None:
        return f"n{found['n']} < {MIN_SEEDS}" if found["n"] else "none"
    return f"{100 * found['baseline_mean']:+.1f}±{100 * found['baseline_sd']:.1f} n{found['n']}"


def diff_cell(found: Mapping) -> str:
    """A decision's diff (pp) and z, with its suites agreeing in sign (``*``: counted over every suite)."""
    signs = f"{found['agree']}/{found['suites']}{'*' if found['basis'] == 'all' else ''}"
    if found["diff"] is None:
        return f"n/a {signs}"
    z = "z n/a" if found["z"] is None else f"z{found['z']:+.1f}"
    return f"{100 * found['diff']:+.1f} {z} {signs}"


def verdict_text(found: Mapping) -> str:
    """A decision's verdict; a mixed one names the suites it regresses on."""
    if found["decision"] == MIXED:
        return f"{MIXED}: {','.join(found['regressions'])}"
    return found["decision"] or UNCALIBRATED


def suite_z_cell(found: Mapping, suite: str) -> str:
    """One suite's z of a decision: ``-`` when the suite does not compare the metric, ``n/a`` without a z, ``!`` when
    it regresses, ``~`` when it has too few pairs or clusters to count."""
    test = found["suite_z"].get(suite)
    if test is None:
        return "-"
    z = "n/a" if test["z"] is None else f"{test['z']:+.1f}"
    return f"{z}{'!' if suite in found['regressions'] else ''}{'~' if test['limited'] else ''}"


def p_text(p: float | None) -> str:
    """A p-value to three decimals, ``<0.001`` below that, ``n/a`` without one."""
    return "n/a" if p is None else "<0.001" if p < .001 else f"{p:.3f}"


def band_text(found: Mapping) -> str:
    return "none" if found["band"] is None else f"{100 * found['band']:.1f}"


def regressions_line(decisions: Mapping[str, Mapping]) -> str:
    """The ``REGRESSIONS`` line of the headline metrics' decisions (``noise.decide``): each regression's metric,
    suite, diff (pp) and z, whatever the metric's verdict; none; or not measured without a calibration."""
    if not any(decisions[metric]["suite_z"] for metric, _ in HEADLINE):
        return f"{REGRESSIONS} not measured, uncalibrated."

    def item(label: str, suite: str, test: Mapping) -> str:
        text = f"{label} {suite} {100 * test['diff']:+.1f} pp"
        if test["regresses"]:
            text += " (read-level verdict worse)" if test["z"] is None else f" z {test['z']:+.1f}"
        return text

    def listed(field: str) -> list[str]:
        return [item(label, suite, decisions[metric]["suite_z"][suite])
                for metric, label in HEADLINE for suite in decisions[metric][field]]

    counted, unmeasured = listed("regressions"), listed("unmeasured")
    line = f"{REGRESSIONS} {'; '.join(counted) if counted else 'none'}."
    return line + (f" Spread unmeasured: {'; '.join(unmeasured)}." if unmeasured else "")


def secondary_lines(decisions: Mapping[str, Mapping]) -> list[str]:
    """One line per ``SECONDARY`` metric the decisions hold: its pooled Δ / band, the baseline seeds, the diff, z and
    signs, and the verdict, as the noise rows give them; ``not scored`` when no suite compares it; then the
    ``secondary_notes``."""
    lines = []
    for metric, label in SECONDARY:
        found = decisions.get(metric)
        if found is None:
            continue
        if found["pooled"] is None:
            lines.append(f"{label} ({metric}): not scored.")
            continue
        lines.append(f"{label} ({metric}): pooled {100 * found['pooled']:+.1f} / band {band_text(found)}; seeds "
                     f"{seeds_cell(found)}; diff {diff_cell(found)}; {verdict_text(found)}.")
    return lines + secondary_notes(decisions)


def secondary_notes(decisions: Mapping[str, Mapping]) -> list[str]:
    """Why a ``SECONDARY`` metric is uncalibrated where the headline metrics are not: the calibration holds no values
    of it (its pairs predate it, and its cached scores could not give them, ``baselines.calibration_seeds``)."""
    calibrated = any(decisions[metric]["n"] >= MIN_SEEDS for metric, _ in HEADLINE if metric in decisions)
    return [f"{label} is uncalibrated: the calibration holds no {metric} values, neither in its pairs nor in its "
            "cached scores." for metric, label in SECONDARY
            if calibrated and metric in decisions and decisions[metric]["pooled"] is not None
            and not decisions[metric]["n"]]


def _noise_rows(decisions: Mapping[str, Mapping], metrics: Sequence[tuple[str, str]]) -> list:
    rows = {POOLED: [], SEEDS: [], DIFF: [], NOISE: []}
    for metric, _ in metrics:
        found = decisions[metric]
        if found["pooled"] is None:
            for cells in rows.values():
                cells.append("n/a")
            continue
        rows[POOLED].append(f"{100 * found['pooled']:+.1f} / {band_text(found)}")
        rows[SEEDS].append(seeds_cell(found))
        rows[DIFF].append(diff_cell(found))
        rows[NOISE].append(verdict_text(found))
    return list(rows.items())


def _table(suites: dict, metrics: Sequence[tuple[str, str]], *, ci: bool, extra=()) -> list[str]:
    columns = [label for _, label in metrics]
    rows = [(name, [_cell(suite["headline"].get(metric), ci=ci) for metric, _ in metrics])
            for name, suite in suites.items()]
    rows.append((TALLY, [_tally([suite["headline"].get(metric) for suite in suites.values()]) for metric, _ in metrics]))
    rows += list(extra)
    first = max(len("Suite"), *(len(name) for name, _ in rows))
    widths = [max(len(column), *(len(cells[i]) for _, cells in rows)) for i, column in enumerate(columns)]

    def line(label, cells):
        return _SEPARATOR.join([label.ljust(first), *(cell.ljust(width) for cell, width in zip(cells, widths))]).rstrip()

    return [line("Suite", columns), *(line(name, cells) for name, cells in rows)]


def wrapped(lines: Sequence[str], width: int = WIDTH) -> list[str]:
    """Title lines wrapped at ``width`` columns between words, never inside one (a path stays whole), each
    continuation indented by two spaces; an empty line stays one."""
    found = []
    for line in lines:
        found += textwrap.wrap(line, width, subsequent_indent="  ", break_long_words=False,
                               break_on_hyphens=False) or [""]
    return found


def _grid(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [max(len(row[i]) for row in (header, *rows)) for i in range(len(header))]
    return [_SEPARATOR.join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in (header, *rows)]


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:+.1f}"


def candidate_cell(found: Mapping) -> str:
    """A decision's candidate seeds (``noise.decide``): mean±SD k in pp, or mean k1 for one seed."""
    spread = "" if found["candidate_sd"] is None else f"±{100 * found['candidate_sd']:.1f}"
    return f"{100 * found['pooled']:+.1f}{spread} k{found['k']}"


def _suite_z_block(comparisons: Mapping[int, Mapping], decisions: Mapping[str, Mapping]) -> list[str]:
    """``SUITES_HEADING`` and one row per suite of the comparisons, one z per headline metric (``suite_z_cell``)."""
    if not any(decisions[metric]["suite_z"] for metric, _ in HEADLINE):
        return ["Per-suite z: none, uncalibrated."]
    suites = list(dict.fromkeys(name for comparison in comparisons.values() for name in comparison["suites"]))
    rows = [[suite, *(suite_z_cell(decisions[metric], suite) for metric, _ in HEADLINE)] for suite in suites]
    return [SUITES_HEADING, *_grid(["Suite", *(label for _, label in HEADLINE)], rows)]


def render_pooled_headline(comparisons: Mapping[int, Mapping], *, decisions: Mapping[str, Mapping],
                           boundaries: tuple[int, str], title: Sequence[str] = ()) -> str:
    """The headline of a candidate's replicates (``replicates``), after the lines of ``title``: ``comparisons``, init
    seed -> its screen's comparison summary, each against the same baseline screen, and ``decisions``, their decisions
    (``noise.decide`` over the comparisons, made once by the caller). ``POOLED_HEADING``: per headline metric, the
    candidate's seeds (mean±SD k), the baseline seeds (mean±SD n), the diff, Welch's z, its df and p, the suites
    agreeing in sign, the null band (a reference) and the decision (a mixed one names its regressing suites);
    ``REGRESSIONS``; ``SEEDS_HEADING``: each seed's pooled Δ; ``SUITES_HEADING``: each suite's own z per metric
    (``suite_z_cell``); then the boundary tables of the screen of one seed, ``boundaries`` (its init seed and its
    name), labelled as that seed's alone."""
    seeds = list(comparisons)
    weighed = [(metric, label) for metric, label in (*HEADLINE, *SECONDARY) if metric in decisions]
    rows = []
    for metric, label in weighed:
        found = decisions[metric]
        if found["pooled"] is None:
            rows.append([label, *["n/a"] * 9])
            continue
        signs = f"{found['agree']}/{found['suites']}{'*' if found['basis'] == 'all' else ''}"
        z = "n/a" if found["z"] is None else f"{found['z']:+.1f}"
        df = "n/a" if found["df"] is None else f"{found['df']:.1f}"
        rows.append([label, candidate_cell(found), seeds_cell(found), _percent(found["diff"]), z, df,
                     p_text(found["p"]), signs, band_text(found), verdict_text(found)])
    per_seed = [[label, *(_percent(pooled(comparisons[seed], metric)) for seed in seeds)] for metric, label in weighed]
    seed, screen = boundaries
    lines = [*wrapped(title), *POOLED_LEGEND, "", POOLED_HEADING,
             *_grid(["Metric", "candidate", "baseline seeds", "diff", "z", "df", "p", "agree", "band", "noise"],
                    rows),
             regressions_line(decisions), *secondary_notes(decisions), "", SEEDS_HEADING,
             *_grid(["Metric", *(f"init {found}" for found in seeds)], per_seed), "",
             *_suite_z_block(comparisons, decisions), "",
             f"Boundary exact rates of init seed {seed} alone ({screen}): one seed against the baseline screen, "
             "not pooled."]
    for decoder, heading in DECODERS:
        lines += ["", heading, *_table(comparisons[seed]["suites"],
                                       [(f"{metric}.{decoder}", label) for metric, label in BOUNDARIES], ci=False)]
    return "\n".join(lines)


def render_headline(comparison: dict, *, decisions: Mapping[str, Mapping], title: Sequence[str] = ()) -> str:
    """The headline of a comparison summary (see the module docstring), after the lines of ``title``. ``decisions``:
    the comparison's decisions (``noise.decide([comparison], ...)``, made once by the caller, against the calibration's
    baseline seeds or uncalibrated)."""
    lines = [*wrapped(title), *LEGEND]
    if comparison.get("same_model"):
        lines.append("The candidate and the baseline are the same model: every Δ is 0.")
    if (comparison.get("code") or {}).get("same") is False:
        lines.append("Warning: different AlignAIR code for the two models, so every Δ holds the code change as well.")
    suites = comparison["suites"]
    lines += ["", *_table(suites, HEADLINE, ci=True, extra=_noise_rows(decisions, HEADLINE)),
              regressions_line(decisions), *secondary_lines(decisions)]
    for decoder, heading in DECODERS:
        lines += ["", heading,
                  *_table(suites, [(f"{metric}.{decoder}", label) for metric, label in BOUNDARIES], ci=False)]
    return "\n".join(lines)
