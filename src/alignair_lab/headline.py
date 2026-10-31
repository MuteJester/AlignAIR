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
   baseline seeds;
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

from collections.abc import Mapping, Sequence

from alignair_lab.noise import MIN_SEEDS, MIXED, UNCALIBRATED, decide, pooled

HEADLINE = (("clone_key.exact", "clone_key.exact"), ("junction.nt_exact", "junction.nt_exact"), ("v.top1_in_set", "V top-1"),
            ("d.top1_in_set", "D top-1"), ("j.top1_in_set", "J top-1"))
BOUNDARIES = tuple((f"head.{gene}.{edge}_exact", f"{gene.upper()} {edge}") for gene in "vdj" for edge in ("start", "end"))
DECODERS = (("soft", "Boundary exact rates, the soft-argmax decoder:"),
            ("argmax", "Boundary exact rates, the position logits' argmax decoder (the product's):"))
MARKS = {"b_better": "+", "a_better": "-", "tie_or_inconclusive": "="}
LEGEND = ("Δ = candidate − baseline (pp) [95% CI], verdict (+ candidate better, - worse, = tie, ? too few clusters), "
          "reads fixed/broken.",
          "CIs resample reads and leave out run-to-run training noise, weighed below: pooled Δ / null band "
          "(lab-null-v1 screens; a reference);",
          "seeds: baseline seeds' Δs, mean±SD n; diff: pooled − their mean, Welch z, suites agreeing (*: all). "
          "Better: z>=2, no suite z<=-2.")
POOLED_LEGEND = (
    "candidate: each init seed's Δ against the baseline screen (read-weighted over the suites), mean±SD k; "
    "baseline seeds: the",
    "calibration's baseline screens' Δs against that screen, mean±SD n. diff: their difference; z: diff / "
    "sqrt(sc²/k + sb²/n), with",
    "sc = sb for one seed; agree: suites whose diff has its sign (*: none has enough clusters, so all; information); "
    "band: the",
    "largest Δ between baseline screens, a reference. Better beyond noise: z >= 2 and no suite regressing (its own z "
    "<= -2), else",
    "mixed, naming the suites; worse: z <= -2; a suite with too few pairs or clusters never counts. Uncalibrated "
    "without 3 seeds.")
POOLED_HEADING = "The candidate's init seeds against the calibration's baseline seeds (pp):"
SEEDS_HEADING = "Each init seed's pooled Δ against the baseline screen (pp):"
SUITES_HEADING = ("Per-suite z, the candidate's seeds against the baseline seeds in each suite (! regresses: z <= -2; "
                  "~ too few pairs or clusters):")
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
    if test["z"] is None:
        return "n/a"
    return f"{test['z']:+.1f}{'!' if suite in found['regressions'] else ''}{'~' if test['limited'] else ''}"


def band_text(found: Mapping) -> str:
    return "none" if found["band"] is None else f"{100 * found['band']:.1f}"


def _noise_rows(comparison: dict, metrics: Sequence[tuple[str, str]], calibration: Mapping | None) -> list:
    calibration = calibration or {}
    decisions = decide([comparison], calibration.get("seeds"), [metric for metric, _ in metrics],
                       band=calibration.get("band"))
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


def render_pooled_headline(comparisons: Mapping[int, Mapping], *, calibration: Mapping | None,
                           boundaries: tuple[int, str], title: Sequence[str] = ()) -> str:
    """The headline of a candidate's replicates (``replicates``), after the lines of ``title``: ``comparisons``, init
    seed -> its screen's comparison summary, each against the same baseline screen; ``calibration`` as for
    ``render_headline``. ``POOLED_HEADING``: per headline metric, the candidate's seeds (mean±SD k), the baseline
    seeds (mean±SD n), the diff, Welch's z, the suites agreeing in sign, the null band (a reference) and the decision
    (``noise.decide``; a mixed one names its regressing suites); ``SEEDS_HEADING``: each seed's pooled Δ;
    ``SUITES_HEADING``: each suite's own z per metric (``suite_z_cell``); then the boundary tables of the screen of
    one seed, ``boundaries`` (its init seed and its name), labelled as that seed's alone."""
    seeds = list(comparisons)
    calibration = calibration or {}
    decisions = decide([comparisons[seed] for seed in seeds], calibration.get("seeds"),
                       [metric for metric, _ in HEADLINE], band=calibration.get("band"))
    rows = []
    for metric, label in HEADLINE:
        found = decisions[metric]
        if found["pooled"] is None:
            rows.append([label, *["n/a"] * 7])
            continue
        signs = f"{found['agree']}/{found['suites']}{'*' if found['basis'] == 'all' else ''}"
        z = "n/a" if found["z"] is None else f"{found['z']:+.1f}"
        rows.append([label, candidate_cell(found), seeds_cell(found), _percent(found["diff"]), z, signs,
                     band_text(found), verdict_text(found)])
    per_seed = [[label, *(_percent(pooled(comparisons[seed], metric)) for seed in seeds)] for metric, label in HEADLINE]
    seed, screen = boundaries
    lines = [*title, *POOLED_LEGEND, "", POOLED_HEADING,
             *_grid(["Metric", "candidate", "baseline seeds", "diff", "z", "agree", "band", "noise"], rows), "",
             SEEDS_HEADING, *_grid(["Metric", *(f"init {found}" for found in seeds)], per_seed), "",
             *_suite_z_block(comparisons, decisions), "",
             f"Boundary exact rates of init seed {seed} alone ({screen}): one seed against the baseline screen, "
             "not pooled."]
    for decoder, heading in DECODERS:
        lines += ["", heading, *_table(comparisons[seed]["suites"],
                                       [(f"{metric}.{decoder}", label) for metric, label in BOUNDARIES], ci=False)]
    return "\n".join(lines)


def render_headline(comparison: dict, *, title: Sequence[str] = (), calibration: Mapping | None = None) -> str:
    """The headline of a comparison summary (see the module docstring), after the lines of ``title``.
    ``calibration``: the ``band`` (each metric's null band, ``noise.null_band``, Δs as fractions) and the ``seeds``
    (each baseline seed's values against the comparison's baseline screen, ``noise.seed_values``); None: uncalibrated,
    nothing marked."""
    lines = [*title, *LEGEND]
    if comparison.get("same_model"):
        lines.append("The candidate and the baseline are the same model: every Δ is 0.")
    if (comparison.get("code") or {}).get("same") is False:
        lines.append("Warning: different AlignAIR code for the two models, so every Δ holds the code change as well.")
    suites = comparison["suites"]
    lines += ["", *_table(suites, HEADLINE, ci=True, extra=_noise_rows(comparison, HEADLINE, calibration))]
    for decoder, heading in DECODERS:
        lines += ["", heading,
                  *_table(suites, [(f"{metric}.{decoder}", label) for metric, label in BOUNDARIES], ci=False)]
    return "\n".join(lines)
