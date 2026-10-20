"""Markdown rendering of a run report: the plain summary dict ``workflow.summary.collect_run_summary`` builds.

Sections, in order: failed runs and oracle check failures (when there are any, since they qualify everything
below), the summary of verdicts, the per-suite tables, S10 against S1, S11 orientation parity, the genotype arms,
the excess alleles of suites with donors and speed. A section without data is left out.
"""
from __future__ import annotations

from alignair_assay.report.markdown import (
    MISSING,
    excess_note,
    format_cell,
    format_excess,
    format_value,
    headline_table,
    markdown_table,
)

RUN_REPORT_KIND = "alignair_assay.run_report"
RUN_REPORT_VERSION = "1"
# A forward-against-reverse comparison's verdicts (forward is tool a of the comparison).
_PARITY_VERDICTS = {"a_better": "forward better", "b_better": "reverse better", "tie_or_inconclusive": "tie"}


def run_report_markdown(summary: dict) -> str:
    """The run report of ``summary`` as markdown (see the module docstring for its sections)."""
    lines = [f"# AlignAIR assay run: {summary['a']} vs {summary['b']}", "", _configuration(summary)]
    for section in (_failed_runs, _oracle_failures, _summary, _suites, _s10_vs_s1, _s11_parity, _genotype_arms,
                    _excess_alleles, _speed):
        lines += section(summary)
    return "\n".join(lines) + "\n"


def _text(value) -> str:
    return MISSING if value is None else str(value)


def _configuration(summary: dict) -> str:
    config = summary["config"]
    digest = summary.get("config_hash")
    return (f"Configuration: split {config['split']}, seed {config['seed']}, scale {config['scale']}, reference "
            f"{config['reference']}; config hash {digest[:12] if digest else MISSING}. The oracle (the truth itself) "
            f"runs on every suite as the check that truth and scoring agree.")


def _failed_runs(summary: dict) -> list[str]:
    runs = summary.get("failed_runs") or []
    if not runs:
        return []
    rows = [[run["tool"], run["suite"], run["arm"], run["status"], _text(run.get("returncode")),
             _text(run.get("failed_donor")), _text(run.get("error")), _text(run.get("run_json"))] for run in runs]
    return ["", "## Failed runs", "",
            "These runs did not succeed, so they are not scored, and every comparison that needs one of them is left "
            "out of this report. Fix the cause and run `all` again: completed runs are kept and these are redone.",
            "", *markdown_table(["Tool", "Suite", "Arm", "Status", "Return code", "Failed donor", "Error",
                                 "Run record"], rows)]


def _oracle_text(oracle: dict) -> str:
    if oracle["status"] == "imperfect":
        return f"imperfect ({oracle['problems']} problems)"
    return oracle["status"]


def _oracle_failures(summary: dict) -> list[str]:
    failures = summary.get("oracle_failures") or []
    if not failures:
        return []
    rows = [[failure["suite"], failure["status"], str(failure.get("problems", 0)),
             "; ".join(f"{case} {metric} = {'nan' if value is None else value}"
                       for case, metric, value in failure.get("examples", [])[:3]) or MISSING]
            for failure in failures]
    return ["", "## Oracle check failures", "",
            "On every suite the oracle must score 1.0 on each compared metric, 0.0 on each lower-is-better one and "
            "never nan. A suite that fails has a truth defect: do not trust its scores.",
            "", *markdown_table(["Suite", "Status", "Problems", "First problems"], rows)]


def _summary(summary: dict) -> list[str]:
    a, b = summary["a"], summary["b"]
    rows = []
    for suite in summary["suites"]:
        for arm, data in suite["arms"].items():
            if data["compared"]:
                verdicts = data["verdicts"]
                counts = [str(verdicts["a_better"]), str(verdicts["b_better"]), str(verdicts["tie_or_inconclusive"])]
            else:
                counts = ["not compared", MISSING, MISSING]
            rows.append([suite["name"], arm, _text(suite["cases"]), _text(suite["excluded"]),
                         _oracle_text(suite["oracle"]), *counts])
    return ["", "## Summary", "",
            "Verdicts on each suite's primary metrics: Holm-corrected paired tests across every compared metric of "
            "the suite and arm.",
            "", *markdown_table(["Suite", "Arm", "Cases", "Excluded", "Oracle", f"{a} better", f"{b} better",
                                 "Tie or inconclusive"], rows)]


def _suites(summary: dict) -> list[str]:
    lines = ["", "## Suites"]
    for suite in summary["suites"]:
        lines += ["", f"### {suite['name']}: {suite['title']}", "",
                  f"{suite['purpose']} {_text(suite['cases'])} cases, {_text(suite['excluded'])} excluded; oracle "
                  f"{_oracle_text(suite['oracle'])}."]
        for arm, data in suite["arms"].items():
            if not data["compared"]:
                lines += ["", f"**{arm} arm.** Not compared: a run failed, is missing or could not be scored (see "
                              "Failed runs)."]
                continue
            table = headline_table({"a": summary["a"], "b": summary["b"], "comparison": data["rows"]},
                                   headline=suite["primary"])
            lines += ["", f"**{arm} arm** ([comparison]({data['link']}))", "", table.rstrip("\n")]
    return lines


def _s10_vs_s1(summary: dict) -> list[str]:
    data = summary.get("s10_vs_s1")
    if not data:
        return []
    settings, baseline = data["settings"], data["baseline"]
    lines = ["", "## S10 vs S1 (held-out settings)", "",
             f"These means are unpaired and informational (no test): S10's held-out settings against {baseline}, "
             "the same read design and SHM rate under the default distributions. Each mean shows its n and "
             "coverage; Δ = S10 − S1 is the difference of the means."]
    for tool, rows in data["tools"].items():
        table = [[row["metric"], format_cell(row["metric"], row["s1"]), format_cell(row["metric"], row["s10"]),
                  format_value(row["metric"], row["delta"], delta=True),
                  *(format_cell(row["metric"], row["by_setting"].get(setting)) for setting in settings)]
                 for row in rows]
        lines += ["", f"**{tool}**", "",
                  *markdown_table(["Metric", baseline, "S10", "Δ (S10 − S1)", *settings], table)]
    return lines


def _s11_parity(summary: dict) -> list[str]:
    data = summary.get("s11_parity")
    if not data:
        return []
    lines = ["", "## S11 orientation parity", "",
             "Each tool's forward reads against their reverse-complemented twins, paired by `groups[\"pair\"]`: exact "
             f"McNemar for 0/1 metrics, paired z otherwise, Holm across the tool's metrics at α = {data['alpha']}. "
             "Δ = reverse − forward."]
    for tool, parity in data["tools"].items():
        differ = parity["differ"]
        verdict = (f"differs on {len(differ)} of {parity['family_size']} metrics: {', '.join(differ)}" if differ
                   else "no metric differs")
        rows = [[row["metric"], format_value(row["metric"], row["forward"]),
                 format_value(row["metric"], row["reverse"]), format_value(row["metric"], row["delta"], delta=True),
                 str(row["n_paired"]), _PARITY_VERDICTS.get(row["verdict"], _text(row["verdict"]))]
                for row in parity["rows"]]
        lines += ["", f"**{tool}** ({parity['pairs']} pairs): {verdict}", "",
                  *markdown_table(["Metric", "Forward", "Reverse", "Δ (rc − fwd)", "n pairs", "Verdict"], rows)]
    return lines


def _genotype_arms(summary: dict) -> list[str]:
    suites = summary.get("genotype_arms") or []
    if not suites:
        return []
    tools = (summary["a"], summary["b"])
    lines = ["", "## Genotype arms", "",
             "Means over each arm's cases, with n and coverage: the shared arm gives every tool the full "
             "reference, the genotype arm restricts it to each donor's alleles. A dash: no value (the run failed or "
             "is missing, see Failed runs, or the tool covered no case)."]
    for suite in suites:
        arms = suite["arms"]
        rows = [[row["metric"], *(format_cell(row["metric"], row["values"].get(tool, {}).get(arm))
                                  for tool in tools for arm in arms)] for row in suite["rows"]]
        lines += ["", f"### {suite['suite']}", "",
                  *markdown_table(["Metric", *(f"{tool} {arm}" for tool in tools for arm in arms)], rows)]
    return lines


def _excess_alleles(summary: dict) -> list[str]:
    suites = summary.get("excess_alleles") or []
    if not suites:
        return []
    tables = [row["table"] for suite in suites for row in suite["rows"] if row["table"]]
    min_reads = ", ".join(str(k) for k in sorted({table["min_reads"] for table in tables})) or MISSING
    rows = [[suite["suite"], row["tool"], row["arm"],
             *(format_excess((row["table"] or {}).get(g), avoidable=avoidable)
               for g in ("v", "j") for avoidable in (False, True))]
            for suite in suites for row in suite["rows"]]
    return ["", "## Excess alleles", "",
            f"{excess_note(min_reads)} Per suite with donors and per run, both shares over the same n donor genes. "
            "The oracle calls each read's generative allele, so it never has an excess; the avoidable share leaves "
            "out the same calls the per-read genotype metrics excuse. A dash with n=0: no donor gene had enough "
            "reads; a dash alone: no value (the run failed or is missing, see Failed runs).",
            "", *markdown_table(["Suite", "Tool", "Arm", "V, all calls", "V, avoidable", "J, all calls",
                                 "J, avoidable"], rows)]


def _number(value, spec: str) -> str:
    return MISSING if value is None else format(value, spec)


def _speed(summary: dict) -> list[str]:
    speed = summary.get("speed")
    return speed_lines(speed) if speed else []


def _host(host: dict) -> str:
    """The measuring host (``workflow.speed.host_info``) in words; what it does not record is left out."""
    words = []
    if host.get("cpu"):
        words.append(host["cpu"])
    if host.get("logical_cpus") is not None:
        words.append(f"{host['logical_cpus']} logical CPUs")
    if host.get("ram_gb") is not None:
        words.append(f"{host['ram_gb']} GB RAM")
    if "gpu" in host:
        words.append(f"GPU {host['gpu'] or 'none'}")
    if "docker" in host:
        words.append(f"Docker {host['docker'] or 'not available'}")
    if host.get("kernel"):
        words.append(f"kernel {host['kernel']}")
    return ", ".join(words)


def speed_lines(speed: dict) -> list[str]:
    """The ``## Speed`` section of a run's speed measurements (``workflow.summary.speed_summary``, or a
    ``speed.json``): the host, how the protocol measures, one row per configuration (its measurements, then its
    threads, CPU list and device; a failed one shows dashes), the failures with their errors and the stale
    configurations (``stale_results``), named with why instead of shown. A stale file (``stale``) is named instead,
    with its config hash and the reason."""
    if speed.get("stale"):
        digest = speed.get("config_hash")
        reason = speed.get("reason") or ("measured for another configuration" if digest else
                                         "measured by the speed command, outside this configuration")
        return ["", "## Speed", "",
                f"speed/speed.json was {reason} (config hash {digest[:12] if digest else 'none'}), so it is not "
                "shown. Run all --speed with this configuration to measure it."]
    n_tiny, n_full = _text(speed.get("n_tiny")), _text(speed.get("n_full"))
    results = speed.get("results") or {}
    rows = [[name, _number(result.get("cold_start_s"), ".2f"), _number(result.get("steady_reads_per_s"), ".1f"),
             _number(result.get("median_full_s"), ".2f"),
             _number(None if result.get("peak_rss_kb") is None else result["peak_rss_kb"] / 1024, ".1f"),
             _text(result.get("threads")), _text(result.get("cpus")), _text(result.get("device"))]
            for name, result in results.items()]
    failures = [f"- {name} failed: {result['error']}" for name, result in results.items() if "error" in result]
    failures += [f"- {row['name']}: not shown, {row['reason']}; run all --speed to measure it again"
                 for row in speed.get("stale_results") or []]
    return ["", "## Speed", "",
            f"Host: {_host(speed.get('host') or {}) or MISSING}.", "",
            f"Each configuration annotated {n_tiny} and {n_full} reads evenly spaced across S1, "
            f"{_text(speed.get('repeats'))} times each, alternating, every run a fresh process. Cold start is the "
            f"median wall time of the {n_tiny}-read runs: starting, loading the model or databases and {n_tiny} "
            f"reads. Steady reads/s is ({n_full} − {n_tiny}) / (median {n_full}-read run − median {n_tiny}-read run), "
            "the rate once started (a dash when the full runs were not slower). A configuration runs on its threads, "
            "pinned to its CPUs. Peak RSS is the largest over the runs: the process's (GNU time) for IgBLAST and the "
            "local CLI, the container's cgroup memory.peak (resident memory plus page cache) for the AlignAIR "
            "image. The AlignAIR image timings include container start.",
            "", *markdown_table(["Configuration", "Cold start (s)", "Steady reads/s", "Median full run (s)",
                                 "Peak RSS (MB)", "Threads", "CPUs", "Device"], rows),
            *([""] + failures if failures else [])]
