"""The real-data report: the plain summary dict ``workflow.real.run_real`` writes (``real_summary.json``), rendered
as markdown (``real_report.md``), the page AlignAIR is judged against IgBLAST on real reads by.

**The summary page** comes first:

1. a three-line header: the donors, split, cap, date and reference; the data sources (each donor's run with its read
   quality, ``full`` or ``lite``, the genomic genotypes, the IUIS map, IGenotyper's gene coordinates and the SD3
   cohort); and the pinned images' digests, the IgBLAST binary's and the checkpoint's;
2. **R1 headline (shared arm)**: the bottom line per protocol (race, multiplex), led by the pre-declared primary, the
   paired R1 Δ of the AlignAIR image against IgBLAST, with the known_top1 Δ beside it (no exclusion touches it) and,
   per tool, the V reads its own call alone takes out of R1 with the documented known answers among them; then the
   pool over both protocols, which depends on their mix of reads; then that every CI is conditional on these donors
   and every other verdict exploratory. A table gives each AlignAIR run's paired Δ against IgBLAST (paired means, Δ,
   its 95% CI, the verdict and ``n paired (K clonotypes)``) per protocol, donor and pool; the components (unsupported
   allele, unsupported gene, no call) of the primary pair follow, then the one-line statistics note. Each run's own
   R1, whose denominator differs from tool to tool, is in the detail;
3. **Genotype arm**: R1 is 0 there by construction, so the arm's headline is known-answer accuracy (``known_top1``,
   ``known_gene_top1``), missing calls and the calls that move into uncaptured genes, shared → genotype per tool
   with the paired arm delta (informational), then the tools against each other on that arm;
4. **Flags**: at most ``MAX_FLAGS`` bullets, the most severe first, on what limits the numbers above.

**The detail** follows, every section of ``DETAIL_SECTIONS`` always present (one without data says so): it is
``report.real_detail``. Both read the formatting and scope helpers of ``report.real_format``.

A pooled scope is shown when two or more donors are scored; with one, that donor's numbers are the headline's. The
dict is ``report.build.clean``-ed first, so a NaN shows as a dash; this module reads plain dicts and imports nothing
of ``real``.
"""
from __future__ import annotations

from alignair_assay.report.build import clean
from alignair_assay.report.markdown import MISSING
from alignair_assay.report.real_detail import DETAIL_SECTIONS, detail_markdown  # noqa: F401 (the sections)
from alignair_assay.report.real_format import (
    COMPONENTS,
    HEADLINE,
    IMAGE_LABELS,
    KNOWN,
    KNOWN_YIELD_FLAG,
    NO_CALL,
    NO_TRUTH,
    PRIMARY,
    _arm_tools,
    _class_share,
    _delta,
    _donor_names,
    _evidence,
    _find,
    _int,
    _label,
    _listed,
    _mean,
    _nothing_scored,
    _pair_cell,
    _pair_header,
    _pairs,
    _pct,
    _pp,
    _protocol_scopes,
    _quality,
    _scopes,
    _scored_blocks,
    _short,
    _table,
    _tool_block,
    _verdict,
)

SUMMARY_SECTIONS = ("R1 headline (shared arm)", "Genotype arm", "Flags")
MAX_FLAGS = 8
# A donor where this share of a tool's shared-arm V calls, or more, is in genes without truth (not captured, or
# uncalled) is flagged.
UNCAPTURED_FLAG = 0.25
R1_NOTE = ("R1 (`real.v.unsupported`, lower is better): the share of a tool's top-1 V calls that the donor's genome "
           "does not carry, as an unsupported allele (the gene is carried, not this allele), an unsupported gene (a "
           "gene every capturing contig shows deleted) or no call; a call is supported when it equals a donor "
           "allele exactly or through IMGT position 318 (Peres et al. 2023). Each Δ (AlignAIR − igblast, 95% CI, "
           "verdict) is on the paired reads, those neither call excludes. Each run's own R1, over the reads its own "
           "call does not exclude (a different set for each tool), is in R1 per donor.")
BOTTOM_LEAD = ("The pre-declared primary is the paired shared-arm R1 Δ, {pair}, per protocol. Beside it: the "
               "known_top1 Δ on the documented known answers, which no exclusion touches, and per tool the V reads "
               "its own call alone takes out of R1 (a call into an excluded gene removes the read for both tools), "
               "with the documented known answers among them and how many of those it called wrong.")
POOL_NOTE = ("Pooled over both protocols (secondary: it depends on how many reads each protocol contributes, not on "
             "the tools alone)")
CONDITIONAL_NOTE = ("Every CI is conditional on these donors: they are fixed, not sampled, so it covers the variation "
                    "between clonotypes only. Verdicts outside the pre-declared primary are exploratory: Holm runs "
                    "within each comparison, not across the report.")
STATISTICS_NOTE = ("Statistics: paired tests clustered by clonotype (clustered McNemar with a cluster-robust 95% CI; "
                   "under 10 clonotypes no test: too few clonotypes); Holm within each comparison (one pair on one "
                   "scope, stratum or variant), not across the whole report; no variation: the tools agree on every "
                   "paired read (Δ and interval 0), so there is nothing to test.")
GENOTYPE_LEAD = ("R1 is 0 on this arm by construction, so the arm is judged on known answers, missing calls and the "
                 "calls that move into genes without truth (not captured, or not called).")
GENOTYPE_NOTE = ("Each tool may call only the donor's alleles in the genes the truth resolves, so every call R1 does "
                 "not exclude is supported, and only a missing call could count against it. Calls in genes without "
                 "truth (uncaptured, or spanned and not called) are excluded from R1; both shares are of all reads. "
                 "Each Δ is the tool's arm delta, genotype − shared on the same reads, and is informational.")


def real_report_markdown(summary: dict) -> str:
    """The real-data report of ``summary`` (``workflow.real``'s ``real_summary.json``) as markdown: the summary page
    (``SUMMARY_SECTIONS``), a rule, then the detail (``DETAIL_SECTIONS``). ``summary`` is not changed."""
    summary = clean(summary)
    scopes = _scopes(summary)
    split = (summary.get("config") or {}).get("split") or MISSING
    lines = [f"# AlignAIR real-data report: {', '.join(_donor_names(summary)) or 'no donor'} ({split})", "",
             *_header(summary)]
    summary_page = (_headline, _genotype_arm, _flags_section)
    for title, render in zip(SUMMARY_SECTIONS, summary_page):
        lines += ["", f"## {title}", "", *render(summary, scopes)]
    lines += ["", "---"]
    lines += detail_markdown(summary, scopes)
    return "\n".join(lines) + "\n"


def _header(summary: dict) -> list[str]:
    """The three header lines: donors and run; data sources; images and binaries."""
    config, provenance = summary.get("config") or {}, summary.get("provenance") or {}
    donors = summary.get("donors") or {}
    names = []
    for name in _donor_names(summary):
        block = donors.get(name) or {}
        stopped = (block.get("stopped") or {}).get("step") if block.get("status") == "stopped" else None
        names.append(f"{name} (stopped at {stopped})" if stopped else name)
    reference = provenance.get("reference") or {}
    created = summary.get("created_utc") or ""
    date = f"{created[:10]} {created[11:16]} UTC" if len(created) >= 16 else MISSING
    git = provenance.get("git") or {}
    mode = "; preprocessed only (`--stop-after preprocess`)" if config.get("stop_after") else ""
    first = (f"- Donors {', '.join(names) or MISSING} on the {config.get('split', MISSING)} split, cap "
             f"{_int(config.get('cap'))} reads per donor{mode}; run {date}; reference "
             f"{reference.get('name', config.get('reference', MISSING))} ({_short(reference.get('digest'))}); "
             f"config {_short(summary.get('config_hash'))}; code {_short(git.get('git_commit'), 8)}"
             f"{' (dirty)' if git.get('git_dirty') else ''}.")
    runs = []
    for name, block in donors.items():
        quality = _quality(summary, block)
        source = (block.get("fetch") or {}).get("source")
        runs.append(f"{name} {block.get('run', MISSING)} {'**lite**' if quality == 'lite' else quality}"
                    + (f" ({source.upper()})" if source else ""))
    samples = ", ".join(f"{name} {block['sample']}" for name, block in donors.items() if block.get("sample"))
    ogrdb = provenance.get("ogrdb") or next(((block.get("genotype") or {}).get("ogrdb") for block in donors.values()
                                             if (block.get("genotype") or {}).get("ogrdb")), None) or {}
    cohort = summary.get("sensitivity") or {}
    second = (f"- Data: PRJNA555323 reads {', '.join(runs) or MISSING}; genotypes from VDJbase genomic P25 "
              f"({samples or MISSING}) with IGenotyper {_short(provenance.get('igenotyper_commit'), 8)} gene "
              f"coordinates and IUIS names from OGRDB IGH_VDJ rev {ogrdb.get('release_version', MISSING)} "
              f"({ogrdb.get('release_date', MISSING)})")
    if cohort:
        second += (f"; SD3 cohort {_int((cohort.get('cohort') or {}).get('subjects'))} subjects (VDJbase API, OGRDB "
                   "rev 9, Rodriguez et al. 2023 SD3)")
    images = provenance.get("images") or {}
    shown = [f"{IMAGE_LABELS.get(name, name)} {_image(entry)}" for name, entry in images.items()]
    tools = provenance.get("tools") or {}
    binaries = []
    if tools.get("igblast"):
        binaries.append(f"igblastn {_short(tools['igblast'].get('igblastn_sha256'))}")
    if tools.get("alignair-api"):
        api = tools["alignair-api"]
        binaries.append(f"alignair-api checkpoint {_short(api.get('checkpoint_sha256'))} ({api.get('device', '?')}, "
                        f"torch {api.get('torch_version', '?')})")
    third = f"- Images, pinned by digest: {', '.join(shown) or MISSING}" + (f"; {'; '.join(binaries)}" if binaries
                                                                          else "") + "."
    return [first, second + ".", third]


def _image(entry) -> str:
    """A pinned image as ``<repository>:<tag>@sha256:<first 12>``."""
    ref = entry.get("image") if isinstance(entry, dict) else entry
    if not ref:
        return MISSING
    name, _, digest = str(ref).partition("@")
    return f"{name}@sha256:{digest.split(':')[-1][:12]}" if digest else name


def _headline(summary: dict, scopes: list[dict]) -> list[str]:
    if not scopes:
        return [_nothing_scored(summary)]
    pairs, protocols = _pairs(scopes, "shared"), _protocol_scopes(summary)
    primary = PRIMARY if PRIMARY in pairs else (pairs[0] if pairs else None)
    lines = [*_bottom_line(scopes, protocols, primary), ""]
    # a protocol of one donor is that donor's row
    shown = [*(scope for scope in protocols if len(scope["donors"]) > 1), *scopes]
    header = ["Scope", *(_pair_header(a, b) for a, b in pairs)]
    rows = [[scope["label"], *(_pair_cell(_find(scope, a, b, "shared"), HEADLINE, means=True) for a, b in pairs)]
            for scope in shown]
    lines += ["R1 Δ per scope (paired means, igblast → AlignAIR, then Δ):", "", *_table(header, rows), "", R1_NOTE]
    if primary is not None:
        lead = [*protocols, *([scopes[-1]] if scopes[-1]["block"] is None or not protocols else [])]
        components = [[label, *(_pair_cell(_find(scope, *primary, "shared"), metric, means=True) for scope in lead)]
                      for metric, label in COMPONENTS.items()]
        lines += ["", f"Components, {_pair_header(*primary)}:", "",
                  *_table(["Component", *(scope["label"] for scope in lead)], components)]
    return [*lines, "", STATISTICS_NOTE]


def _bottom_line(scopes: list[dict], protocols: list[dict], primary: tuple[str, str] | None) -> list[str]:
    """The bottom line: per protocol, the pre-declared primary (the paired R1 Δ of ``primary``), the known_top1 Δ and
    the reads each tool's call alone takes out of R1; then the pool over both protocols, which their mix of reads
    sets; then the note that every CI is conditional on these donors. Without per-protocol pools (a summary written
    before them), the last scope stands in."""
    if primary is None:
        return ["**Bottom line.** No pair was compared on the shared arm."]
    a, b = primary
    pair = _pair_header(a, b)
    lead = protocols or [scopes[-1]]
    rows = [[scope["label"], _pair_cell(_find(scope, a, b, "shared"), HEADLINE, means=True),
             _pair_cell(_find(scope, a, b, "shared"), KNOWN[0], means=True), _one_sided(_find(scope, a, b, "shared"))]
            for scope in lead]
    lines = [f"**Bottom line.** {BOTTOM_LEAD.format(pair=pair)}", "",
             *_table(["Protocol (donors)" if protocols else "Scope", f"R1 Δ, {pair} (primary)",
                      f"known_top1 Δ, {pair}", "V reads out by one call alone"], rows)]
    pooled = scopes[-1] if scopes[-1]["block"] is None else None
    if len(protocols) > 1 and pooled is not None:
        comparison = _find(pooled, a, b, "shared")
        lines += ["", f"{POOL_NOTE}: R1 {_pair_cell(comparison, HEADLINE, means=True)}; known_top1 "
                      f"{_pair_cell(comparison, KNOWN[0], means=True)}."]
    changed = _sl15_changes(lead, a, b)
    if changed is not None:
        lines += ["", "Clustered by SL15 instead of exact junctions, "
                      + (f"a verdict changes: {'; '.join(changed)} (see Sensitivity)." if changed
                         else "every verdict above holds (see Sensitivity).")]
    return [*lines, "", CONDITIONAL_NOTE]


def _sl15_changes(scopes: list[dict], a: str, b: str) -> list[str] | None:
    """The scopes and metrics (R1, known_top1) of pair ``a``-``b`` whose verdict differs between the exact and the
    SL15 clusterings, as ``<scope> <metric> <exact> → <sl15>``; None when no comparison holds both."""
    changes, seen = [], False
    for scope in scopes:
        clusterings = (_find(scope, a, b, "shared") or {}).get("clusterings") or {}
        exact = (clusterings.get("exact") or {}).get("comparison") or {}
        sl15 = (clusterings.get("sl15") or {}).get("comparison") or {}
        for metric, label in ((HEADLINE, "R1"), (KNOWN[0], "known_top1")):
            if metric in exact and metric in sl15:
                seen = True
                before, after = _verdict(exact[metric], a, b), _verdict(sl15[metric], a, b)
                if before != after:
                    changes.append(f"{scope['label']} {label} {before} → {after}")
    return changes if seen else None


def _one_sided(comparison: dict | None) -> str:
    """Per tool of a comparison, the V reads its own call alone takes out of R1, with the documented known answers
    among them and how many of those it called wrong."""
    sides = (comparison or {}).get("one_sided")
    if not sides or comparison.get("status", "compared") != "compared":
        return MISSING
    parts = []
    for side in ("a", "b"):
        counts = sides.get(side) or {}
        parts.append(f"{_label(comparison[side])} {_int(counts.get('reads'))} ({_int(counts.get('documented'))} "
                     f"documented known answers, {_int(counts.get('documented_wrong'))} of them called wrong)")
    return "; ".join(parts)


def _genotype_arm(summary: dict, scopes: list[dict]) -> list[str]:
    if not scopes:
        return [_nothing_scored(summary)]
    lead, tools = scopes[-1], _arm_tools(summary, "genotype")
    if not tools:
        return ["No tool ran on the genotype arm."]
    deltas = {delta["tool"]: delta.get("overall") or {} for delta in lead["arm_deltas"]}
    rows = []
    for tool in tools:
        shared, genotype = _tool_block(lead, "shared", tool), _tool_block(lead, "genotype", tool)
        known = [_arm_known(shared, genotype, (deltas.get(tool) or {}).get(metric), metric) for metric in KNOWN]
        rows.append([_label(tool), *known, _arm_share(shared, genotype, NO_CALL),
                     _arm_share(shared, genotype, *NO_TRUTH, delta=True),
                     f"{_pct(_mean(shared, HEADLINE))} → {_pct(_mean(genotype, HEADLINE))} (0 by construction)"])
    header = ["Tool", "known_top1, shared → genotype", "known_gene_top1, shared → genotype",
              "No V call, of all reads", "Calls in genes without truth, of all reads", "R1, shared → genotype"]
    lines = [f"{GENOTYPE_LEAD} Scope: {lead['words']}.", "", *_table(header, rows), "", GENOTYPE_NOTE]
    for a, b in _pairs([lead], "genotype"):
        comparison = _find(lead, a, b, "genotype")
        cells = " · ".join(f"{metric.removeprefix('real.v.')} {_pair_cell(comparison, metric)}" for metric in KNOWN)
        lines += ["", f"{_pair_header(a, b)} on the genotype arm: {cells}."]
    return lines


def _arm_known(shared: dict | None, genotype: dict | None, entry: dict | None, metric: str) -> str:
    text = f"{_pct(_mean(shared, metric))} → {_pct(_mean(genotype, metric))}"
    if entry and entry.get("delta") is not None:
        text += f" (Δ {_delta(entry, 'shared', 'genotype', verdict=False)})"
    return text


def _arm_share(shared: dict | None, genotype: dict | None, *names: str, delta: bool = False) -> str:
    before, after = _class_share(shared, *names), _class_share(genotype, *names)
    text = f"{_pct(before)} → {_pct(after)}"
    return text + (f" ({_pp(after - before)})" if delta and before is not None and after is not None else "")


def _flags_section(summary: dict, scopes: list[dict]) -> list[str]:
    flags = _flags(summary, scopes)
    return [f"- {flag}" for flag in flags] if flags else ["None: nothing qualifies the numbers above."]


def _flags(summary: dict, scopes: list[dict]) -> list[str]:
    """The flags, the most severe first; at most ``MAX_FLAGS``, the last counting those left out."""
    donors = summary.get("donors") or {}
    flags = []
    if summary.get("oracle_failures"):
        flags.append(_listed("Truth self-check failed (a documented known answer's own allele is not right and "
                             "supported: the truth contradicts itself, so do not trust these donors' R1)",
                             list(summary["oracle_failures"])))
    stopped = summary.get("stopped_donors") or []
    if stopped:
        items = []
        for entry in stopped:
            evidence = ((donors.get(entry["donor"]) or {}).get("stopped") or {}).get("evidence")
            items.append(f"{entry['donor']} stopped at {entry['step']}: {entry['reason']}"
                         + (f" ({_evidence(evidence)})" if evidence else ""))
        flags.append(_listed("Donors stopped, not scored", items))
    failed = summary.get("failed_runs") or []
    if failed:
        flags.append(_listed("Failed runs, not scored (their pairs are not compared)",
                             [f"{run['tool']} on {run['donor']} {run['arm']}: {run.get('error') or MISSING}"
                              + (f" ({run['note']})" if run.get("note") else "") for run in failed]))
    gaps = [f"{name} {gene} ({_int(entry['clonotypes'])} clonotypes)"
            for name, block in donors.items() for gene in ((block.get("truth_health") or {}).get("flagged") or [])
            for entry in (block["truth_health"].get("genes") or []) if entry.get("gene") == gene]
    if gaps:
        flags.append(_listed("Truth health: carried genes whose reads exactly match, in many clonotypes, an allele the "
                             "genotype lacks (an allele VDJbase missed, or a haplotype it did not resolve; see Truth "
                             "health)", gaps))
    if summary.get("oracle_unchecked"):
        flags.append(_listed("Truth self-check unchecked, a warning (no documented known answer is scored: there "
                             "is none, or every one is excluded, e.g. not_captured)",
                             list(summary["oracle_unchecked"])))
    lite = [f"{name} {block.get('run', MISSING)}" for name, block in donors.items()
            if _quality(summary, block) == "lite"]
    if lite:
        flags.append(_listed("SRA-Lite reads (no per-base quality, so the quality filter did not run)", lite))
    low = []
    for name, block in donors.items():
        truth = block.get("truth") or {}
        share = truth.get("known_share")
        if truth and truth.get("low_known_yield", share is not None and share < KNOWN_YIELD_FLAG):
            low.append(f"{name} {_pct(share)} ({_int(truth.get('known'))} of {_int(truth.get('reads'))} reads)")
    if low:
        flags.append(_listed(f"Known-answer yield below {KNOWN_YIELD_FLAG:.0%} (known_top1 rests on few reads)", low))
    tools = _arm_tools(summary, "shared")
    uncaptured = []
    for name, block in _scored_blocks(summary):
        shares = [(tool, share) for tool in tools
                  if (share := _class_share((block["tools"].get("shared") or {}).get(tool), *NO_TRUTH)) is not None]
        if shares and max(share for _, share in shares) >= UNCAPTURED_FLAG:
            uncaptured.append(f"{name} " + ", ".join(f"{_label(tool)} {_pct(share)}" for tool, share in shares))
    if uncaptured:
        flags.append(_listed(f"At least {UNCAPTURED_FLAG:.0%} of a tool's shared-arm V calls are to genes without "
                             "truth (excluded:not_captured, the donor's contigs miss the gene, or excluded:uncalled, "
                             "they hold it and the genotype does not call it), so R1 rests on the other reads",
                             uncaptured))
    over = [f"{name} {_pct(acceptance.get('failure_rate'))}" for name, block in donors.items()
            if (acceptance := (block.get("preprocess") or {}).get("assembly_acceptance") or {}).get("override")
            and acceptance.get("within_line") is False]
    if over:
        flags.append(_listed("Assembly loss over the acceptance line, let through by --accept-assembly-loss", over))
    airr = [f"{tool} on {name} {arm} {_int(assembly['failed_assemblies'])} of {_int(assembly.get('written_records'))}"
            for name, block in donors.items() for arm, runs in (block.get("runs") or {}).items()
            for tool, run in runs.items()
            if (assembly := run.get("assembly") or {}).get("failed_assemblies")]
    if airr:
        flags.append(_listed("AlignAIR's AIRR assembly failed on some reads (see AlignAIR's AIRR output)", airr))
    few = [f"{scope['label']} {_pair_header(c['a'], c['b'])}" for scope in scopes for c in scope["comparisons"]
           if c.get("arm") == "shared" and ((c.get("overall") or {}).get(HEADLINE) or {}).get("test")
           == "too_few_clusters"]
    if few:
        flags.append(_listed("Too few clonotypes for an R1 verdict (under 10: not tested)", few))
    if len(flags) > MAX_FLAGS:
        left = len(flags) - (MAX_FLAGS - 1)
        flags = [*flags[:MAX_FLAGS - 1], f"And {left} more flags, less severe: see the detail sections."]
    return flags
