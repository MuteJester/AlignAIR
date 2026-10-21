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

**The detail** follows (``DETAIL_SECTIONS``), every section always present (one without data says so): R1 per donor
with its components and paired means; exclusions per tool and reason, what each reason means and the invariant;
known answers (the yield per donor, flagged below ``KNOWN_YIELD_FLAG``, and the accuracy per tool and arm); where the
calls of novel-allele reads go; the two sensitivity variants beside the primary; D and J; protocol × isotype strata;
preprocessing (the classifier's evidence, the steps, the assembly's failure rate against its line, its failures by
kind and the deciles of the pairs that passed and failed); AlignAIR's AIRR output (its assembly counts, the image's
gate, and each tool's rows the AIRR reader flags); the truth self-check; failed runs and refusals; and the provenance
in full.

A pooled scope is shown when two or more donors are scored; with one, that donor's numbers are the headline's. The
dict is ``report.build.clean``-ed first, so a NaN shows as a dash; this module reads plain dicts and imports nothing
of ``real``.
"""
from __future__ import annotations

from collections import Counter

from alignair_assay.report.build import clean
from alignair_assay.report.markdown import MISSING, markdown_table

SUMMARY_SECTIONS = ("R1 headline (shared arm)", "Genotype arm", "Flags")
DETAIL_SECTIONS = ("R1 per donor", "Exclusions", "Known answers", "Novel-allele reads", "Sensitivity",
                   "D and J (informational)", "Stratification: protocol × isotype", "Preprocessing",
                   "AlignAIR's AIRR output", "Truth self-check", "Failed runs and refusals", "Provenance")
MAX_FLAGS = 8
# Items named per flag before the rest are counted.
FLAG_ITEMS = 4
# A donor with fewer documented known answers than this share of its reads is flagged. The summary's own
# ``low_known_yield`` (``workflow.real.KNOWN_YIELD_FLAG``) decides; this is used only where it is missing.
KNOWN_YIELD_FLAG = 0.05
# A donor where this share of a tool's shared-arm V calls, or more, is in genes without truth (not captured, or
# uncalled) is flagged.
UNCAPTURED_FLAG = 0.25
HEADLINE = "real.v.unsupported"
# The pre-declared primary: the paired shared-arm R1 of the AlignAIR image against IgBLAST, per protocol.
PRIMARY = ("igblast", "alignair")
COMPONENTS = {"real.v.unsupported_allele": "unsupported allele", "real.v.unsupported_gene": "unsupported gene",
              "real.v.no_call": "no call"}
KNOWN = ("real.v.known_top1", "real.v.known_gene_top1")
ANY_LISTED = "real.v.any_listed_supported"
# The tie-aware metrics (informational): what the read's observed V window can tell apart.
UNSUPPORTED_OBSERVABLE = "real.v.unsupported_observable"
KNOWN_OBSERVABLE = "real.v.known_top1_observable"
# D and J (informational): each metric and the run's field of its classes. Named in full: a metric name built from
# pieces would read as a name suffix to the registry's check (tests/alignair_assay/real/test_metrics.py).
D_AND_J = (("real.d.unsupported", "d_classes"), ("real.j.unsupported", "j_classes"))
ARMS = ("shared", "genotype")
TOOL_ORDER = ("igblast", "alignair", "alignair-api")
TOOL_LABELS = {"alignair": "alignair (image)"}
IMAGE_LABELS = {"sra_tools": "sra-tools", "samtools": "samtools", "immcantation": "Immcantation",
                "alignair": "AlignAIR"}
NOT_CAPTURED = "excluded:not_captured"
UNCALLED = "excluded:uncalled"
# The exclusions of a call to a gene the truth says nothing about: its region is not captured, or it is not called.
NO_TRUTH = (NOT_CAPTURED, UNCALLED)
NO_CALL = "no_call"
DECILES = ("p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90")
# The exclusions, in the order a call's genes are checked (``real.metrics.EXCLUSIONS``), and what each means.
EXCLUSION_NOTES = {
    "excluded:dropped_truth": "D only: the gene's only genomic calls were dropped by the D-call filter, so its truth "
                              "is unknown, not absent.",
    "excluded:not_captured": "the donor's genomic contigs do not resolve the gene's region, so the call may be an "
                             "allele the long reads missed.",
    "excluded:uncalled": "the contigs span the gene and hold sequence there, but VDJbase's genotype calls no allele "
                         "of it: a calling gap, not a gene the donor lacks. The price: a tool's real error into such a "
                         "gene leaves R1 too (e.g. a call to IGHV4-30-2 on a read that exactly matches IGHV4-59).",
    "excluded:unmapped": "a guard: the call has no IUIS gene, labelled or inferred (none on HUMAN_IGH_OGRDB).",
    "excluded:novel_gene": "the donor carries an allele of the gene that the shared reference lacks, which a call of "
                           "a carried shared allele could hide (not applied to a documented known answer).",
    "excluded:nonfunctional": "the gene is carried only as nonfunctional alleles, so its truth is in doubt (Mikelov "
                              "et al. 2024 leave such alleles out).",
    "excluded:beyond_318": "a functional donor allele of the gene has no usable IMGT-gapped form, so no call can be "
                           "compared with it through IMGT 318, past which rearranged reads cannot tell alleles "
                           "apart (Peres et al. 2023).",
    "excluded:half_resolved": "the gene is carried, but an allele on one haplotype is unknown: its region is not "
                              "spanned, or spanned but not called (a contig there holds a sequence the genotype does "
                              "not call), so that allele could be any call.",
    "excluded:low_support": "in the gene-support variant only: the V gene is below Mikelov et al. 2024's thresholds "
                            "among the donor's known answers (see Sensitivity).",
}
ASSEMBLY_KINDS = {"mismatches": "mismatches", "not_significant": "not significant", "no_overlap": "no overlap"}

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
# The clusterings the pre-declared primary is compared under: the primary clonotypes, the candidate for P3b, a stress.
CLUSTERINGS = {"exact": "exact junction (primary)", "sl15": "SL15",
               "junction_length": "donor × junction length (stress)"}
CLUSTERING_NOTE = ("Clustering: the pre-declared primary pair's R1 and known_top1 compared under three clusterings of "
                   "the same paired reads, each its own Holm family of those two metrics: exact junctions (the primary "
                   "clonotypes in P3a), SL15 (single linkage at 15% or less nt Hamming among a donor's junctions of "
                   "one length; the candidate primary for P3b) and, as a stress test, one cluster per donor and "
                   "junction length.")
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
# The words for a verdict that is not significant, in the tables and the bottom line alike; a comparison under 10
# clonotypes is not tested, and one whose paired differences are all 0 has nothing to test.
NOT_SIGNIFICANT = "not significant"
TOO_FEW_CLONOTYPES = "too few clonotypes"
NO_VARIATION = "no variation"
INVARIANT = ("Whether a call is excluded never depends on whether it is correct: a call is excluded by its own IUIS "
             "genes and the read's tool-neutral truth alone, so a right call and a wrong call to the same genes are "
             "excluded alike; a call into an excluded gene still takes the read out of R1 for both tools. A read "
             "enters a paired comparison only when neither tool's call is excluded, so a tool's errors into excluded "
             "genes leave the comparison with their reads. The counts below are each run's own, as a share of all "
             "its reads, and each pair's reads that one call alone takes out follow, so a tool steering reads into "
             "excluded genes stays visible.")
AIRR_NOTE = ("AIRR rows the assay's reader flags (informational): rows with a field that disagrees with the row's "
             "coordinates, out of the rows written. AlignAIR writes a read's leading unaligned bases as an `I` "
             "(insertion) in its CIGARs where the AIRR standard has a soft clip (`S`), so most of its rows are "
             "flagged (on the AlignAIR fix-list). R1 reads only the calls, so this does not change R1.")


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
    detail = (_r1_detail, _exclusions, _known_answers, _novel_reads, _sensitivity, _d_and_j, _strata,
              _preprocessing, _airr_output, _self_check, _failures, _provenance)
    for title, render in zip(DETAIL_SECTIONS, detail):
        lines += ["", f"## {title}", "", *render(summary, scopes)]
    return "\n".join(lines) + "\n"


# --- formatting --------------------------------------------------------------------------------------------------

def _pct(value) -> str:
    return MISSING if value is None else f"{100 * value:.1f}%"


def _pp(value) -> str:
    return MISSING if value is None else f"{100 * value:+.1f} pp"


def _ci(ci) -> str:
    if not ci or ci[0] is None or ci[1] is None:
        return f"[{MISSING}]"
    return f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"


def _int(value) -> str:
    return MISSING if value is None else f"{value:,}" if isinstance(value, int) else str(value)


def _short(text, n: int = 12) -> str:
    return MISSING if not text else str(text)[:n]


def _scalar(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return _int(value)


def _label(tool: str) -> str:
    return TOOL_LABELS.get(tool, tool)


def _tool_key(tool: str) -> tuple:
    return (TOOL_ORDER.index(tool) if tool in TOOL_ORDER else len(TOOL_ORDER), tool)


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    """A markdown table whose cells hold no line break."""
    return markdown_table(header, [[str(cell).replace("\n", " ") for cell in row] for row in rows])


def _listed(head: str, items: list[str]) -> str:
    shown = items[:FLAG_ITEMS]
    more = f"; and {len(items) - FLAG_ITEMS} more" if len(items) > FLAG_ITEMS else ""
    return f"{head}: {'; '.join(shown)}{more}."


def _evidence(evidence) -> str:
    """Evidence of a stop (a classifier's fractions, an assembly's acceptance) in words: its reason, then its other
    plain fields."""
    if not isinstance(evidence, dict):
        return MISSING if evidence is None else str(evidence)
    parts = [str(evidence["reason"])] if evidence.get("reason") else []
    parts += [f"{key} {_scalar(value)}" for key, value in evidence.items()
              if key != "reason" and not isinstance(value, (dict, list))]
    return "; ".join(parts) or MISSING


# --- what is scored ----------------------------------------------------------------------------------------------

def _scopes(summary: dict) -> list[dict]:
    """Each scored donor, then the pooled donors when two or more are scored: ``label``, ``words`` (for prose),
    ``tools`` (arm -> tool -> its own R1 block), ``comparisons``, ``arm_deltas`` and the donor's ``block``."""
    scopes = [{"label": name, "words": name, "tools": block["tools"], "comparisons": block.get("comparisons") or [],
               "arm_deltas": block.get("arm_deltas") or [], "block": block}
              for name, block in (summary.get("donors") or {}).items() if block.get("tools")]
    pooled = summary.get("pooled") or {}
    donors = pooled.get("donors") or []
    if len(donors) > 1:
        scopes.append({"label": f"Pooled ({len(donors)} donors)", "words": f"pooled over {len(donors)} donors",
                       "tools": pooled.get("tools") or {}, "comparisons": pooled.get("comparisons") or [],
                       "arm_deltas": pooled.get("arm_deltas") or [], "block": None})
    return scopes


def _protocol_scopes(summary: dict) -> list[dict]:
    """The comparisons pooled over each protocol's donors (``pooled["protocols"]``), as scopes, in protocol order:
    where the bottom line reads the pre-declared primary."""
    protocols = (summary.get("pooled") or {}).get("protocols") or {}
    return [{"label": f"{protocol} ({', '.join(entry.get('donors') or [])})", "words": f"{protocol} reads",
             "tools": {}, "comparisons": entry.get("comparisons") or [], "arm_deltas": [], "block": None,
             "donors": entry.get("donors") or []}
            for protocol, entry in protocols.items()]


def _donor_names(summary: dict) -> list[str]:
    return list(summary.get("donors") or {}) or list((summary.get("config") or {}).get("donors") or [])


def _scored_blocks(summary: dict) -> list[tuple[str, dict]]:
    return [(name, block) for name, block in (summary.get("donors") or {}).items() if block.get("tools")]


def _arm_tools(summary: dict, arm: str) -> list[str]:
    """The tools that ran on ``arm`` for any donor, in display order."""
    seen = {tool for block in (summary.get("donors") or {}).values()
            for tool in ((block.get("runs") or {}).get(arm) or {})}
    seen |= set((((summary.get("pooled") or {}).get("tools") or {}).get(arm)) or {})
    return sorted(seen, key=_tool_key)


def _pairs(scopes: list[dict], arm: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for scope in scopes:
        for comparison in scope["comparisons"]:
            if comparison.get("arm") == arm and (comparison["a"], comparison["b"]) not in pairs:
                pairs.append((comparison["a"], comparison["b"]))
    return pairs


def _find(scope: dict, a: str, b: str, arm: str) -> dict | None:
    return next((c for c in scope["comparisons"] if (c["a"], c["b"], c.get("arm")) == (a, b, arm)), None)


def _tool_block(scope: dict, arm: str, tool: str) -> dict | None:
    return (scope["tools"].get(arm) or {}).get(tool)


def _mean(block: dict | None, metric: str):
    return (((block or {}).get("summary") or {}).get(metric) or {}).get("mean")


def _own(scope: dict, arm: str, tool: str, metric: str) -> str:
    """A run's own value of ``metric`` with its n; a dash without the run, ``— (n=0)`` when no read is scored."""
    entry = ((_tool_block(scope, arm, tool) or {}).get("summary") or {}).get(metric)
    if not entry:
        return MISSING
    if entry.get("mean") is None:
        return f"{MISSING} (n={entry.get('n', 0)})"
    return f"{_pct(entry['mean'])} (n={entry['n']})"


def _class_share(block: dict | None, *names: str) -> float | None:
    """The share of a run's reads whose V class is one of ``names``."""
    classes = (block or {}).get("v_classes") or {}
    total = sum(classes.values())
    return sum(classes.get(name, 0) for name in names) / total if total else None


def _quality(summary: dict, block: dict) -> str:
    quality = (block.get("fetch") or {}).get("quality")
    if quality is None:
        quality = ((summary.get("provenance") or {}).get("fetch_quality") or {}).get(block.get("run"))
    return quality or "unknown"


def _nothing_scored(summary: dict) -> str:
    if (summary.get("config") or {}).get("stop_after"):
        return ("Nothing was scored: the run stopped after preprocessing (`--stop-after preprocess`); see "
                "Preprocessing for each donor's classification and yields.")
    return "No donor was scored: see Failed runs and refusals."


# --- comparison cells --------------------------------------------------------------------------------------------

def _no_variation(entry: dict) -> bool:
    """Every paired difference is 0 (both tools at 0% or 100%, or agreeing read by read): Δ 0 in a [0, 0] interval."""
    ci = entry.get("ci95") or (None, None)
    return entry.get("delta") == 0 and ci[0] == 0 and ci[1] == 0


def _verdict(entry: dict, a: str, b: str) -> str:
    """``<tool> better`` when significant; otherwise ``no variation`` (nothing to test), ``too few clonotypes`` (under
    10: no test; the paired reads hold at least the clonotypes, so too few pairs is too few clonotypes) or ``not
    significant``."""
    better = {"a_better": f"{_label(a)} better", "b_better": f"{_label(b)} better"}.get(entry.get("verdict"))
    if better:
        return better
    if _no_variation(entry):
        return NO_VARIATION
    return TOO_FEW_CLONOTYPES if entry.get("test") in ("too_few_clusters", "too_few_pairs") else NOT_SIGNIFICANT


def _paired(entry: dict) -> str:
    n, k = entry.get("n_paired"), entry.get("n_clusters")
    return f"{n} paired" if k is None else f"{n} paired ({k} clonotypes)"


def _delta(entry: dict, a: str, b: str, *, means: bool = False, verdict: bool = True) -> str:
    """A paired entry as ``Δ [CI], verdict; n paired (K clonotypes)``, Δ and verdict bold when significant, led by
    the paired means (``a → b``) with ``means``; without ``verdict`` (an informational delta) none is shown."""
    value, said = _pp(entry["delta"]), _verdict(entry, a, b)
    if verdict and entry.get("verdict") in ("a_better", "b_better"):
        value, said = f"**{value}**", f"**{said}**"
    text = f"{value} {_ci(entry.get('ci95'))}" + (f", {said}; " if verdict else ", ") + _paired(entry)
    if means:
        text = f"{_pct(entry.get('mean_a_paired'))} → {_pct(entry.get('mean_b_paired'))}: {text}"
    return text


def _pair_cell(comparison: dict | None, metric: str, *, entries: dict | None = None, means: bool = False) -> str:
    """A comparison's cell for ``metric``, from its overall entries or ``entries`` (a stratum, a variant): not
    compared with its reason, or no paired read when every read is excluded."""
    if comparison is None:
        return MISSING
    if comparison.get("status", "compared") != "compared":
        return f"not compared: {comparison.get('reason') or 'a run did not succeed'}"
    entry = ((comparison.get("overall") or {}) if entries is None else entries).get(metric)
    if entry is None or entry.get("delta") is None:
        if entries is not None:
            return "no paired read"
        excluded = (comparison.get("excluded") or {}).get("v")
        return f"no paired read ({_int(excluded)} of {_int(comparison.get('n_reads'))} V-excluded by either call)"
    return _delta(entry, comparison["a"], comparison["b"], means=means)


def _pair_header(a: str, b: str) -> str:
    return f"{_label(b)} − {_label(a)}"


# --- the summary page --------------------------------------------------------------------------------------------

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


# --- the detail --------------------------------------------------------------------------------------------------

def _r1_detail(summary: dict, scopes: list[dict]) -> list[str]:
    if not scopes:
        return [_nothing_scored(summary)]
    tools, pairs = _arm_tools(summary, "shared"), _pairs(scopes, "shared")
    lines = ["The shared arm's R1 and its components per scope: each run's own value (n), and per pair the paired "
             "means (igblast → AlignAIR), then Δ, its 95% CI, the verdict and the paired count with its clonotypes. "
             "Consensus-weighted: each run's own R1 with every sequence weighted by the reads behind it. Any listed "
             "allele supported (informational, paired means): some V allele the tool lists is supported, so a tie "
             "list hides a supported allele behind an unsupported top-1. Unsupported, observably (informational, "
             "paired means): a call not supported whose germline window over the read's observed V (its first "
             "observed base to codon 104, its Ns left out) equals no donor allele's, so the read itself shows the "
             "error; a call the read cannot tell from a donor allele is a tie, not counted."]
    header = ["Metric", *map(_label, tools), *(_pair_header(a, b) for a, b in pairs)]
    for scope in scopes:
        rows = [[COMPONENTS.get(name, "R1 (unsupported)"), *(_own(scope, "shared", tool, name) for tool in tools),
                 *(_pair_cell(_find(scope, a, b, "shared"), name, means=True) for a, b in pairs)]
                for name in (HEADLINE, *COMPONENTS)]
        weighted = []
        for tool in tools:
            entry = (_tool_block(scope, "shared", tool) or {}).get("consensus_weighted")
            weighted.append(MISSING if not entry else
                            f"{_pct(entry.get(HEADLINE))} ({_int(entry.get('weight'))} reads)")
        rows.append(["consensus-weighted R1", *weighted, *(MISSING for _ in pairs)])
        rows.append(["any listed allele supported", *(_own(scope, "shared", tool, ANY_LISTED) for tool in tools),
                     *(_informational(_find(scope, a, b, "shared"), ANY_LISTED) for a, b in pairs)])
        rows.append(["unsupported, observably (informational)", *(MISSING for _ in tools),
                     *(_informational(_find(scope, a, b, "shared"), UNSUPPORTED_OBSERVABLE) for a, b in pairs)])
        rows.append(["V reads out of the pair", *(MISSING for _ in tools),
                     *(_reads_out(_find(scope, a, b, "shared")) for a, b in pairs)])
        lines += ["", f"**{scope['label']}**", "", *_table(header, rows)]
    return lines


def _informational(comparison: dict | None, metric: str) -> str:
    entry = ((comparison or {}).get("informational") or {}).get(metric)
    if not entry:
        return MISSING
    a, b = entry.get("a") or {}, entry.get("b") or {}
    return f"{_pct(a.get('mean'))} → {_pct(b.get('mean'))} (n={a.get('n', MISSING)})"


def _reads_out(comparison: dict | None) -> str:
    if not comparison or comparison.get("status", "compared") != "compared":
        return MISSING
    return f"{_int((comparison.get('excluded') or {}).get('v'))} of {_int(comparison.get('n_reads'))}"


def _exclusions(summary: dict, scopes: list[dict]) -> list[str]:
    lines = [INVARIANT, "", "What each reason means (a call is excluded by the first that applies to any of its "
                            "genes, in this order):", "",
             *(f"- `{reason}`: {note}" for reason, note in EXCLUSION_NOTES.items())]
    if not scopes:
        return [*lines, "", _nothing_scored(summary)]
    columns = [(tool, arm) for arm in ARMS for tool in _arm_tools(summary, arm)]
    header = ["Reason", *(f"{_label(tool)} {arm}" for tool, arm in columns)]
    for scope in scopes:
        blocks = [_tool_block(scope, arm, tool) for tool, arm in columns]
        reasons = [reason for reason in EXCLUSION_NOTES
                   if any(((block or {}).get("v_classes") or {}).get(reason) for block in blocks)]
        rows = [[reason, *(_count_share(block, reason) for block in blocks)] for reason in reasons]
        rows.append(["not excluded (scored)", *(_count_share(block, None) for block in blocks)])
        out = [f"{_pair_header(c['a'], c['b'])} {c.get('arm')} {_reads_out(c)}" for c in scope["comparisons"]
               if c.get("status", "compared") == "compared"]
        lines += ["", f"**{scope['label']}**", "", *_table(header, rows)]
        if out:
            lines += ["", f"V reads out of each pair (either call excluded): {'; '.join(out)}."]
        alone = [f"{_pair_header(c['a'], c['b'])} {c.get('arm')}: {_one_sided_reasons(c)}"
                 for c in scope["comparisons"] if c.get("status", "compared") == "compared" and c.get("one_sided")]
        if alone:
            lines += ["", f"Of those, the reads one call alone takes out, by reason: {'; '.join(alone)}."]
    return [*lines, "", *_truth_gaps(summary)]


def _truth_gaps(summary: dict) -> list[str]:
    """Per donor, where its genomic truth stops: the genes its contigs hold and its genotype does not call
    (``excluded:uncalled``), each with what a capturing contig holds there; the genes deleted on every capturing contig
    (the only ones a call can be ``unsupported_gene`` for); and the carried genes with an unknown haplotype
    (``excluded:half_resolved``)."""
    rows = []
    for name, block in (summary.get("donors") or {}).items():
        capture = (block.get("genotype") or {}).get("capture")
        if not capture:
            continue
        uncalled = []
        for gene, evidence in (capture.get("uncalled") or {}).items():
            shown = [f"h{entry.get('haplotype')} {entry.get('holds') or entry.get('state')}"
                     + (f": {entry['label']} at {_pct(entry.get('identity'))}" if entry.get("label") else "")
                     for entry in evidence if entry.get("state") != "deleted"]
            uncalled.append(f"{gene} ({'; '.join(shown) or MISSING})")
        rows.append([name, ", ".join(uncalled) or "none", ", ".join(capture.get("absent") or []) or "none",
                     ", ".join(capture.get("half_resolved") or []) or "none"])
    if not rows:
        return []
    return ["Where the genomic truth stops, per donor: the genes the contigs hold and the genotype does not call "
            "(uncalled; what a capturing contig holds there, by its closest OGRDB allele), the genes every capturing "
            "contig shows deleted (absent: the only genes a call can be an unsupported gene for), and the carried "
            "genes with an allele on one haplotype unknown (half resolved).", "",
            *_table(["Donor", "Uncalled", "Absent", "Half resolved"], rows)]


def _one_sided_reasons(comparison: dict) -> str:
    parts = []
    for side in ("a", "b"):
        counts = comparison["one_sided"].get(side) or {}
        reasons = ", ".join(f"{reason.removeprefix('excluded:')} {_int(count)}"
                            for reason, count in (counts.get("reasons") or {}).items())
        parts.append(f"{_label(comparison[side])} {_int(counts.get('reads'))}" + (f" ({reasons})" if reasons else ""))
    return ", ".join(parts)


def _count_share(block: dict | None, reason: str | None) -> str:
    """A run's reads of V class ``reason`` (None: every class not excluded) with their share of its reads."""
    classes = (block or {}).get("v_classes")
    if not classes:
        return MISSING
    total = sum(classes.values())
    count = (classes.get(reason, 0) if reason else
             sum(value for name, value in classes.items() if not name.startswith("excluded:")))
    return f"{_int(count)} ({_pct(count / total if total else None)})"


def _known_answers(summary: dict, scopes: list[dict]) -> list[str]:
    blocks = [(name, block) for name, block in (summary.get("donors") or {}).items() if block.get("truth")]
    if not blocks:
        return [_nothing_scored(summary)]
    rows = []
    for name, block in blocks:
        truth = block["truth"]
        share = truth.get("known_share")
        low = truth.get("low_known_yield", share is not None and share < KNOWN_YIELD_FLAG)
        known = (f"{_int(truth.get('known'))} (**{_pct(share)}**, below {KNOWN_YIELD_FLAG:.0%})" if low
                 else f"{_int(truth.get('known'))} ({_pct(share)})")
        anchor = truth.get("anchor") or {}
        rows.append([name, _int(truth.get("reads")), known, _int(truth.get("novel_known")), _int(truth.get("naive")),
                     f"{_int(truth.get('clonotypes'))} ({_int(truth.get('multi_read_clonotypes'))})",
                     " / ".join(_int(anchor.get(key, 0)) for key in ("donor", "shared", "none"))])
    lines = [f"A read is a documented known answer when exactly one donor allele's region through Cys104 (at least "
             f"200 nt of it observed) occurs in it exactly, and that allele is in the shared reference; known_top1 is "
             f"scored on every one, excluded from R1 or not (its truth is tool-neutral). A yield below "
             f"{KNOWN_YIELD_FLAG:.0%} is flagged.", "",
             *_table(["Donor", "Reads", "Documented known answers", "Novel known answers", "Naive",
                      "Clonotypes (multi-read)", "Cys104 anchor: donor / shared / none"], rows)]
    if not scopes:
        return lines
    runs = [(tool, arm) for arm in ARMS for tool in _arm_tools(summary, arm)]
    accuracy = [[f"{_label(tool)} {arm}", *(_known_cell(_tool_block(scope, arm, tool)) for scope in scopes)]
                for tool, arm in runs]
    lines += ["", "known_top1 / known_gene_top1 per run (n):", "",
              *_table(["Run", *(scope["label"] for scope in scopes)], accuracy)]
    paired = [[scope["label"], f"{_pair_header(a, b)}, {arm}",
               *(_pair_cell(_find(scope, a, b, arm), metric) for metric in KNOWN),
               _informational(_find(scope, a, b, arm), KNOWN_OBSERVABLE)]
              for scope in scopes for arm in ARMS for a, b in _pairs(scopes, arm) if _find(scope, a, b, arm)]
    if paired:
        lines += ["", "Paired (known_top1, observably: whether the call's germline window over the read's observed V "
                      "is the known alleles' window, so a call the read cannot tell from the known allele counts as "
                      "right; informational):", "",
                  *_table(["Scope", "Pair", "known_top1", "known_gene_top1",
                           "known_top1, observably (informational: paired means)"], paired)]
    return lines


def _known_cell(block: dict | None) -> str:
    entry = ((block or {}).get("summary") or {}).get(KNOWN[0])
    if not entry:
        return MISSING
    return f"{_pct(entry.get('mean'))} / {_pct(_mean(block, KNOWN[1]))} (n={entry.get('n', MISSING)})"


def _novel_reads(summary: dict, scopes: list[dict]) -> list[str]:
    blocks = _scored_blocks(summary)
    if not blocks:
        return [_nothing_scored(summary)]
    rows = []
    for arm in ARMS:
        for tool in _arm_tools(summary, arm):
            reads, calls = 0, Counter()
            for _, block in blocks:
                novel = ((block["tools"].get(arm) or {}).get(tool) or {}).get("novel_known") or {}
                reads += novel.get("reads") or 0
                calls.update(novel.get("calls") or {})
            where = ", ".join(f"{call} {count}" for call, count in calls.most_common(6)) or MISSING
            rows.append([f"{_label(tool)} {arm}", _int(reads),
                         _pct(calls["known gene"] / reads if reads else None), where])
    return ["Reads whose known answer is a donor allele the shared reference lacks: no tool can call that allele, so "
            "the question is whether the call lands in the right gene. Summed over the scored donors.", "",
            *_table(["Run", "Novel known answers", "Called to the known gene", "Where the calls go"], rows)]


def _sensitivity(summary: dict, scopes: list[dict]) -> list[str]:
    sensitivity = summary.get("sensitivity") or {}
    genes = sensitivity.get("genes") or []
    thresholds = next((((c.get("sensitivity") or {}).get("gene_support") or {}).get("thresholds")
                       for scope in scopes for c in scope["comparisons"]
                       if ((c.get("sensitivity") or {}).get("gene_support") or {}).get("thresholds")), None) or {}
    lines = ["Two variants of R1 beside the primary, per shared-arm pair:", "",
             f"- Without SD3-only genes: R1 leaves out the reads whose top-1 V call, by either tool, is to a gene of "
             f"the paper's novel alleles (Rodriguez et al. 2023, Supplementary Data 3) that no subject of the VDJbase "
             f"genomic "
             f"cohort carries ({_int((sensitivity.get('cohort') or {}).get('subjects'))} subjects): "
             f"{_int(sensitivity.get('missing_alleles'))} alleles of its {_int(sensitivity.get('sd3_alleles'))}, in "
             f"{len(genes)} genes: {', '.join(genes) or MISSING}.",
             f"- With Mikelov gene-support thresholds (≥{thresholds.get('min_clonotypes', 10)} clonotypes and "
             f"≥{thresholds.get('min_naive_clonotypes', 3)} naive clonotypes per V gene among the donor's known "
             "answers; Mikelov et al. 2024): a call to a gene below them is excluded (`excluded:low_support`). The "
             "primary applies none: a gene the donor does not carry has no known answers, so the thresholds could "
             "only drop calls to carried genes, supported ones among them."]
    if not scopes:
        return [*lines, "", _nothing_scored(summary)]
    pairs = _pairs(scopes, "shared")
    rows = []
    for scope in scopes:
        found = [_find(scope, a, b, "shared") for a, b in pairs]
        first = next((c for c in found if c and c.get("sensitivity")), None) or {}
        sd3 = (first.get("sensitivity") or {}).get("sd3") or {}
        support = (first.get("sensitivity") or {}).get("gene_support") or {}
        rows += [[scope["label"], "R1 primary", *(_pair_cell(c, HEADLINE) for c in found)],
                 [scope["label"], f"Without SD3-only genes ({_int(sd3.get('excluded_reads'))} reads called to them)",
                  *(_variant_cell(c, "sd3") for c in found)],
                 [scope["label"], f"With Mikelov gene-support thresholds "
                                  f"({_int((support.get('excluded') or {}).get('v'))} V reads out)",
                  *(_variant_cell(c, "gene_support") for c in found)]]
    lines = [*lines, "", *_table(["Scope", "Variant", *(_pair_header(a, b) for a, b in pairs)], rows)]
    return [*lines, *_clusterings(summary, scopes, pairs)]


def _clusterings(summary: dict, scopes: list[dict], pairs: list[tuple[str, str]]) -> list[str]:
    """The pre-declared primary pair's R1 and known_top1 per protocol (and pooled) under each clustering."""
    primary = PRIMARY if PRIMARY in pairs else (pairs[0] if pairs else None)
    lead = _protocol_scopes(summary) + [scope for scope in scopes if scope["block"] is None]
    rows = []
    for scope in lead if primary else ():
        clusterings = (_find(scope, *primary, "shared") or {}).get("clusterings") or {}
        if not clusterings:
            continue
        for metric, label in ((HEADLINE, "R1"), (KNOWN[0], "known_top1")):
            rows.append([scope["label"], label,
                         *(_pair_cell(_find(scope, *primary, "shared"), metric,
                                      entries=((clusterings.get(name) or {}).get("comparison") or {}))
                           for name in CLUSTERINGS)])
    if not rows:
        return []
    return ["", f"{CLUSTERING_NOTE} Pair: {_pair_header(*primary)}.", "",
            *_table(["Scope", "Metric", *CLUSTERINGS.values()], rows)]


def _variant_cell(comparison: dict | None, key: str) -> str:
    """A comparison's R1 cell in sensitivity variant ``key`` (``sd3``, ``gene_support``); a dash when the comparison
    has no such variant."""
    block = ((comparison or {}).get("sensitivity") or {}).get(key)
    if comparison is not None and comparison.get("status", "compared") == "compared" and block is None:
        return MISSING
    return _pair_cell(comparison, HEADLINE, entries=(block or {}).get("comparison") or {})


def _d_and_j(summary: dict, scopes: list[dict]) -> list[str]:
    blocks = [(name, block) for name, block in _scored_blocks(summary)
              if (block.get("genotype") or {}).get("d_truth") or (block.get("genotype") or {}).get("j_truth")]
    if not blocks:
        return ["No donor has D or J truth in this run (genomic D and J calls are thin), so there is nothing to show. "
                "Where a donor has it, `real.d.unsupported` and `real.j.unsupported` are informational: the same "
                "rule by exact sequence, with the same exclusions per segment."]
    rows = []
    for name, block in blocks:
        for arm in ARMS:
            for tool, run in (block["tools"].get(arm) or {}).items():
                cells = []
                for metric, field in D_AND_J:
                    entry = (run.get("summary") or {}).get(metric) or {}
                    classes = run.get(field)
                    cells += [MISSING if not entry else f"{_pct(entry.get('mean'))} (n={entry.get('n', 0)})",
                              ", ".join(f"{key} {value}" for key, value in (classes or {}).items()) or MISSING]
                rows.append([name, _label(tool), arm, *cells])
    return ["Informational: a D or J call against the donor's genomic D or J alleles, by exact sequence, with the "
            "same exclusions per segment; never compared.", "",
            *_table(["Donor", "Tool", "Arm", "D unsupported", "D classes", "J unsupported", "J classes"], rows)]


def _strata(summary: dict, scopes: list[dict]) -> list[str]:
    if not scopes:
        return [_nothing_scored(summary)]
    pairs = _pairs(scopes, "shared")
    rows = []
    for scope in scopes:
        found = [_find(scope, a, b, "shared") for a, b in pairs]
        labels = sorted({label for c in found if c for label in (c.get("strata") or {})})
        for label in labels:
            reads = _stratum_reads(summary, scope, label)
            rows.append([scope["label"], label, _int(reads),
                         *(_pair_cell(c, HEADLINE, entries=((c or {}).get("strata") or {}).get(label) or {})
                           for c in found)])
    return ["R1 per protocol (multiplex or 5'RACE) × isotype (IGHM, IGHG, other), per shared-arm pair; each stratum "
            "is its own Holm family. Reads: the stratum's sequences, excluded or not.", "",
            *_table(["Scope", "Stratum", "Reads", *(_pair_header(a, b) for a, b in pairs)], rows)]


def _stratum_reads(summary: dict, scope: dict, label: str):
    donors = [scope["block"]] if scope["block"] else [block for _, block in _scored_blocks(summary)]
    counts = [((block.get("truth") or {}).get("strata") or {}).get(label) for block in donors]
    counts = [count for count in counts if count is not None]
    return sum(counts) if counts else None


def _preprocessing(summary: dict, scopes: list[dict]) -> list[str]:
    blocks = [(name, block) for name, block in (summary.get("donors") or {}).items() if block.get("preprocess")]
    if not blocks:
        return ["No donor reached preprocessing: see Failed runs and refusals."]
    lines = ["Each run's reads to the unique sequences the tools are given (pRESTO in the pinned Immcantation image, "
             "reference-free). Assembly failures are counted by AssemblePairs' own reason: mismatches, the best "
             "overlap tried is over the error limit (mates that overlap with too many mismatches, or that do not "
             "meet, whose best overlap is chance); not significant, an overlap within the limit but too short to be "
             "significant; no overlap, none tried. A failed pair is lost, not wrong, and both tools see the same "
             "set. A failed pair's mates' combined length is a lower bound on its insert only when its mates do not "
             "meet; its overlap and error show whether they did."]
    for name, block in blocks:
        record = block["preprocess"]
        classifier = record.get("classifier") or {}
        lines += ["", f"**{name}** ({block.get('run', MISSING)}, {record.get('protocol', MISSING)}, "
                      f"{record.get('status', MISSING)}): classifier: "
                      f"{_evidence(classifier.get('evidence'))}. Quality {record.get('quality', MISSING)}: "
                      f"{record.get('quality_steps') or MISSING}. Wall time {_scalar(record.get('wall_seconds'))} s."]
        acceptance = record.get("assembly_acceptance") or {}
        if acceptance:
            line = acceptance.get("max_failure_rate")
            where = ("within it" if acceptance.get("within_line") else
                     "over it, let through by --accept-assembly-loss" if acceptance.get("accepted") else
                     "over it: the run is rejected")
            lines += ["", f"Assembly: {_int(acceptance.get('failed'))} of {_int(acceptance.get('pairs'))} pairs "
                          f"({_pct(acceptance.get('failure_rate'))}) failed to assemble; the line is "
                          f"{MISSING if line is None else f'{100 * line:.0f}%'}: {where}."]
        evidence = record.get("assembly") or {}
        kinds = ((evidence.get("fail") or {}).get("kinds")) or record.get("assembly_failures")
        if kinds:
            lines += ["", "Failures by kind: " + ", ".join(f"{ASSEMBLY_KINDS.get(kind, kind)} {_int(count)}"
                                                         for kind, count in kinds.items()) + "."]
        deciles = _deciles(evidence)
        if deciles:
            lines += ["", *_table(["Pairs", "n", *DECILES], deciles)]
        steps = record.get("steps") or []
        if steps:
            lines += ["", *_table(["Step", "Records", "Pass rate", "Seconds"],
                                  [[step.get("name") or MISSING, _int(step.get("records")), _pct(step.get("pass_rate")),
                                    _scalar(step.get("seconds"))] for step in steps])]
        counts = record.get("counts") or {}
        if counts:
            isotypes = ", ".join(f"{key} {_int(value)}" for key, value in (record.get("isotypes") or {}).items())
            lines += ["", f"Raw pairs {_int(counts.get('raw_pairs'))} → unique sequences {_int(counts.get('unique'))}"
                          f" → capped {_int(counts.get('capped'))}" + (f" ({isotypes})" if isotypes else "") + "."]
    return lines


def _deciles(evidence: dict) -> list[list[str]]:
    rows = []
    for label, group, key, number in (("passed: assembled length (nt)", "pass", "insert_length", _int),
                                      ("failed: mates' combined length (nt)", "fail", "mates_length", _int),
                                      ("passed: overlap (nt)", "pass", "overlap", _int),
                                      ("failed: best overlap tried (nt)", "fail", "overlap", _int),
                                      ("passed: overlap error", "pass", "error", _error),
                                      ("failed: overlap error", "fail", "error", _error)):
        found = (evidence.get(group) or {}).get(key)
        if found:
            rows.append([label, _int(found.get("n")), *(number(found.get(decile)) for decile in DECILES)])
    return rows


def _error(value) -> str:
    return MISSING if value is None else f"{value:.3f}"


def _airr_output(summary: dict, scopes: list[dict]) -> list[str]:
    donors = summary.get("donors") or {}
    rows = []
    for name, block in donors.items():
        for arm, runs in (block.get("runs") or {}).items():
            for tool, run in runs.items():
                if not tool.startswith("alignair"):
                    continue
                assembly = run.get("assembly") or {}
                status = "ok" if run.get("status") == "ok" else f"failed ({run.get('returncode', MISSING)})"
                notes = []
                if tool == "alignair":
                    notes.append("the image's ≤1% failed-assembly gate passed" if run.get("status") == "ok" else
                                 f"the image's run failed: {run.get('error') or MISSING}")
                elif run.get("status") != "ok":
                    notes.append(f"failed: {run.get('error') or MISSING}")
                reasons = assembly.get("fail_reasons") or {}
                if reasons:
                    notes.append("failures: " + ", ".join(f"{reason} {_int(count)}" for reason, count in
                                                          reasons.items()))
                if run.get("note"):
                    notes.append(str(run["note"]))
                rows.append([name, tool, arm, status, *(_int(assembly.get(key)) for key in (
                    "written_records", "complete_assemblies", "partial_assemblies", "failed_assemblies")),
                    "; ".join(notes) or MISSING])
    lines = ["AlignAIR assembles each read's AIRR row from its predictions. The image refuses a run where more than 1% "
             "of its records fail to assemble (the run is then failed, with the cause); alignair-api records its "
             "failed assemblies instead and is scored."]
    lines += ["", *_table(["Donor", "Tool", "Arm", "Run", "Written", "Complete", "Partial", "Failed", "Notes"], rows)
              ] if rows else ["", "No AlignAIR run in this summary."]
    columns = [(tool, arm) for arm in ARMS for tool in _arm_tools(summary, arm)]
    validity, totals = [], {column: [0, 0] for column in columns}
    for name, block in _scored_blocks(summary):
        cells = []
        for tool, arm in columns:
            matching = ((block["tools"].get(arm) or {}).get(tool) or {}).get("matching") or {}
            if "invalid_rows" in matching:
                totals[tool, arm][0] += matching["invalid_rows"]
                totals[tool, arm][1] += matching.get("rows") or 0
                cells.append(f"{_int(matching['invalid_rows'])} of {_int(matching.get('rows'))}")
            else:
                cells.append(MISSING)
        validity.append([name, *cells])
    if len(validity) > 1:
        validity.append(["All donors", *(f"{_int(totals[c][0])} of {_int(totals[c][1])}" for c in columns)])
    lines += ["", AIRR_NOTE, ""]
    if validity:
        lines += _table(["Donor", *(f"{_label(tool)} {arm}" for tool, arm in columns)], validity)
    else:
        lines.append(_nothing_scored(summary))
    return lines


def _self_check(summary: dict, scopes: list[dict]) -> list[str]:
    rows = []
    for name, block in (summary.get("donors") or {}).items():
        check = block.get("oracle_self_check")
        if not check:
            continue
        excluded = ", ".join(
            f"{reason} {_int(count)}" + (f" ({', '.join((check.get('excluded_genes') or {}).get(reason) or [])})"
                                          if (check.get("excluded_genes") or {}).get(reason) else "")
            for reason, count in (check.get("classes") or {}).items() if reason.startswith("excluded:"))
        status = check.get("status") or ("passed" if check.get("passed") else "failed")
        rows.append([name, _int(check.get("reads")), _int(check.get("scored")), _pct(check.get("supported")),
                     _pct(check.get("known_top1")), _pct(check.get("known_gene_top1")), excluded or MISSING,
                     {"failed": "**failed**", "unchecked": "unchecked (warning)"}.get(status, status)])
    if not rows:
        return [_nothing_scored(summary)]
    return ["The oracle calls each documented known answer's own shared allele, as a tool would: every one scored "
            "must be supported and right (known_top1 and known_gene_top1 100%), or the truth contradicts itself "
            "(failed). Unchecked: none is scored (no documented known answer, or every one excluded), a warning.", "",
            *_table(["Donor", "Known answers", "Scored", "Supported", "known_top1", "known_gene_top1", "Excluded",
                     "Status"], rows)]


def _failures(summary: dict, scopes: list[dict]) -> list[str]:
    donors = summary.get("donors") or {}
    lines = []
    stopped = summary.get("stopped_donors") or []
    if stopped:
        rows = [[entry["donor"], entry["step"], entry["reason"],
                 _evidence(((donors.get(entry["donor"]) or {}).get("stopped") or {}).get("evidence"))]
                for entry in stopped]
        lines += ["Donors stopped (not scored; the others went on):", "",
                  *_table(["Donor", "Step", "Reason", "Evidence"], rows), ""]
    after = [name for name, block in donors.items() if block.get("stopped_after")]
    if after:
        lines += [f"Stopped after preprocessing, as asked (`--stop-after preprocess`): {', '.join(after)}.", ""]
    failed = summary.get("failed_runs") or []
    if failed:
        rows = [[run["donor"], _label(run["tool"]), run["arm"], run.get("error") or MISSING, run.get("note") or MISSING]
                for run in failed]
        lines += ["Failed runs (not scored, and left out of every pair; a run whose rows hold ids of no read has "
                  "drifted from its reads and is failed too):", "",
                  *_table(["Donor", "Tool", "Arm", "Error", "Note"], rows), ""]
    not_compared = [f"{name}: {_pair_header(c['a'], c['b'])} {c.get('arm')}: {c.get('reason') or MISSING}"
                    for name, block in donors.items() for c in block.get("comparisons") or []
                    if c.get("status", "compared") != "compared"]
    if not_compared:
        lines += ["Pairs not compared:", "", *(f"- {item}" for item in not_compared), ""]
    if summary.get("oracle_failures"):
        lines += [f"Truth self-check failed: {', '.join(summary['oracle_failures'])} (see Truth self-check).", ""]
    if not lines:
        return ["None: every donor was scored, every run succeeded and every pair was compared."]
    return lines[:-1]


def _provenance(summary: dict, scopes: list[dict]) -> list[str]:
    provenance = summary.get("provenance") or {}
    lines = []
    for name, entry in (provenance.get("images") or {}).items():
        entry = entry if isinstance(entry, dict) else {"image": entry}
        lines.append(f"- {IMAGE_LABELS.get(name, name)}: `{entry.get('image', MISSING)}` (image "
                     f"`{entry.get('id', MISSING)}`)")
    for tool, settings in (provenance.get("tools") or {}).items():
        values = ", ".join(f"{key} `{value}`" for key, value in (settings or {}).items())
        lines.append(f"- {_label(tool)}: {values or MISSING}")
    reference = provenance.get("reference") or {}
    if reference:
        lines.append(f"- Shared reference {reference.get('name', MISSING)}: digest "
                     f"`{reference.get('digest', MISSING)}`")
    ogrdb = provenance.get("ogrdb") or {}
    if ogrdb:
        lines.append(f"- IUIS map: {ogrdb.get('germline_set_id', 'OGRDB')} rev {ogrdb.get('release_version', MISSING)}"
                     f" ({ogrdb.get('release_date', MISSING)}), sha256 `{ogrdb.get('sha256', MISSING)}`")
    if provenance.get("igenotyper_commit"):
        lines.append(f"- IGenotyper gene coordinates at commit `{provenance['igenotyper_commit']}`")
    for name, entry in ((summary.get("sensitivity") or {}).get("files") or {}).items():
        lines.append(f"- Sensitivity cohort file {name}: md5 `{(entry or {}).get('md5', MISSING)}`"
                     + (f", {entry['source']}" if (entry or {}).get("source") else ""))
    git = provenance.get("git") or {}
    lines.append(f"- Code: commit `{git.get('git_commit', MISSING)}`{' (dirty)' if git.get('git_dirty') else ''}; "
                 f"config hash `{summary.get('config_hash') or MISSING}`")
    return lines
