"""The detail of the real-data report (``report.real``), after its summary page: every section of ``DETAIL_SECTIONS``,
always present (one without data says so), rendered from the plain summary dict by ``detail_markdown``. R1 per donor
with each run's own R1 and the paired means; exclusions per tool and reason, what each reason means and the
invariant, the reads one call alone takes out of each pair, and where each donor's genomic truth stops; known answers
(the yield per donor, flagged below ``KNOWN_YIELD_FLAG``, and the accuracy per tool and arm); where the calls of
novel-allele reads go; the sensitivity variants and clusterings beside the primary; D and J; protocol × isotype
strata; preprocessing (the classifier's evidence, the steps, the assembly's failure rate against its line, its
failures by kind and the deciles of the pairs that passed and failed); AlignAIR's AIRR output (its assembly counts,
the image's gate, and each tool's rows the AIRR reader flags); the truth self-check; the truth health; failed runs
and refusals; and the provenance in full. It reads plain dicts and imports nothing of ``real``.
"""
from __future__ import annotations

from collections import Counter

from alignair_assay.report.markdown import MISSING
from alignair_assay.report.real_format import (
    ARMS,
    COMPONENTS,
    HEADLINE,
    IMAGE_LABELS,
    KNOWN,
    KNOWN_YIELD_FLAG,
    PRIMARY,
    REF_KNOWN,
    _arm_tools,
    _evidence,
    _find,
    _int,
    _label,
    _mean,
    _nothing_scored,
    _own,
    _pair_cell,
    _pair_header,
    _pairs,
    _pct,
    _protocol_scopes,
    _scalar,
    _scored_blocks,
    _table,
    _tool_block,
)

DETAIL_SECTIONS = ("R1 per donor", "Exclusions", "Known answers", "Novel-allele reads", "Sensitivity",
                   "D and J (informational)", "Stratification: protocol × isotype", "Preprocessing",
                   "AlignAIR's AIRR output", "Truth self-check", "Truth health", "Failed runs and refusals",
                   "Provenance")
# The genes the truth-health section names per donor, the flagged ones always among them.
HEALTH_GENES = 8
GAP_CLASSES = ("carried", "captured", "uncalled", "not_captured")
ANY_LISTED = "real.v.any_listed_supported"
# The tie-aware metrics (informational): what the read's observed V window can tell apart.
UNSUPPORTED_OBSERVABLE = "real.v.unsupported_observable"
KNOWN_OBSERVABLE = "real.v.known_top1_observable"
# D and J (informational): each metric and the run's field of its classes. Named in full: a metric name built from
# pieces would read as a name suffix to the registry's check (tests/alignair_assay/real/test_metrics.py).
D_AND_J = (("real.d.unsupported", "d_classes"), ("real.j.unsupported", "j_classes"))
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
# The clusterings the pre-declared primary is compared under: the primary clonotypes, the candidate for P3b, a stress.
CLUSTERINGS = {"exact": "exact junction (primary)", "sl15": "SL15",
               "junction_length": "donor × junction length (stress)"}
CLUSTERING_NOTE = ("Clustering: the pre-declared primary pair's R1, known_top1 and ref_known_top1 compared under three "
                   "clusterings of the same paired reads, each its own Holm family of those three metrics: exact "
                   "junctions (the primary clonotypes in P3a), SL15 (single linkage at 15% or less nt Hamming among a "
                   "donor's junctions of one length; the candidate primary for P3b) and, as a stress test, one cluster "
                   "per donor and junction length.")
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
        referenced = (MISSING if truth.get("ref_known") is None
                      else f"{_int(truth.get('ref_known'))} ({_pct(truth.get('ref_known_share'))})")
        rows.append([name, _int(truth.get("reads")), known, referenced, _int(truth.get("novel_known")),
                     _int(truth.get("naive")),
                     f"{_int(truth.get('clonotypes'))} ({_int(truth.get('multi_read_clonotypes'))})",
                     " / ".join(_int(anchor.get(key, 0)) for key in ("donor", "shared", "none"))])
    lines = [f"A read is a documented known answer when exactly one donor allele's region through Cys104 (at least "
             f"200 nt of it observed) occurs in it exactly, and that allele is in the shared reference; known_top1 is "
             f"scored on every one, excluded from R1 or not (its truth is tool-neutral). A yield below "
             f"{KNOWN_YIELD_FLAG:.0%} is flagged. A reference-exact known answer is the same rule run against the "
             f"shared reference alone: a read whose observed V is exactly one shared allele's window, whatever the "
             f"donor's genotype, so it covers the genes the donor's assembly misses; ref_known_top1 is scored on "
             f"every one, and no exclusion touches it.", "",
             *_table(["Donor", "Reads", "Documented known answers", "Reference-exact known answers",
                      "Novel known answers", "Naive",
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
               _informational(_find(scope, a, b, arm), KNOWN_OBSERVABLE),
               _pair_cell(_find(scope, a, b, arm), REF_KNOWN)]
              for scope in scopes for arm in ARMS for a, b in _pairs(scopes, arm) if _find(scope, a, b, arm)]
    if paired:
        lines += ["", "Paired (known_top1, observably: whether the call's germline window over the read's observed V "
                      "is the known alleles' window, so a call the read cannot tell from the known allele counts as "
                      "right; informational):", "",
                  *_table(["Scope", "Pair", "known_top1", "known_gene_top1",
                           "known_top1, observably (informational: paired means)", "ref_known_top1"], paired)]
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
    """The pre-declared primary pair's R1, known_top1 and ref_known_top1 per protocol (and pooled) under each
    clustering."""
    primary = PRIMARY if PRIMARY in pairs else (pairs[0] if pairs else None)
    lead = _protocol_scopes(summary) + [scope for scope in scopes if scope["block"] is None]
    rows = []
    for scope in lead if primary else ():
        clusterings = (_find(scope, *primary, "shared") or {}).get("clusterings") or {}
        if not clusterings:
            continue
        for metric, label in ((HEADLINE, "R1"), (KNOWN[0], "known_top1"), (REF_KNOWN, "ref_known_top1")):
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
    lines = ["Informational: a D or J call against the donor's genomic D or J alleles, by exact sequence, with the "
             "same exclusions per segment; never compared.", "",
             *_table(["Donor", "Tool", "Arm", "D unsupported", "D classes", "J unsupported", "J classes"], rows)]
    for name, block in blocks:
        alleles = (block.get("genotype") or {}).get("j_alleles") or []
        if len(alleles) == 1 and alleles[0].get("haplotype") == 0:
            allele = alleles[0]["allele"]
            calls = "; ".join(f"{_label(tool)} {arm} " + ", ".join(f"{call} {_int(count)}" for call, count in
                                                                     (run.get("j_calls") or {}).items())
                              for arm in ARMS for tool, run in (block["tools"].get(arm) or {}).items()
                              if run.get("j_calls"))
            lines += ["", f"{name}: its only J truth is one unphased allele, {allele} (h0): another allele of the "
                          "gene may be on its other haplotype, so a call of one counts as unsupported here. The "
                          f"tools' top J calls: {calls or MISSING}."]
    return lines


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
                     _pipeline_cell(check.get("pipeline")), _decoys_cell(check.get("decoys")),
                     {"failed": "**failed**", "unchecked": "unchecked (warning)"}.get(status, status)])
    if not rows:
        return [_nothing_scored(summary)]
    return ["The oracle calls each documented known answer's own shared allele, as a tool would: every one scored "
            "must be supported and right (known_top1 and known_gene_top1 100%), or the truth contradicts itself "
            "(failed). It proves the truth and the metric wiring on carried genes only: every oracle call is a donor "
            "allele. The pipeline oracle writes those calls as an AIRR TSV and scores it as a tool's predictions "
            "(the AIRR reader, then the scores' step), and must agree read for read. Two decoys check what the oracle "
            "never calls: a non-carried allele of a carried, resolved gene must be unsupported_allele, and a call "
            "into an uncalled gene excluded:uncalled. Unchecked: none is scored (no documented known answer, or every "
            "one excluded), a warning.", "",
            *_table(["Donor", "Known answers", "Scored", "Supported", "known_top1", "known_gene_top1", "Excluded",
                     "Pipeline (AIRR → scores)", "Decoys", "Status"], rows)]


def _pipeline_cell(pipeline: dict | None) -> str:
    if not pipeline:
        return MISSING
    return (f"{_int(pipeline.get('agrees'))} of {_int(pipeline.get('reads'))} agree, known_top1 "
            f"{_pct(pipeline.get('known_top1'))}" + ("" if pipeline.get("passed") else " (**failed**)"))


def _decoys_cell(decoys: dict | None) -> str:
    if not decoys:
        return MISSING
    parts = []
    for key, missing in (("unsupported_allele", "no clean carried gene"), ("uncalled", "no uncalled gene")):
        entry = decoys.get(key)
        verdict = "ok" if entry and entry.get("passed") else f"**expected {(entry or {}).get('expected')}**"
        parts.append(missing if entry is None else f"{entry['call']} → {entry['class']} ({verdict})")
    return "; ".join(parts)


def _truth_health(summary: dict, scopes: list[dict]) -> list[str]:
    blocks = [(name, block["truth_health"]) for name, block in (summary.get("donors") or {}).items()
              if block.get("truth_health")]
    if not blocks:
        return ["No truth audit in this summary: no donor's truth was computed."]
    counts = [[name, _int(health.get("reads")), _int(health.get("shared_exact")), _int(health.get("not_carried")),
               *(_int((health.get("by_class") or {}).get(cls)) for cls in GAP_CLASSES)] for name, health in blocks]
    genes = []
    for name, health in blocks:
        flagged = set(health.get("flagged") or [])
        entries = health.get("genes") or []
        shown = [entry for index, entry in enumerate(entries) if index < HEALTH_GENES or entry["gene"] in flagged]
        for entry in shown:
            alleles = ", ".join(f"{allele} {_int(count)}" for allele, count in (entry.get("alleles") or {}).items())
            genes.append([name, f"**{entry['gene']}**" if entry["gene"] in flagged else entry["gene"], entry["class"],
                          _int(entry.get("reads")), _int(entry.get("clonotypes")), alleles or MISSING,
                          "yes" if entry.get("h0_only") else "no"])
    flag = next((health.get("flag_clonotypes") for _, health in blocks if health.get("flag_clonotypes")), 10)
    lines = ["A tool-neutral audit of the truth: the known-answer rule run against the shared reference's V alleles "
             "instead of the donor's. A read whose V is exactly a shared allele's window (at least 200 nt) that no "
             "donor allele explains (no known answer, and no donor allele's window there is that window; windows "
             "are compared, never names) shows an allele the genotype lacks; its gene is carried (another allele of "
             "it: an allele VDJbase missed, or a haplotype it did not resolve), captured (deleted on every capturing "
             f"contig), uncalled or not captured. A carried gene with {flag} clonotypes of such reads or more is in "
             "bold and flagged.", "",
             *_table(["Donor", "Reads", "Exactly a shared allele", "Not carried by the donor",
                      *(cls.replace("_", " ") for cls in GAP_CLASSES)], counts)]
    if genes:
        lines += ["", f"The genes with the most such reads per donor (up to {HEALTH_GENES}, and every flagged "
                      "one):", "",
                  *_table(["Donor", "Gene", "Class", "Reads", "Clonotypes", "Shared alleles", "Unphased only"],
                          genes)]
    return lines


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


def detail_markdown(summary: dict, scopes: list[dict]) -> list[str]:
    """The detail's lines: each section of ``DETAIL_SECTIONS`` under its heading, in order."""
    lines: list[str] = []
    renderers = (_r1_detail, _exclusions, _known_answers, _novel_reads, _sensitivity, _d_and_j, _strata,
                 _preprocessing, _airr_output, _self_check, _truth_health, _failures, _provenance)
    for title, render in zip(DETAIL_SECTIONS, renderers):
        lines += ["", f"## {title}", "", *render(summary, scopes)]
    return lines
