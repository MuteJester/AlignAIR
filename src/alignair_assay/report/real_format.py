"""The formatting and scope helpers of the real-data report (``report.real``), shared by its summary page and its
detail (``report.real_detail``): percentages, deltas and their CIs, the cell of a paired comparison and the words of
its verdict, the scopes a summary is shown at (each donor, each protocol's pool, the pool), and the metric names and
labels both read. Like the report, it reads plain dicts and imports nothing of ``real``.
"""
from __future__ import annotations

from alignair_assay.report.markdown import MISSING, markdown_table

# Items named per flag before the rest are counted.
FLAG_ITEMS = 4
# A donor with fewer documented known answers than this share of its reads is flagged. The summary's own
# ``low_known_yield`` (``workflow.real.KNOWN_YIELD_FLAG``) decides; this is used only where it is missing.
KNOWN_YIELD_FLAG = 0.05
HEADLINE = "real.v.unsupported"
# The pre-declared primary: the paired shared-arm R1 of the AlignAIR image against IgBLAST, per protocol.
PRIMARY = ("igblast", "alignair")
COMPONENTS = {"real.v.unsupported_allele": "unsupported allele", "real.v.unsupported_gene": "unsupported gene",
              "real.v.no_call": "no call"}
KNOWN = ("real.v.known_top1", "real.v.known_gene_top1")
# The reference-exact known answer: the read against the shared reference alone (compared, higher is better).
REF_KNOWN = "real.v.ref_known_top1"
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
# The words for a verdict that is not significant, in the tables and the bottom line alike; a comparison under 10
# clonotypes is not tested, and one whose paired differences are all 0 has nothing to test.
NOT_SIGNIFICANT = "not significant"
TOO_FEW_CLONOTYPES = "too few clonotypes"
NO_VARIATION = "no variation"


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
