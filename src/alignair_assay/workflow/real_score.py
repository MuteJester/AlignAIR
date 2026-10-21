"""R1 scoring and the paired comparisons of the real-data workflow (``workflow.real``), per donor and pooled.

**One ``SupportIndex`` per donor**, shared by every tool and arm (``DonorScoring``): the donor's genotype with its
captured and resolved genes, the shared reference's IUIS map and identical alleles, and the gene support of its known
answers. Every tool's calls on either arm are classified against the whole shared reference with the same truth per
read (``SupportIndex.read_calls(pred, truth=truth)``).

**Each tool's classes** (``step_scores``) are cached in ``real_scores(donor, arm, tool)``, one row per read: its top-1
V call and the classes of its V, D and J calls, with the read's tool-neutral truth applied. The cache's header records
what they were computed from (the predictions', truth's and genotype's digests, the IUIS map's OGRDB file, the
reference and ``SCORE_CODE``); a cache whose record is not current is computed again.

**Scores** (``real.metrics.real_case_scores``) follow from the classes and the reads excluded:

- a tool alone (``tool_summary``): the reads its own calls exclude, per segment, with its classes and its exclusion
  counts per reason, so steering reads into excluded genes stays visible;
- a pair (``pair_scores``): the reads either tool's call excludes, per segment (``real.metrics.shared_exclusions``),
  so a read enters the paired comparison only when neither call is excluded.

**Comparisons** (``comparison``): ``stats.compare`` of the pair over every non-informational metric either tool has
(Holm at ``ALPHA``), with ``clusters`` the reads' clonotypes (``real.truth.clonotype_clusters``, ids carrying the
donor), overall, per stratum (protocol x isotype: IGHM, IGHG, other) and for the sensitivity analysis: the V R1
metrics again without the reads whose top-1 V call, by either tool, is to a gene of the paper's novel alleles the
cohort lacks (``workflow.real_steps.sensitivity_genes``). The informational metrics are summarized beside it.
Pooled comparisons (``pool``) join the donors' pairs with donor-prefixed read ids (``<donor>/<read id>``) and the union
of their clusters, whose ids never collide. An arm delta is a tool's genotype arm paired with its shared arm
(``pair_scores(data, (tool, "shared"), (tool, "genotype"))``), compared the same way without strata, and only
informational.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from alignair_assay.airr.reader import load_airr_tsv
from alignair_assay.core.jsonl import read_jsonl, write_jsonl
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.prediction import PredSegment, Prediction
from alignair_assay.metrics import CaseScores, is_informational, lower_is_better
from alignair_assay.real.metrics import (
    NO_CALL,
    SEGMENTS,
    ReadCalls,
    SupportIndex,
    exclusion_counts,
    real_case_scores,
    shared_exclusions,
)
from alignair_assay.real.truth import RealTruth, clonotype_clusters
from alignair_assay.stats import compare, summarize, to_dict
from alignair_assay.workflow.real_steps import code_digest, digest, jsonl_header, write_json
from alignair_assay.workflow.summary import ALPHA

SCORES_KIND = "alignair_assay.real.scores"
SCORE_CODE = ("real/metrics.py", "real/truth.py", "real/genotype.py", "airr/reader.py", "core/prediction.py",
              "workflow/real_score.py")
# The V R1 headline and its components: the sensitivity analysis's family.
V_R1 = ("real.v.unsupported", "real.v.unsupported_allele", "real.v.unsupported_gene", "real.v.no_call")
ISOTYPES = ("IGHM", "IGHG")
_EXCLUDED = "excluded:"


@dataclass
class DonorScoring:
    """What one donor's reads are scored with: their truths by read id (in read order), the donor's ``SupportIndex``,
    the reads' clonotypes, strata and consensus counts, the genes the sensitivity analysis leaves out, and each
    scored ``(tool, arm)``'s rows (read id -> row, ``step_scores``)."""

    donor: str
    truths: dict[str, RealTruth]
    index: SupportIndex
    weights: dict[str, int]
    sd3: frozenset[str]
    calls: dict[tuple[str, str], dict[str, dict]] = field(default_factory=dict)

    @property
    def clusters(self) -> dict[str, str]:
        return clonotype_clusters(self.truths.values())

    @property
    def strata(self) -> dict[str, str]:
        return {read_id: stratum(truth) for read_id, truth in self.truths.items()}


@dataclass
class PairScores:
    """Two runs' paired scores on the same reads (read id -> scores, ``a`` and ``b``), with each read's clonotype and
    stratum, the reads excluded per segment, and the reads the sensitivity analysis leaves out."""

    a: dict[str, CaseScores]
    b: dict[str, CaseScores]
    clusters: dict[str, str]
    strata: dict[str, str]
    excluded: dict[str, int]
    sensitivity_out: set[str]


def stratum(truth: RealTruth) -> str:
    """``<protocol>/<isotype>``: the isotype is IGHM, IGHG or ``other``."""
    return f"{truth.protocol}/{truth.isotype if truth.isotype in ISOTYPES else 'other'}"


def step_scores(data: DonorScoring, tool: str, arm: str, layout: RunLayout, inputs: dict) -> tuple[dict, dict]:
    """``(tool, arm)``'s rows on the donor's reads, from the cache while its record is current (``inputs``, the
    digests they are computed from, and ``SCORE_CODE``), else from its predictions; stored in ``data.calls``.
    Returns the rows' ``matching`` (rows, duplicates, invalid rows, reads without a row, rows of no read) and the
    step's ``status``."""
    path = layout.real_scores(data.donor, arm, tool)
    fingerprint = digest({"inputs": inputs, "code_sha256": code_digest(SCORE_CODE)})
    header = jsonl_header(path)
    if header.get("fingerprint") == fingerprint:
        rows = {row["read_id"]: row for row in read_jsonl(path)}
        if rows.keys() == data.truths.keys():
            data.calls[tool, arm] = rows
            return header["matching"], {"status": "reused"}
    predictions, stats = load_airr_tsv(layout.real_predictions(tool, data.donor, arm))
    rows = {}
    for read_id, truth in data.truths.items():
        pred = predictions.get(read_id)
        calls = data.index.read_calls(pred, truth=truth)
        segment = None if pred is None else pred.segments.get("v")
        rows[read_id] = {"read_id": read_id, "v_call": None if segment is None else segment.call, "v": calls.v,
                         "d": calls.d, "j": calls.j, "any": calls.v_any_listed_supported}
    matching = {**stats, "missing": sum(read_id not in predictions for read_id in data.truths),
                "extra": sum(read_id not in data.truths for read_id in predictions)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}")
    write_jsonl(rows.values(), temporary, header={"kind": SCORES_KIND, "donor": data.donor, "tool": tool, "arm": arm,
                                                  "fingerprint": fingerprint, "inputs": inputs,
                                                  "code_sha256": code_digest(SCORE_CODE), "matching": matching})
    temporary.replace(path)
    data.calls[tool, arm] = rows
    return matching, {"status": "done"}


def scores(data: DonorScoring, rows: Mapping[str, dict], excluded: Mapping[str, set[str]]) -> dict[str, CaseScores]:
    """Each read's ``real_case_scores`` from its row, out of R1 for the segments ``excluded`` names for it."""
    out = {}
    for read_id, truth in data.truths.items():
        row = rows[read_id]
        pred = Prediction(sequence_id=read_id, segments={"v": PredSegment(call=row["v_call"])})
        calls = ReadCalls(v=row["v"], d=row["d"], j=row["j"], v_any_listed_supported=row["any"])
        out[read_id] = real_case_scores(pred, truth, calls, excluded=excluded.get(read_id, set()), index=data.index)
    return out


def tool_summary(data: DonorScoring, tool: str, arm: str) -> dict:
    """One run's own R1 (the reads its own calls exclude left out): its V, D and J classes, its V exclusions per reason
    (``exclusions``, with D and J's in ``exclusions_dj`` where the donor has their truth), every metric's summary,
    the V headline weighted by the reads' consensus counts, and where the calls of novel known answers go."""
    rows = data.calls[tool, arm]
    own = scores(data, rows, {read_id: _excluded(row) for read_id, row in rows.items()})
    summary = {"v_classes": _counts(row["v"] for row in rows.values()),
               "exclusions": exclusion_counts([row["v"] for row in rows.values()]),
               "summary": to_dict(summarize(own, list(own))),
               "consensus_weighted": _weighted(own, data.weights),
               "novel_known": _novel_calls(data, rows)}
    for segment in ("d", "j"):
        if data.index.has_truth(segment):
            summary[f"{segment}_classes"] = _counts(row[segment] for row in rows.values())
            summary.setdefault("exclusions_dj", {})[segment] = exclusion_counts([row[segment] for row in rows.values()])
    return summary


def pair_scores(data: DonorScoring, a: tuple[str, str], b: tuple[str, str]) -> PairScores:
    """Runs ``a`` and ``b`` (each ``(tool, arm)``) paired on the donor's reads: a read is out of R1 for a segment, for
    both, when either run's call to it is excluded."""
    rows_a, rows_b = data.calls[a], data.calls[b]
    excluded: dict[str, set[str]] = defaultdict(set)
    counts = {}
    for segment in SEGMENTS:
        out = shared_exclusions({read_id: row[segment] for read_id, row in rows_a.items()},
                                {read_id: row[segment] for read_id, row in rows_b.items()})
        counts[segment] = len(out)
        for read_id in out:
            excluded[read_id].add(segment)
    touched = {read_id for read_id in rows_a
               if _touches(data, rows_a[read_id]) or _touches(data, rows_b[read_id])}
    return PairScores(a=scores(data, rows_a, excluded), b=scores(data, rows_b, excluded), clusters=data.clusters,
                      strata=data.strata, excluded=counts, sensitivity_out=touched)


def comparison(pair: PairScores, *, strata: bool = True, sensitivity: Iterable[str] | None = None) -> dict:
    """The pair's comparison (the module docstring): ``n_reads``, ``n_clusters`` and ``multi_read_clusters`` of its
    reads, the reads ``excluded`` per segment, the ``overall`` comparison, the ``informational`` metrics' summaries,
    and with ``strata`` its ``strata``; with ``sensitivity`` (the genes left out) its ``sensitivity``."""
    members = Counter(pair.clusters.get(read_id, read_id) for read_id in pair.a)
    result = {"n_reads": len(pair.a), "n_clusters": len(members),
              "multi_read_clusters": sum(count > 1 for count in members.values()), "excluded": dict(pair.excluded),
              "overall": _compare(pair.a, pair.b, pair.clusters), "informational": _informational(pair.a, pair.b)}
    if strata:
        result["strata"] = {label: _compare(*_subset(pair, lambda read_id: pair.strata[read_id] == label),
                                            pair.clusters)
                            for label in sorted(set(pair.strata.values()))}
    if sensitivity is not None:
        kept = _subset(pair, lambda read_id: read_id not in pair.sensitivity_out)
        result["sensitivity"] = {"genes": sorted(sensitivity), "excluded_reads": len(pair.sensitivity_out),
                                 "comparison": _compare(*kept, pair.clusters, metrics=V_R1)}
    return result


def pool(pairs: Mapping[str, PairScores]) -> PairScores:
    """The pairs of several donors (donor -> pair) as one: read ids prefixed ``<donor>/``, clusters joined (their ids
    carry the donor), exclusions summed."""
    joined = PairScores(a={}, b={}, clusters={}, strata={}, excluded=Counter(), sensitivity_out=set())
    for donor, pair in pairs.items():
        joined.a.update({f"{donor}/{read_id}": value for read_id, value in pair.a.items()})
        joined.b.update({f"{donor}/{read_id}": value for read_id, value in pair.b.items()})
        joined.clusters.update({f"{donor}/{read_id}": cluster for read_id, cluster in pair.clusters.items()})
        joined.strata.update({f"{donor}/{read_id}": label for read_id, label in pair.strata.items()})
        joined.excluded.update(pair.excluded)
        joined.sensitivity_out.update(f"{donor}/{read_id}" for read_id in pair.sensitivity_out)
    joined.excluded = {segment: joined.excluded.get(segment, 0) for segment in SEGMENTS}
    return joined


def pool_tools(summaries: Iterable[dict]) -> dict:
    """Several donors' ``tool_summary`` of one run as one: the classes and exclusion counts summed, and the reads
    behind each mean added up (``summary`` keeps each metric's pooled mean and counts)."""
    pooled: dict = {"v_classes": Counter(), "exclusions": Counter(), "summary": {}}
    sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0, 0])
    for summary in summaries:
        pooled["v_classes"].update(summary["v_classes"])
        pooled["exclusions"].update(summary["exclusions"])
        for name, value in summary["summary"].items():
            if value["mean"] is not None:
                sums[name][0] += value["mean"] * value["n"]
            sums[name][1] += value["n"]
            sums[name][2] += value["n_applicable"]
    pooled["v_classes"] = dict(sorted(pooled["v_classes"].items()))
    pooled["exclusions"] = dict(pooled["exclusions"])
    pooled["summary"] = {name: {"mean": total / n if n else None, "n": n, "n_applicable": applicable,
                                "coverage": n / applicable if applicable else 0.0}
                         for name, (total, n, applicable) in sorted(sums.items())}
    return pooled


# --- helpers -----------------------------------------------------------------------------------------------------

def _compare(a: dict[str, CaseScores], b: dict[str, CaseScores], clusters: Mapping[str, str], *,
             metrics: Iterable[str] | None = None) -> dict:
    """``stats.compare`` of ``a`` with ``b`` over the non-informational metrics either has (or those of ``metrics``
    either has), Holm at ``ALPHA``, cluster-aware by ``clusters``; JSON-ready."""
    seen = {name for values in (a, b) for case in values.values() for name in case}
    family = (sorted(name for name in seen if not is_informational(name)) if metrics is None
              else [name for name in metrics if name in seen])
    found = compare(a, b, family, alpha=ALPHA, lower_is_better={name for name in family if lower_is_better(name)},
                    clusters=clusters)
    return to_dict(found)


def _informational(a: dict[str, CaseScores], b: dict[str, CaseScores]) -> dict:
    summary_a, summary_b = summarize(a, list(a)), summarize(b, list(b))
    names = sorted({name for name in (*summary_a, *summary_b) if is_informational(name)})
    return {name: to_dict({"a": summary_a.get(name), "b": summary_b.get(name)}) for name in names}


def _subset(pair: PairScores, keep) -> tuple[dict, dict]:
    ids = [read_id for read_id in pair.a if keep(read_id)]
    return {read_id: pair.a[read_id] for read_id in ids}, {read_id: pair.b[read_id] for read_id in ids}


def _excluded(row: dict) -> set[str]:
    return {segment for segment in SEGMENTS if (row[segment] or "").startswith(_EXCLUDED)}


def _touches(data: DonorScoring, row: dict) -> bool:
    """True when the row's top-1 V call is to a gene the sensitivity analysis leaves out."""
    return bool(row["v_call"]) and bool(set(data.index.genes(row["v_call"])) & data.sd3)


def _counts(values: Iterable[str | None]) -> dict[str, int]:
    return dict(sorted(Counter(value for value in values if value is not None).items()))


def _weighted(own: dict[str, CaseScores], weights: Mapping[str, int]) -> dict:
    """The V headline over the reads behind the scored sequences: each read's value weighted by its consensus count."""
    total = weight = 0.0
    for read_id, values in own.items():
        value = values.get("real.v.unsupported")
        if value is not None and math.isfinite(value):
            total += value * weights.get(read_id, 1)
            weight += weights.get(read_id, 1)
    return {"real.v.unsupported": total / weight if weight else None, "weight": int(weight)}


def _novel_calls(data: DonorScoring, rows: Mapping[str, dict]) -> dict:
    """Where the top-1 V calls of the novel known answers go: to the known gene, to another gene (named), or none."""
    where = Counter()
    for read_id, truth in data.truths.items():
        if truth.known is None or not truth.novel_known:
            continue
        call = rows[read_id]["v_call"]
        if data.index.known_gene_top1(call, truth):
            where["known gene"] += 1
        else:
            where[("|".join(data.index.genes(call)) or call) if call else NO_CALL] += 1
    reads = sum(where.values())
    return {"reads": reads, "to_known_gene": where["known gene"] / reads if reads else None,
            "calls": dict(where.most_common(10))}


def write_comparison(path: Path, value: dict) -> None:
    """Write one comparison (``real_comparison_json``) as strict JSON."""
    write_json(path, value)
