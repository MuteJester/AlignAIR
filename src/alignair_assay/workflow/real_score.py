"""R1 scoring and the paired comparisons of the real-data workflow (``workflow.real``), per donor and pooled.

**One ``SupportIndex`` per donor and variant**, shared by every tool and arm (``DonorScoring``): the donor's genotype
with its captured and resolved genes, the shared reference's IUIS map and identical alleles, and the gene support of
its known answers. The R1 primary is ``index``, without gene-support thresholds; the sensitivity variant "R1 with
Mikelov gene-support thresholds" is ``thresholded`` (``SupportIndex(..., thresholds=True)``, which adds
``excluded:low_support`` for V). Every tool's calls on either arm are classified against the whole shared reference
with the same truth per read (``SupportIndex.read_calls(pred, truth=truth)``).

**Each tool's classes** (``step_scores``) are cached in ``real_scores(donor, arm, tool)``, one row per read: its top-1
V call and the classes of its V, D and J calls, with the read's tool-neutral truth applied, and its V class in the
gene-support variant (``v_support``; low support is V's only), and its top-1 D and J calls. The cache's header
records what they were computed from (the predictions', truth's and genotype's digests, the IUIS map's OGRDB file,
the reference and ``SCORE_CODE``); a cache whose record is not current is computed again.

**Scores** (``real.metrics.real_case_scores``) follow from the classes and the reads excluded:

- a tool alone (``tool_summary``): the reads its own calls exclude, per segment, with its classes and its exclusion
  counts per reason, so steering reads into excluded genes stays visible;
- a pair (``pair_scores``): the reads either tool's call excludes, per segment (``real.metrics.shared_exclusions``),
  so a read enters the paired comparison only when neither call is excluded. A right and a wrong call to the same
  genes are excluded alike, but a call into an excluded gene still takes the read out for both tools, so each pair
  counts, per run, the V reads its own call alone takes out (``one_sided``: how many, by reason, and the documented
  known answers among them with those it called wrong): a tool steering its errors into excluded genes shows there.

**Comparisons** (``comparison``): ``stats.compare`` of the pair over every non-informational metric either tool has
(Holm at ``ALPHA``), with ``clusters`` the reads' clonotypes (``real.truth.clonotype_clusters``, ids carrying the
donor), overall, per stratum (protocol x isotype: IGHM, IGHG, other) and for two sensitivity analyses of the V R1
metrics: ``sd3``, without the reads whose top-1 V call, by either tool, is to a gene of the paper's novel alleles the
cohort lacks (``workflow.real_steps.sensitivity_genes``), and ``gene_support``, the pair again in the gene-support
variant (``pair_scores(..., variant="gene_support")``). The informational metrics are summarized beside it.
With ``clusterings``, the pre-declared primary's metrics (``CLUSTERED``: R1, known_top1 and ref_known_top1) are
compared again under each clustering (``clusterings``): ``exact`` junctions (the primary clonotypes), ``sl15``
(single linkage at 15% junction Hamming, the candidate primary for P3b) and ``junction_length`` (one cluster per
donor and junction length, a stress test), each its own Holm family of those metrics, on the same paired reads.
Pooled comparisons (``pool``) join the donors' pairs with donor-prefixed read ids (``<donor>/<read id>``) and the union
of their clusters, whose ids never collide, and sum their one-sided counts. An arm delta is a tool's genotype arm
paired with its shared arm (``pair_scores(data, (tool, "shared"), (tool, "genotype"))``), compared the same way
without strata, and only informational.
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
    MIN_CLONOTYPES,
    SUPPORTED,
    MIN_NAIVE_CLONOTYPES,
    NO_CALL,
    SEGMENTS,
    ReadCalls,
    SupportIndex,
    exclusion_counts,
    is_documented,
    oracle_prediction,
    real_case_scores,
    shared_exclusions,
)
from alignair_assay.real.truth import RealTruth, clonotype_clusters, junction_length_clusters, sl15_clusters
from alignair_assay.stats import compare, read_changes, summarize, to_dict
from alignair_assay.core.provenance import sha256_file
from alignair_assay.core.records import digest, write_json, write_text
from alignair_assay.workflow.real_steps import code_digest, jsonl_header
from alignair_assay.workflow.summary import ALPHA

SCORES_KIND = "alignair_assay.real.scores"
SCORE_CODE = ("real/metrics.py", "real/truth.py", "real/genotype.py", "airr/reader.py", "core/prediction.py",
              "workflow/real_score.py")
# The V R1 headline and its components: the sensitivity analysis's family.
V_R1 = ("real.v.unsupported", "real.v.unsupported_allele", "real.v.unsupported_gene", "real.v.no_call")
ISOTYPES = ("IGHM", "IGHG")
# The ids of rows matching no read a run's matching names.
EXTRA_EXAMPLES = 3
# The top D and J calls a run's summary names, where the donor has their truth.
TOP_CALLS = 5
# The pipeline oracle's run name (``pipeline_oracle``): its AIRR TSV and scores sit where a tool's would.
ORACLE_TOOL = "oracle"
# The pre-declared primary's metrics, compared again under each clustering of the reads.
CLUSTERED = ("real.v.unsupported", "real.v.known_top1", "real.v.ref_known_top1")
# The R1 primary and the gene-support sensitivity variant: which row field holds the V class.
VARIANTS = {"primary": "v", "gene_support": "v_support"}
_EXCLUDED = "excluded:"


@dataclass
class DonorScoring:
    """What one donor's reads are scored with: their truths by read id (in read order), the donor's ``SupportIndex``
    of the R1 primary (``index``) and of the gene-support variant (``thresholded``), the reads' clonotypes, strata and
    consensus counts, the genes the SD3 sensitivity analysis leaves out, and each scored ``(tool, arm)``'s rows (read
    id -> row, ``step_scores``)."""

    donor: str
    truths: dict[str, RealTruth]
    index: SupportIndex
    thresholded: SupportIndex
    weights: dict[str, int]
    sd3: frozenset[str]
    calls: dict[tuple[str, str], dict[str, dict]] = field(default_factory=dict)

    @property
    def clusters(self) -> dict[str, str]:
        return clonotype_clusters(self.truths.values())

    @property
    def strata(self) -> dict[str, str]:
        return {read_id: stratum(truth) for read_id, truth in self.truths.items()}

    @property
    def cluster_sets(self) -> dict[str, dict[str, str]]:
        """The other clusterings of the reads (read id -> cluster): ``sl15`` and ``junction_length``."""
        return {"sl15": sl15_clusters(self.truths.values()),
                "junction_length": junction_length_clusters(self.truths.values())}


@dataclass
class PairScores:
    """Two runs' paired scores on the same reads (read id -> scores, ``a`` and ``b``), with each read's clonotype and
    stratum, the reads excluded per segment, the reads the sensitivity analysis leaves out, and per run (``a``,
    ``b``) the V reads its own call alone excludes (``one_sided``)."""

    a: dict[str, CaseScores]
    b: dict[str, CaseScores]
    clusters: dict[str, str]
    strata: dict[str, str]
    excluded: dict[str, int]
    sensitivity_out: set[str]
    one_sided: dict[str, dict] = field(default_factory=dict)
    # the other clusterings of the reads: name -> read id -> cluster
    cluster_sets: dict[str, dict[str, str]] = field(default_factory=dict)


def stratum(truth: RealTruth) -> str:
    """``<protocol>/<isotype>``: the isotype is IGHM, IGHG or ``other``."""
    return f"{truth.protocol}/{truth.isotype if truth.isotype in ISOTYPES else 'other'}"


def step_scores(data: DonorScoring, tool: str, arm: str, layout: RunLayout, inputs: dict) -> tuple[dict, dict]:
    """``(tool, arm)``'s rows on the donor's reads, from the cache while its record is current (``inputs``, the
    digests they are computed from, and ``SCORE_CODE``), else from its predictions; stored in ``data.calls``.
    Returns the rows' ``matching`` (rows, duplicates, invalid rows, ``missing``: reads without a row, scored as no
    call; ``extra``: rows whose id matches no read, with up to ``EXTRA_EXAMPLES`` of their ids, sorted, in
    ``extra_ids``) and the step's ``status``. A run with extra rows is not scored (``workflow.real``): its ids
    drifted from the reads it was given."""
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
                         "d": calls.d, "j": calls.j, "any": calls.v_any_listed_supported,
                         "d_call": _call(pred, "d"), "j_call": _call(pred, "j"),
                         "v_support": data.thresholded.classify_v(pred, truth=truth)}
    extra = sorted(read_id for read_id in predictions if read_id not in data.truths)
    matching = {**stats, "missing": sum(read_id not in predictions for read_id in data.truths), "extra": len(extra),
                "extra_ids": extra[:EXTRA_EXAMPLES]}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}")
    write_jsonl(rows.values(), temporary, header={"kind": SCORES_KIND, "donor": data.donor, "tool": tool, "arm": arm,
                                                  "fingerprint": fingerprint, "inputs": inputs,
                                                  "code_sha256": code_digest(SCORE_CODE), "matching": matching})
    temporary.replace(path)
    data.calls[tool, arm] = rows
    return matching, {"status": "done"}


def pipeline_oracle(data: DonorScoring, layout: RunLayout, inputs: dict) -> dict:
    """The oracle through the tools' path: each documented known answer's own shared alleles (``oracle_prediction``:
    the first, sorted, as the top-1) written as an AIRR TSV (``real_predictions(ORACLE_TOOL, donor, "shared")``),
    read back by the AIRR reader and classified as a tool's rows are (``step_scores``, keyed on ``inputs`` and the
    TSV's digest). ``reads`` (the documented known answers), ``classes`` (class -> reads), ``agrees`` (reads whose
    class is the direct oracle's, ``SupportIndex.classify_v``), ``known_top1`` (its share among them through the
    scores) and ``passed`` (every read agrees, every scored one is supported, every top-1 is right)."""
    documented = [truth for truth in data.truths.values() if is_documented(truth)]
    lines = ["sequence_id\tsequence\tv_call\td_call\tj_call"]
    for truth in documented:
        lines.append(f"{truth.read_id}\t\t{','.join(oracle_prediction(truth).segments['v'].call_set)}\t\t")
    path = layout.real_predictions(ORACLE_TOOL, data.donor, "shared")
    write_text(path, "\n".join(lines) + "\n")
    step_scores(data, ORACLE_TOOL, "shared", layout, {**inputs, "predictions_sha256": sha256_file(path)})
    rows = data.calls.pop((ORACLE_TOOL, "shared"))
    classes: Counter[str] = Counter()
    agrees = right = 0
    for truth in documented:
        row = rows[truth.read_id]
        classes[row["v"]] += 1
        agrees += row["v"] == data.index.classify_v(oracle_prediction(truth), truth=truth)
        right += data.index.known_top1(row["v_call"], truth)
    reads = len(documented)
    return {"reads": reads, "classes": dict(sorted(classes.items())), "agrees": agrees,
            "known_top1": right / reads if reads else None,
            "passed": agrees == reads and right == reads
                      and all(cls == SUPPORTED or cls.startswith(_EXCLUDED) for cls in classes)}


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
    the V headline weighted by the reads' consensus counts, where the calls of novel known answers go, and its V
    classes, exclusions and summary in the gene-support variant (``gene_support``)."""
    rows = data.calls[tool, arm]
    summary, own = _own(data, rows)
    summary.update(consensus_weighted=_weighted(own, data.weights), novel_known=_novel_calls(data, rows),
                   gene_support=_own(data, _variant(rows, "gene_support"))[0])
    for segment in ("d", "j"):
        if data.index.has_truth(segment):
            summary[f"{segment}_classes"] = _counts(row[segment] for row in rows.values())
            summary[f"{segment}_calls"] = dict(Counter(row[f"{segment}_call"] for row in rows.values()
                                                       if row[f"{segment}_call"]).most_common(TOP_CALLS))
            summary.setdefault("exclusions_dj", {})[segment] = exclusion_counts([row[segment] for row in rows.values()])
    return summary


def pair_scores(data: DonorScoring, a: tuple[str, str], b: tuple[str, str], *,
                variant: str = "primary") -> PairScores:
    """Runs ``a`` and ``b`` (each ``(tool, arm)``) paired on the donor's reads, in the R1 ``variant`` (``VARIANTS``:
    the primary, or the gene-support variant): a read is out of R1 for a segment, for both, when either run's call to
    it is excluded."""
    rows_a, rows_b = _variant(data.calls[a], variant), _variant(data.calls[b], variant)
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
                      strata=data.strata, excluded=counts, sensitivity_out=touched,
                      one_sided={"a": one_sided(data, rows_a, rows_b), "b": one_sided(data, rows_b, rows_a)},
                      cluster_sets=data.cluster_sets)


def one_sided(data: DonorScoring, mine: Mapping[str, dict], other: Mapping[str, dict]) -> dict:
    """The V reads run ``mine``'s call excludes while run ``other``'s does not (each read id -> row): ``reads``, by
    ``reasons`` (the exclusions), and the ``documented`` known answers among them, with how many of those its top-1
    call got wrong (``documented_wrong``, ``SupportIndex.known_top1``). Each is out of R1 for both runs because of this
    run's call alone."""
    reasons: Counter[str] = Counter()
    documented = wrong = 0
    for read_id, row in mine.items():
        if not _is_excluded(row["v"]) or _is_excluded(other[read_id]["v"]):
            continue
        reasons[row["v"]] += 1
        truth = data.truths[read_id]
        if is_documented(truth):
            documented += 1
            wrong += not data.index.known_top1(row["v_call"], truth)
    return {"reads": sum(reasons.values()), "reasons": dict(sorted(reasons.items())), "documented": documented,
            "documented_wrong": wrong}


def change_groups(pair: PairScores, donor: str | None = None) -> dict[str, str]:
    """Each read's donor and protocol, ``<donor>/<protocol>`` (the protocol is its stratum's first part): the
    ``donor`` given for one donor's pair, else (a pool, ``pool``) the donor its read id is prefixed with."""
    groups = {}
    for read_id in pair.a:
        owner = donor if donor is not None else read_id.partition("/")[0]
        groups[read_id] = f"{owner}/{pair.strata.get(read_id, 'unknown').partition('/')[0]}"
    return groups


def comparison(pair: PairScores, *, strata: bool = True, sensitivity: Iterable[str] | None = None,
               gene_support: PairScores | None = None, clusterings: bool = False, changes: bool = False,
               donor: str | None = None) -> dict:
    """The pair's comparison (the module docstring): ``n_reads``, ``n_clusters`` and ``multi_read_clusters`` of its
    reads, the reads ``excluded`` per segment, the ``overall`` comparison, the ``informational`` metrics' summaries,
    and with ``strata`` its ``strata``. ``sensitivity`` holds ``sd3`` with ``sensitivity`` (the SD3-only genes left
    out) and ``gene_support`` with ``gene_support`` (the same pair in the gene-support variant: its thresholds, the
    reads it excludes per segment and the V R1 comparison). With ``clusterings``, ``clusterings`` holds the
    ``CLUSTERED`` metrics' comparison under each clustering, with its cluster count (the module docstring). With
    ``changes`` (two checkpoints of one tool), ``changes`` holds the reads ``b`` fixes and breaks against ``a`` on
    each 0/1 metric of the overall comparison, with examples (``stats.read_changes``; read ids carry the donor in a
    pool) and the reads changed per donor and protocol (``change_groups``, ``donor`` naming one donor's pair's)."""
    members = Counter(pair.clusters.get(read_id, read_id) for read_id in pair.a)
    result = {"n_reads": len(pair.a), "n_clusters": len(members),
              "multi_read_clusters": sum(count > 1 for count in members.values()), "excluded": dict(pair.excluded),
              "one_sided": pair.one_sided,
              "overall": _compare(pair.a, pair.b, pair.clusters), "informational": _informational(pair.a, pair.b)}
    if strata:
        result["strata"] = {label: _compare(*_subset(pair, lambda read_id: pair.strata[read_id] == label),
                                            pair.clusters)
                            for label in sorted(set(pair.strata.values()))}
    if sensitivity is not None:
        kept = _subset(pair, lambda read_id: read_id not in pair.sensitivity_out)
        result.setdefault("sensitivity", {})["sd3"] = {
            "genes": sorted(sensitivity), "excluded_reads": len(pair.sensitivity_out),
            "comparison": _compare(*kept, pair.clusters, metrics=V_R1)}
    if clusterings:
        result["clusterings"] = {name: {"n_clusters": len({clusters.get(read_id, read_id) for read_id in pair.a}),
                                        "comparison": _compare(pair.a, pair.b, clusters, metrics=CLUSTERED)}
                                 for name, clusters in (("exact", pair.clusters), *pair.cluster_sets.items())}
    if changes:
        family = sorted(result["overall"])
        result["changes"] = read_changes(pair.a, pair.b, family,
                                         lower_is_better={name for name in family if lower_is_better(name)},
                                         groups=change_groups(pair, donor))
    if gene_support is not None:
        result.setdefault("sensitivity", {})["gene_support"] = {
            "thresholds": {"min_clonotypes": MIN_CLONOTYPES, "min_naive_clonotypes": MIN_NAIVE_CLONOTYPES},
            "excluded": dict(gene_support.excluded),
            "comparison": _compare(gene_support.a, gene_support.b, gene_support.clusters, metrics=V_R1)}
    return result


def pool(pairs: Mapping[str, PairScores]) -> PairScores:
    """The pairs of several donors (donor -> pair) as one: read ids prefixed ``<donor>/``, clusters joined (their ids
    carry the donor, as do the other clusterings'), exclusions and one-sided counts summed."""
    joined = PairScores(a={}, b={}, clusters={}, strata={}, excluded=Counter(), sensitivity_out=set(),
                        one_sided={side: {"reads": 0, "reasons": Counter(), "documented": 0, "documented_wrong": 0}
                                   for side in ("a", "b")})
    for donor, pair in pairs.items():
        for name, clusters in pair.cluster_sets.items():
            joined.cluster_sets.setdefault(name, {}).update(
                {f"{donor}/{read_id}": cluster for read_id, cluster in clusters.items()})
        for side, counts in pair.one_sided.items():
            total = joined.one_sided[side]
            for key in ("reads", "documented", "documented_wrong"):
                total[key] += counts[key]
            total["reasons"].update(counts["reasons"])
        joined.a.update({f"{donor}/{read_id}": value for read_id, value in pair.a.items()})
        joined.b.update({f"{donor}/{read_id}": value for read_id, value in pair.b.items()})
        joined.clusters.update({f"{donor}/{read_id}": cluster for read_id, cluster in pair.clusters.items()})
        joined.strata.update({f"{donor}/{read_id}": label for read_id, label in pair.strata.items()})
        joined.excluded.update(pair.excluded)
        joined.sensitivity_out.update(f"{donor}/{read_id}" for read_id in pair.sensitivity_out)
    joined.excluded = {segment: joined.excluded.get(segment, 0) for segment in SEGMENTS}
    for counts in joined.one_sided.values():
        counts["reasons"] = dict(sorted(counts["reasons"].items()))
    return joined


def pool_tools(summaries: Iterable[dict]) -> dict:
    """Several donors' ``tool_summary`` of one run as one: the classes and exclusion counts summed, and the reads
    behind each mean added up (``summary`` keeps each metric's pooled mean and counts), in the R1 primary and in the
    gene-support variant (``gene_support``); and the consensus-weighted R1 over every donor's reads
    (``consensus_weighted``: each donor's weighted R1 by the reads behind it)."""
    summaries = list(summaries)
    weighted = [summary.get("consensus_weighted") or {} for summary in summaries]
    weight = sum(entry.get("weight") or 0 for entry in weighted if entry.get("real.v.unsupported") is not None)
    total = sum(entry["real.v.unsupported"] * entry["weight"] for entry in weighted
                if entry.get("real.v.unsupported") is not None)
    return {**_pool_block(summaries), "gene_support": _pool_block([summary["gene_support"] for summary in summaries]),
            "consensus_weighted": {"real.v.unsupported": total / weight if weight else None, "weight": weight}}


def _pool_block(summaries: list[dict]) -> dict:
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


def _own(data: DonorScoring, rows: Mapping[str, dict]) -> tuple[dict, dict[str, CaseScores]]:
    """A run's own R1 on ``rows`` (its own exclusions left out): its V classes, V exclusion counts and every metric's
    summary, and the scores themselves."""
    own = scores(data, rows, {read_id: _excluded(row) for read_id, row in rows.items()})
    return {"v_classes": _counts(row["v"] for row in rows.values()),
            "exclusions": exclusion_counts([row["v"] for row in rows.values()]),
            "summary": to_dict(summarize(own, list(own)))}, own


def _variant(rows: Mapping[str, dict], variant: str) -> Mapping[str, dict]:
    """``rows`` with their V class of R1 ``variant`` (``VARIANTS``) as ``v``. ValueError for another variant."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown R1 variant {variant!r}: {', '.join(VARIANTS)}")
    key = VARIANTS[variant]
    return rows if key == "v" else {read_id: {**row, "v": row[key]} for read_id, row in rows.items()}


def _excluded(row: dict) -> set[str]:
    return {segment for segment in SEGMENTS if _is_excluded(row[segment])}


def _call(pred: Prediction | None, segment: str) -> str | None:
    found = None if pred is None else pred.segments.get(segment)
    return None if found is None else found.call


def _is_excluded(cls: str | None) -> bool:
    return (cls or "").startswith(_EXCLUDED)


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
