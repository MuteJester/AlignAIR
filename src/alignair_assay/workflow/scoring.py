"""Score reports and comparison reports of tools against a truth set, and the files they are written to.

The ``score`` and ``compare`` commands and ``orchestrate.run_all`` build their reports here, so the files ``all``
writes under ``scores/<S>/<arm>/`` are the ones those commands write.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from alignair_assay.core.genotype import DonorGenotype
from alignair_assay.core.headline import HEADLINE
from alignair_assay.core.provenance import git_state, sha256_file
from alignair_assay.core.reference import Reference
from alignair_assay.metrics import CaseScores, is_informational, lower_is_better
from alignair_assay.metrics.case import MIN_OBSERVED
from alignair_assay.metrics.genotype import excess_alleles
from alignair_assay.report import build_comparison_report, build_score_report, to_markdown
from alignair_assay.scoring.clusters import CLUSTER_KEYS, cluster_ids
from alignair_assay.scoring.pipeline import load_predictions, score_predictions
from alignair_assay.scoring.slices import slices
from alignair_assay.stats import compare, read_changes, summarize
from alignair_assay.truth.model import ObservableTruth
from alignair_assay.truth.summary import simulator_calls

# Case ids named for the reads fixed and broken of a comparison of two checkpoints, per metric and overall.
CHANGE_EXAMPLES = 10
STATISTICAL_TESTS = ("exact McNemar for 0/1 metrics, paired z-test otherwise, p = 1 below 10 pairs; reads grouped "
                     "by clone or pair (cluster_keys) are compared by cluster: the clustered McNemar (0/1 metrics) "
                     "or clustered z score test on per-cluster sums, p = 1 below 10 clusters (too_few_clusters), "
                     "with cluster-robust (CR1) CIs; Holm across the family; normal-theory CIs (95% and "
                     "simultaneous Bonferroni)")


@dataclass(frozen=True)
class ToolScores:
    """One tool's per-case scores over a truth set's scored cases and its prediction/truth ``matching`` counts
    (``scoring.pipeline.score_tool``), and, when it was scored against donor genotypes, its ``excess_alleles``
    table (``metrics.genotype.excess_alleles``; None otherwise). ``score_file`` builds it."""

    tool: str
    scores: dict[str, CaseScores]
    matching: dict
    excess_alleles: dict | None = None


def score_file(tool: str, truths: list[ObservableTruth], predictions_path: str | Path, reference: Reference, *,
               genotypes: Mapping[str, DonorGenotype] | None = None) -> ToolScores:
    """Score ``tool``'s AIRR TSV against ``truths`` as ``scoring.pipeline.score_tool`` does (the genotype metrics
    included when donor ``genotypes`` are given) and, with ``genotypes``, its donor-level ``excess_alleles``
    table. ValueError for a file matching no scored case, or a case whose subject has no genotype."""
    preds, matching = load_predictions(truths, predictions_path)
    scores = score_predictions(truths, preds, reference, genotypes)
    excess = None if genotypes is None else excess_alleles(truths, preds, genotypes)
    return ToolScores(tool, scores, matching, excess)


def provenance(command: str, inputs: dict, **extra) -> dict:
    """Provenance of a report: the command, each input's path and SHA-256, the assay checkout's git state, a UTC
    timestamp, the minimum observed lengths in force, and ``extra``."""
    return {
        "command": command,
        "inputs": {k: {"path": str(p), "sha256": sha256_file(p)} for k, p in inputs.items()},
        **git_state(Path(__file__).resolve().parent),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "min_observed": dict(MIN_OBSERVED),
        **extra,
    }


def score_report(scored: ToolScores, truths: list[ObservableTruth], header: dict, *, command: str, inputs: dict,
                 reference_name: str) -> dict:
    """The score report of one tool: its overall and per-slice summaries, with the provenance of ``command`` run on
    ``inputs`` (name -> path) against ``truths`` (whose file header is ``header``), and its ``excess_alleles``
    table when it was scored against donor genotypes."""
    prov = provenance(command, inputs, tool=scored.tool, reference=reference_name,
                      reference_digest=header["reference_digest"], truth_header=header, matching=scored.matching,
                      simulator_calls=simulator_calls(truths))
    return build_score_report(scored.tool, summarize(scored.scores, list(scored.scores)),
                              slices(scored.scores, truths), prov, excess_alleles=scored.excess_alleles)


def comparison_report(a: ToolScores, b: ToolScores, truths: list[ObservableTruth], header: dict, *, command: str,
                      inputs: dict, reference_name: str, alpha: float = 0.05, changes: bool = False) -> dict:
    """The paired comparison report of tool ``a`` against tool ``b``. The family is every non-informational metric
    either tool produced, Holm-corrected at ``alpha``; every informational one either tool produced is reported
    beside it, each tool's own summary, untested. Reads of one clone or one orientation pair are compared as a
    cluster (``scoring.clusters.cluster_ids``, recorded as ``cluster_keys``). With ``changes`` (two checkpoints),
    ``changes`` holds the reads ``b`` fixes and breaks against ``a`` on each 0/1 metric of the family, with
    ``CHANGE_EXAMPLES`` examples and the counts per stratum (``stats.read_changes``). The provenance is as in
    ``score_report``. ValueError when the two tools have the same name (their files would collide)."""
    if a.tool == b.tool:
        raise ValueError(f"the two compared tools must differ, both are {a.tool!r}")
    seen = {name for scores in (a.scores, b.scores) for case in scores.values() for name in case}
    family = sorted(name for name in seen if not is_informational(name))
    comparison = compare(a.scores, b.scores, family, alpha=alpha,
                         lower_is_better={name for name in family if lower_is_better(name)},
                         clusters=cluster_ids(truths))
    summary_a, summary_b = summarize(a.scores, list(a.scores)), summarize(b.scores, list(b.scores))
    informational = {name: {"a": summary_a.get(name), "b": summary_b.get(name)}
                     for name in sorted(seen) if is_informational(name)}
    prov = provenance(command, inputs, a=a.tool, b=b.tool, reference=reference_name,
                      reference_digest=header["reference_digest"], truth_header=header, alpha=alpha,
                      tests=STATISTICAL_TESTS, cluster_keys=list(CLUSTER_KEYS), family_size=len(family),
                      matching={"a": a.matching, "b": b.matching}, simulator_calls=simulator_calls(truths))
    changed = None
    if changes:
        changed = read_changes(a.scores, b.scores, family, examples=CHANGE_EXAMPLES,
                               lower_is_better={name for name in family if lower_is_better(name)},
                               groups={truth.case_id: truth.stratum for truth in truths})
    return build_comparison_report(a.tool, b.tool, comparison, slices(a.scores, truths), slices(b.scores, truths),
                                   prov, informational=informational, changes=changed)


def write_report(report: dict, json_path: Path, md_path: Path, *, headline: Sequence[str] = HEADLINE) -> None:
    """Write a score or comparison report as strict JSON and as its markdown summary, led by the ``headline``
    metrics (``report.to_markdown``)."""
    Path(json_path).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    Path(md_path).write_text(to_markdown(report, headline=headline), encoding="utf-8")
