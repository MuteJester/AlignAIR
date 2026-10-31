"""Truth-substitution ablations: how much of AlignAIR's end-to-end error each stage contributes.

A variant replays AlignAIR's predict pipeline (``diagnostics.replay``) with the observable truth fed in place of
some stages' output, then scores its AIRR rows exactly as the assay scores a tool (``score_predictions``, with the
donors' genotypes when given).

**Variants.** The cumulative chain (``VARIANTS``) adds the truth of one more stage at each step:

1. ``baseline``: the product pipeline, nothing substituted.
2. ``true_orientation``: the truth orientation, teacher-forced in the network and used to canonicalize the read.
3. ``true_spans``: the truth read spans of the observed segments, as the segments and as the classifier's input (a
   model with a core reads the whole read for its calls: only its segments change).
4. ``true_calls``: the truth (primary) allele as each observed segment's call.
5. ``true_germline``: the truth germline alignment (allele, read and germline coordinates, CIGAR) of each observed
   segment, in place of the germline matcher.

What still differs from the oracle after ``true_germline`` comes from the metadata heads (the productive and indel
predictions that gate the assembly) and the AIRR assembly itself. A model with anchor heads takes its junction and
flags from its anchors instead: only ``true_orientation`` moves them (the anchors are recomputed in the true frame),
the later steps move only the coding span the stop is searched over, and what differs after ``true_germline`` is the
anchor heads' error (the README's "Truth-substitution ablations" has the detail). The single swaps (``SINGLE_SWAPS``) change one
thing on the baseline: ``soft_decoder`` (segments from the soft-argmax expectations, the decoder the product used before it decoded
by argmax; the classifier unchanged) and ``true_classifier_spans`` (truth spans into the classifier only; the segments stay the model's).

**Truth in AlignAIR's terms** (``substitutions_for``). AlignAIR works on the read it saw: the first 576 bases of the
presented read (the canonical prefix of a forward read, the canonical suffix of a reverse-complemented one), in its
forward frame, 0-based and end-exclusive, with ungapped germline coordinates. A truth segment's observed read span
is moved into that window; a span is a ``(start, end)`` pair; a call is ``GeneCall((primary,), (1.0,))`` (the
likelihood of a substituted call is 1); an alignment is ``GermlineAlignment(allele, seq_start, seq_end, germ_start,
germ_end, cigar)`` from the observed segment (its germline coordinates and CIGAR are already ungapped and
gap-free at the edges).

**Fallbacks.** Truth the read does not have is never invented: the read keeps the model's value for that segment,
and ``fallbacks`` counts it as ``"{field}.{gene}.{reason}"`` (field ``spans``, ``calls`` or ``germline``), with the
first reason that applies:

- ``unscorable``: the truth of the segment could not be built;
- ``unobserved``: the read shows no base of the segment;
- ``orientation``: (spans, germline) the variant does not substitute the orientation and the model's is wrong, so the
  truth coordinates are not in the model's frame;
- ``cropped``: the segment is not wholly inside the window AlignAIR saw of an over-length read;
- ``not_in_model``: (calls, germline) the model cannot call the primary allele;
- ``inverted``: (germline) an inverted D, whose germline frame (the reverse-complemented allele) AlignAIR's
  alignments and assembly cannot represent; its read span and call are still substituted.

Excluded cases and reads without a truth keep every model value and are not counted (they are not scored).

**Scoring** (ruling D2). A variant's rows become the text rows AlignAIR's AIRR writer puts in a TSV
(``airr_text_rows``: ``alignair.io.airr._build_row`` over its full column set) and are read with the assay's AIRR
reader (``airr.reader.rows_to_predictions``, the parser ``load_airr_tsv`` uses), without a file. The tests pin this
against a TSV written by ``alignair.io.airr.write_airr``. Rows are scored against the truth reference the caller
gives, and ``run_variants`` refuses a model whose own reference is not that one (``replay.check_reference``), so an
in-memory model never reaches scoring unchecked.

**Stage contributions.** ``stage_contributions`` compares each consecutive pair of the chain (paired, Holm-corrected
over the metrics: ``stats.compare``): the change when the truth of one more stage is fed in is the error that stage
contributes, given perfect upstream stages. ``swap_effects`` compares each single swap with the baseline. Each
comparison gives the paired means (over the cases both variants cover) beside its delta, which is their difference,
and each variant's own mean over its own cases with its coverage. Given ``clusters`` (``scoring.clusters``: the reads
of one clone or one orientation pair), the comparisons are tested by cluster, not by read.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from alignair.io.airr import COLUMNS, _build_row
from alignair.predict.state import GeneCall, GermlineAlignment

from alignair_assay.airr.reader import rows_to_predictions
from alignair_assay.core.genotype import DonorGenotype
from alignair_assay.core.headline import HEADLINE
from alignair_assay.core.reference import Reference
from alignair_assay.diagnostics.heads import read_window
from alignair_assay.diagnostics.replay import StageOutputs, Substitutions, check_reference, replay
from alignair_assay.metrics import CaseScores, lower_is_better
from alignair_assay.scoring.pipeline import score_predictions
from alignair_assay.stats import compare
from alignair_assay.truth.model import ObservableTruth

# The cumulative chain, in order, and the single swaps on the baseline.
VARIANTS: tuple[str, ...] = ("baseline", "true_orientation", "true_spans", "true_calls", "true_germline")
SINGLE_SWAPS: tuple[str, ...] = ("soft_decoder", "true_classifier_spans")
DEFAULT_VARIANTS: tuple[str, ...] = VARIANTS + SINGLE_SWAPS
# The stage whose truth each step of the chain adds.
STAGE_OF = {"true_orientation": "orientation", "true_spans": "segmentation", "true_calls": "allele_calls",
            "true_germline": "germline_alignment"}
# What each variant substitutes ("classifier": the spans into the classifier).
_CHAIN = ({"orientation"}, {"spans", "classifier"}, {"calls"}, {"germline"})
_FIELDS: dict[str, frozenset[str]] = {
    **{name: frozenset().union(*_CHAIN[:k]) for k, name in enumerate(VARIANTS)},
    "soft_decoder": frozenset({"decoder"}),
    "true_classifier_spans": frozenset({"classifier"}),
}
# The likelihood written for a substituted (truth) call.
TRUE_CALL_LIKELIHOOD = 1.0
# The metrics a stage-contribution table leads with: the headline metrics, then the junction translation, the read
# coordinates and the orientation.
CONTRIBUTION_METRICS: tuple[str, ...] = (
    *HEADLINE, "junction.aa_exact",
    *(f"{g}.{edge}_exact" for g in ("v", "d", "j") for edge in ("seq_start", "seq_end")),
    "orientation.correct",
)
_VERDICT = {"b_better": "to_better", "a_better": "from_better", "tie_or_inconclusive": "tie_or_inconclusive"}


@dataclass(frozen=True)
class VariantResult:
    """One variant's replay, scored."""

    variant: str
    # the variant's AIRR rows (build_airr output, one per case, in case order)
    rows: list[dict]
    # per scored case, keyed by case id (scoring.pipeline.score_predictions)
    scores: dict[str, CaseScores]
    # "{field}.{gene}.{reason}" -> reads that kept the model's value (see the module docstring)
    fallbacks: dict[str, int]
    # the replay's wall time per stage (replay.STAGES)
    seconds: dict[str, float]
    # what the variant substituted (Substitutions.active())
    substituted: tuple[str, ...]


def substitutions_for(variant: str, truths: Sequence[ObservableTruth], outputs_baseline: StageOutputs
                      ) -> tuple[Substitutions, dict[str, int]]:
    """The ``Substitutions`` of ``variant`` for the reads of ``outputs_baseline`` (a pure replay, which gives the
    reads, their crop flags, the model's genes, allele names and orientation), built from ``truths`` (matched by
    case id), and the fallback counts (see the module docstring). ValueError for an unknown variant, or a read whose
    length does not match its truth read (after AlignAIR's crop)."""
    if variant not in _FIELDS:
        raise ValueError(f"unknown variant {variant!r}; known: {', '.join(DEFAULT_VARIANTS)}")
    fields = _FIELDS[variant]
    decoder = "soft" if "decoder" in fields else "argmax"
    if not fields - {"decoder"}:
        return Substitutions(decoder=decoder), {}
    out = outputs_baseline
    n = len(out.ids)
    genes = list(out.segments.start)
    by_id = {t.case_id: t for t in truths}
    fallbacks: Counter[str] = Counter()
    orientation = [None] * n if "orientation" in fields else None
    spans = {g: ([None] * n, [None] * n) for g in genes} if fields & {"spans", "classifier"} else None
    calls = {g: [None] * n for g in genes} if "calls" in fields else None
    germline = {g: [None] * n for g in genes} if "germline" in fields else None
    built = {field: target for field, target in (("spans", spans), ("calls", calls), ("germline", germline))
             if target is not None}

    for i, case_id in enumerate(out.ids):
        truth = by_id.get(case_id)
        if truth is None or truth.excluded is not None:
            continue
        lo, hi = _window(truth, out, i)
        if orientation is not None:
            orientation[i] = truth.orientation
        framed = orientation is not None or int(out.predictions.orientation[i]) == truth.orientation
        for g in genes:
            reasons = _fallback_reasons(truth, g, (lo, hi), framed, out.allele_names.get(g, ()))
            for field, target in built.items():
                if reasons[field] is not None:
                    fallbacks[f"{field}.{g}.{reasons[field]}"] += 1
                    continue
                seg = truth.segments[g]
                o = seg.observed
                if field == "spans":
                    target[g][0][i], target[g][1][i] = o.seq_start - lo, o.seq_end - lo
                elif field == "calls":
                    target[g][i] = GeneCall((seg.alleles.primary,), (TRUE_CALL_LIKELIHOOD,))
                else:
                    target[g][i] = GermlineAlignment(o.allele, o.seq_start - lo, o.seq_end - lo, o.germ_start,
                                                     o.germ_end, o.cigar)

    if orientation is not None:
        # The model's own orientation fills the entries without truth: the same value the replay would use, and it
        # keeps the replay from running a second forward pass for those reads.
        own = out.predictions.orientation
        orientation = [int(own[i]) if o is None else o for i, o in enumerate(orientation)]
    substitutions = Substitutions(orientation=orientation, spans=spans, segment_spans="spans" in fields,
                                  classifier_spans="classifier" in fields, calls=calls, germline=germline,
                                  decoder=decoder)
    return substitutions, dict(sorted(fallbacks.items()))


def run_variants(model, reference_set, cases: Sequence[dict], truths: Sequence[ObservableTruth], *,
                 reference: Reference, device: str, batch_size: int, variants: Sequence[str] = DEFAULT_VARIANTS,
                 genotypes: Mapping[str, DonorGenotype] | None = None, _baseline: StageOutputs | None = None
                 ) -> dict[str, VariantResult]:
    """Replay and score each of ``variants`` (in the given order) on the reads of ``cases`` (the suite's case dicts:
    ``case_id`` and the presented ``sequence``).

    A pure replay (the baseline) comes first: it gives the substitutions their reads and the model's own values.
    Each variant's rows are scored against the truths of the replayed cases (others are ignored), with the donor
    ``genotypes`` when given, and against ``reference``, the truth reference. ValueError, before anything is
    replayed, when the model's reference is not ``reference`` (``replay.check_reference``: it names the first
    differing allele), for an unknown variant, or when no truth is scored.

    ``_baseline`` is private to ``workflow.diagnose``, which has already replayed the cases for the head metrics: it
    must be that pure replay, of this ``model`` and ``reference_set`` on these cases' reads with this device and batch
    size (``replay(model, reference_set, sequences, ids, ...)`` without substitutions), and is then not run again.
    Only its reads are checked (ValueError for other reads than the cases', in their order); a replay of another
    model would silently become the baseline."""
    check_reference(reference_set, reference)
    unknown = [name for name in variants if name not in _FIELDS]
    if unknown:
        raise ValueError(f"unknown variant {unknown[0]!r}; known: {', '.join(DEFAULT_VARIANTS)}")
    ids = [case["case_id"] for case in cases]
    sequences = [case["sequence"] for case in cases]
    replayed = set(ids)
    truths = [t for t in truths if t.case_id in replayed]
    if not any(t.excluded is None for t in truths):
        raise ValueError("no truth to score: none of the truths is a scored case among the replayed cases")
    if _baseline is not None and list(_baseline.ids) != ids:
        raise ValueError(f"the baseline replay is of other reads: {len(_baseline.ids)} reads, not the {len(ids)} "
                         "cases' in their order")
    locus = _locus(reference_set)
    baseline = (replay(model, reference_set, sequences, ids, device=device, batch_size=batch_size)
                if _baseline is None else _baseline)

    results: dict[str, VariantResult] = {}
    for name in dict.fromkeys(variants):
        substitutions, fallbacks = substitutions_for(name, truths, baseline)
        outputs = baseline if name == "baseline" else replay(model, reference_set, sequences, ids, device=device,
                                                             batch_size=batch_size, substitutions=substitutions)
        preds, _ = rows_to_predictions(airr_text_rows(ids, sequences, outputs.airr_rows, locus))
        results[name] = VariantResult(variant=name, rows=outputs.airr_rows,
                                      scores=score_predictions(truths, preds, reference, genotypes),
                                      fallbacks=fallbacks, seconds=dict(outputs.seconds),
                                      substituted=substitutions.active())
    return results


def airr_text_rows(ids: Sequence[str], sequences: Sequence[str], rows: Sequence[dict], locus: str
                   ) -> list[dict[str, str]]:
    """The TSV rows ``alignair predict`` writes for ``rows`` (``build_airr`` output) of the reads ``sequences``
    named ``ids``: AlignAIR's AIRR writer row (``alignair.io.airr._build_row``) over its full column set, each cell as
    the text a TSV reader reads back (None empty, anything else ``str``)."""
    return [{column: "" if row.get(column) is None else str(row[column]) for column in COLUMNS}
            for row in (_build_row(sid, seq, record, locus) for sid, seq, record in zip(ids, sequences, rows))]


def stage_contributions(results: Mapping[str, VariantResult], metrics: Sequence[str], *, alpha: float = 0.05,
                        clusters: Mapping[str, str] | None = None) -> list[dict]:
    """For each consecutive pair of the cumulative chain present in ``results`` (in chain order; a missing variant
    is skipped, so its pair spans its stages too): ``from``, ``to``, the ``stages`` whose truth the step adds and
    the per-metric comparison (``_comparison``, clustered by ``clusters``)."""
    chain = [name for name in VARIANTS if name in results]
    return [{"from": a, "to": b,
             "stages": [STAGE_OF[name] for name in VARIANTS[VARIANTS.index(a) + 1:VARIANTS.index(b) + 1]],
             "metrics": _comparison(results[a], results[b], metrics, alpha, clusters)}
            for a, b in zip(chain, chain[1:])]


def swap_effects(results: Mapping[str, VariantResult], metrics: Sequence[str], *, alpha: float = 0.05,
                 clusters: Mapping[str, str] | None = None) -> list[dict]:
    """Each single swap present in ``results`` against the baseline: ``from`` (``baseline``), ``to`` and the
    per-metric comparison (``_comparison``, clustered by ``clusters``); empty without a baseline."""
    if "baseline" not in results:
        return []
    return [{"from": "baseline", "to": name,
             "metrics": _comparison(results["baseline"], results[name], metrics, alpha, clusters)}
            for name in SINGLE_SWAPS if name in results]


def _comparison(a: VariantResult, b: VariantResult, metrics: Sequence[str], alpha: float,
                clusters: Mapping[str, str] | None = None) -> dict[str, dict]:
    """Per metric, ``stats.compare`` of ``a`` (``from``) against ``b`` (``to``), Holm-corrected over ``metrics``
    and clustered by ``clusters`` (case id -> cluster, ``scoring.clusters.cluster_ids``): each variant's own mean,
    value count and coverage (``mean_from``, ``n_from``, ``coverage_from`` and the ``_to`` twins); the means over the
    ``n_paired`` cases both cover, in ``n_clusters`` clusters (``mean_from_paired``, ``mean_to_paired``; None
    without a pair) and the paired ``delta``, their difference (to - from), with its ``ci95``; the ``test``,
    ``p_value`` and ``verdict`` (``to_better``, ``from_better`` or ``tie_or_inconclusive``, oriented by the metric
    registry's lower-is-better rule)."""
    metrics = list(metrics)
    found = compare(a.scores, b.scores, metrics, alpha=alpha,
                    lower_is_better={name for name in metrics if lower_is_better(name)}, clusters=clusters)
    return {name: {"mean_from": e["a"].mean if e["a"] else None, "n_from": e["a"].n if e["a"] else 0,
                   "coverage_from": e["coverage_a"],
                   "mean_to": e["b"].mean if e["b"] else None, "n_to": e["b"].n if e["b"] else 0,
                   "coverage_to": e["coverage_b"],
                   "mean_from_paired": e["mean_a_paired"], "mean_to_paired": e["mean_b_paired"],
                   "delta": e["delta"], "n_paired": e["n_paired"], "n_clusters": e["n_clusters"],
                   "ci95": e["ci95"], "test": e["test"], "p_value": e["p_value"], "verdict": _VERDICT[e["verdict"]]}
            for name, e in found.items()}


def _fallback_reasons(truth: ObservableTruth, g: str, window: tuple[int, int], framed: bool, names
                      ) -> dict[str, str | None]:
    """Per field (``spans``, ``calls``, ``germline``), the first fallback reason that applies to gene ``g`` of the
    read (None: the truth is substituted); ``framed``: the read is in the truth's frame."""
    seg = truth.segments.get(g)
    missing = "unscorable" if g in truth.unscorable else "unobserved" if seg is None else None
    if missing:
        return dict.fromkeys(("spans", "calls", "germline"), missing)
    lo, hi = window
    cropped = None if lo <= seg.observed.seq_start and seg.observed.seq_end <= hi else "cropped"
    frame = None if framed else "orientation"
    call = cropped or (None if seg.alleles.primary in names else "not_in_model")
    return {"spans": frame or cropped, "calls": call,
            "germline": frame or call or ("inverted" if seg.inverted else None)}


def _window(truth: ObservableTruth, outputs: StageOutputs, i: int) -> tuple[int, int]:
    """``(start, end)`` of the truth's canonical read that replayed read ``i`` holds (the window AlignAIR saw);
    ValueError when the lengths show they are not the same read."""
    seen, n = len(outputs.canonical[i]), len(truth.sequence)
    cropped = bool(outputs.records[i].get("length_cropped"))
    if seen > n or (seen < n) != cropped:
        raise ValueError(f"case {truth.case_id!r}: the replayed read has {seen} bases, its truth read {n}; they are "
                         f"not the same read")
    lo, hi, _ = read_window(truth, seen)
    return lo, hi


def _locus(reference_set) -> str:
    """The locus ``alignair predict`` writes: the model's one locus, else IGH."""
    loci = reference_set.locus_names() if hasattr(reference_set, "locus_names") else ()
    return loci[0] if len(loci) == 1 else "IGH"
