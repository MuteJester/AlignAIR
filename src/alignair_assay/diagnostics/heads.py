"""Head-level metrics: AlignAIR's raw head outputs, as the replay captured them, scored against observable truth.

``head_scores(outputs, truths, cases=None)`` gives each scored case a ``CaseScores`` of ``head.*`` metrics (absent
key: not applicable), and ``calibration_table`` the reliability of one gene's top-1 allele probability. Both are
pure functions of a ``StageOutputs`` and the truths; a case is matched to its replayed read by id.

**Frame.** The segment heads work on the read the model saw, in its forward frame, padded on the right to the logits
width (576 for the released model). A read longer than that is cropped by AlignAIR to the first ``width`` bases of
the presented read, which is the canonical prefix of a forward read and the canonical suffix of a reverse-complemented
one. Model positions are moved into the truth's canonical frame by that window's start. With a wrong orientation the
model's frame is not the truth's, and its boundaries are scored as wrong, as the product would place them.

**Boundaries** (``head.{g}.{edge}_{kind}.{decoder}``, ``edge`` ``start`` or ``end``, 0-based, end-exclusive like
``correct_segments``):

- ``soft``: the floor of the head's soft-argmax expectation (``{g}_start``/``{g}_end``), clipped to the read: the
  per-segment step of ``correct_segments`` without its min-span and ordering repair.
- ``argmax``: the argmax of the logits over the read's coordinates: a start lies on a read base (``0..n-1``), an
  end-exclusive end on ``0..n`` (position ``n`` exists when the read is shorter than the window). A segment that runs
  to the read end has end ``n``, which the model's soft-argmax, masked to ``0..n-1``, cannot reach.
- ``exact``: 1.0 when the decoded edge is inside the truth equivalence interval; ``abs_err``: its distance to the
  interval; ``err`` (informational): the signed distance (negative before the interval).

An edge is scored for an observed segment with at least ``MIN_OBSERVED`` germline bases (the end-to-end boundary
and allele metrics' rule), when its whole interval lies inside the cropped window without touching the cut.

**Alleles** (the calibrated probabilities ``predictions.allele``, which the selector sees, ranked by probability over
the read's allowed alleles, ties to the lower index): ``head.{g}.allele_top{1,3,5}`` (the top k meet the scored
truth set), ``head.{g}.allele_top1_generative`` (informational: the top-1 is in the generative set) and
``head.{g}.n_above_threshold`` (informational: allowed alleles at probability >= 0.5), for the same segments as the
boundaries. ``head.d.short_d_prob_ok`` applies only to a model whose D head has a Short-D output (the released IGH
model has none): P(Short-D) >= 0.5 exactly when the read's D is absent or shorter than ``MIN_OBSERVED["d"]``.

**Metadata.** ``head.mutation_rate_abs_err.read``: against the truth V mismatches over the observed V germline bases
(V observed with at least ``MIN_OBSERVED["v"]`` bases). ``head.mutation_rate_abs_err.record`` (informational, only
with ``cases``): against the GenAIRR record's ``mutation_rate``, the training target. ``head.indel_count_abs_err``:
against the inserted plus deleted bases of the truth's observed V, D and J CIGARs (GenAIRR's indels are single
bases), when every gene's truth was built. ``head.productive.correct``: probability >= 0.5 matches the truth, where
it is defined. ``head.orientation.correct``: the orientation logits' argmax is the truth orientation (0 or 1).
``head.cropped`` (informational): 1.0 for a read longer than the window, so its sum counts the crops.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from alignair.predict.airr.constants import SHORT_D_SENTINEL

from alignair_assay.core.cigar import parse_cigar
from alignair_assay.diagnostics.replay import DECODERS, StageOutputs, argmax_boundaries
from alignair_assay.metrics import CaseScores
from alignair_assay.metrics.boundaries import signed_interval_distance
from alignair_assay.metrics.case import MIN_OBSERVED
from alignair_assay.truth.boundaries import Interval
from alignair_assay.truth.model import ObservableTruth, SegmentTruth

GENES = ("v", "d", "j")
EDGES = ("start", "end")
TOP_K = (1, 3, 5)
# AlignAIR's allele selection threshold (``select_alleles``' default) and its productivity threshold.
THRESHOLD = 0.5

# Every name head_scores can give.
METRIC_NAMES: tuple[str, ...] = (
    *(f"head.{g}.{edge}_{kind}.{decoder}" for g in GENES for edge in EDGES for kind in ("exact", "abs_err", "err")
      for decoder in DECODERS),
    *(f"head.{g}.allele_top{k}" for g in GENES for k in TOP_K),
    *(f"head.{g}.{name}" for g in GENES for name in ("allele_top1_generative", "n_above_threshold")),
    "head.d.short_d_prob_ok",
    "head.mutation_rate_abs_err.read", "head.mutation_rate_abs_err.record", "head.indel_count_abs_err",
    "head.productive.correct", "head.orientation.correct", "head.cropped",
)


def decode_boundaries(outputs: StageOutputs, gene: str, decoder: str) -> tuple[np.ndarray, np.ndarray]:
    """``(starts, ends)`` of ``gene`` per read in the model's frame (0-based, end-exclusive), decoded by ``decoder``
    (see the module docstring); ValueError for an unknown decoder."""
    if decoder not in DECODERS:
        raise ValueError(f"decoder {decoder!r} is not one of {', '.join(DECODERS)}")
    lengths = np.array([len(read) for read in outputs.canonical])
    if decoder == "soft":
        starts, ends = (np.clip(np.floor(_flat(outputs.raw[f"{gene}_{edge}"])).astype(int), 0, lengths)
                        for edge in EDGES)
        return starts, ends
    return argmax_boundaries(outputs.raw[f"{gene}_start_logits"], outputs.raw[f"{gene}_end_logits"], lengths)


def allele_ranking(outputs: StageOutputs, gene: str) -> list[np.ndarray]:
    """Per read, the indices of the alleles ``gene``'s selector could call (all of them, or the read's ``allowed``
    row), by decreasing calibrated probability, ties to the lower index; empty when none is allowed. The first is
    the selector's top-1 call; the selector breaks ties the same way unless more than 16 alleles pass its threshold
    (numpy's default sort is stable only up to 16 elements)."""
    probs = np.asarray(outputs.predictions.allele[gene])
    mask = None if outputs.allowed is None else outputs.allowed.get(gene)
    mask = None if mask is None else np.asarray(mask, dtype=bool)
    ranking = []
    for i, row in enumerate(probs):
        candidates = np.arange(row.shape[0]) if mask is None else np.flatnonzero(mask[i] if mask.ndim == 2 else mask)
        ranking.append(candidates[np.argsort(-row[candidates], kind="stable")])
    return ranking


def head_scores(outputs: StageOutputs, truths: list[ObservableTruth],
                cases: Iterable[dict] | None = None) -> dict[str, CaseScores]:
    """Head-level metrics per non-excluded truth, in truth order, keyed by case id (see the module docstring).

    ``cases`` (the suite's case dicts) add ``head.mutation_rate_abs_err.record``. ValueError when a scored case has
    no replayed read, its replayed read is not as long as its truth read would be after the crop, or ``cases`` is
    given without one of the scored cases.
    """
    pairs = _pairs(outputs, truths)
    records = _records(cases, pairs)
    width = _width(outputs)
    raw = {key: _flat(outputs.raw[key]) for key in ("mutation_rate", "indel_count", "productive")}
    genes = [g for g in GENES if f"{g}_start_logits" in outputs.raw]
    decoded = {g: {decoder: decode_boundaries(outputs, g, decoder) for decoder in DECODERS} for g in genes}
    alleles = [g for g in genes if g in outputs.allele_names]
    ranking = {g: allele_ranking(outputs, g) for g in alleles}
    probs = {g: np.asarray(outputs.predictions.allele[g]) for g in alleles}
    short_d = _short_d_index(outputs)

    scores: dict[str, CaseScores] = {}
    for truth, i in pairs:
        case: CaseScores = {"head.cropped": float(len(truth.sequence) > width)}
        window = read_window(truth, width)
        for g in genes:
            seg = _scored_segment(truth, g)
            if seg is None:
                continue
            case.update(_boundary_scores(g, seg, i, decoded[g], window))
            if g in ranking:
                case.update(_allele_scores(g, seg, ranking[g][i], probs[g][i], outputs.allele_names[g]))
        if short_d is not None and "d" not in truth.unscorable:
            d = truth.segments.get("d")
            short = d is None or d.observed_length < MIN_OBSERVED["d"]
            case["head.d.short_d_prob_ok"] = float((probs["d"][i, short_d] >= THRESHOLD) == short)
        v = _scored_segment(truth, "v")
        if v is not None and truth.v_mismatches is not None:
            case["head.mutation_rate_abs_err.read"] = abs(float(raw["mutation_rate"][i])
                                                          - truth.v_mismatches / v.observed_length)
        rate = None if records is None else records[truth.case_id].get("mutation_rate")
        if rate is not None:
            case["head.mutation_rate_abs_err.record"] = abs(float(raw["mutation_rate"][i]) - float(rate))
        if not truth.unscorable:
            indels = sum(n for seg in truth.segments.values() for n, op in parse_cigar(seg.observed.cigar)
                         if op in "ID")
            case["head.indel_count_abs_err"] = abs(float(raw["indel_count"][i]) - indels)
        if truth.productive is not None:
            case["head.productive.correct"] = float((float(raw["productive"][i]) >= THRESHOLD) == truth.productive)
        orientation = int(np.argmax(outputs.raw["orientation_logits"][i]))
        case["head.orientation.correct"] = float(orientation == truth.orientation)
        scores[truth.case_id] = case
    return scores


def calibration_table(outputs: StageOutputs, truths: list[ObservableTruth], gene: str, bins: int = 10) -> dict:
    """Reliability of ``gene``'s top-1 calibrated allele probability against the top-1 being in the scored truth
    set, over the cases its allele head metrics apply to (a read with no allowed allele has no top-1 and is left
    out).

    The ``bins`` equal-width bins are closed on the right and the first is closed on both sides, as in AlignAIR's
    ``calibrate.ece``. Returns ``gene``, ``n``, ``bins`` (per bin ``lo``, ``hi``, ``n``, mean ``confidence`` and
    ``accuracy``, None when empty), the expected calibration error ``ece`` (the bins' count-weighted
    |accuracy - confidence|) and the overall ``confidence`` and ``accuracy`` (None without a case). ValueError for
    fewer than one bin or a gene without an allele head.
    """
    if bins < 1:
        raise ValueError(f"a calibration table needs at least one bin, got {bins}")
    if gene not in outputs.allele_names:
        raise ValueError(f"the model has no allele head for gene {gene!r}")
    ranking = allele_ranking(outputs, gene)
    probs = np.asarray(outputs.predictions.allele[gene], dtype=np.float64)
    names = outputs.allele_names[gene]
    confidence, correct = [], []
    for truth, i in _pairs(outputs, truths):
        seg = _scored_segment(truth, gene)
        if seg is None or ranking[i].size == 0:
            continue
        top = int(ranking[i][0])
        confidence.append(probs[i, top])
        correct.append(float(names[top] in seg.alleles.alleles))
    conf, corr = np.asarray(confidence), np.asarray(correct)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        members = which == b
        k = int(members.sum())
        mean_conf = float(conf[members].mean()) if k else None
        mean_acc = float(corr[members].mean()) if k else None
        table.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": k, "confidence": mean_conf,
                      "accuracy": mean_acc})
        if k:
            ece += k / len(conf) * abs(mean_acc - mean_conf)
    n = len(conf)
    return {"gene": gene, "n": n, "bins": table, "ece": ece if n else None,
            "confidence": float(conf.mean()) if n else None, "accuracy": float(corr.mean()) if n else None}


def _flat(values) -> np.ndarray:
    return np.asarray(values).reshape(-1)


def _width(outputs: StageOutputs) -> int:
    """The model's window: the length of its position logits."""
    return int(np.asarray(outputs.raw["v_start_logits"]).shape[1])


def _pairs(outputs: StageOutputs, truths: list[ObservableTruth]) -> list[tuple[ObservableTruth, int]]:
    """``(truth, read index)`` per non-excluded truth, in truth order, checked against the replayed reads."""
    position = {case_id: i for i, case_id in enumerate(outputs.ids)}
    width = _width(outputs)
    pairs = []
    for truth in truths:
        if truth.excluded is not None:
            continue
        i = position.get(truth.case_id)
        if i is None:
            raise ValueError(f"case {truth.case_id!r} has no replayed read")
        if len(outputs.canonical[i]) != min(len(truth.sequence), width):
            raise ValueError(f"case {truth.case_id!r}: the replayed read has {len(outputs.canonical[i])} bases, its "
                             f"truth read {len(truth.sequence)} (model window {width}); they are not the same read")
        pairs.append((truth, i))
    return pairs


def _records(cases: Iterable[dict] | None, pairs: list[tuple[ObservableTruth, int]]) -> dict[str, dict] | None:
    """Case id -> GenAIRR record for the scored cases; None without ``cases``."""
    if cases is None:
        return None
    records = {case["case_id"]: case["record"] for case in cases}
    for truth, _ in pairs:
        if truth.case_id not in records:
            raise ValueError(f"case {truth.case_id!r} has no case record")
    return records


def read_window(truth: ObservableTruth, width: int) -> tuple[int, int, int | None]:
    """``(start, end, cut)`` of the canonical read bases the model saw; ``cut`` is the canonical position where a
    cropped read was cut (None for a read that fits)."""
    n = len(truth.sequence)
    if n <= width:
        return 0, n, None
    if truth.orientation == 1:
        return n - width, n, n - width
    return 0, width, width


def _scored_segment(truth: ObservableTruth, g: str) -> SegmentTruth | None:
    """The segment when its head metrics apply: observed, scorable and at least ``MIN_OBSERVED`` bases long."""
    if g in truth.unscorable:
        return None
    seg = truth.segments.get(g)
    if seg is None or seg.observed_length < MIN_OBSERVED[g]:
        return None
    return seg


def _seen(interval: Interval, window: tuple[int, int, int | None]) -> bool:
    lo, hi, cut = window
    return lo <= interval[0] and interval[1] <= hi and not (cut is not None and interval[0] <= cut <= interval[1])


def _boundary_scores(g: str, seg: SegmentTruth, i: int, decoded: dict, window: tuple[int, int, int | None]
                     ) -> CaseScores:
    scores: CaseScores = {}
    for decoder, (starts, ends) in decoded.items():
        for edge, values, interval in (("start", starts, seg.boundaries.seq_start),
                                       ("end", ends, seg.boundaries.seq_end)):
            if not _seen(interval, window):
                continue
            signed = signed_interval_distance(int(values[i]) + window[0], interval)
            scores[f"head.{g}.{edge}_exact.{decoder}"] = float(signed == 0)
            scores[f"head.{g}.{edge}_abs_err.{decoder}"] = float(abs(signed))
            scores[f"head.{g}.{edge}_err.{decoder}"] = float(signed)
    return scores


def _allele_scores(g: str, seg: SegmentTruth, ranking: np.ndarray, probs: np.ndarray, names: list[str]
                   ) -> CaseScores:
    top = [names[k] for k in ranking[:max(TOP_K)]]
    scores: CaseScores = {f"head.{g}.allele_top{k}": float(any(name in seg.alleles.alleles for name in top[:k]))
                          for k in TOP_K}
    scores[f"head.{g}.allele_top1_generative"] = float(bool(top) and top[0] in seg.alleles.generative)
    scores[f"head.{g}.n_above_threshold"] = float(np.count_nonzero(probs[ranking] >= THRESHOLD))
    return scores


def _short_d_index(outputs: StageOutputs) -> int | None:
    """The D head's Short-D column, or None when the model has no such output."""
    names = outputs.allele_names.get("d") or []
    return names.index(SHORT_D_SENTINEL) if SHORT_D_SENTINEL in names else None
