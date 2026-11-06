"""Head-level metrics: AlignAIR's raw head outputs, as the replay captured them, scored against observable truth.

``head_scores(outputs, truths, cases=None)`` gives each scored case a ``CaseScores`` of ``head.*`` metrics (absent
key: not applicable), a pure function of a ``StageOutputs`` and the truths; a case is matched to its replayed read
by id.

**Frame.** The segment heads work on the read the model saw, in its forward frame, padded on the right to the logits
width (576 for the released model). A read longer than that is cropped by AlignAIR to the first ``width`` bases of
the presented read, which is the canonical prefix of a forward read and the canonical suffix of a reverse-complemented
one. Model positions are moved into the truth's canonical frame by that window's start.

**Misoriented reads.** When the model's orientation (``predictions.orientation``, the frame the replay canonicalized
the read in) is not the truth's, its positions are on another strand or in another direction than the truth's, so
no boundary of the read is scored: comparing them would score a frame mismatch, which can also land on the truth by
chance, as a boundary error. ``head.misoriented`` (informational) is 1.0 for such a read, so its sum counts them; the
orientation head's own error is ``head.orientation.correct``, and the end-to-end cost of a wrong orientation is the
``true_orientation`` step of the ablations. The read's allele and metadata heads are still scored.

**The window's last position.** The logits have ``width`` positions, ``0..width-1``. An end-exclusive edge at
``width`` itself (a segment that runs to the end of a read of exactly ``width`` bases, or, on a cropped
reverse-complemented read, to the canonical end, which is the end of the window) cannot be decoded by either decoder,
yet it is scored when its interval lies inside the window: such an edge is always at least one base off. It is rare
(no released-model suite has many reads of exactly 576 bases) and is left in, not excluded.

**Cropped reads.** Only the boundaries a crop hides are left out. The allele and metadata heads of a cropped read
are scored against the truth of the whole read, although the model saw only its window: a segment cut by the crop is
still scored on its alleles, and the mutation rate and indel count still count the bases past the cut.

**Boundaries** (``head.{g}.{edge}_{kind}.{decoder}``, ``edge`` ``start`` or ``end``, 0-based, end-exclusive like
``correct_segments``):

- ``soft``: the floor of the head's soft-argmax expectation (``{g}_start``/``{g}_end``), clipped to the read: the
  per-segment step of ``correct_segments`` without its min-span and ordering repair.
- ``argmax``: the product's decoder (``alignair.predict.decode``) over the model's own half-open boundary slots: a
  start lies on a read base (``0..n-1``), an end-exclusive end on ``1..n``. A segment that runs to the read end has
  end ``n``.
- ``exact``: 1.0 when the decoded edge is inside the truth equivalence interval; ``abs_err``: its distance to the
  interval; ``err`` (informational): the signed distance (negative before the interval).

An edge is scored for an observed segment with at least ``MIN_OBSERVED`` germline bases (the end-to-end boundary
and allele metrics' rule), when its whole interval lies inside the cropped window without touching the cut.

**Alleles** (the allele scores ``predictions.allele``, which the selector sees, ranked by score over the bank's
alleles, ties to the lower index): ``head.{g}.allele_top{1,3,5}`` (the top k meet the scored truth set),
``head.{g}.allele_top1_generative`` (informational: the top-1 is in the generative set) and
``head.{g}.n_above_threshold`` (informational: alleles scoring >= 0.5), for the same segments as the boundaries.

**D presence** (a model with a D presence head, ``d_presence_logits``; the product gates the D call on its argmax):
``head.d_presence.correct``, the head's class is the truth's (``d_class``: absent, short below ``MIN_OBSERVED["d"]``
observed germline bases, callable), on every read whose D truth was built and that the window does not crop; and
``head.d_presence.callable_gated``, on the truth-callable reads, 1.0 when the head called them absent or short, so the
gate dropped a D call the assay scores (its sum counts them). ``d_presence_confusion`` gives the truth × head counts.

**Metadata.** ``head.mutation_rate_abs_err.read``: against the truth V mismatches over the observed V germline bases
(V observed with at least ``MIN_OBSERVED["v"]`` bases). ``head.mutation_rate_abs_err.record`` (informational, only
with ``cases``): against the GenAIRR record's ``mutation_rate``, the training target. ``head.indel_count_abs_err``:
against the inserted plus deleted bases of the truth's observed V, D and J CIGARs (GenAIRR's indels are single
bases), when every gene's truth was built. (The chip's productivity is derived from its anchors and scored on the
AIRR row as ``productive.correct``.) ``head.orientation.correct``: the orientation logits' argmax is the truth
orientation (0 or 1).
``head.cropped`` (informational): 1.0 for a read longer than the window, so its sum counts the crops;
``head.misoriented`` (informational): 1.0 for a read in another frame than the truth's, whose boundaries are not
scored.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from alignair.core.boundaries import read_window as model_window
from alignair.core.d_presence import ABSENT, CALLABLE, D_CLASSES, D_PRESENCE_LOGITS, SHORT
from alignair.predict.decode import boundary_slots, decode_boundaries as product_decode

from alignair_assay.core.cigar import parse_cigar
from alignair_assay.diagnostics.replay import DECODERS, StageOutputs
from alignair_assay.metrics import CaseScores
from alignair_assay.metrics.boundaries import signed_interval_distance
from alignair_assay.metrics.case import MIN_OBSERVED
from alignair_assay.truth.boundaries import Interval
from alignair_assay.truth.model import ObservableTruth, SegmentTruth

GENES = ("v", "d", "j")
EDGES = ("start", "end")
TOP_K = (1, 3, 5)
# AlignAIR's allele selection threshold (``select_alleles``' default).
THRESHOLD = 0.5

# Every name head_scores can give.
METRIC_NAMES: tuple[str, ...] = (
    *(f"head.{g}.{edge}_{kind}.{decoder}" for g in GENES for edge in EDGES for kind in ("exact", "abs_err", "err")
      for decoder in DECODERS),
    *(f"head.{g}.allele_top{k}" for g in GENES for k in TOP_K),
    *(f"head.{g}.{name}" for g in GENES for name in ("allele_top1_generative", "n_above_threshold")),
    "head.d_presence.correct", "head.d_presence.callable_gated",
    "head.mutation_rate_abs_err.read", "head.mutation_rate_abs_err.record", "head.indel_count_abs_err",
    "head.orientation.correct", "head.cropped", "head.misoriented",
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
    start_slots, end_slots = _slots(outputs)
    starts, ends = product_decode(torch.as_tensor(outputs.raw[f"{gene}_start_logits"]),
                                  torch.as_tensor(outputs.raw[f"{gene}_end_logits"]), start_slots, end_slots)
    return starts.numpy(), ends.numpy()


def allele_ranking(outputs: StageOutputs, gene: str) -> list[np.ndarray]:
    """Per read, the indices of ``gene``'s alleles by decreasing score, ties to the lower index. The first is
    the selector's top-1 call; the selector breaks ties the same way unless more than 16 alleles pass its threshold
    (numpy's default sort is stable only up to 16 elements)."""
    probs = np.asarray(outputs.predictions.allele[gene])
    return [np.argsort(-row, kind="stable") for row in probs]


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
    raw = {key: _flat(outputs.raw[key]) for key in ("mutation_rate", "indel_count") if key in outputs.raw}
    genes = [g for g in GENES if f"{g}_start_logits" in outputs.raw]
    decoded = {g: {decoder: decode_boundaries(outputs, g, decoder) for decoder in DECODERS} for g in genes}
    alleles = [g for g in genes if g in outputs.allele_names]
    ranking = {g: allele_ranking(outputs, g) for g in alleles}
    probs = {g: np.asarray(outputs.predictions.allele[g]) for g in alleles}
    presence = _d_presence(outputs)
    frames = model_frames(outputs)

    scores: dict[str, CaseScores] = {}
    for truth, i in pairs:
        misoriented = int(frames[i]) != truth.orientation
        case: CaseScores = {"head.cropped": float(len(truth.sequence) > width), "head.misoriented": float(misoriented)}
        window = read_window(truth, width)
        for g in genes:
            seg = _scored_segment(truth, g)
            if seg is None:
                continue
            if not misoriented:
                case.update(_boundary_scores(g, seg, i, decoded[g], window))
            if g in ranking:
                case.update(_allele_scores(g, seg, ranking[g][i], probs[g][i], outputs.allele_names[g]))
        want = None if presence is None else d_class(truth, width)
        if want is not None:
            got = int(presence[i])
            case["head.d_presence.correct"] = float(got == want)
            if want == CALLABLE:
                case["head.d_presence.callable_gated"] = float(got != CALLABLE)
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
        orientation = int(np.argmax(outputs.raw["orientation_logits"][i]))
        case["head.orientation.correct"] = float(orientation == truth.orientation)
        scores[truth.case_id] = case
    return scores


def d_class(truth: ObservableTruth, width: int) -> int | None:
    """The truth's D class in AlignAIR's terms (``alignair.core.d_presence``): absent without an observed D, short
    below ``MIN_OBSERVED["d"]`` observed germline bases, callable otherwise. None when the D truth was not built, or
    when the read is longer than the window ``width``, whose crop may cut its D."""
    if "d" in truth.unscorable or len(truth.sequence) > width:
        return None
    segment = truth.segments.get("d")
    if segment is None:
        return ABSENT
    return SHORT if segment.observed_length < MIN_OBSERVED["d"] else CALLABLE


def d_presence_confusion(outputs: StageOutputs, truths: list[ObservableTruth]) -> dict | None:
    """For a model with a D presence head, the counts of truth class (rows) × the head's class (columns) over the
    reads ``d_class`` scores, in ``D_CLASSES`` order: ``classes``, ``counts`` (3 × 3), ``n`` and ``callable_gated``
    (truth callable, called absent or short: the reads whose D call the gate drops). None for a model without the
    head."""
    presence = _d_presence(outputs)
    if presence is None:
        return None
    width = _width(outputs)
    counts = [[0] * len(D_CLASSES) for _ in D_CLASSES]
    for truth, i in _pairs(outputs, truths):
        want = d_class(truth, width)
        if want is not None:
            counts[want][int(presence[i])] += 1
    return {"classes": list(D_CLASSES), "counts": counts, "n": sum(map(sum, counts)),
            "callable_gated": sum(counts[CALLABLE][k] for k in (ABSENT, SHORT))}


def _d_presence(outputs: StageOutputs) -> np.ndarray | None:
    """The D presence head's class per read (the argmax of its logits, the class the product gates on), or None."""
    logits = outputs.raw.get(D_PRESENCE_LOGITS)
    return None if logits is None else np.asarray(logits).argmax(-1)


def model_frames(outputs: StageOutputs) -> np.ndarray:
    """Per read, the orientation id the replay canonicalized it with (``predictions.orientation``; 0 for every read
    of a model without an orientation head, as ``pipeline._canonicalize`` treats it)."""
    orientation = getattr(outputs.predictions, "orientation", None)
    if orientation is None:
        return np.zeros(len(outputs.ids), dtype=int)
    return np.asarray(orientation, dtype=int).reshape(-1)


def _flat(values) -> np.ndarray:
    return np.asarray(values).reshape(-1)


def _slots(outputs: StageOutputs) -> tuple[torch.Tensor, torch.Tensor]:
    """The product's boundary slots of every read (``alignair.predict.decode.boundary_slots``) from the model's own
    masks in ``raw``; for outputs without masks, the half-open slots of each replayed read's length."""
    raw = outputs.raw
    if "start_mask" in raw and "end_mask" in raw:
        return boundary_slots(raw)
    width = int(np.asarray(raw["v_start_logits"]).shape[1])
    lengths = torch.tensor([len(read) for read in outputs.canonical])
    slots = torch.arange(width)[None, :]
    return slots < lengths[:, None], (slots >= 1) & (slots <= lengths[:, None])     # ends lie on 1..n


def _width(outputs: StageOutputs) -> int:
    """The window the model saw: the length of its position logits, less the terminal end slot."""
    return model_window(int(np.asarray(outputs.raw["v_start_logits"]).shape[1]))


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

