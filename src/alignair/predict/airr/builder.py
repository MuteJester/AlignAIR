"""Assemble full AIRR rearrangement records from each read's typed result (``predict.results.ReadResult``) +
reference.

Produces IMGT-gapped sequence_alignment, np1/np2, germline alignment, IMGT-frame positions,
per-segment alignments, FWR/CDR regions, the junction and CDR3 from the V and J anchors, and quality flags.
Per-record ``try/except`` isolates edge cases (fall back to a bare row).

A row starts from the read's light record (``predict.results.light_record``) and adds the assembly. A read takes its
junction, CDR3, FWR4, frame, stop codon and productivity from its predicted V and J anchors (``ReadResult.junction``,
over its V start and J end), in read coordinates; a read without anchors (from a core without anchor heads) has
none of them. The rest of the assembly is the same for both. Every field it reads is the read's typed field; a gene's
placement is its ``placed`` alignment (none for an absent gene).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from alignair.errors import AirrAssemblyError

from ...core.d_presence import ABSENT, D_CLASSES, SHORT
from ...reference.genes import GENES
from ...reference.loci import FALLBACK_LOCUS
from ..results import ReadResult, light_record
from . import quality
from .alignment import (
    GermlineMaps,
    build_germline_alignment,
    compute_alignment_positions,
    compute_np_regions,
    extract_segment_alignments,
    sequence_alignment_columns,
    translate_alignment,
)
from .regions import anchor_junction_fields, extract_regions

if TYPE_CHECKING:
    from ...reference.reference_set import ReferenceSet

# Exceptions edge-case *data* can legitimately raise inside assembly (bad coords, missing anchors,
# out-of-range slices). These are tagged as per-record assembly failures and tracked; anything else is
# treated as a programming defect and re-raised loudly (see build_airr).
_EXPECTED_ASSEMBLY_ERRORS = (ValueError, IndexError, KeyError)


def _partial(out: dict, reason: str) -> dict:
    """Mark a record as a PARTIAL assembly (valid calls, but one or more expected alignment products
    could not be derived) with a machine-readable reason code — never reported as a clean ``ok``."""
    out["airr_assembly_status"] = "partial"
    out["airr_assembly_reason"] = reason
    return out


def _first(call_field: str) -> str:
    """The first allele of a comma-joined call ("" for no call)."""
    return call_field.split(",")[0]


def _build_one(read: ReadResult, light: dict, germlines: GermlineMaps, has_d: bool) -> dict:
    seq = read.sequence
    out = dict(light)                  # preserve the light record (calls/coords/cigar/orientation/likelihoods)
    out.setdefault("locus", FALLBACK_LOCUS)          # a read that carries no locus
    # a D the D presence head gated (predict.results): an absent one is assembled as a read without a D (one NP region
    # between V and J); a short one keeps its span, but as it has no call the bases between V and J are untemplated
    presence = read.d_presence
    has_d = has_d and presence != D_CLASSES[ABSENT]
    short_d = presence == D_CLASSES[SHORT]
    d_gene = read.genes.get("d")
    v, d, j = read.genes["v"].placed, None if d_gene is None else d_gene.placed, read.genes["j"].placed
    junction = read.junction()
    predicted_nonproductive = False
    if junction is not None:
        # the junction, frame, stop codon and productivity come from the predicted V and J anchors over the read's V
        # start and J end (ReadResult.junction), not from the alignment below; `productive_prediction` is the same call
        out.update(anchor_junction_fields(seq, junction, None if j is None else j.seq_end))
        out["productive_prediction"] = junction.productive
        predicted_nonproductive = junction.productive is False
    else:
        out["productive_prediction"] = out["productive"] = None
    out["ar_indels"] = read.indel_count
    if not read.v_present:
        # a read without a V (predict.v_presence): no junction or alignment hangs on a V; its D and J calls stand
        return _partial(out, "no_v")
    # skip the heavy alignment math for clearly-garbage reads (predicted non-productive with multiple
    # indels); the junction above is already attached.
    # These are PARTIAL, not ok — germline_alignment / identity are not derivable.
    if predicted_nonproductive and (read.indel_count or 0) > 1:
        return _partial(out, "nonproductive_indel")
    v_call = read.genes["v"].call_field
    if not v_call or v is None or j is None:
        return _partial(out, "missing_calls_or_coordinates")
    if read.low_quality:                           # V/J collapsed -> no feasible alignment
        return _partial(out, "collapsed_segment")

    names = {"v": _first(v_call), "j": _first(read.genes["j"].call_field),
             "d": "" if d_gene is None else _first(d_gene.call_field)}
    v_ref_gapped = germlines.v_gapped.get(names["v"], "")
    # a D without a call (too short to call) keeps its span but is no D of the germline: the bases between V and J are
    # one NP region, as Change-O reads a row without a D call
    d_called = has_d and not short_d and bool(names["d"]) and d is not None

    seq_alignment, read_positions = sequence_alignment_columns(seq, v_ref_gapped, v, j.seq_end) or (None, None)
    np1, np2 = compute_np_regions(seq, v, d, j, d_called)
    germ_alignment = build_germline_alignment(germlines, names, v, d, j, np1, np2, d_called=d_called)
    positions = compute_alignment_positions(v_ref_gapped, v, d, j, has_d)
    if seq_alignment is None:
        # no gapped alignment (a called V the reference holds no gapped form of): no positions in one
        positions = dict.fromkeys(positions)
    seq_aa = translate_alignment(seq_alignment)
    germ_aa = translate_alignment(germ_alignment)
    seg = (extract_segment_alignments(seq_alignment, germ_alignment, seq_aa, germ_aa, positions, has_d)
           if seq_alignment and germ_alignment else {})
    regions = extract_regions(seq_alignment, seq_aa, read_positions)

    out.update({"sequence_alignment": seq_alignment, "germline_alignment": germ_alignment,
                "sequence_alignment_aa": seq_aa, "germline_alignment_aa": germ_aa,
                "np1": np1, "np1_length": len(np1) if np1 else 0,
                "np2": np2, "np2_length": len(np2) if np2 else 0})
    out.update(positions)
    out.update(seg)
    out.update(regions)
    for g in GENES:                                    # per-segment identity
        ident = quality.segment_identity(seg.get(f"{g}_sequence_alignment"),
                                         seg.get(f"{g}_germline_alignment"))
        if ident is not None:
            out[f"{g}_identity"] = ident
    if has_d and not d_called:
        # a D without a call (a short one) names no germline: no statistic of one, only its read alignment
        out.update(d_identity=None, d_germline_alignment=None, d_germline_alignment_aa=None)
    if not germ_alignment or not seq_alignment:    # ran the math but a core product is missing
        return _partial(out, "incomplete_alignment")
    out["airr_assembly_status"] = "complete"
    return out


def build_airr(reads: Sequence[ReadResult], reference: ReferenceSet, *, has_d: bool = True,
               strict: bool = False) -> list[dict]:
    """Assemble full AIRR rows from each read's typed result. Each returned row carries ``airr_assembly_status``:
    ``"complete"`` (all expected products derived), ``"partial"`` (valid calls but a product could not
    be assembled — with a machine-readable ``airr_assembly_reason``), or ``"failed"`` (an exception,
    keeping the light fields + an ``airr_assembly_error``). Incomplete records are never reported as a
    clean success and the assembly is never *silently* dropped.

    Expected data-edge exceptions become tagged failures; any other exception is a programming defect
    and is re-raised as :class:`AirrAssemblyError` with the record identifier. ``strict=True`` re-raises
    even the expected data failures (used to gate release fixtures / tests that must assemble cleanly). ``has_d``
    says whether the model calls a D: without one no read is assembled with a D; with one, a read is assembled with
    its locus's D, and a read of a locus the reference does not describe with a D."""
    germlines = GermlineMaps.of(reference)
    # each read is assembled with its own locus's D (a reference of several loci may mix loci with and without one),
    # and never with a D the model does not call
    locus_hasd = {info.locus: info.has_d for info in reference.loci}
    out = []
    for i, read in enumerate(reads):
        ident = f"record[{i}]"
        light = light_record(read)
        try:
            row = _build_one(read, light, germlines, has_d and locus_hasd.get(read.locus, True))
            row.setdefault("airr_assembly_status", "complete")   # _build_one sets complete/partial
            out.append(row)
        except _EXPECTED_ASSEMBLY_ERRORS as e:
            if strict:
                raise AirrAssemblyError(ident, e) from e
            row = dict(light)              # keep the light record (calls/coords) but TAG the failure
            row["airr_assembly_status"] = "failed"
            row["airr_assembly_error"] = f"{type(e).__name__}: {e}"
            out.append(row)
        except Exception as e:             # unexpected -> a programming defect, never silent
            raise AirrAssemblyError(ident, e) from e
    return out
