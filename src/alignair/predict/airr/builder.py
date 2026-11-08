"""Assemble full AIRR rearrangement records from predict() records + reference.

Produces IMGT-gapped sequence_alignment, np1/np2, germline alignment, IMGT-frame positions,
per-segment alignments, FWR/CDR regions, junction/CDR3 via the J anchor, and quality flags.
Per-record ``try/except`` isolates edge cases (fall back to a bare row).

A record takes its junction, CDR3, FWR4, frame, stop codon and productivity from its predicted V and J anchors
(``predict.junction.derive_junction``, over the record's V start and J end), in read coordinates; a record without
anchors (a model built without anchor heads) has none of them. The rest of the assembly is the same for both.
"""
from __future__ import annotations

from alignair.errors import InferenceError

from . import quality
from .alignment import (build_germline_alignment, build_sequence_alignment,
                        compute_alignment_positions, compute_np_regions,
                        extract_segment_alignments, translate_alignment)
from ...core.anchors import ANCHOR_GENES, ANCHOR_KEYS
from ...core.d_presence import ABSENT, D_CLASSES, D_PRESENCE, SHORT
from .constants import SHORT_D_SENTINEL
from ..junction import derive_junction
from ..v_presence import V_ABSENT, V_PRESENCE
from .regions import anchor_junction_fields, extract_regions

# Exceptions edge-case *data* can legitimately raise inside assembly (bad coords, missing anchors,
# out-of-range slices). These are tagged as per-record assembly failures and tracked; anything else is
# treated as a programming defect and re-raised loudly (see build_airr).
_EXPECTED_ASSEMBLY_ERRORS = (ValueError, IndexError, KeyError)


class AirrAssemblyError(InferenceError):
    """AIRR assembly failed for one record. Carries the record identifier and the originating error so
    an unexpected (non-data) failure surfaces with context instead of silently dropping AIRR fields."""
    code = "ALIGNAIR_AIRR_ASSEMBLY_FAILED"

    def __init__(self, ident, original):
        self.ident = ident
        self.original = original
        super().__init__(f"AIRR assembly failed for {ident}: "
                         f"{type(original).__name__}: {original}",
                         context={"stage": "airr_assembly", "record_id": ident})


def _germline_maps(reference):
    v, j = reference.gene("V"), reference.gene("J")
    v_gapped = v.gapped or {n: s for n, s in zip(v.names, v.sequences)}
    j_ung = {n: s for n, s in zip(j.names, j.sequences)}
    d_ung = {}
    if "D" in reference.genes:
        d = reference.gene("D")
        d_ung = {n: s for n, s in zip(d.names, d.sequences)}
    return v_gapped, j_ung, d_ung


def _partial(out: dict, reason: str) -> dict:
    """Mark a record as a PARTIAL assembly (valid calls, but one or more expected alignment products
    could not be derived) with a machine-readable reason code — never reported as a clean ``ok``."""
    out["airr_assembly_status"] = "partial"
    out["airr_assembly_reason"] = reason
    return out


def _build_one(rec, v_gapped, j_ung, d_ung, chain) -> dict:
    seq = rec["sequence"]
    out = dict(rec)                    # preserve the light record (calls/coords/cigar/orientation/likelihoods)
    out.setdefault("locus", "IGH")
    # a D the D presence head gated (predict.stages.gate_d): an absent one is assembled as a read without a D (the
    # light-chain path: one NP region between V and J); a short one keeps its span and takes the Short-D path for its
    # germline alignment (the read's bases between V and J), as it has no call
    presence = rec.get(D_PRESENCE)
    chain = "light" if presence == D_CLASSES[ABSENT] else chain
    d_call = SHORT_D_SENTINEL if presence == D_CLASSES[SHORT] else rec.get("d_call")
    anchored = all(key in rec for key in ANCHOR_KEYS.values())
    predicted_nonproductive = False
    if anchored:
        # the junction, frame, stop codon and productivity come from the predicted V and J anchors over the read's V
        # start and J end (predict.junction), not from the alignment below; `productive_prediction` is the same call
        call = derive_junction(seq, *(rec[ANCHOR_KEYS[g]] for g in ANCHOR_GENES),
                               coding_span=(rec.get("v_sequence_start"), rec.get("j_sequence_end")))
        out.update(anchor_junction_fields(seq, call, rec.get("j_sequence_end")))
        out["productive_prediction"] = call.productive
        predicted_nonproductive = call.productive is False
    else:
        out["productive_prediction"] = out["productive"] = None
    out["ar_indels"] = rec.get("indel_count")
    if rec.get(V_PRESENCE) == V_ABSENT:
        # a read without a V (predict.v_presence): no junction or alignment hangs on a V; its D and J calls stand
        return _partial(out, "no_v")
    # skip the heavy alignment math for clearly-garbage reads (predicted non-productive with multiple
    # indels); the junction above is already attached.
    # These are PARTIAL, not ok — germline_alignment / identity are not derivable.
    if predicted_nonproductive and (rec.get("indel_count") or 0) > 1:
        return _partial(out, "nonproductive_indel")
    v_call = rec.get("v_call", "")
    if not v_call or rec.get("v_sequence_start") is None or rec.get("j_sequence_start") is None:
        return _partial(out, "missing_calls_or_coordinates")
    if rec.get("segmentation_low_quality"):        # V/J collapsed -> no feasible alignment
        return _partial(out, "collapsed_segment")

    vss, vse = rec["v_sequence_start"], rec["v_sequence_end"]
    vgs, vge = rec["v_germline_start"], rec["v_germline_end"]
    jss, jse = rec["j_sequence_start"], rec["j_sequence_end"]
    jgs, jge = rec.get("j_germline_start", 0), rec.get("j_germline_end", 0)
    dss, dse = rec.get("d_sequence_start"), rec.get("d_sequence_end")
    dgs, dge = rec.get("d_germline_start"), rec.get("d_germline_end")
    v_ref_gapped = v_gapped.get(v_call.split(",")[0], "")

    seq_alignment = build_sequence_alignment(seq, v_ref_gapped, vss, vse, vgs, vge, jse, v_cigar=rec.get("v_cigar"))
    np1, np2 = compute_np_regions(seq, vse, jss, dss, dse, chain)
    germ_alignment = build_germline_alignment(seq, v_gapped, j_ung, d_ung, v_call,
                                              rec.get("j_call", ""), d_call, vge, jgs,
                                              jge, dgs, dge, np1, np2, vse, jss, chain, j_seq_end=jse,
                                              d_cigar=rec.get("d_cigar"), j_cigar=rec.get("j_cigar"))
    positions = compute_alignment_positions(v_ref_gapped, vge, vse, dss, dse, jss, jse, chain)
    seq_aa = translate_alignment(seq_alignment)
    germ_aa = translate_alignment(germ_alignment)
    seg = (extract_segment_alignments(seq_alignment, germ_alignment, seq_aa, germ_aa, positions, chain)
           if seq_alignment and germ_alignment else {})
    regions = extract_regions(seq_alignment, seq_aa)

    out.update({"sequence_alignment": seq_alignment, "germline_alignment": germ_alignment,
                "sequence_alignment_aa": seq_aa, "germline_alignment_aa": germ_aa,
                "np1": np1, "np1_length": len(np1) if np1 else 0,
                "np2": np2, "np2_length": len(np2) if np2 else 0})
    out.update(positions)
    out.update(seg)
    out.update(regions)
    for g in ("v", "d", "j"):                          # per-segment identity (was V-only)
        ident = quality.segment_identity(seg.get(f"{g}_sequence_alignment"),
                                         seg.get(f"{g}_germline_alignment"))
        if ident is not None:
            out[f"{g}_identity"] = ident
    if presence == D_CLASSES[SHORT]:
        # the short D's germline alignment is the read's own bases (the Short-D path): no statistic of a germline it
        # does not name, only its read alignment
        out.update(d_identity=None, d_germline_alignment=None, d_germline_alignment_aa=None)
    if not germ_alignment or not seq_alignment:    # ran the math but a core product is missing
        return _partial(out, "incomplete_alignment")
    out["airr_assembly_status"] = "complete"
    return out


def build_airr(records: list, reference, chain: str = "heavy", strict: bool = False) -> list:
    """Assemble full AIRR rows. Each returned record carries ``airr_assembly_status``:
    ``"complete"`` (all expected products derived), ``"partial"`` (valid calls but a product could not
    be assembled — with a machine-readable ``airr_assembly_reason``), or ``"failed"`` (an exception,
    keeping the light fields + an ``airr_assembly_error``). Incomplete records are never reported as a
    clean success and the assembly is never *silently* dropped.

    Expected data-edge exceptions become tagged failures; any other exception is a programming defect
    and is re-raised as :class:`AirrAssemblyError` with the record identifier. ``strict=True`` re-raises
    even the expected data failures (used to gate release fixtures / tests that must assemble cleanly)."""
    v_gapped, j_ung, d_ung = _germline_maps(reference)
    # per-record chain: a multi-chain union may mix loci, so assemble each record with ITS locus's
    # has-D / heavy-vs-light policy rather than one global chain.
    locus_hasd = {l.locus: l.has_d for l in getattr(reference, "loci", [])}
    out = []
    for i, rec in enumerate(records):
        ident = rec.get("sequence_id") or f"record[{i}]"
        rec_chain = ("heavy" if locus_hasd[rec["locus"]] else "light") if rec.get("locus") in locus_hasd \
            else chain
        try:
            row = _build_one(rec, v_gapped, j_ung, d_ung, rec_chain)
            row.setdefault("airr_assembly_status", "complete")   # _build_one sets complete/partial
            out.append(row)
        except _EXPECTED_ASSEMBLY_ERRORS as e:
            if strict:
                raise AirrAssemblyError(ident, e) from e
            row = dict(rec)                # keep the light record (calls/coords) but TAG the failure
            row["airr_assembly_status"] = "failed"
            row["airr_assembly_error"] = f"{type(e).__name__}: {e}"
            out.append(row)
        except Exception as e:             # unexpected -> a programming defect, never silent
            raise AirrAssemblyError(ident, e) from e
    return out
