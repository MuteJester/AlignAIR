"""Assemble full AIRR rearrangement records from predict() records + reference (Phase B).

Faithful port of TF Pipeline/AIRR/builder: IMGT-gapped sequence_alignment, np1/np2, germline
alignment, IMGT-frame positions, per-segment alignments, FWR/CDR regions, junction/CDR3 via the J
anchor, and quality flags. Per-record ``try/except`` isolates edge cases (fall back to a bare row).
"""
from __future__ import annotations

from . import quality
from .alignment import (build_germline_alignment, build_sequence_alignment,
                        compute_alignment_positions, compute_np_regions,
                        extract_segment_alignments, translate_alignment)
from .regions import cigar_has_indel, compute_junction, compute_junction_cigar, extract_regions

# Exceptions edge-case *data* can legitimately raise inside assembly (bad coords, missing anchors,
# out-of-range slices). These are tagged as per-record assembly failures and tracked; anything else is
# treated as a programming defect and re-raised loudly (see build_airr / the pre-launch audit P0-7).
_EXPECTED_ASSEMBLY_ERRORS = (ValueError, IndexError, KeyError)


class AirrAssemblyError(Exception):
    """AIRR assembly failed for one record. Carries the record identifier and the originating error so
    an unexpected (non-data) failure surfaces with context instead of silently dropping AIRR fields."""

    def __init__(self, ident, original):
        self.ident = ident
        self.original = original
        super().__init__(f"AIRR assembly failed for {ident}: "
                         f"{type(original).__name__}: {original}")


def _germline_maps(reference):
    v, j = reference.gene("V"), reference.gene("J")
    v_gapped = v.gapped or {n: s for n, s in zip(v.names, v.sequences)}
    j_ung = {n: s for n, s in zip(j.names, j.sequences)}
    d_ung = {}
    if "D" in reference.genes:
        d = reference.gene("D")
        d_ung = {n: s for n, s in zip(d.names, d.sequences)}
    return v_gapped, j_ung, d_ung, (j.anchors or {})


def _apply_cigar_junction(out, rec, seq, v_gapped, j_anchors, force=False) -> bool:
    """Fix A: attach the indel-robust, read-coordinate junction (Cys / J-anchor mapped through the
    CIGAR) when the read carries a V/J/D indel and the anchors can be placed — OR always when
    ``force`` is set (TCR loci: their IMGT gapping breaks the gapped column-309 junction math, and the
    read-coordinate path is exact for TCR; see the locus gate in :func:`build_airr`). Needs only coords
    + CIGAR (no gapped `sequence_alignment`), so it runs before the guards below — even reads whose
    heavy alignment math is skipped still get a correct junction. Returns True when applied."""
    if not force and not (cigar_has_indel(rec.get("v_cigar")) or cigar_has_indel(rec.get("j_cigar"))
                          or cigar_has_indel(rec.get("d_cigar"))):
        return False
    v_call = rec.get("v_call", "")
    v_ref_gapped = v_gapped.get(v_call.split(",")[0], "") if v_call else ""
    j_anchor = (j_anchors or {}).get(rec.get("j_call", "").split(",")[0], 0)
    cj = compute_junction_cigar(seq, v_ref_gapped, rec.get("v_germline_start"),
                                rec.get("v_sequence_start"), rec.get("v_sequence_end"),
                                rec.get("v_cigar", ""), j_anchor, rec.get("j_germline_start"),
                                rec.get("j_sequence_start"), rec.get("j_sequence_end"),
                                rec.get("j_cigar", ""))
    if cj is None:
        return False
    out.update(cj)
    jl = cj.get("junction_length")
    out["vj_in_frame"] = (jl % 3 == 0) if jl else None      # in-frame <=> junction length divisible by 3
    return True


def _partial(out: dict, reason: str) -> dict:
    """Mark a record as a PARTIAL assembly (valid calls, but one or more expected alignment products
    could not be derived) with a machine-readable reason code — never reported as a clean ``ok``."""
    out["airr_assembly_status"] = "partial"
    out["airr_assembly_reason"] = reason
    return out


def _build_one(rec, v_gapped, j_ung, d_ung, j_anchors, chain, is_tcr=False) -> dict:
    seq = rec["sequence"]
    out = dict(rec)                    # preserve the light record (calls/coords/cigar/orientation/likelihoods)
    out.setdefault("locus", "IGH")
    # the model's neural call is kept as `productive_prediction`; AIRR `productive` is a DERIVED fact
    # (in-frame + no stop codon) set below once the alignment math runs. If it cannot be derived (this
    # record skipped assembly), `productive` stays **blank/unknown** rather than presenting the neural
    # guess as a definitive fact (audit #6).
    neural_productive = bool(rec.get("productive", True))
    out["productive_prediction"] = neural_productive
    out["productive"] = None
    out["ar_indels"] = rec.get("indel_count")
    cigar_junction = _apply_cigar_junction(out, rec, seq, v_gapped, j_anchors, force=is_tcr)
    # skip the heavy alignment math for clearly-garbage reads (predicted non-productive with multiple
    # indels); the indel-robust junction above is already attached, and AIRR `productive` stays blank.
    # These are PARTIAL, not ok — germline_alignment / identity are not derivable (audit #5).
    if (not neural_productive) and (rec.get("indel_count") or 0) > 1:
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

    seq_alignment = build_sequence_alignment(seq, v_ref_gapped, vss, vse, vgs, vge, jse)
    np1, np2 = compute_np_regions(seq, vse, jss, dss, dse, chain)
    germ_alignment = build_germline_alignment(seq, v_gapped, j_ung, d_ung, v_call,
                                              rec.get("j_call", ""), rec.get("d_call"), vge, jgs,
                                              jge, dgs, dge, np1, np2, vse, jss, chain)
    positions = compute_alignment_positions(v_ref_gapped, vge, vss, dss, dse, jss, jse, chain)
    seq_aa = translate_alignment(seq_alignment)
    germ_aa = translate_alignment(germ_alignment)
    seg = (extract_segment_alignments(seq_alignment, germ_alignment, seq_aa, germ_aa, positions, chain)
           if seq_alignment and germ_alignment else {})
    regions = extract_regions(seq_alignment, seq_aa)
    # the indel-robust junction (if any) is already on `out`; otherwise derive it from the IMGT column
    if cigar_junction:
        junction = {}
        vj_in_frame = out.get("vj_in_frame")
    else:
        junction = compute_junction(seq_alignment, seq_aa, rec.get("j_call", ""), j_anchors,
                                    jss, vss, jgs, positions.get("j_alignment_end"))
        vj_in_frame = quality.vj_in_frame(junction.get("cdr3_start"), junction.get("cdr3_end"),
                                          positions.get("v_alignment_start"))

    out.update({"sequence_alignment": seq_alignment, "germline_alignment": germ_alignment,
                "sequence_alignment_aa": seq_aa, "germline_alignment_aa": germ_aa,
                "np1": np1, "np1_length": len(np1) if np1 else 0,
                "np2": np2, "np2_length": len(np2) if np2 else 0})
    out.update(positions)
    out.update(seg)
    out.update(regions)
    out.update(junction)
    out["stop_codon"] = quality.stop_codon(seq_aa)
    out["vj_in_frame"] = vj_in_frame
    for g in ("v", "d", "j"):                          # per-segment identity (was V-only; P0-14)
        ident = quality.segment_identity(seg.get(f"{g}_sequence_alignment"),
                                         seg.get(f"{g}_germline_alignment"))
        if ident is not None:
            out[f"{g}_identity"] = ident
    derived = quality.airr_productive(vj_in_frame, out["stop_codon"])   # AIRR productive = derived fact
    if derived is not None:
        out["productive"] = derived
    if not germ_alignment or not seq_alignment:    # ran the math but a core product is missing
        return _partial(out, "incomplete_alignment")
    out["airr_assembly_status"] = "complete"
    return out


def build_airr(records: list, reference, chain: str = "heavy", strict: bool = False) -> list:
    """Assemble full AIRR rows. Each returned record carries ``airr_assembly_status``:
    ``"complete"`` (all expected products derived), ``"partial"`` (valid calls but a product could not
    be assembled — with a machine-readable ``airr_assembly_reason``), or ``"failed"`` (an exception,
    keeping the light fields + an ``airr_assembly_error``). Incomplete records are never reported as a
    clean success and the assembly is never *silently* dropped (P0-7 / AIRR-review #5).

    Expected data-edge exceptions become tagged failures; any other exception is a programming defect
    and is re-raised as :class:`AirrAssemblyError` with the record identifier. ``strict=True`` re-raises
    even the expected data failures (used to gate release fixtures / tests that must assemble cleanly)."""
    v_gapped, j_ung, d_ung, j_anchors = _germline_maps(reference)
    v_names = reference.gene("V").names                          # locus gate: TCR (TRxV...) IMGT gapping
    is_tcr = bool(v_names) and str(v_names[0]).upper().startswith("TR")  # breaks the gapped column-309
    # per-record chain: a multi-chain union may mix loci, so assemble each record with ITS locus's
    # has-D / heavy-vs-light policy rather than one global chain (P0-6).
    locus_hasd = {l.locus: l.has_d for l in getattr(reference, "loci", [])}
    out = []                                                     # junction -> use the read-coordinate path
    for i, rec in enumerate(records):
        ident = rec.get("sequence_id") or f"record[{i}] {str(rec.get('sequence', ''))[:24]!r}"
        rec_chain = ("heavy" if locus_hasd[rec["locus"]] else "light") if rec.get("locus") in locus_hasd \
            else chain
        try:
            row = _build_one(rec, v_gapped, j_ung, d_ung, j_anchors, rec_chain, is_tcr)
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
