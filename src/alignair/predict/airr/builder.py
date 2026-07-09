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
from .regions import compute_junction, extract_regions


def _germline_maps(reference):
    v, j = reference.gene("V"), reference.gene("J")
    v_gapped = v.gapped or {n: s for n, s in zip(v.names, v.sequences)}
    j_ung = {n: s for n, s in zip(j.names, j.sequences)}
    d_ung = {}
    if "D" in reference.genes:
        d = reference.gene("D")
        d_ung = {n: s for n, s in zip(d.names, d.sequences)}
    return v_gapped, j_ung, d_ung, (j.anchors or {})


def _build_one(rec, v_gapped, j_ung, d_ung, j_anchors, chain) -> dict:
    seq = rec["sequence"]
    out = dict(rec)                    # preserve the light record (calls/coords/cigar/orientation/likelihoods)
    out.setdefault("locus", "IGH")
    out["productive"] = bool(rec.get("productive", True))
    out["ar_indels"] = rec.get("indel_count")
    # skip alignment math for clearly-garbage reads (non-productive with multiple indels)
    if (not out["productive"]) and (rec.get("indel_count") or 0) > 1:
        return out
    v_call = rec.get("v_call", "")
    if not v_call or rec.get("v_sequence_start") is None or rec.get("j_sequence_start") is None:
        return out

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
    junction = compute_junction(seq_alignment, seq_aa, rec.get("j_call", ""), j_anchors,
                                jss, vss, jgs, positions.get("j_alignment_end"))

    out.update({"sequence_alignment": seq_alignment, "germline_alignment": germ_alignment,
                "sequence_alignment_aa": seq_aa, "germline_alignment_aa": germ_aa,
                "np1": np1, "np1_length": len(np1) if np1 else 0,
                "np2": np2, "np2_length": len(np2) if np2 else 0})
    out.update(positions)
    out.update(seg)
    out.update(regions)
    out.update(junction)
    out["stop_codon"] = quality.stop_codon(seq_aa)
    out["vj_in_frame"] = quality.vj_in_frame(junction.get("cdr3_start"), junction.get("cdr3_end"),
                                             positions.get("v_alignment_start"))
    out["v_identity"] = quality.v_identity(seg.get("v_sequence_alignment"),
                                           seg.get("v_germline_alignment"))
    return out


def build_airr(records: list, reference, chain: str = "heavy") -> list:
    v_gapped, j_ung, d_ung, j_anchors = _germline_maps(reference)
    out = []
    for rec in records:
        try:
            out.append(_build_one(rec, v_gapped, j_ung, d_ung, j_anchors, chain))
        except Exception:
            out.append(dict(rec))          # AIRR assembly failed -> keep the light record (calls/coords)
    return out
