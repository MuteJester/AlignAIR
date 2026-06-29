"""Derive the per-sample ground-truth bundle from a GenAIRR stream_records dict
plus the ReferenceSet germline sequences (forward-orientation frame)."""
import numpy as np

from ..data.tokenizer import TOKEN_DICT
from ..nn.heads.region import REGION_INDEX
from ..nn.heads.state import STATE_INDEX

_GENES = ("v", "d", "j")


def _tok(seq: str) -> np.ndarray:
    n = TOKEN_DICT["N"]
    return np.array([TOKEN_DICT.get(c, n) for c in seq.upper()], dtype=np.int64)


def _parse_cigar(cigar: str) -> list:
    """'41M13D' -> [(41,'M'), (13,'D')]. Empty/None -> []."""
    if not cigar:
        return []
    ops, num = [], ""
    for ch in str(cigar):
        if ch.isdigit():
            num += ch
        else:
            ops.append((int(num or 0), ch))
            num = ""
    return ops


def _label_segment_states(state, seq, ss, gref, gs, cigar):
    """Walk a segment's CIGAR to assign exact per-observed-position edit states.

    M/=/X consume an observed base and a germline base -> germline (match) or
    substitution (mismatch). I/S consume an observed base only -> insertion.
    D/N consume a germline base only (no observed position) -> the next observed
    base is tagged ``deletion`` (a learnable 'a gap precedes me' signal, since a
    deletion has no observed token of its own). Returns the deletion count."""
    obs, germ = ss, gs
    pending_del = False
    n_del = 0
    for count, op in _parse_cigar(cigar):
        if op in ("M", "=", "X"):
            for _ in range(count):
                if obs >= len(seq):
                    break
                if pending_del:
                    state[obs] = STATE_INDEX["deletion"]
                    pending_del = False
                elif germ < len(gref) and seq[obs] != gref[germ]:
                    state[obs] = STATE_INDEX["substitution"]
                # else leaves germline (0)
                obs += 1
                germ += 1
        elif op in ("I", "S"):
            for _ in range(count):
                if obs >= len(seq):
                    break
                state[obs] = STATE_INDEX["insertion"]
                obs += 1
        elif op in ("D", "N"):
            germ += count
            n_del += count
            pending_del = True
    return n_del


def build_targets(record: dict, reference_set, has_d: bool) -> dict:
    seq = str(record["sequence"]).upper()
    L = len(seq)
    coords = {g: (int(record[f"{g}_sequence_start"]), int(record[f"{g}_sequence_end"]))
              for g in _GENES if record.get(f"{g}_sequence_start") is not None}
    germ = {g: (int(record[f"{g}_germline_start"]), int(record[f"{g}_germline_end"]))
            for g in _GENES if record.get(f"{g}_germline_start") is not None}

    # ---- region labels (presence-aware: one-sided/adaptive crops can drop V and/or D) ----
    region = np.full(L, REGION_INDEX["pre"], dtype=np.int64)
    has_v = "v" in coords
    if has_v:
        vs, ve = coords["v"]
        region[vs:ve] = REGION_INDEX["V"]
    js, je = coords.get("j", (L, L))
    left = ve if has_v else 0                       # start of the inter-segment (np/D) region
    if has_d and "d" in coords:
        ds, de = coords["d"]
        region[left:ds] = REGION_INDEX["N1"]
        region[ds:de] = REGION_INDEX["D"]
        region[de:js] = REGION_INDEX["N2"]
    else:
        region[left:js] = REGION_INDEX["N1"]
    region[js:je] = REGION_INDEX["J"]
    region[je:L] = REGION_INDEX["post"]

    # ---- per-position edit state from each segment's CIGAR (exact, indel-aware) ----
    state = np.zeros(L, dtype=np.int64)  # germline = 0
    for g in _GENES:
        if g not in coords:
            continue
        call = str(record[f"{g}_call"]).split(",")[0]
        ref = reference_set.gene(g.upper())
        idx = ref.index.get(call)
        if idx is None:
            continue
        ss, _ = coords[g]
        gs, _ = germ[g]
        cigar = record.get(f"{g}_cigar")
        _label_segment_states(state, seq, ss, ref.sequences[idx], gs, cigar)

    calls = {g.upper(): set(str(record[f"{g}_call"]).split(","))
             for g in _GENES if record.get(f"{g}_call")}
    # the allele the germline coordinates actually belong to (GenAIRR lists it
    # first); teacher-forced germline alignment must use THIS allele's reps, not an
    # arbitrary co-listed one whose germline length differs.
    primary = {g.upper(): str(record[f"{g}_call"]).split(",")[0]
               for g in _GENES if record.get(f"{g}_call")}

    return {
        "tokens": _tok(seq),
        "region_labels": region,
        "state_labels": state,
        "germline": germ,
        "inseq": coords,
        "calls": calls,
        "primary": primary,
        # inverted-D reads contain RC(germline) at the D locus; we cannot yet
        # supervise D-match/D-germline against the forward reference, so flag them
        # and mask those terms (RC reference modelling is deferred).
        "d_inverted": bool(record.get("d_inverted", False)),
        "orientation_id": 0,  # forward-only gym for now
        "noise_count": float(record["n_quality_errors"] + record.get("n_pcr_errors", 0)),
        "mutation_rate": float(record["mutation_rate"]),
        "indel_count": float(record["n_indels"]),
        "productive": 1.0 if record["productive"] else 0.0,
    }
