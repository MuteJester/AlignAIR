"""Derive the per-sample ground-truth bundle from a GenAIRR stream_records dict
plus the ReferenceSet germline sequences (forward-orientation frame)."""
import numpy as np

from ..data.tokenizer import TOKEN_DICT
from ..nn.region_head import REGION_INDEX
from ..nn.state_head import STATE_INDEX

_GENES = ("v", "d", "j")


def _tok(seq: str) -> np.ndarray:
    n = TOKEN_DICT["N"]
    return np.array([TOKEN_DICT.get(c, n) for c in seq.upper()], dtype=np.int64)


def _substitution_offsets(obs: str, gref: str) -> list:
    """obs-relative indices that differ from the germline, over the best of a 5'- or
    3'-anchored comparison (robust to small coordinate-convention/trim offsets so a
    1-base span mismatch doesn't frame-shift the whole comparison)."""
    n = min(len(obs), len(gref))
    if n == 0:
        return []
    five = [k for k in range(n) if obs[k] != gref[k]]
    lo, lg = len(obs), len(gref)
    three = [lo - 1 - k for k in range(n) if obs[lo - 1 - k] != gref[lg - 1 - k]]
    return five if len(five) <= len(three) else three


def build_targets(record: dict, reference_set, has_d: bool) -> dict:
    seq = str(record["sequence"]).upper()
    L = len(seq)
    coords = {g: (int(record[f"{g}_sequence_start"]), int(record[f"{g}_sequence_end"]))
              for g in _GENES if record.get(f"{g}_sequence_start") is not None}
    germ = {g: (int(record[f"{g}_germline_start"]), int(record[f"{g}_germline_end"]))
            for g in _GENES if record.get(f"{g}_germline_start") is not None}

    vs, ve = coords["v"]
    js, je = coords["j"]

    # ---- region labels ----
    region = np.full(L, REGION_INDEX["pre"], dtype=np.int64)
    region[vs:ve] = REGION_INDEX["V"]
    if has_d and "d" in coords:
        ds, de = coords["d"]
        region[ve:ds] = REGION_INDEX["N1"]
        region[ds:de] = REGION_INDEX["D"]
        region[de:js] = REGION_INDEX["N2"]
    else:
        region[ve:js] = REGION_INDEX["N1"]
    region[js:je] = REGION_INDEX["J"]
    region[je:L] = REGION_INDEX["post"]

    # ---- per-position state (germline vs substitution over equal-length gene spans) ----
    state = np.zeros(L, dtype=np.int64)  # germline = 0
    for g in _GENES:
        if g not in coords:
            continue
        call = str(record[f"{g}_call"]).split(",")[0]
        ref = reference_set.gene(g.upper())
        idx = ref.index.get(call)
        if idx is None:
            continue
        ss, ee = coords[g]
        gs, ge = germ[g]
        obs = seq[ss:ee]
        gref = ref.sequences[idx][gs:ge]
        for k in _substitution_offsets(obs, gref):
            state[ss + k] = STATE_INDEX["substitution"]

    calls = {g.upper(): set(str(record[f"{g}_call"]).split(","))
             for g in _GENES if record.get(f"{g}_call")}

    return {
        "tokens": _tok(seq),
        "region_labels": region,
        "state_labels": state,
        "germline": germ,
        "inseq": coords,
        "calls": calls,
        "orientation_id": 0,  # forward-only gym for now
        "noise_count": float(record["n_quality_errors"] + record.get("n_pcr_errors", 0)),
        "mutation_rate": float(record["mutation_rate"]),
        "indel_count": float(record["n_indels"]),
        "productive": 1.0 if record["productive"] else 0.0,
    }
