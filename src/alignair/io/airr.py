"""Write DNAlignAIR predictions as an AIRR rearrangement TSV (+ calibrated-uncertainty
extension columns the AIRR schema does not cover: per-gene equivalence set, resolution
level, and set confidence)."""
from __future__ import annotations

import csv
from typing import List

GENES = ("v", "d", "j")
# AIRR-standard core columns we populate, then our extensions.
_CORE = ["sequence_id", "sequence", "locus", "v_call", "d_call", "j_call", "productive"]
_COORDS = [f"{g}_{k}" for g in GENES
           for k in ("sequence_start", "sequence_end", "germline_start", "germline_end")]
_EXT = [f"{g}_{k}" for g in GENES for k in ("call_set", "call_level", "set_confidence")]
COLUMNS = _CORE + _COORDS + _EXT


def _airr_start(v):
    """0-based (GenAIRR/our convention) -> 1-based (AIRR)."""
    return (int(v) + 1) if v is not None else None


def write_airr(path: str, ids: List[str], sequences: List[str], preds: List[dict],
               locus: str = "IGH") -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for sid, seq, p in zip(ids, sequences, preds):
            row = {"sequence_id": sid, "sequence": seq, "locus": locus,
                   "productive": p.get("productive")}
            for g in GENES:
                row[f"{g}_call"] = p.get(f"{g}_call")
                row[f"{g}_sequence_start"] = _airr_start(p.get(f"{g}_sequence_start"))
                row[f"{g}_sequence_end"] = p.get(f"{g}_sequence_end")
                row[f"{g}_germline_start"] = _airr_start(p.get(f"{g}_germline_start"))
                row[f"{g}_germline_end"] = p.get(f"{g}_germline_end")
                cset = p.get(f"{g}_call_set") or ([p.get(f"{g}_call")] if p.get(f"{g}_call") else [])
                row[f"{g}_call_set"] = ",".join(c for c in cset if c)
                row[f"{g}_call_level"] = p.get(f"{g}_call_level")
                conf = p.get(f"{g}_set_confidence")
                row[f"{g}_set_confidence"] = f"{conf:.4f}" if conf is not None else None
            w.writerow(row)
