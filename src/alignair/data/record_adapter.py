"""Adapt a raw CSV row (+ pad offset) into a canonical per-sample record."""
from ast import literal_eval


def _to_float_bool(v) -> float:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return 1.0 if v != 0 else 0.0
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return 1.0
    if s in {"false", "0", "no", "n", "f", ""}:
        return 0.0
    return 0.0


def _indel_count(item) -> float:
    if isinstance(item, (dict, list, tuple)):
        return float(len(item))
    if isinstance(item, str) and item.strip():
        try:
            parsed = literal_eval(item)
            return float(len(parsed)) if isinstance(parsed, (dict, list, tuple)) else 0.0
        except Exception:
            return 0.0
    return 0.0


class RecordAdapter:
    def __init__(self, has_d: bool):
        self.has_d = has_d
        self.genes = ["v", "j"] + (["d"] if has_d else [])

    def adapt(self, row: dict, pad_left: int) -> dict:
        rec: dict = {}
        for g in self.genes:
            rec[f"{g}_start"] = float(int(row[f"{g}_sequence_start"]) + pad_left)
            rec[f"{g}_end"] = float(int(row[f"{g}_sequence_end"]) + pad_left)
            rec[f"{g}_call_set"] = set(str(row[f"{g}_call"]).split(","))
        rec["mutation_rate"] = float(row["mutation_rate"])
        rec["indel_count"] = _indel_count(row.get("indels", ""))
        rec["productive"] = _to_float_bool(row.get("productive", 1.0))
        return rec
