from __future__ import annotations

import math
from typing import Any

from ...core.schema import GENES


def compact_summary(scores: dict[str, Any]) -> dict[str, Any]:
    """Return a small high-signal summary suitable for tables/logging."""

    genes = scores.get("genes", {})
    out = {"n_cases": scores.get("n_cases", 0), "frame": scores.get("frame")}
    for g in GENES:
        gm = genes.get(g, {})
        out[g] = {
            "call": gm.get("call_top1_in_set", math.nan),
            "set_f1": gm.get("call_set_f1", math.nan),
            "gene": gm.get("gene_top1_in_set", math.nan),
            "seq_mae": [gm.get("ss_mae", math.nan), gm.get("se_mae", math.nan)],
            "germ_mae": [gm.get("gs_mae", math.nan), gm.get("ge_mae", math.nan)],
            "segment_iou": gm.get("seq_span_iou", math.nan),
        }
    out["global"] = scores.get("global", {})
    return out
