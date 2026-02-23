"""SegmentCorrectionStage — remove center padding and clamp segment boundaries."""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from AlignAIR.Pipeline.Stage.protocol import Stage, StageContext

logger = logging.getLogger("AlignAIR.Pipeline")


class SegmentCorrectionStage(Stage):
    """Removes center-padding offsets from positions and enforces V <= D <= J ordering."""

    reads = frozenset({"processed_predictions", "sequences", "model"})
    writes = frozenset({"corrected_segments", "processed_predictions"})

    def run(self, context: StageContext) -> Dict[str, Any]:
        preds = context["processed_predictions"]
        sequences = context["sequences"]
        model = context["model"]
        has_d = model.has_d_gene
        max_length = model.max_seq_length

        logger.info("Correcting segments for paddings...")

        # Calculate per-sequence padding
        paddings = np.array([(max_length - len(s)) // 2 for s in sequences], dtype=np.int32)
        seq_lengths = np.array([len(s) for s in sequences], dtype=np.int32)

        def sanitize_bounds(raw_start, raw_end):
            s_raw = np.squeeze(raw_start)
            e_raw = np.squeeze(raw_end)
            s = np.floor(s_raw - paddings).astype(np.int32)
            e = np.floor(e_raw - paddings).astype(np.int32)
            s = np.clip(s, 0, seq_lengths - 1)
            e = np.clip(e, 1, seq_lengths)
            e = np.maximum(e, s + 1)
            return s, e

        v_start, v_end = sanitize_bounds(preds['v_start'], preds['v_end'])
        j_start, j_end = sanitize_bounds(preds['j_start'], preds['j_end'])

        d_start, d_end = None, None
        if has_d and preds.get('d_start') is not None and preds.get('d_end') is not None:
            d_start, d_end = sanitize_bounds(preds['d_start'], preds['d_end'])

        # Enforce monotonic ordering: V <= D <= J
        if d_start is not None and d_end is not None:
            d_start = np.maximum(d_start, v_end)
            d_end = np.maximum(d_end, d_start + 1)
            j_start = np.maximum(j_start, d_end)
            j_end = np.maximum(j_end, j_start + 1)
        else:
            j_start = np.maximum(j_start, v_end)
            j_end = np.maximum(j_end, j_start + 1)

        corrected = {
            'v_start': v_start, 'v_end': v_end,
            'd_start': d_start, 'd_end': d_end,
            'j_start': j_start, 'j_end': j_end,
        }

        # Also update processed_predictions with corrected positions
        # (needed by downstream stages: allele threshold, germline alignment, finalization)
        updated_preds = dict(preds)
        for key, val in corrected.items():
            updated_preds[key] = val

        return {
            "corrected_segments": corrected,
            "processed_predictions": updated_preds,
        }
