"""CleanAndExtractStage — merge batch predictions, extract positions from logits."""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from AlignAIR.Pipeline.Stage.protocol import Stage, StageContext

logger = logging.getLogger("AlignAIR.Pipeline")


class CleanAndExtractStage(Stage):
    """Merges batch predictions, extracts argmax positions from logits.

    Converts the list-of-batch-dicts from BatchInferenceStage into
    a single dict of stacked arrays, with logits replaced by argmax positions.
    """

    reads = frozenset({"raw_predictions", "model"})
    writes = frozenset({"processed_predictions"})

    def run(self, context: StageContext) -> Dict[str, Any]:
        predictions = context["raw_predictions"]
        model = context["model"]
        has_d = model.has_d_gene

        def extract_values(key):
            return [batch[key] for batch in predictions]

        def stack_or_none(key):
            if key in predictions[0]:
                return np.vstack(extract_values(key))
            return None

        # Scalars
        mutation_rate = np.vstack(extract_values('mutation_rate'))
        indel_count = np.vstack(extract_values('indel_count'))
        productive = np.vstack(extract_values('productive')) > 0.5

        # Allele probabilities
        v_allele = np.vstack(extract_values('v_allele'))
        j_allele = np.vstack(extract_values('j_allele'))

        # Position extraction: prefer logits (argmax) over scalar expectations
        v_start_logits = stack_or_none('v_start_logits')
        v_end_logits = stack_or_none('v_end_logits')
        j_start_logits = stack_or_none('j_start_logits')
        j_end_logits = stack_or_none('j_end_logits')

        if v_start_logits is not None and v_end_logits is not None and \
           j_start_logits is not None and j_end_logits is not None:
            v_start = np.argmax(v_start_logits, axis=-1)[:, None].astype(np.float32)
            v_end = np.argmax(v_end_logits, axis=-1)[:, None].astype(np.float32)
            j_start = np.argmax(j_start_logits, axis=-1)[:, None].astype(np.float32)
            j_end = np.argmax(j_end_logits, axis=-1)[:, None].astype(np.float32)
        else:
            v_start = np.vstack(extract_values('v_start'))
            v_end = np.vstack(extract_values('v_end'))
            j_start = np.vstack(extract_values('j_start'))
            j_end = np.vstack(extract_values('j_end'))

        # D gene
        d_allele = None
        d_start = None
        d_end = None
        if has_d:
            d_allele = np.vstack(extract_values('d_allele'))
            d_start_logits = stack_or_none('d_start_logits')
            d_end_logits = stack_or_none('d_end_logits')
            if d_start_logits is not None and d_end_logits is not None:
                d_start = np.argmax(d_start_logits, axis=-1)[:, None].astype(np.float32)
                d_end = np.argmax(d_end_logits, axis=-1)[:, None].astype(np.float32)
            else:
                d_start = np.vstack(extract_values('d_start'))
                d_end = np.vstack(extract_values('d_end'))

        # Diagnostics
        self._log_diagnostics(v_allele, j_allele, d_allele)

        output = {
            'v_allele': v_allele,
            'j_allele': j_allele,
            'v_start': v_start,
            'v_end': v_end,
            'j_start': j_start,
            'j_end': j_end,
            'mutation_rate': mutation_rate,
            'indel_count': indel_count,
            'productive': productive,
        }
        if has_d:
            output['d_allele'] = d_allele
            output['d_start'] = d_start
            output['d_end'] = d_end

        # Multi-chain type output
        if 'chain_type' in predictions[0]:
            output['type_'] = np.vstack(extract_values('chain_type'))

        return {"processed_predictions": output}

    def _log_diagnostics(self, v_allele, j_allele, d_allele):
        """Log allele distribution diagnostics pre-threshold."""
        def summarize(name, arr):
            if arr is None:
                return
            n, c = arr.shape
            row_max = np.max(arr, axis=1)
            q25, q50, q75 = np.percentile(row_max, [25, 50, 75])
            logger.info(
                "[Diag] %s: shape=(%d,%d) | mean=%.4f std=%.4f min=%.4f max=%.4f | "
                "row_max mean=%.4f std=%.4f min=%.4f q25=%.4f med=%.4f q75=%.4f max=%.4f",
                name, n, c, arr.mean(), arr.std(), arr.min(), arr.max(),
                row_max.mean(), row_max.std(), row_max.min(), q25, q50, q75, row_max.max(),
            )

        try:
            summarize('V allele (pre-threshold)', v_allele)
            summarize('J allele (pre-threshold)', j_allele)
            if d_allele is not None:
                summarize('D allele (pre-threshold)', d_allele)
        except Exception as e:
            logger.debug("Pre-threshold diagnostics skipped: %s", e)
