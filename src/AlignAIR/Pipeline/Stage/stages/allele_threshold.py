"""AlleleThresholdStage — apply max likelihood percentage threshold to distill allele calls."""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from AlignAIR.Pipeline.Stage.protocol import Stage, StageContext

logger = logging.getLogger("AlignAIR.Pipeline")


class AlleleThresholdStage(Stage):
    """Applies percentage threshold + cap to select allele calls."""

    reads = frozenset({"config", "processed_predictions", "model"})
    writes = frozenset({"selected_allele_calls", "likelihoods_of_selected_alleles"})

    def run(self, context: StageContext) -> Dict[str, Any]:
        config = context.config
        preds = context["processed_predictions"]
        model = context["model"]
        has_d = model.has_d_gene

        logger.info("Applying max likelihood thresholds...")

        from AlignAIR.PostProcessing.AlleleSelector import MaxLikelihoodPercentageThreshold
        extractor = MaxLikelihoodPercentageThreshold(dataconfig=model.dataconfig)

        alleles = {
            'v': preds['v_allele'],
            'j': preds['j_allele'],
        }
        thresholds = {
            'v': config.thresholds.v_threshold,
            'j': config.thresholds.j_threshold,
        }
        caps = {
            'v': config.thresholds.v_cap,
            'j': config.thresholds.j_cap,
        }

        if has_d and preds.get('d_allele') is not None:
            alleles['d'] = preds['d_allele']
            thresholds['d'] = config.thresholds.d_threshold
            caps['d'] = config.thresholds.d_cap

        predicted_alleles = {}
        predicted_likelihoods = {}

        for gene in thresholds:
            selected = extractor.get_alleles(
                alleles[gene],
                percentage=thresholds[gene],
                cap=caps[gene],
                allele=gene,
                verbose=True,
            )
            predicted_alleles[gene] = [item[0] for item in selected]
            predicted_likelihoods[gene] = [item[1] for item in selected]

        # Diagnostics
        try:
            for gene in predicted_likelihoods:
                count = len(predicted_likelihoods[gene])
                logger.info("Selected allele likelihoods — %s: count=%d", gene.upper(), count)
        except Exception:
            pass

        return {
            "selected_allele_calls": predicted_alleles,
            "likelihoods_of_selected_alleles": predicted_likelihoods,
        }
