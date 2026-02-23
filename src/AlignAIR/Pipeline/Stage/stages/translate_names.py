"""TranslateNamesStage — translate ASC allele names back to IMGT names."""
from __future__ import annotations

import logging
from typing import Any, Dict

from AlignAIR.Pipeline.Stage.protocol import Stage, StageContext

logger = logging.getLogger("AlignAIR.Pipeline")


class TranslateNamesStage(Stage):
    """Translates allele names from ASC format to IMGT format.

    Note: In the current pipeline, this is a no-op when translate_to_asc is True
    (the default) — the names stay as ASC names. It only translates when
    translate_to_asc is False. The logic matches the existing TranslationStep.
    """

    reads = frozenset({"config", "model", "selected_allele_calls"})
    writes = frozenset({"selected_allele_calls"})

    def run(self, context: StageContext) -> Dict[str, Any]:
        config = context.config
        model = context["model"]
        allele_calls = context["selected_allele_calls"]

        logger.info("Translating allele names...")

        # The existing logic: translate only when translate_to_asc is False
        if not config.translate_to_asc:
            from AlignAIR.PostProcessing import TranslateToIMGT
            translator = TranslateToIMGT(model.dataconfig.packaged_config())

            updated = dict(allele_calls)
            updated['v'] = [
                [translator.translate(name) for name in call]
                for call in allele_calls['v']
            ]
            return {"selected_allele_calls": updated}

        # Default: no translation needed
        return {"selected_allele_calls": allele_calls}
