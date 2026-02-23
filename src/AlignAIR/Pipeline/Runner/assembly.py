"""Pipeline assembly — builds the stage list from configuration."""
from __future__ import annotations

from typing import List

from AlignAIR.Pipeline.Models.config import PipelineConfig
from AlignAIR.Pipeline.Models.enums import OutputFormat
from AlignAIR.Pipeline.Stage.protocol import Stage


def assemble_pipeline(config: PipelineConfig) -> List[Stage]:
    """Build the standard prediction pipeline from config.

    This is the single place where pipeline structure is defined.
    Conditional stages are included unconditionally but will self-skip at runtime.
    """
    from AlignAIR.Pipeline.Stage.stages.load_model import LoadModelStage
    from AlignAIR.Pipeline.Stage.stages.batch_inference import BatchInferenceStage
    from AlignAIR.Pipeline.Stage.stages.clean_and_extract import CleanAndExtractStage
    from AlignAIR.Pipeline.Stage.stages.genotype_adjustment import GenotypeAdjustmentStage
    from AlignAIR.Pipeline.Stage.stages.segment_correction import SegmentCorrectionStage
    from AlignAIR.Pipeline.Stage.stages.allele_threshold import AlleleThresholdStage
    from AlignAIR.Pipeline.Stage.stages.germline_alignment import GermlineAlignmentStage
    from AlignAIR.Pipeline.Stage.stages.translate_names import TranslateNamesStage
    from AlignAIR.Pipeline.Stage.stages.serialize import CSVSerializeStage, AIRRSerializeStage

    stages: List[Stage] = [
        LoadModelStage("LoadModel"),
        BatchInferenceStage("BatchInference"),
        CleanAndExtractStage("CleanAndExtract"),
        GenotypeAdjustmentStage("GenotypeAdjustment"),   # conditional: skips if no genotype
        SegmentCorrectionStage("SegmentCorrection"),
        AlleleThresholdStage("AlleleThreshold"),
        GermlineAlignmentStage("GermlineAlignment"),
    ]

    if config.translate_to_asc:
        stages.append(TranslateNamesStage("TranslateNames"))

    if config.output_format == OutputFormat.AIRR:
        stages.append(AIRRSerializeStage("Serialize"))
    else:
        stages.append(CSVSerializeStage("Serialize"))

    return stages
