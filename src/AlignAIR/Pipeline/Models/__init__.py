from AlignAIR.Pipeline.Models.enums import GeneType, OutputFormat, ThresholdMethod
from AlignAIR.Pipeline.Models.config import (
    PipelineConfig,
    AlleleThresholdConfig,
    OrientationConfig,
    MemoryConfig,
    CheckpointConfig,
    ReproducibilityConfig,
)
from AlignAIR.Pipeline.Models.slots import (
    FileInfo,
    LoadedModel,
    RawPredictions,
    ProcessedPredictions,
    CorrectedSegments,
    AlleleCall,
    SelectedAlleles,
    GermlineAlignment,
    GermlineAlignments,
    PipelineResult,
)
