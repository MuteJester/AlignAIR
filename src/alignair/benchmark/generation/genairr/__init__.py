"""GenAIRR integration boundary for benchmark generation."""
from __future__ import annotations

from .capabilities import (
    ALLOWED_GENAIRR_FEATURE_STATUSES,
    GENAIRR_FEATURES,
    GenAIRRFeature,
    genairr_feature_catalog,
    validate_genairr_feature_catalog,
)
from .experiments import (
    ResolvedGenAIRRExperiment,
    apply_record_metadata,
    build_stratum_experiment,
    dataconfig_by_name,
    resolve_stratum_params,
    run_records_kwargs_from_params,
    sampling_kwargs_from_params,
    stream_stratum_records,
    validate_stratum_records,
)
from .transforms import apply_benchmark_crop

__all__ = [
    "ALLOWED_GENAIRR_FEATURE_STATUSES",
    "GENAIRR_FEATURES",
    "GenAIRRFeature",
    "ResolvedGenAIRRExperiment",
    "apply_record_metadata",
    "apply_benchmark_crop",
    "build_stratum_experiment",
    "dataconfig_by_name",
    "genairr_feature_catalog",
    "resolve_stratum_params",
    "run_records_kwargs_from_params",
    "sampling_kwargs_from_params",
    "stream_stratum_records",
    "validate_stratum_records",
    "validate_genairr_feature_catalog",
]
