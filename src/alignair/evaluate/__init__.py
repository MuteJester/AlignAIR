"""The model benchmark API: generate labelled GenAIRR reads, predict them, score per stratum."""
from .benchmark import default_strata, format_text, generate_labeled, run_benchmark, score

__all__ = ["default_strata", "format_text", "generate_labeled", "run_benchmark", "score"]
