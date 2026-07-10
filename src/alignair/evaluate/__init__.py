"""In-package model evaluation: score a model on freshly-generated, labelled GenAIRR reads.

This ships in the wheel (unlike the wheel-excluded ``alignair_benchmark`` head-to-head suite). It
generates labelled reads via the training gym's GenAIRR stream, runs the model, and reports per-stratum
call accuracy, coordinate MAE and junction_nt_exact — a fast "how good is this model" self-check.
"""
from .benchmark import default_strata, format_text, generate_labeled, run_benchmark, score

# genotype constraint study moved to alignair.genotype.study
__all__ = ["default_strata", "format_text", "generate_labeled", "run_benchmark", "score"]
