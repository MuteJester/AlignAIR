"""Per-case metrics: one tool's prediction for one read, scored against observable truth.

``CaseScores`` maps a metric name (``"<scope>.<metric>"``, e.g. ``"v.top1_in_set"``) to a value. An absent key
means the metric does not apply to the case. An applicable metric the tool did not cover is ``nan``; accuracy-type
metrics score a missing prediction as 0.0 instead, except germline-frame edges, which cannot be compared without
a top-1 call in the truth set (``nan``).
"""
from __future__ import annotations

CaseScores = dict[str, float]

# Metric-name suffixes (a full name is its own suffix) that are reported but excluded from tool comparisons' Holm
# family (a comparison report shows them, untested, beside it). The signed junction edge errors are compared
# through their ``*_abs_err`` twins; SHM metrics of a V with an indel are not compared because indel placement is
# not unique; the novel SNPs a tool reports as mutations describe how it handles an allele missing from its
# reference, which no fixed-reference tool can avoid; and true-allele accuracy (``top1_generative``) cannot reach
# 1.0 for a perfect sequence-only aligner when SHM has made the read a sibling allele's (``top1_in_set`` is compared).
# Of the head-level metrics (``head.*``, ``diagnostics.heads``), the signed boundary errors are compared through their
# ``*_abs_err`` twins; true-allele accuracy is informational for the same reason as above; the number of alleles over
# the selection threshold describes the head; GenAIRR's record mutation rate is the training target, not observable
# truth; and the crop flag describes the read.
INFORMATIONAL: frozenset[str] = frozenset({".set_size", ".top1_generative", "junction.start_err", "junction.end_err",
                                           "v_indel.mutation_count_err", "v_indel.germline_errors",
                                           "v_novel.snp_mutations",
                                           ".start_err.soft", ".start_err.argmax", ".end_err.soft", ".end_err.argmax",
                                           ".allele_top1_generative", ".n_above_threshold",
                                           "head.mutation_rate_abs_err.record", "head.cropped"})

# Errors, failure rates and counts; the genotype metrics (``metrics.genotype``) are rates of calls outside the donor's
# genotype and of calls of its deleted genes. A head-level error names its decoder or its truth source after the
# ``_err`` (``head.v.start_abs_err.soft``, ``head.mutation_rate_abs_err.read``).
_LOWER_IS_BETTER = ("_err", ".no_call", ".spurious_call", ".germline_errors", ".mutation_count_err",
                    ".indel_false", ".outside_genotype", ".deleted_gene_call",
                    "_err.soft", "_err.argmax", "_err.read", "_err.record")


def is_informational(name: str) -> bool:
    """True for a metric that is reported but not compared: a name listed in ``INFORMATIONAL``, or one ending
    with a listed suffix that starts with ``.`` (``v.set_size``)."""
    return name in INFORMATIONAL or any(s.startswith(".") and name.endswith(s) for s in INFORMATIONAL)


def lower_is_better(name: str) -> bool:
    """True for metrics where a smaller value is better (errors, failure rates and counts).

    Every ``_err`` metric, including the absolute ``*_abs_err``, is an error to minimise.
    """
    return name.endswith(_LOWER_IS_BETTER)
