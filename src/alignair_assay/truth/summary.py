"""What a truth set could not score: excluded cases, warnings, unscorable genes, simulator re-calls that differ from
the generative allele, and primaries that fell back to the simulator's call."""
from __future__ import annotations

from collections import Counter

from alignair_assay.core.prediction import GENES
from alignair_assay.truth.model import ObservableTruth


def truth_summary(truths: list[ObservableTruth]) -> dict:
    """Counts describing a truth set, as strict JSON: ``cases``; ``excluded`` (reason -> cases, sorted by reason);
    ``warnings`` (case id -> its warnings, for every case with any, in truth order); ``unscorable`` (gene ->
    cases whose truth for that gene could not be built); ``simulator_disagreements`` (gene -> cases where the
    simulator re-call differs, see that function); ``primary_fallbacks`` (gene -> cases whose primary is the
    simulator's first call because the record does not name the generative allele, see that function)."""
    excluded = Counter(t.excluded for t in truths if t.excluded is not None)
    return {
        "cases": len(truths),
        "excluded": dict(sorted(excluded.items())),
        "warnings": {t.case_id: list(t.warnings) for t in truths if t.warnings},
        "unscorable": {g: sum(g in t.unscorable for t in truths) for g in GENES},
        "simulator_disagreements": simulator_disagreements(truths),
        "primary_fallbacks": primary_fallbacks(truths),
    }


def simulator_disagreements(truths: list[ObservableTruth]) -> dict[str, int]:
    """Per gene, the non-excluded cases where the simulator re-call differs: GenAIRR's post-SHM call list is not
    the generative allele's observable set (``AlleleTruth.disagrees_with_simulator``)."""
    return {g: sum(t.segments[g].alleles.disagrees_with_simulator for t in truths
                   if t.excluded is None and g in t.segments) for g in GENES}


def primary_fallbacks(truths: list[ObservableTruth]) -> dict[str, int]:
    """Per gene, the non-excluded cases whose primary allele is the simulator's first call because the record does
    not name the generative allele (``AlleleTruth.primary_fallback``): bench cases, and records simulated without
    ``expose_provenance``."""
    return {g: sum(t.segments[g].alleles.primary_fallback for t in truths
                   if t.excluded is None and g in t.segments) for g in GENES}
