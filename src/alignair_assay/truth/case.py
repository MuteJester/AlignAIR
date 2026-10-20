"""Observable truth for a case (``core.cases``): the record's truth plus the case's suite, labels and groups."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from alignair_assay.core.genotype import NovelAllele
from alignair_assay.core.reference import Reference
from alignair_assay.core.seqops import revcomp
from alignair_assay.truth.build import truth_from_genairr_record
from alignair_assay.truth.model import ObservableTruth


def truth_from_case(case: dict, reference: Reference, *,
                    novel: Mapping[str, NovelAllele] | None = None) -> ObservableTruth:
    """Observable truth of ``case`` against ``reference``, carrying its suite, labels and groups. ``novel`` maps
    the novel alleles of the case's donor by name; a novel primary V is re-based onto its base allele
    (``truth.build.truth_from_genairr_record``).

    Raises ValueError, naming the case, when the presented sequence is not the record sequence (orientation 0)
    or its reverse complement (orientation 1), so a tampered or mismatched case is never scored.
    """
    record = case["record"]
    forward = record["sequence"].upper()
    expected = revcomp(forward) if case["orientation"] else forward
    if case["sequence"] != expected:
        raise ValueError(f"{case['case_id']}: presented sequence does not match the record "
                         f"(orientation {case['orientation']})")
    truth = truth_from_genairr_record(record, reference, case_id=case["case_id"], stratum=case["stratum"],
                                      orientation=case["orientation"], novel=novel)
    return replace(truth, suite=case["suite"], labels=dict(case["labels"]), groups=dict(case["groups"]))
