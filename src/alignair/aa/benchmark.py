"""Stratified benchmark of the AA model, reusing the REAL benchmark harness' validated strata.

The benchmark module generates GenAIRR cases per stratum (clean/moderate/hard full, FR1/FR2/
J-anchored amplicons, high-SHM, high-indel, noisy, trimmed, orientation). Each case carries the
full GenAIRR ``record`` (with ``sequence_aa``) and its ``stratum`` label, so we drive the AA model
on ``sequence_aa`` and score allele recovery per stratum.

Two strata classes are inherently NOT AA-evaluable and are reported as such (findings, not errors):
- junction-trimmed amplicons (V-anchored / FR / J-anchored) have no ``sequence_aa``;
- orientation-transformed reads (RC etc.) — proteins have no reverse complement.
"""
from __future__ import annotations

from collections import defaultdict

from ..evaluate.cohort import collect_alignment_observations
from ..evaluate.alleles import (allele_observation_counts, empty_allele_counts, finalize_gene,
                                merge_allele_counts, validate_allele_request)
from ..data.adapters import ProteinAlleleBatchAdapter
from ..data.projection import translated_record_usable as aa_usable


def run_aa_benchmark(model, reference_set, cfg, class_maps, *, spec=None, dataconfig=None,
                     device: str = "cpu", threshold: float = 0.5, batch_size: int = 128) -> dict:
    """Return per-stratum + overall allele-recovery metrics for the AA model.

    ``class_maps``: {"V": {name: class_id}, "J": {name: class_id}} (no D). ``spec`` defaults to the
    canonical ``default_igh_spec()``."""
    from alignair_benchmark import default_igh_spec, stream_benchmark

    batch_size = validate_allele_request(cfg, threshold, batch_size)
    spec = spec or default_igh_spec()
    genes = ["v", "j"] + (["d"] if cfg.has_d else [])

    groups = defaultdict(list)
    for case in stream_benchmark(spec, dataconfig=dataconfig, reference_set=reference_set):
        groups[case.stratum].append(case)

    overall = {g: empty_allele_counts() for g in genes}
    ov_total = ov_aa = 0
    by_stratum: dict = {}

    for stratum, cases in groups.items():
        n_non_forward = sum(1 for c in cases if c.orientation_id != 0)
        aa_recs = [c.record for c in cases if c.orientation_id == 0 and aa_usable(c.record)]
        observed = collect_alignment_observations(
            model, aa_recs, ProteinAlleleBatchAdapter(reference_set, cfg, device), cfg,
            batch_size=batch_size, prediction_set_threshold=threshold,
        )
        counts = {}
        for g in genes:
            counts[g] = allele_observation_counts(
                observed.classification[g], observed.prediction_sets[g],
                reference_set.gene(g.upper()).names, class_maps.get(g.upper()),
            )
            overall[g] = merge_allele_counts(overall[g], counts[g])
        by_stratum[stratum] = {
            "n_total": len(cases), "n_aa": len(aa_recs), "n_non_forward": n_non_forward,
            "no_aa_rate": 1.0 - len(aa_recs) / max(1, len(cases)),
            "alleles": {g: finalize_gene(counts[g], class_maps.get(g.upper()) is not None)
                        for g in genes},
        }
        ov_total += len(cases)
        ov_aa += len(aa_recs)

    return {"spec": spec.name, "n_total": ov_total, "n_aa": ov_aa,
            "overall_no_aa_rate": 1.0 - ov_aa / max(1, ov_total),
            "overall_alleles": {g: finalize_gene(overall[g], class_maps.get(g.upper()) is not None)
                                for g in genes},
            "by_stratum": by_stratum}
