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

import torch

from .eval_gym import _merge, finalize_gene, gene_allele_counts
from .gym_bridge import aa_usable, build_batch_aa

_GENES = ("v", "d", "j")


def _empty_counts():
    return gene_allele_counts(torch.zeros(0, 1), torch.zeros(0, 1), [], None)


def run_aa_benchmark(model, reference_set, cfg, class_maps, *, spec=None, dataconfig=None,
                     device: str = "cpu", threshold: float = 0.5, batch_size: int = 128) -> dict:
    """Return per-stratum + overall allele-recovery metrics for the AA model.

    ``class_maps``: {"V": {name: class_id}, "J": {name: class_id}} (no D). ``spec`` defaults to the
    canonical ``default_igh_spec()``."""
    from alignair_benchmark import default_igh_spec, stream_benchmark

    spec = spec or default_igh_spec()
    model.eval()
    genes = ["v", "j"] + (["d"] if cfg.has_d else [])

    groups = defaultdict(list)
    for case in stream_benchmark(spec, dataconfig=dataconfig, reference_set=reference_set):
        groups[case.stratum].append(case)

    overall = {g: _empty_counts() for g in genes}
    ov_total = ov_aa = 0
    by_stratum: dict = {}

    with torch.no_grad():
        for stratum, cases in groups.items():
            n_non_forward = sum(1 for c in cases if c.orientation_id != 0)
            aa_recs = [c.record for c in cases if c.orientation_id == 0 and aa_usable(c.record)]
            counts = {g: _empty_counts() for g in genes}
            for i in range(0, len(aa_recs), batch_size):
                bi, tg = build_batch_aa(aa_recs[i:i + batch_size], reference_set, cfg, device)
                out = model(bi)
                for g in genes:
                    names = reference_set.gene(g.upper()).names
                    c = gene_allele_counts(out[f"{g}_allele"].cpu(), tg[f"{g}_allele"].cpu(),
                                           names, class_maps.get(g.upper()), threshold)
                    counts[g] = _merge(counts[g], c)
                    overall[g] = _merge(overall[g], c)
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
