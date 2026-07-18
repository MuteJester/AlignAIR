"""Stage 3 — SHM vs. germline polymorphism (the arbiter).

Aggregate per-position mismatch over the (low-SHM-weighted) reads assigned to an allele. A position
mismatched in a HIGH fraction of reads is a germline polymorphism (systematic), not SHM (random,
per-read). Resolve the variant ONLY where the diagnostic sites are covered:
reassign to a known allele → "compatible with" (sites uncovered) → novel candidate (no known match).
"""
from __future__ import annotations

from collections import defaultdict

from .observe import germline_observations


def polymorphism_profile(reads, germline: str, weights=None) -> dict:
    """``reads`` = list of ``(read_seq, cigar, seq_start, germ_start)``. Returns
    ``{germline_pos: {coverage, mismatch_fraction, alt}}`` (``alt`` = consensus non-reference base)."""
    weights = weights if weights is not None else [1.0] * len(reads)
    cov: dict = defaultdict(float)
    mism: dict = defaultdict(float)
    alt_mass: dict = defaultdict(lambda: defaultdict(float))
    for (seq, cigar, ss, gs), w in zip(reads, weights):
        for g, base in germline_observations(seq, cigar, ss, gs).items():
            if g >= len(germline):
                continue
            cov[g] += w
            if base is not None and base != germline[g]:
                mism[g] += w
                alt_mass[g][base] += w
    profile = {}
    for g, c in cov.items():
        alt = max(alt_mass[g], key=alt_mass[g].get) if alt_mass[g] else None
        profile[g] = {"coverage": c, "mismatch_fraction": (mism[g] / c if c > 0 else 0.0), "alt": alt}
    return profile


def resolve(profile: dict, allele_name: str, reference, gene: str, *,
            poly_thr: float = 0.6, min_cov: float = 1.0) -> dict:
    """Decide what the reads assigned to ``allele_name`` really are, from the polymorphism profile."""
    ref = reference.gene(gene.upper())
    a = ref.sequences[ref.index[allele_name]]
    poly = {g: p["alt"] for g, p in profile.items()
            if p["mismatch_fraction"] > poly_thr and p["coverage"] >= min_cov and p["alt"]}
    if not poly:
        return {"call": "confirm", "allele": allele_name}

    def covered(g):
        return g in profile and profile[g]["coverage"] >= min_cov

    def obs_at(g):
        return poly.get(g, a[g] if g < len(a) else None)

    for b_name in ref.names:                       # does a known allele explain the polymorphism?
        if b_name == allele_name:
            continue
        b = ref.sequences[ref.index[b_name]]
        dist = [g for g in range(min(len(a), len(b))) if a[g] != b[g]]
        if not dist:
            continue
        covered_dist = [g for g in dist if covered(g)]
        uncovered_dist = [g for g in dist if not covered(g)]
        if covered_dist and all(obs_at(g) == b[g] for g in covered_dist):
            return {"call": "reassign" if not uncovered_dist else "compatible_with", "allele": b_name}

    prov = list(a)                                  # no known match -> provisional novel candidate
    mask = ["ref"] * len(prov)
    for g in range(len(prov)):
        if g not in profile:
            mask[g] = "uncovered"
    for g, alt in poly.items():
        if g < len(prov):
            prov[g] = alt
            mask[g] = "observed"
    return {"call": "novel", "positions": sorted(poly), "sequence": "".join(prov), "source_mask": mask}
