"""Shared repertoire recipes: read designs, SHM classes and the one GenAIRR chain the repertoire suites build.

``repertoire`` applies, in this order: ``recombine`` (NP lengths optionally shifted), ``trim`` (pooled trim
marginals optionally shifted; GenAIRR requires it immediately after ``recombine``), ``mutate`` (skipped at rate
0), one noise step, then 5' and 3' end loss. Every distribution comes from ``ctx.config``, the DataConfig the
truth reference is built from, so an unshifted recipe samples exactly what GenAIRR samples by default.

The targeted suites build their own chains from ``mutated`` (the SHM step) and ``symmetric_d_trim`` (a fixed trim
at both D ends that GenAIRR can always apply; see ``d_alleles_for_trim``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from GenAIRR import Experiment

if TYPE_CHECKING:
    from alignair_assay.suites.base import RecipeContext

# Read design -> (5' end-loss range, 3' end-loss range), inclusive, in nucleotides; (0, 0) skips that step. The
# amplicon designs are FRx -> J products: fr1, fr2 and fr3 start in that framework region and keep the junction.
READ_DESIGNS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "full": ((0, 0), (0, 0)),
    "fr1": ((0, 25), (0, 5)),
    "fr2": ((105, 135), (0, 5)),
    "fr3": ((225, 255), (0, 5)),
}
# SHM class -> Poisson per-base mutation rate (0 to 1) of the S5F ``hh_s5f`` kernel.
SHM_CLASSES: dict[str, float] = {"igm": 0.03, "igg": 0.075, "high": 0.20}

# ``trim()`` keyword -> (``DataConfig.trim_dicts`` key, segment, trimmed at both ends of the segment).
_TRIM_SIDES = {"v_3": ("V_3", "v", False), "d_5": ("D_5", "d", True), "d_3": ("D_3", "d", True),
               "j_5": ("J_5", "j", False)}
# GenAIRR's trim cap keeps this many bases of the shortest allele of a segment (its anchor codon).
_TRIM_MARGIN = 3
# A fixed trim at both D ends keeps at least this many bases of every D allele it is applied to.
_D_TRIM_KEEP = 2


def repertoire(ctx: RecipeContext, *, design: str = "full", rate: float = 0.075, model: str = "s5f",
               s5f_model: str = "hh_s5f", np_shift: int = 0, trim_shift: int = 0,
               noise: tuple[str, dict] | None = None) -> Experiment:
    """A fresh ``Experiment.on(ctx.config)`` repertoire chain: recombination, SHM, one noise step and the end loss
    of read ``design`` (a ``READ_DESIGNS`` key; ValueError otherwise).

    ``rate`` is the Poisson per-base SHM rate (0 to 1; 0 skips ``mutate``) of ``model`` (``"s5f"`` or
    ``"uniform"``) with S5F kernel ``s5f_model``. ``np_shift`` adds that many nucleotides to every length of the
    config's NP1 (and, with D alleles, NP2) distribution; ``trim_shift`` adds that many to every length of the
    pooled trim marginals (``pooled_trim_marginals``), each clamped to GenAIRR's cap for its side so no allele is
    over-trimmed. ``noise`` is a ``(step name, keyword arguments)`` pair applied verbatim, such as
    ``("sequencing_errors", {"rate": 0.01})``.
    """
    if design not in READ_DESIGNS:
        raise ValueError(f"unknown read design {design!r}; known designs: {', '.join(READ_DESIGNS)}")
    config = ctx.config
    if np_shift:
        np_lengths = {"np1_lengths": _shift(_np_lengths(config, "NP1"), np_shift)}
        if config.d_alleles:
            np_lengths["np2_lengths"] = _shift(_np_lengths(config, "NP2"), np_shift)
        experiment = Experiment.on(config).recombine(**np_lengths)
    else:
        experiment = Experiment.on(config).recombine()
    if trim_shift:
        caps = _trim_caps(config)
        experiment = experiment.trim(**{side: _shift(pairs, trim_shift, cap=caps.get(side))
                                        for side, pairs in pooled_trim_marginals(config).items()})
    experiment = mutated(experiment, rate, model=model, s5f_model=s5f_model)
    if noise is not None:
        step, kwargs = noise
        experiment = getattr(experiment, step)(**kwargs)
    five_prime, three_prime = READ_DESIGNS[design]
    if five_prime != (0, 0):
        experiment = experiment.end_loss_5prime(length=five_prime)
    if three_prime != (0, 0):
        experiment = experiment.end_loss_3prime(length=three_prime)
    return experiment


def mutated(experiment: Experiment, rate: float, *, model: str = "s5f", s5f_model: str = "hh_s5f") -> Experiment:
    """``experiment`` followed by GenAIRR's SHM step: Poisson per-base ``rate`` (0 to 1) of ``model`` (``"s5f"`` or
    ``"uniform"``) with S5F kernel ``s5f_model``; ``experiment`` unchanged at rate 0."""
    if not rate:
        return experiment
    return experiment.mutate(model=model, rate=rate, s5f_model=s5f_model)


def d_alleles_for_trim(config, per_side: int) -> list[str] | None:
    """The D alleles of DataConfig ``config`` that a trim of ``per_side`` nucleotides at both ends leaves at least
    2 bases of (``_D_TRIM_KEEP``): their sorted names when some D allele is shorter than ``2 * per_side + 2`` nt,
    else None (every allele qualifies and the D pool needs no restriction).

    GenAIRR's engine aborts the whole run (a Rust panic while assembling D) when the two D trims exceed the sampled
    allele. Its default trims stay under a cap (``pooled_trim_marginals``: 4 per side on human IGH, whose shortest D
    allele is 11 nt), but ``trim()`` applies an explicit value as given, whatever the allele, so a deeper trim must
    restrict D to these alleles.
    """
    lengths = {allele.name: len(allele.ungapped_seq)
               for alleles in (config.d_alleles or {}).values() for allele in alleles}
    fitting = sorted(name for name, length in lengths.items() if length >= 2 * per_side + _D_TRIM_KEEP)
    return None if len(fitting) == len(lengths) else fitting


def symmetric_d_trim(ctx: RecipeContext, per_side: int) -> Experiment:
    """A fresh ``Experiment.on(ctx.config)`` recombination whose D is trimmed exactly ``per_side`` nucleotides at
    both ends, every other trim and NP length keeping GenAIRR's defaults. When some D allele is too short for that
    trim, D is restricted to the alleles that fit (``d_alleles_for_trim``), sampled uniformly; otherwise the D pool
    is GenAIRR's default. Every record then carries ``d_trim_5 == d_trim_3 == per_side``."""
    experiment = Experiment.on(ctx.config)
    fitting = d_alleles_for_trim(ctx.config, per_side)
    if fitting is not None:
        experiment = experiment.restrict_alleles(d=fitting)
    return experiment.recombine().trim(d_5=[(per_side, 1.0)], d_3=[(per_side, 1.0)])


def pooled_trim_marginals(config) -> dict[str, list[tuple[int, float]]]:
    """Per ``trim()`` keyword (``v_3``, ``d_5``, ``d_3``, ``j_5``; sides without trim data are left out), the trim
    length distribution GenAIRR's ``recombine()`` samples by default from DataConfig ``config``: the mean over
    genes of each gene's ``{length: probability}`` in ``config.trim_dicts``, restricted to lengths up to GenAIRR's
    cap for the side (the shortest allele of the segment minus a 3-base anchor margin, halved for the two D
    sides) and normalised to sum to 1. Pairs are ``(length in nucleotides, probability)`` sorted by length, so
    ``trim(**pooled_trim_marginals(config))`` reproduces GenAIRR's default trimming exactly."""
    caps = _trim_caps(config)
    marginals = {}
    for side, (key, _, _) in _TRIM_SIDES.items():
        genes = [distribution for family in (config.trim_dicts.get(key) or {}).values()
                 for distribution in family.values() if distribution]
        if not genes:
            continue
        totals: dict[int, float] = {}
        for distribution in genes:
            for length, probability in distribution.items():
                totals[int(length)] = totals.get(int(length), 0.0) + float(probability)
        cap = caps.get(side)
        kept = {length: total for length, total in totals.items()
                if total > 0 and (cap is None or length <= cap)}
        norm = sum(kept.values())
        marginals[side] = [(length, kept[length] / norm) for length in sorted(kept)]
    return marginals


def _np_lengths(config, segment: str) -> list[tuple[int, float]]:
    """The config's empirical ``segment`` (``"NP1"`` or ``"NP2"``) length distribution as ``(length, weight)``
    pairs with positive weight, sorted by length."""
    distribution = config.NP_lengths.get(segment) or {}
    return sorted((int(length), float(weight)) for length, weight in distribution.items() if weight > 0)


def _trim_caps(config) -> dict[str, int]:
    """Per ``trim()`` keyword, the longest trim GenAIRR allows by default: the segment's shortest allele minus
    the anchor margin, halved for a segment trimmed at both ends."""
    tables = {"v": config.v_alleles, "d": config.d_alleles, "j": config.j_alleles}
    caps = {}
    for side, (_, segment, two_sided) in _TRIM_SIDES.items():
        lengths = [len(allele.ungapped_seq) for alleles in (tables[segment] or {}).values() for allele in alleles]
        if lengths:
            cap = max(0, min(lengths) - _TRIM_MARGIN)
            caps[side] = cap // 2 if two_sided else cap
    return caps


def _shift(pairs: list[tuple[int, float]], shift: int, *, cap: int | None = None) -> list[tuple[int, float]]:
    """The distribution of ``length + shift`` clamped to ``[0, cap]``: lengths that clamp to the same value pool
    their weights. Pairs are sorted by length."""
    shifted: dict[int, float] = {}
    for length, weight in pairs:
        moved = max(0, length + shift)
        if cap is not None:
            moved = min(moved, cap)
        shifted[moved] = shifted.get(moved, 0.0) + weight
    return sorted(shifted.items())
