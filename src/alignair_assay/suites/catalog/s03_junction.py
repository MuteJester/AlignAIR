"""S3 ``junction``: full-length, lightly mutated reads with one junction variable fixed per stratum.

Each stratum fixes one variable of the sweep it belongs to and leaves the others at GenAIRR's defaults: the NP1
or NP2 length (``recombine(np1_lengths=...)``), the V 3' or J 5' trim (``trim(v_3=...)``; ``trim()`` keeps the
default distribution of every side it is not given), or the same trim at both D ends (``symmetric_d_trim``, which
restricts D to the alleles long enough for it). Trims are exact on every record. A record's ``np1_length`` /
``np2_length`` is GenAIRR's annotation, which is one base shorter than the sampled NP on about 1% (NP1) to 2.5%
(NP2) of records: GenAIRR claims an NP base for the neighbouring V, D or J when it matches that segment's trimmed
germline (or, for a segment GenAIRR re-called, the re-called allele's). The labels name the requested value.
"""
from __future__ import annotations

from functools import partial

from GenAIRR import Experiment

from alignair_assay.suites.base import RecipeContext, Stratum, Suite
from alignair_assay.suites.recipes import experiment_on, mutated, symmetric_d_trim

# Poisson per-base SHM rate (0 to 1) of the S5F ``hh_s5f`` kernel.
_RATE = 0.02
# Sweep -> its values, in nucleotides: NP lengths, or the trim of one end (both D ends for ``d_both``).
_SWEEPS = {"np1": (0, 3, 6, 12, 20), "np2": (0, 3, 6, 12, 20), "v3": (0, 2, 5, 10), "j5": (0, 4, 8, 14),
           "d_both": (0, 4, 8)}
# Trim sweep -> its ``trim()`` keyword.
_TRIM_SIDES = {"v3": "v_3", "j5": "j_5"}
# Record fields of GenAIRR's P-nucleotide lengths, one per segment end.
_P_FIELDS = ("p_v_3_length", "p_d_5_length", "p_d_3_length", "p_j_5_length")


def junction_repertoire(ctx: RecipeContext, *, sweep: str, value: int) -> Experiment:
    """Recombination with ``sweep`` (a key of ``_SWEEPS``) fixed at ``value`` nucleotides, then SHM at Poisson
    per-base rate 0.02; no end loss."""
    if sweep == "d_both":
        experiment = symmetric_d_trim(ctx, value)
    elif sweep in ("np1", "np2"):
        experiment = experiment_on(ctx.config).recombine(**{f"{sweep}_lengths": [(value, 1.0)]})
    else:
        experiment = experiment_on(ctx.config).recombine().trim(**{_TRIM_SIDES[sweep]: [(value, 1.0)]})
    return mutated(experiment, _RATE, config=ctx.config)


def p_nucleotide_label(record: dict, ctx: RecipeContext) -> dict[str, str]:
    """The case label ``p_nt``: ``yes`` when the record has P-nucleotides at any segment end, else ``no``. GenAIRR
    2.3.1 cannot set P-nucleotide lengths, and a DataConfig without a P-nucleotide model (human IGH OGRDB) never
    adds any."""
    return {"p_nt": "yes" if any(record.get(field) for field in _P_FIELDS) else "no"}


S3 = Suite(
    name="S3",
    slug="junction",
    title="Junction sweeps",
    purpose="Measures junction and segment-edge accuracy as NP lengths and V, J and D trims are swept one at a time.",
    primary=("junction.nt_exact", "junction.length_exact", "clone_key.exact", "junction.start_abs_err",
             "junction.end_abs_err", "v.seq_end_exact", "j.seq_start_exact"),
    strata=tuple(Stratum(f"{sweep}_{value}", 150, partial(junction_repertoire, sweep=sweep, value=value),
                         labels={"sweep": sweep, "value": str(value)}, case_labels=p_nucleotide_label)
                 for sweep, values in _SWEEPS.items() for value in values),
)
