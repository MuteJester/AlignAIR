"""Declared protein convergence cohort construction and content identity."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np

from ...data.augmentation.protein import aa_indel, aa_substitute, aa_truncate, il_ambiguity
from ...data.projection import translated_record_usable as aa_usable
from ...errors import ConfigurationError, InputError, MissingDependencyError, TrainingDataError
from ...reference import build_protein_reference

COHORT_SCHEMA = "alignair.convergence.cohort.v2"
STRATA_ORDER = ("clean", "moderate_full", "aa_subs_7pct", "aa_single_indel", "aa_il_ambiguity",
                "aa_truncate_50_80", "short_janchor", "dj_indel", "v_indel_1", "v_indel_multi")


def cohort_fingerprint(cohort: Mapping) -> str:
    """Hash versioned finite JSON content, including source records and ordered row lists.

    Mapping insertion order is irrelevant. This checksum does not freeze objects or validate
    scientific eligibility; recompute it after editing a cohort, and bind model/reference identity
    separately when comparing evaluations.
    """
    context = {"stage": "convergence_cohort", "field": "cohort"}
    if not isinstance(cohort, Mapping) or cohort.get("schema") != COHORT_SCHEMA:
        raise InputError("cohort fingerprint requires the current cohort schema", context=context)
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    try:
        for chunk in encoder.iterencode(cohort):
            digest.update(chunk.encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise InputError("cohort fingerprint requires finite JSON content", context=context) from exc
    return digest.hexdigest()


def _request(n_per_stratum, seed, cal_frac):
    for field, value, minimum in (("n_per_stratum", n_per_stratum, 1), ("seed", seed, 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ConfigurationError("cohort counts and seeds must be valid integers",
                                     context={"stage": "convergence_cohort", "field": field})
    if (isinstance(cal_frac, bool) or not isinstance(cal_frac, Real)
            or not 0 < cal_frac < 1 or not math.isfinite(cal_frac)):
        raise ConfigurationError("calibration fraction must be finite and strictly between zero and one",
                                 context={"stage": "convergence_cohort", "field": "cal_frac"})
    return int(n_per_stratum), int(seed), float(cal_frac)


def _first(call) -> str:
    return str(call or "").split(",")[0].strip()


def build_cohort(dc_name: str = "HUMAN_IGH_OGRDB", n_per_stratum: int = 200,
                 seed: int = 20250722, *, cal_frac: float = 0.3):
    """Build a FROZEN simulator eval cohort + a disjoint calibration slice from one seeded stream.

    Constructs the declared evaluation strata: a clean pool
    from ``stream_benchmark(default_igh_spec(...))`` (AA-usable, orientation 0, known V+J allele,
    ``n_indels==0``, stratum == ``clean_full`` ONLY -- this matches the frozen baseline/§5 gates,
    which were computed against ``clean_full`` alone), AA-native augmentation strata built from
    that pool (``aa_substitute``/``aa_indel``/``il_ambiguity``/``aa_truncate``), plus native-indel
    buckets (``v_indel_1``/``v_indel_multi``/``dj_indel``) and a J-anchored short stratum
    (``short_janchor``, from the ``amplicon_janchor`` stratum). ``n_indels==0`` rows from the
    ``moderate_full`` stratum are collected SEPARATELY into their own ``moderate_full`` bucket
    (scored like any other stratum in ``by_stratum`` -- not pooled into ``clean``).

    A fixed ``cal_frac`` slice of the clean (``clean_full``-only) pool is carved off as
    ``calibration`` (never scored as ``"clean"``) so conformal calibration in
    ``evaluate_checkpoint`` never touches the accuracy-eval rows. The pool is deterministically
    shuffled (seeded on ``seed``) before the calibration/eval split, since the underlying stream
    is not randomly ordered (front-of-stream rows skew easier) -- both halves must be
    representative, not just disjoint.

    Returns ``(cohort, fingerprint_sha256)``:
    - ``cohort = {"by_stratum": {name: [row, ...]}, "calibration": [row, ...], "dc_name": ...,
      "seed": ..., "n_per_stratum": ...}``; every ``row`` is ``{"aa", "v_call", "j_call", "d_call",
      "d_inverted", "record"}`` (``record`` is the raw GenAIRR record, unaugmented, used only for
      boundary-MAE projection on the ``clean`` stratum).
    - ``fingerprint_sha256``: versioned SHA-256 over the complete returned finite JSON content,
      including D inversion, source records, split parameters and reference hash. It differs from
      the historical sequence/call-only digest. A checksum does not make the returned dict immutable.
    """
    n_per_stratum, seed, cal_frac = _request(n_per_stratum, seed, cal_frac)
    if not isinstance(dc_name, str) or not dc_name:
        raise ConfigurationError("cohort requires a dataconfig name",
                                 context={"stage": "convergence_cohort", "field": "dc_name"})
    try:
        from alignair_benchmark import default_igh_spec, stream_benchmark
    except ModuleNotFoundError as exc:
        if exc.name != "alignair_benchmark":
            raise
        raise MissingDependencyError("convergence cohort generation requires alignair_benchmark",
                                     hint="Run generation from a source checkout containing the benchmark package.",
                                     context={"stage": "convergence_cohort"}) from exc
    import GenAIRR as ga

    reference = build_protein_reference(dc_name)
    dc = getattr(ga, dc_name)
    class_maps = reference.class_maps
    vclass, jclass = class_maps["V"].allele_class, class_maps["J"].allele_class

    rng = np.random.default_rng(seed)
    spec = default_igh_spec(n_per_stratum=n_per_stratum, seed=seed)

    def _row(r, aa):
        return {
            "aa": aa,
            "v_call": str(r.get("v_call") or ""),
            "j_call": str(r.get("j_call") or ""),
            "d_call": str(r.get("d_call") or ""),
            "d_inverted": bool(r.get("d_inverted", False)),
            "record": r,
        }

    buckets: dict = defaultdict(list)
    clean_pool = []
    for c in stream_benchmark(spec, dataconfig=dc):
        r = c.record
        if c.orientation_id != 0 or not aa_usable(r):
            continue
        vc, jc = _first(r["v_call"]), _first(r["j_call"])
        if vc not in vclass or jc not in jclass:
            continue
        row = _row(r, r["sequence_aa"])
        n_indels, n_v_indels = int(r.get("n_indels", 0)), int(r.get("n_v_indels", 0))
        if n_indels == 0 and c.stratum == "clean_full":
            clean_pool.append(row)
        elif n_indels == 0 and c.stratum == "moderate_full":
            buckets["moderate_full"].append(row)
        elif n_v_indels == 1:
            buckets["v_indel_1"].append(row)
        elif n_v_indels >= 2:
            buckets["v_indel_multi"].append(row)
        elif n_indels > 0 and n_v_indels == 0:
            buckets["dj_indel"].append(row)
        if c.stratum == "amplicon_janchor":
            buckets["short_janchor"].append(row)

    # Disjoint split of the clean (clean_full-only) pool: a calibration slice (never scored as
    # "clean") + the eval/augmentation source. The underlying stream is NOT randomly ordered
    # (progress-ramped generation skews front-of-stream rows easier), so a deterministically
    # seeded shuffle runs before the index split -- otherwise both the calibration slice and the
    # "clean" eval slice would be systematically biased by their position in the stream.
    shuffle_rng = np.random.default_rng(seed)
    if len(clean_pool) < 2:
        raise TrainingDataError("cohort requires at least two eligible clean records for a disjoint split",
                                context={"stage": "convergence_cohort", "clean_records": len(clean_pool)})
    order = shuffle_rng.permutation(len(clean_pool))
    clean_pool = [clean_pool[i] for i in order]
    n_cal = min(len(clean_pool) - 1, max(1, int(round(cal_frac * len(clean_pool)))))
    calibration = clean_pool[:n_cal]
    eval_source = clean_pool[n_cal:]
    buckets["clean"] = list(eval_source)

    for row in eval_source:
        aa = row["aa"]
        for name, aug in (
            ("aa_subs_7pct", aa_substitute(aa, 0.07, rng)),
            ("aa_single_indel", aa_indel(aa, rng)),
            ("aa_il_ambiguity", il_ambiguity(aa)),
            ("aa_truncate_50_80", aa_truncate(aa, 0.5, 0.8, rng)),
        ):
            # v_call/j_call/d_call truth carries over (the augmentation corrupts the AA string,
            # not the germline truth); "record" is dropped -- it no longer matches the corrupted
            # AA string, so it must not be fed to boundary-MAE projection.
            buckets[name].append({k: v for k, v in row.items() if k != "record"} | {"aa": aug})

    by_stratum = {name: buckets[name] for name in STRATA_ORDER if buckets.get(name)}

    cohort = {"schema": COHORT_SCHEMA, "strata_order": list(STRATA_ORDER),
              "by_stratum": by_stratum, "calibration": calibration,
              "dc_name": dc_name, "seed": seed, "n_per_stratum": n_per_stratum,
              "cal_frac": cal_frac, "reference_db_hash": reference.db_hash}
    return cohort, cohort_fingerprint(cohort)
