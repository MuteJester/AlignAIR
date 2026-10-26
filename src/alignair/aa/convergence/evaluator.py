"""Unified checkpoint evaluator over a frozen, hash-fingerprinted simulator cohort.

``build_cohort`` freezes a strata cohort (clean-pool + AA-native augmentation strata +
native-indel buckets + a J-anchored short stratum)
plus a disjoint calibration slice, all from ONE seeded ``stream_benchmark`` pass so the whole thing
is reproducible from ``(dc_name, n_per_stratum, seed)`` alone. It is fingerprinted with SHA-256 so
every checkpoint scored in a convergence sweep is compared against byte-identical data.

``evaluate_checkpoint`` runs one checkpoint's V/J/D heads over that frozen cohort (via the shared
hardened inference path ``alignair.aa.infer.class_probs``) and reports the metric set used to gate
checkpoint selection: clean-stratum V/J accuracy (micro + macro), D allele/gene accuracy, per-stratum
V accuracy, softmax NLL + Brier, allele-frequency-binned accuracy, boundary MAE (reusing
``evaluate.py::evaluate_frozen``), and a per-checkpoint conformal (APS) coverage/set-size summary.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np

from alignair_benchmark import default_igh_spec, stream_benchmark

from ..augment import aa_indel, aa_substitute, aa_truncate, il_ambiguity
from ..conformal import Conformal
from ..data import build_class_maps, record_to_sample
from ..evaluate import evaluate_frozen
from ..gym_bridge import aa_usable
from ..infer import class_probs
from ..tokenizer import AAProteinTokenizer
from .metrics import brier_score, macro_accuracy, per_frequency_bins, softmax_nll

# Fixed stratum order for cohort fingerprints and checkpoint comparisons.
STRATA_ORDER = ("clean", "moderate_full", "aa_subs_7pct", "aa_single_indel", "aa_il_ambiguity",
                "aa_truncate_50_80", "short_janchor", "dj_indel", "v_indel_1", "v_indel_multi")

_DEFAULT_FREQ_EDGES = (0.0, 0.02, 0.05, 1.0)


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
    - ``fingerprint_sha256``: SHA-256 hex digest over the ordered ``(aa, v_call, j_call, d_call)``
      tuples of every row (strata in ``STRATA_ORDER``, then ``calibration``) — deterministic for a
      fixed ``(dc_name, n_per_stratum, seed)``.
    """
    import GenAIRR as ga

    dc = getattr(ga, dc_name)
    class_maps, _meta = build_class_maps(dc_name)
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
    order = shuffle_rng.permutation(len(clean_pool)) if clean_pool else np.array([], dtype=int)
    clean_pool = [clean_pool[i] for i in order]
    n_cal = max(1, int(round(cal_frac * len(clean_pool)))) if clean_pool else 0
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

    h = hashlib.sha256()
    for name in STRATA_ORDER:
        for row in by_stratum.get(name, []):
            for field in (name, row["aa"], row["v_call"], row["j_call"], row["d_call"]):
                h.update(field.encode()); h.update(b"\0")
    for row in calibration:
        for field in ("__calibration__", row["aa"], row["v_call"], row["j_call"], row["d_call"]):
            h.update(field.encode()); h.update(b"\0")

    cohort = {"by_stratum": by_stratum, "calibration": calibration,
              "dc_name": dc_name, "seed": seed, "n_per_stratum": n_per_stratum}
    return cohort, h.hexdigest()


def _freq_for(rows: list, gene_key: str, allele_class: dict, dc_name: str) -> np.ndarray:
    """Per-row TRUE-allele frequency for freq-binning. Uses ``dc.gene_use_dict`` when the
    dataconfig exposes it; otherwise falls back to a UNIFORM prior over all classifiable alleles
    (``1 / n_alleles``) -- documented here since ``gene_use_dict`` is not guaranteed on every
    GenAIRR dataconfig."""
    import GenAIRR as ga
    dc = getattr(ga, dc_name, None)
    table = getattr(dc, "gene_use_dict", None) if dc is not None else None
    table = (table or {}).get(gene_key)
    names = [_first(r["v_call" if gene_key == "V" else "j_call"]) for r in rows]
    if table:
        return np.array([table.get(n, 0.0) for n in names], dtype=float)
    n_alleles = len(allele_class) or 1
    return np.full(len(names), 1.0 / n_alleles, dtype=float)


def _boundary_mae(model, cfg, class_maps, clean_rows: list, device: str) -> dict:
    """Boundary MAE (V/J start+end) on the clean stratum, reusing ``evaluate.py::evaluate_frozen``
    end-to-end (the model's soft-argmax boundary heads vs the record's projected AA-space germline
    coordinates via ``data.record_to_sample``). Best-effort/informational: returns nulls with a note
    if no clean-stratum record survives ``record_to_sample``'s stricter filters (productive,
    no-stop-codon, exact translation length)."""
    tok = AAProteinTokenizer()
    samples = []
    for row in clean_rows:
        rec = row.get("record")
        if rec is None:
            continue
        sample, _why = record_to_sample(rec, class_maps, tok, cfg.max_seq_length)
        if sample is not None:
            samples.append(sample)
    if not samples:
        return {"v": None, "j": None,
                "note": "deferred: no clean-stratum record survived record_to_sample filtering"}
    res = evaluate_frozen(model, samples, cfg, class_maps, device=device)
    b = res["boundaries"]
    return {"v": {"start": b["v_start"]["mae"], "end": b["v_end"]["mae"]},
            "j": {"start": b["j_start"]["mae"], "end": b["j_end"]["mae"]},
            "n": len(samples)}


def _conformal_v(model, cfg, device: str, class_maps, calibration_rows: list, clean_rows: list,
                 alpha: float, seed: int = 0) -> dict:
    """Per-checkpoint APS conformal fit (V gene -- the selection-critical head) on the fixed
    calibration slice, evaluated (coverage + set-size stats) on the clean-stratum test rows."""
    if not calibration_rows or not clean_rows:
        return {"coverage": None, "set_mean": None, "set_p95": None,
                "note": "insufficient rows for conformal fit"}
    vclass = class_maps["V"].allele_class
    y_cal = np.array([vclass[_first(r["v_call"])] for r in calibration_rows])
    p_cal = class_probs(model, [r["aa"] for r in calibration_rows], cfg, device, "v_class_logits")
    conf = Conformal(alpha=alpha, seed=seed).fit(p_cal, y_cal)

    y_tst = np.array([vclass[_first(r["v_call"])] for r in clean_rows])
    p_tst = class_probs(model, [r["aa"] for r in clean_rows], cfg, device, "v_class_logits")
    sizes, covered = [], []
    for i in range(len(clean_rows)):
        s = conf.predict_set(p_tst[i])
        sizes.append(len(s))
        covered.append(y_tst[i] in s)
    sizes = np.asarray(sizes, dtype=float)
    return {"gene": "V", "n_cal": len(calibration_rows), "n_test": len(clean_rows),
            "coverage": float(np.mean(covered)),
            "set_mean": float(sizes.mean()), "set_p95": float(np.percentile(sizes, 95))}


def evaluate_checkpoint(model, cfg, class_maps, meta, cohort, *, device: str = "cpu",
                        alpha: float = 0.05, freq_edges=_DEFAULT_FREQ_EDGES) -> dict:
    """Evaluate one checkpoint (``model``/``cfg``/``class_maps``/``meta`` as returned by
    ``strategic_setup``) over a frozen cohort from ``build_cohort``. Pure function of
    (model weights, cohort) -- no mutation of the cohort.
    """
    model.to(device).eval()
    vclass, jclass = class_maps["V"].allele_class, class_maps["J"].allele_class
    d_index = meta["d_index"]
    d_name = {i: n for n, i in d_index.items()}

    by_stratum_rows = cohort["by_stratum"]
    calibration_rows = cohort["calibration"]

    if not by_stratum_rows.get("clean"):
        raise ValueError("evaluate_checkpoint: cohort has no 'clean' stratum rows")

    # ---- per-stratum V accuracy (selection-critical: "clean" feeds clean_v_acc/v_macro below) ----
    by_stratum: dict = {}
    v_probs_cache: dict = {}
    for name, rows in by_stratum_rows.items():
        if not rows:
            continue
        seqs = [r["aa"] for r in rows]
        y_v = np.array([vclass[_first(r["v_call"])] for r in rows])
        pv = class_probs(model, seqs, cfg, device, "v_class_logits")
        pred_v = pv.argmax(-1)
        by_stratum[name] = {"n": len(rows), "v_acc": float((pred_v == y_v).mean())}
        v_probs_cache[name] = {"probs": pv, "y": y_v}

    clean_rows = by_stratum_rows["clean"]
    pv_clean = v_probs_cache["clean"]["probs"]
    y_v_clean = v_probs_cache["clean"]["y"]
    pred_v_clean = pv_clean.argmax(-1)

    y_j_clean = np.array([jclass[_first(r["j_call"])] for r in clean_rows])
    pj_clean = class_probs(model, [r["aa"] for r in clean_rows], cfg, device, "j_class_logits")
    pred_j_clean = pj_clean.argmax(-1)

    clean_v_acc = float((pred_v_clean == y_v_clean).mean())
    clean_j_acc = float((pred_j_clean == y_j_clean).mean())
    v_macro = macro_accuracy(y_v_clean, pred_v_clean)
    j_macro = macro_accuracy(y_j_clean, pred_j_clean)

    # ---- D on clean (allele + gene; gene = allele name before "*") ----
    d_rows = [r for r in clean_rows if not r["d_inverted"] and _first(r["d_call"]) in d_index]
    if d_rows:
        y_d = np.array([d_index[_first(r["d_call"])] for r in d_rows])
        pd_ = class_probs(model, [r["aa"] for r in d_rows], cfg, device, "d_class_logits")
        pred_d = pd_.argmax(-1)
        d_allele_acc = float((pred_d == y_d).mean())
        d_allele_macro = macro_accuracy(y_d, pred_d)
        true_gene = np.array([d_name[i].split("*")[0] for i in y_d])
        pred_gene = np.array([d_name[i].split("*")[0] for i in pred_d])
        d_gene_acc = float((pred_gene == true_gene).mean())
        d_nll, d_brier = softmax_nll(pd_, y_d), brier_score(pd_, y_d)
    else:
        d_allele_acc = d_allele_macro = d_gene_acc = d_nll = d_brier = None

    nll = {"v": softmax_nll(pv_clean, y_v_clean), "j": softmax_nll(pj_clean, y_j_clean), "d": d_nll}
    brier = {"v": brier_score(pv_clean, y_v_clean), "j": brier_score(pj_clean, y_j_clean), "d": d_brier}

    freq_bins = {
        "v": per_frequency_bins(y_v_clean, pred_v_clean,
                                _freq_for(clean_rows, "V", vclass, cohort["dc_name"]), freq_edges),
        "j": per_frequency_bins(y_j_clean, pred_j_clean,
                                _freq_for(clean_rows, "J", jclass, cohort["dc_name"]), freq_edges),
    }

    boundary_mae = _boundary_mae(model, cfg, class_maps, clean_rows, device)
    conformal = _conformal_v(model, cfg, device, class_maps, calibration_rows, clean_rows, alpha)

    return {
        "clean_v_acc": clean_v_acc, "clean_j_acc": clean_j_acc,
        "v_macro": v_macro, "j_macro": j_macro,
        "d_allele_acc": d_allele_acc, "d_allele_macro": d_allele_macro, "d_gene_acc": d_gene_acc,
        "by_stratum": by_stratum,
        "freq_bins": freq_bins,
        "boundary_mae": boundary_mae,
        "nll": nll, "brier": brier,
        "conformal": conformal,
    }
