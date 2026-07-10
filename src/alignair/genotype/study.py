"""Research study: does constraining the classifier to a known genotype improve it?

Generate labelled reads, take the genotype ``G`` = the distinct true alleles they use (a proper
subset of the trained reference — the "donor genotype" that produced them, so every truth lies in
``G``), then compare UNCONSTRAINED vs genotype-CONSTRAINED prediction on accuracy and confidence.
"""
from __future__ import annotations

from ..api import predict_sequences
from ..evaluate.benchmark import _call_correct, _mean, generate_labeled, default_strata


def _observed_genotype(truth: list[dict]) -> dict:
    genotype: dict[str, set] = {}
    for g in ("v", "d", "j"):
        names: set[str] = set()
        for r in truth:
            for a in str(r.get(f"{g}_call") or "").split(","):
                if a.strip():
                    names.add(a.strip())
        if names:
            genotype[g] = names
    return genotype


def _metrics(truth: list[dict], preds: list[dict]) -> dict:
    m: dict = {}
    for g in ("v", "d", "j"):
        pairs = [(t, p) for t, p in zip(truth, preds) if t.get(f"{g}_call")]
        if not pairs:
            continue
        m[f"{g}_acc"] = _mean([_call_correct(p.get(f"{g}_call"), t.get(f"{g}_call")) for t, p in pairs])
        confs = [p[f"{g}_likelihoods"][0] for _, p in pairs if p.get(f"{g}_likelihoods")]
        m[f"{g}_confidence"] = _mean(confs)                     # confidence assigned to the top-1 call
    return m


def _in(call, allowed) -> bool:
    parts = [a.strip() for a in str(call or "").split(",") if a.strip()]
    return bool(parts) and all(a in allowed for a in parts)


def _sample_to_genotype(dataconfig, reference, stratum, n, seed, v_size):
    """Reject-sample ``n`` reads whose V truth lies in a random V genotype of size ``v_size``
    (a realistic donor carries ~40-60 of the ~200 V alleles); D/J default to the observed set."""
    import random
    rng = random.Random(seed)
    v_names = list(reference.gene("V").names)
    g_v = set(rng.sample(v_names, min(v_size, len(v_names))))
    truth, s = [], seed
    while len(truth) < n and s < seed + 200:
        for r in generate_labeled(dataconfig, default_strata()[stratum], n, s):
            if _in(r.get("v_call"), g_v):
                truth.append(r)
                if len(truth) >= n:
                    break
        s += 1
    genotype = _observed_genotype(truth)
    genotype["v"] = g_v                                        # tight V genotype (truth guaranteed inside)
    return truth[:n], genotype


def genotype_study(model, reference, dataconfig, *, n: int = 200, seed: int = 0,
                   method: str = "renormalize", stratum: str = "moderate", v_genotype_size=None,
                   device: str = "cpu", batch_size: int = 64) -> dict:
    if v_genotype_size:
        truth, genotype = _sample_to_genotype(dataconfig, reference, stratum, n, seed, v_genotype_size)
    else:
        truth = generate_labeled(dataconfig, default_strata()[stratum], n, seed)
        genotype = _observed_genotype(truth)
    seqs = [r["sequence"] for r in truth]

    base = predict_sequences(model, reference, seqs, device=device, batch_size=batch_size, airr=False)
    cons = predict_sequences(model, reference, seqs, device=device, batch_size=batch_size, airr=False,
                             genotype=genotype, genotype_method=method)
    return {
        "n": n, "method": method, "stratum": stratum,
        "genotype_sizes": {g: len(v) for g, v in genotype.items()},
        "reference_sizes": {g: len(reference.gene(g.upper())) for g in genotype},
        "unconstrained": _metrics(truth, base),
        "constrained": _metrics(truth, cons),
    }


def format_study(s: dict) -> str:
    L = [f"Genotype constraint study — n={s['n']}, method={s['method']}, stratum={s['stratum']}", "=" * 62]
    L.append("  genotype vs reference alleles: " +
             ", ".join(f"{g.upper()} {s['genotype_sizes'][g]}/{s['reference_sizes'][g]}"
                       for g in s["genotype_sizes"]))
    L.append(f"  {'metric':16s} {'unconstrained':>14s} {'constrained':>12s} {'delta':>8s}")
    for g in ("v", "d", "j"):
        for k in (f"{g}_acc", f"{g}_confidence"):
            u, c = s["unconstrained"].get(k), s["constrained"].get(k)
            if u is None or c is None:
                continue
            L.append(f"  {k:16s} {u:>14.3f} {c:>12.3f} {c - u:>+8.3f}")
    return "\n".join(L)
