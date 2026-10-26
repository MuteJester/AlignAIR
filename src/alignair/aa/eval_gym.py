"""Evaluate the AA model and, crucially, whether allele-call AMBIGUITY EMERGED on its own.

The decisive comparison is nt-allele accuracy vs protein-class accuracy: if the model resolves the
protein identity (class_top1 high, ~ the DNA model) but its nt-allele errors are almost all
WITHIN-class synonymous confusions — with the true allele still sitting in its high-probability set
— then the sigmoid heads learned to spread across synonymous-indistinguishable alleles, exactly the
hypothesised behaviour, with no equivalence-class targets and no reverse-translation layer."""
from __future__ import annotations

import torch

from .gym_bridge import build_batch_aa
from .tokenizer import AAProteinTokenizer


def gene_allele_counts(probs, target, names, class_of, threshold: float = 0.5) -> dict:
    """Accumulate per-gene counts over a batch. ``class_of``: {allele_name: class_id} or None (D)."""
    c = {"n": 0, "nt_top1": 0, "class_top1": 0, "errors": 0, "errors_within_class": 0,
         "true_in_predicted_set": 0, "set_size": 0}
    B = probs.shape[0]
    for b in range(B):
        true_idx = set((target[b] > 0.5).nonzero().flatten().tolist())
        if not true_idx:                                   # unsupervised row (e.g. inverted D)
            continue
        c["n"] += 1
        pred = int(probs[b].argmax())
        c["nt_top1"] += int(pred in true_idx)
        predset = set((probs[b] > threshold).nonzero().flatten().tolist())
        c["set_size"] += len(predset)
        c["true_in_predicted_set"] += int(bool(true_idx & predset))
        if class_of is not None:
            pred_c = class_of.get(names[pred])
            true_c = {class_of.get(names[t]) for t in true_idx}
            c["class_top1"] += int(pred_c in true_c)
            if pred not in true_idx:                        # an nt-allele error
                c["errors"] += 1
                c["errors_within_class"] += int(pred_c in true_c)   # ...but synonymous?
    return c


def finalize_gene(counts: dict, has_class: bool) -> dict:
    n = max(1, counts["n"])
    r = {"n": counts["n"], "nt_top1": counts["nt_top1"] / n,
         "true_in_predicted_set": counts["true_in_predicted_set"] / n,
         "mean_pred_set_size": counts["set_size"] / n}
    if has_class:
        r["class_top1"] = counts["class_top1"] / n
        r["nt_error_rate"] = counts["errors"] / n
        r["frac_errors_within_class"] = counts["errors_within_class"] / max(1, counts["errors"])
    return r


def _merge(a: dict, b: dict) -> dict:
    return {k: a[k] + b[k] for k in a}


def evaluate_aa(model, records, reference_set, cfg, class_maps, *, device="cpu",
                threshold: float = 0.5, batch_size: int = 256) -> dict:
    """Whole-set evaluation. ``class_maps``: {"V": {name:class}, "J": {name:class}} (no D)."""
    model.eval()
    genes = ["v", "j"] + (["d"] if cfg.has_d else [])
    tok = AAProteinTokenizer()
    acc = {g: gene_allele_counts(torch.zeros(0, 1), torch.zeros(0, 1), [], None) for g in genes}
    seg_abs = {f"{g}_{b}": 0.0 for g in genes for b in ("start", "end")}
    mut_abs = prod_ok = ntot = 0.0

    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            chunk = records[i:i + batch_size]
            bi, tg = build_batch_aa(chunk, reference_set, cfg, device, tok)
            out = model(bi)
            for g in genes:
                names = reference_set.gene(g.upper()).names
                cof = class_maps.get(g.upper())
                acc[g] = _merge(acc[g], gene_allele_counts(
                    out[f"{g}_allele"].cpu(), tg[f"{g}_allele"].cpu(), names, cof, threshold))
                for bnd in ("start", "end"):
                    seg_abs[f"{g}_{bnd}"] += float(
                        (out[f"{g}_{bnd}"].squeeze(-1) - tg[f"{g}_{bnd}"].squeeze(-1)).abs().sum())
            mut_abs += float((out["mutation_rate"] - tg["mutation_rate"]).abs().sum())
            prod_ok += float(((out["productive"] > 0.5) == (tg["productive"] > 0.5)).float().sum())
            ntot += len(chunk)

    ntot = max(1.0, ntot)
    res = {"n": int(ntot),
           "alleles": {g: finalize_gene(acc[g], has_class=class_maps.get(g.upper()) is not None)
                       for g in genes},
           "segmentation_mae": {k: v / ntot for k, v in seg_abs.items()},
           "mutation_mae": mut_abs / ntot,
           "productive_acc": prod_ok / ntot}
    return res
