"""Genotype-constrained inference — restrict the allele classifier to a known allele subset.

Given a genotype ``G`` (a subset of the trained reference's alleles that is known to contain the
true allele), we constrain the model's per-allele beliefs to ``G`` at inference — no retraining.

Methods (see the module tests for the guarantees):
  * ``mask``        — zero every out-of-genotype allele. Since the true allele ``a* in G`` is never
                      removed and every out-of-genotype false positive is, precision can only rise
                      (recall of ``a*`` unchanged). Kept probabilities are otherwise untouched.
                      The safe default: airtight accuracy gain, no confidence distortion.
  * ``softmax``     — mask, recover per-allele logits from the calibrated sigmoids, and softmax over
                      the allowed set: the categorical posterior ``P(a | read, a in G)`` (one true
                      allele among ``G``). Concentrates mass on ``a*`` → SHARPER confidence, and can
                      promote a borderline ``a*`` over the threshold. Use for a confidence boost.
  * ``renormalize`` — mask, then divide surviving probabilities by their sum (each allele's share of
                      the allowed mass). Note: for the multi-label SIGMOID head the allowed probs do
                      NOT sum to ~1, so this can LOWER confidence; ``softmax`` is the calibrated choice.
  * ``redistribute``— the legacy TF bounded redistribution (kept for comparison).
"""
from __future__ import annotations

import numpy as np

from ..predict.state import Predictions

METHODS = ("mask", "softmax", "renormalize", "redistribute")


class NovelAlleleUnsupportedError(ValueError):
    """A genotype (or supplied reference) names an allele the model was not trained on. A fixed-head
    model's classification indices are tied to its trained allele catalog, so it cannot call a novel
    allele at inference — retrain/fine-tune to add alleles. Subclasses ``ValueError`` so existing
    ``except ValueError`` handlers keep working while the type documents the fixed-reference contract."""


def genotype_allowed_mask(genotype: dict, reference, genes=None) -> dict:
    """Compute ``{gene: bool mask over head indices}`` for the alleles a genotype allows, validating
    that every constrained gene present in both the genotype and the reference retains **at least one**
    supported allele. Raises ``ValueError`` if a gene's allowed set is empty (e.g. an all-novel-allele
    file) — this must fail before inference rather than silently calling a disallowed allele."""
    out: dict[str, np.ndarray] = {}
    for gene, allowed in genotype.items():
        g = gene.lower()
        if genes is not None and g not in genes:
            continue
        try:
            names = reference.gene(g.upper()).names
        except (KeyError, AttributeError):               # gene absent from this model's reference
            continue
        keep = np.array([n in allowed for n in names], dtype=bool)
        if not keep.any():
            raise NovelAlleleUnsupportedError(
                f"genotype constraint for gene {g!r} allows no allele in the model's reference "
                f"(a fixed-reference model cannot call alleles it was not trained on). "
                f"Supplied: {sorted(allowed)[:8]}{'...' if len(allowed) > 8 else ''}")
        out[g] = keep
    return out


def adjust_for_genotype(preds: Predictions, genotype: dict, reference, method: str = "renormalize") -> Predictions:
    if method not in METHODS:
        raise ValueError(f"unknown genotype method {method!r}; choose from {METHODS}")
    masks = genotype_allowed_mask(genotype, reference, genes=set(preds.allele))   # validates non-empty
    for gene, allowed in genotype.items():
        if gene.lower() not in preds.allele:
            continue
        gene = gene.lower()
        keep = masks[gene]
        probs = preds.allele[gene].astype(np.float64, copy=True)
        if method == "mask":
            probs[:, ~keep] = 0.0
        elif method == "softmax":
            eps = 1e-6
            pc = np.clip(preds.allele[gene].astype(np.float64), eps, 1.0 - eps)
            logits = np.log(pc) - np.log(1.0 - pc)         # recover logits from calibrated sigmoids
            allowed = np.where(keep[None, :], logits, -np.inf)
            ex = np.where(keep[None, :], np.exp(allowed - allowed.max(axis=1, keepdims=True)), 0.0)
            probs = ex / ex.sum(axis=1, keepdims=True)      # softmax over the allowed set
        elif method == "renormalize":
            probs[:, ~keep] = 0.0
            s = probs[:, keep].sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                probs[:, keep] = np.where(s > 0, probs[:, keep] / s, probs[:, keep])
        else:                                              # redistribute (legacy)
            for row in probs:
                total_geno, total_non = row[keep].sum(), row[~keep].sum()
                if total_geno > 0:
                    row[keep] = np.minimum(1.0, row[keep] + row[keep] * (total_non / total_geno))
                row[~keep] = 0.0
        preds.allele[gene] = probs
    return preds


def load_genotype(path: str, *, reference=None, drop_unknown: bool = True):
    """Load a genotype from JSON or YAML: ``{gene: [allele names]}`` (a subset of the trained
    reference). Returns ``{gene: set(names)}``; if ``reference`` is given, validates the names are in
    it and returns ``(genotype, unknown)`` — dropping (``drop_unknown``) or raising on unknown names."""
    with open(path) as f:
        text = f.read()
    data = _parse(text, path)
    genotype = {str(g).lower(): {str(n) for n in names} for g, names in data.items()}
    if reference is None:
        return genotype
    unknown: dict[str, set] = {}
    for g, names in list(genotype.items()):
        try:
            ref_names = set(reference.gene(g.upper()).names)
        except (KeyError, AttributeError):                 # gene not in this reference -> skip validation
            continue
        bad = {n for n in names if n not in ref_names}
        if bad:
            unknown[g] = bad
            if drop_unknown:
                genotype[g] = names - bad
            else:
                raise NovelAlleleUnsupportedError(
                    f"unknown alleles in genotype for {g}: {sorted(bad)} (not in the model reference; "
                    f"a fixed-reference model cannot call alleles it was not trained on)")
    return genotype, unknown


def _parse(text: str, path: str):
    import json
    if path.endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import yaml
        return yaml.safe_load(text)
