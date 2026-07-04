"""Per-gene convergence telemetry derived from the training forward pass.

The scalar loss hides which gene is failing (V converges fast; D is the canary), so we
track per-gene: selection loss, top-1-in-set accuracy, and the true-vs-best-sibling
margin (the SNP-discrimination health signal).
"""
import torch

from .loss import selection_ce


def batch_metrics(gene_scores, tau: float, margin: float):
    """gene_scores: {gene: [(scores[C], target_mask[C] bool), ...]} (one entry per read).

    Returns {gene: {"loss", "acc", "margin", "n"}}; genes with no true-target reads omitted.
    """
    out = {}
    for g, recs in gene_scores.items():
        losses, correct, margins, n = [], 0, [], 0
        for scores, tgt in recs:
            if not bool(tgt.any()):
                continue
            n += 1
            losses.append(float(selection_ce(scores, tgt, tau, margin)))
            pred = int(scores.argmax())
            correct += int(bool(tgt[pred]))
            true_max = float(scores[tgt].max())
            wrong = scores[~tgt]
            wrong_max = float(wrong.max()) if wrong.numel() else float("-inf")
            margins.append(true_max - wrong_max)
        if n:
            out[g] = {"loss": sum(losses) / n, "acc": correct / n,
                      "margin": sum(margins) / len(margins), "n": n}
    return out
