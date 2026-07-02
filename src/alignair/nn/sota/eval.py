"""Allele-call accuracy + the dynamic-genotype contract eval.

`call_accuracy` scores one collated batch under a given CandidateBank and returns per-gene top-1
accuracy over the present samples. `contract_eval` runs it under three references built from the
SAME held-out reads — canonical, renamed+shuffled, and SNP-perturbed novel — and reports all three;
a model that "aligns to the reference" (rather than memorizing) holds accuracy across the row.
"""
import numpy as np
import torch

from .data import CandidateBank, detector_inputs
from .augment import perturb_reference_snps, rename_and_shuffle
from .query_decoder import GENES


@torch.no_grad()
def call_accuracy(model, bank: CandidateBank, collated: dict, top_k: int | None = None,
                  remap: dict | None = None) -> dict:
    """Per-gene allele-call accuracy over present samples: a call is correct when the true allele
    is the top-scoring candidate. This is order-invariant (unlike strict argmax, which tie-breaks by
    index) and set-aware — under heavy SHM several alleles are genuinely indistinguishable, so any of
    the tied top scorers counts. `remap[G][old]=new` re-points the true index when the candidate
    order changed (rename+shuffle)."""
    model.eval()
    # honest retrieval: no forced positive (we don't know the answer at inference time).
    read_tokens, read_mask, cands, targets = detector_inputs(
        collated, bank, device=bank.tokens[GENES[0]].device, force_positive=False)
    out = model(read_tokens, read_mask, cands, top_k=top_k)
    acc = {}
    for G in GENES:
        present = targets[G]["present"].bool()
        true = collated[f"{G.lower()}_primary_idx"].to(present.device)
        if remap is not None:
            true = remap[G].to(true.device)[true.clamp(min=0)]
        scores = out[G]["allele_scores"]
        top = scores.max(dim=-1).values
        true_score = scores.gather(1, true.clamp(min=0).unsqueeze(1)).squeeze(1)
        correct = true_score >= top - 1e-6                 # true allele is (tied) top scorer
        n = int(present.sum())
        acc[G] = float(correct[present].float().mean()) if n else float("nan")
    return acc


def contract_eval(model, reference_set, collated: dict, top_k: int | None = None,
                  n_snps: int = 3, seed: int = 0) -> dict:
    """{condition: {gene: accuracy}} for canonical / renamed / novel-SNP references.

    Leave top_k=None (score the full reference) for a clean measurement: the model's name/order
    invariance is exact only over the full candidate set, whereas the lossy top-k prefilter can
    reorder near-tied candidates, so `renamed` would differ from `canonical` for a spurious reason."""
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device

    canon = CandidateBank(reference_set).to(device)
    renamed_ref, remap = rename_and_shuffle(reference_set, rng)
    renamed = CandidateBank(renamed_ref).to(device)
    novel_ref = perturb_reference_snps(reference_set, n_snps, rng)
    novel = CandidateBank(novel_ref).to(device)

    return {
        "canonical": call_accuracy(model, canon, collated, top_k),
        "renamed": call_accuracy(model, renamed, collated, top_k, remap=remap),
        f"novel_snp{n_snps}": call_accuracy(model, novel, collated, top_k),
    }
