"""Stage 0-1: align a repertoire and aggregate weighted per-allele usage.

Stage 0 runs the predict pipeline while KEEPING the raw per-allele probability vectors (not just the
capped top-k) plus the per-position state logits (when the model has the state head). Stage 1 turns
that into weighted usage per allele, up-weighting germline-informative reads (low SHM, long, confident).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AlignedRepertoire:
    sequences: list
    allele_probs: dict            # {gene: (N, n_alleles) raw sigmoid probs}
    records: list                 # predict() records (calls / coords / cigars) for the polymorphism stage
    mutation_rate: np.ndarray     # (N,)
    read_lengths: np.ndarray      # (N,)
    state_logits: object          # (N, L, 4) or None
    gene_names: dict              # {gene: [allele names], index-aligned to allele_probs columns}
    reference: object


def read_weights(aligned: AlignedRepertoire) -> np.ndarray:
    """Per-read germline-informativeness: low SHM and longer reads weigh more. In [~0, 1]."""
    mut = np.asarray(aligned.mutation_rate, dtype=np.float64).reshape(-1)
    ln = np.asarray(aligned.read_lengths, dtype=np.float64).reshape(-1)
    length_factor = ln / (ln.max() + 1e-9)
    return np.clip((1.0 - mut) * length_factor, 1e-6, None)


def weighted_usage(aligned: AlignedRepertoire, gene: str) -> dict:
    """{allele_name: {mass, count}} — ``mass`` = Σ w_r·p_{r,a} (soft), ``count`` = top-1 hard count."""
    probs = np.asarray(aligned.allele_probs[gene])
    names = aligned.gene_names[gene]
    w = read_weights(aligned)
    top1 = probs.argmax(axis=1)
    return {name: {"mass": float((w * probs[:, i]).sum()), "count": int((top1 == i).sum())}
            for i, name in enumerate(names)}


def align_repertoire(model, reference, sequences, *, device: str = "cpu", batch_size: int = 64) -> AlignedRepertoire:
    """Run Stage 0: raw allele probs + state logits + predict records for the repertoire."""
    from ..api import predict_sequences
    from ..predict.clean import clean
    from ..predict.config import PredictConfig
    from ..predict.forward import run_model
    cfg = PredictConfig(max_seq_length=model.cfg.max_seq_length, has_d=model.cfg.has_d, batch_size=batch_size)
    genes = ("v", "d", "j") if cfg.has_d else ("v", "j")
    preds = clean(run_model(model, list(sequences), cfg, device), genes)      # raw probs + state logits
    records = predict_sequences(model, reference, list(sequences), device=device,
                                batch_size=batch_size, airr=False)            # calls / coords / cigars
    return AlignedRepertoire(
        sequences=list(sequences),
        allele_probs={g: preds.allele[g] for g in genes},
        records=records,
        mutation_rate=preds.mutation_rate,
        read_lengths=np.array([len(s) for s in sequences]),
        state_logits=preds.state_logits,
        gene_names={g: list(reference.gene(g.upper()).names) for g in genes},
        reference=reference,
    )
