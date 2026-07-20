"""AA-nearest-germline oracle: the provable information ceiling for allele identity from protein.

Given only the amino-acid sequence, the best any classifier can do is pick the germline allele whose
translation is most consistent with the observed residues. This module computes that oracle so error
analysis can separate:
  - information-forced errors (even the oracle misses the true allele -> unrecoverable from AA), from
  - model-deficiency errors (the oracle recovers it but the trained model does not -> fixable).
The oracle is a ceiling, not a proposed aligner (it peeks at the covered window)."""
from __future__ import annotations

import numpy as np

from .tokenizer import AAProteinTokenizer


def germline_id_matrix(reference_set, aa_by_name: dict, gene: str = "V", tokenizer=None):
    """Encode germline proteins into an (A, Lmax) int matrix in the REFERENCE allele order (so row i
    lines up with the model's allele-head index i). Returns (names, G, glen)."""
    tok = tokenizer or AAProteinTokenizer()
    names = list(reference_set.gene(gene.upper()).names)
    encoded = [tok.encode(aa_by_name.get(n, "")) for n in names]
    lmax = max((len(e) for e in encoded), default=1)
    G = np.zeros((len(names), lmax), dtype=np.int64)      # pad 0 never matches a real residue (1..22)
    glen = np.zeros(len(names), dtype=np.int64)
    for i, e in enumerate(encoded):
        G[i, :len(e)] = e
        glen[i] = len(e)
    return names, G, glen


def nearest_germline(obs_ids: np.ndarray, G: np.ndarray):
    """Prefix-anchored AA identity of the observed V residues against every germline.

    Both the observed V span and the germline proteins start at V residue 0, so a prefix-anchored
    comparison over the observed window is the correct alignment for indel-free reads. Returns
    (argmax index, identity array over A alleles)."""
    obs = np.asarray(obs_ids, dtype=np.int64)
    m = len(obs)
    if m == 0:
        return -1, np.zeros(G.shape[0])
    Gm = G[:, :m]                                          # (A, m); germlines shorter than m pad with 0
    ident = (Gm == obs[None, :]).sum(axis=1) / m
    return int(ident.argmax()), ident
