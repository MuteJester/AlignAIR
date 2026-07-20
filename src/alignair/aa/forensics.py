"""AA-nearest-germline oracle: the provable information ceiling for allele identity from protein.

Given only the amino-acid sequence, the best any classifier can do is pick the germline allele whose
translation is most consistent with the observed residues. This oracle lets error analysis separate:
  - information-forced errors (even the oracle misses the true allele -> unrecoverable from AA), from
  - model-deficiency errors (the oracle recovers it but the trained model does not -> fixable).

Alignment is anchored at the conserved Cys landmark, which is present in BOTH the observed protein
(at residue ``junction_start // 3``) and every germline (at its ``anchor_aa``). Anchoring there is
robust to reading-frame offset and 5' V-trimming; a naive prefix (residue-0) anchor is not. Indel
reads frameshift the region up to the anchor and must be handled separately (the oracle is invalid
for them). The oracle is a ceiling, not a proposed aligner (it peeks at the covered window)."""
from __future__ import annotations

import numpy as np

from .tokenizer import AAProteinTokenizer


def germline_id_matrix(reference_set, germlines: dict, gene: str = "V", tokenizer=None):
    """Encode germline proteins into an (A, Lmax) int matrix in REFERENCE allele order (row i lines
    up with the model's allele-head index i). ``germlines``: {name: AAGermline}. Returns
    (names, G, glen, ganchor) where ``ganchor[i]`` is the Cys residue index of allele i."""
    tok = tokenizer or AAProteinTokenizer()
    names = list(reference_set.gene(gene.upper()).names)
    encoded = [tok.encode(germlines[n].aa_seq) if n in germlines else tok.encode("") for n in names]
    ganchor = [germlines[n].anchor_aa if n in germlines else 0 for n in names]
    lmax = max((len(e) for e in encoded), default=1)
    G = np.zeros((len(names), lmax), dtype=np.int64)
    glen = np.zeros(len(names), dtype=np.int64)
    for i, e in enumerate(encoded):
        G[i, :len(e)] = e
        glen[i] = len(e)
    return names, G, glen, np.asarray(ganchor, dtype=np.int64)


def nearest_germline(obs_ids: np.ndarray, obs_cys: int, G: np.ndarray, ganchor: np.ndarray,
                     min_overlap: int = 15):
    """Cys-anchored AA identity of the observed V residues against every germline.

    ``obs_ids`` is the full encoded ``sequence_aa``; ``obs_cys = junction_start // 3`` is the Cys
    residue. For allele a, obs residue ``obs_cys`` aligns to germline residue ``ganchor[a]``, so we
    compare the ``t = min(obs_cys, ganchor[a])`` V residues ending at the anchor. Returns
    (argmax index, identity array); alleles with < ``min_overlap`` comparable residues get -1."""
    obs = np.asarray(obs_ids, dtype=np.int64)
    A = G.shape[0]
    ident = np.full(A, -1.0)
    for a in range(A):
        t = min(int(obs_cys), int(ganchor[a]))
        if t < min_overlap:
            continue
        o = obs[obs_cys - t:obs_cys]
        g = G[a, int(ganchor[a]) - t:int(ganchor[a])]
        ident[a] = float((o == g).mean())
    return int(ident.argmax()), ident
