"""Reference perturbations for the dynamic-genotype contract.

The detector scores alleles by sequence similarity alone — it has no name or index channel — so
the LLM's failure mode (memorizing read->canonical-name) is structurally impossible. What must be
tested (and can be trained against) is sequence-level generalization:

  - `rename_and_shuffle` : new names + a shuffled candidate order. Since the model ignores both,
    call accuracy must be unchanged. Returns a per-gene old->new index remap so targets follow.
  - `perturb_reference_snps` : replace each germline with a SNP-mutated "novel" allele. Reads were
    simulated from the originals, so the reference now differs by a few SNPs — the model must still
    rank the corresponding allele first (align to whatever reference it is handed). Order preserved.
"""
import numpy as np
import torch

from ...reference.reference_set import ReferenceSet
from .query_decoder import GENES

_BASES = "ACGT"


def perturb_reference_snps(reference_set, n_snps: int, rng: np.random.Generator, genes=GENES):
    """New ReferenceSet with each allele's germline mutated at `n_snps` random positions."""
    out = {}
    for G in genes:
        gref = reference_set.gene(G)
        m = {}
        for name, seq in zip(gref.names, gref.sequences):
            s = list(seq)
            if n_snps > 0 and s:
                for p in rng.choice(len(s), size=min(n_snps, len(s)), replace=False):
                    alt = [b for b in _BASES if b != s[p]]
                    s[p] = alt[int(rng.integers(len(alt)))]
            m[name] = "".join(s)
        out[G] = m
    return ReferenceSet.from_genotype(out)


def rename_and_shuffle(reference_set, rng: np.random.Generator, genes=GENES):
    """New ReferenceSet with renamed alleles in a shuffled order. Returns (ref, remap) where
    remap[G][old_index] = new_index, so multi-hot / primary-index targets can be re-pointed."""
    out, remap = {}, {}
    for G in genes:
        gref = reference_set.gene(G)
        K = len(gref.names)
        perm = rng.permutation(K)                            # new position i holds old allele perm[i]
        out[G] = {f"AL_{G}_{i}": gref.sequences[int(perm[i])] for i in range(K)}
        inv = np.empty(K, dtype=np.int64)
        inv[perm] = np.arange(K)                             # old index -> new position
        remap[G] = torch.from_numpy(inv)
    return ReferenceSet.from_genotype(out), remap
