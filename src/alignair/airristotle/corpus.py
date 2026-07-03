"""Multi-reference training corpus for AIRRistotle v2.

The dynamic-reference design generalizes only if the reference *varies* during training, so instead of
one locus we sample across GenAIRR's whole reference bank — 100+ dataconfigs spanning many species and
both BCR (IGH/IGK/IGL) and TCR (TCRA/B/D/G). Each example uses its own config's reference in the
prompt (never a union), exactly as at inference; balance is uniform over configs. Holding out whole
species/loci gives the definitive novel-reference generalization test.

Owns the multi-reference orchestration; the per-example building blocks live in `data.py`.
"""
from __future__ import annotations

import random

import GenAIRR.data as gdata

from ..reference.reference_set import ReferenceSet
from ..gym.gym import build_experiment
from .data import build_v2_example, make_retrievers


def all_dataconfigs() -> dict:
    """{name: dataconfig} for every V(D)J reference GenAIRR ships (IMGT/OGRDB)."""
    names = [n for n in dir(gdata) if ("IMGT" in n or "OGRDB" in n) and not n.startswith("_")]
    return {n: getattr(gdata, n) for n in names}


def select_configs(species=None, loci=None, exclude=None) -> dict:
    """Filter `all_dataconfigs` by species and/or locus name (case-insensitive), dropping `exclude`
    (a set of config names). e.g. select_configs(species=["HUMAN", "MOUSE"], exclude=heldout)."""
    sp = {s.upper() for s in species} if species else None
    lo = {l.upper() for l in loci} if loci else None
    ex = set(exclude or ())
    out = {}
    for name, dc in all_dataconfigs().items():
        s, l = name.split("_")[0], name.split("_")[1]
        if sp and s not in sp:
            continue
        if lo and l not in lo:
            continue
        if name in ex:
            continue
        out[name] = dc
    return out


class ReferenceCorpus:
    """A balanced multi-reference example stream. Caches each config's ReferenceSet + retriever, and
    (lazily, once per stream) its GenAIRR experiment."""

    def __init__(self, dataconfigs: dict, tokenizer, v_shortlist: int = 16, k: int = 11):
        self.names = list(dataconfigs)
        self.dcs = dataconfigs
        self.tok = tokenizer
        self.v_shortlist = v_shortlist
        self.k = k
        self.n_configs = len(dataconfigs)
        self.sampled: set = set()            # distinct references actually drawn (diversity telemetry)
        self._rs: dict = {}
        self._retr: dict = {}

    def _prepare(self, name):
        if name not in self._rs:
            rs = ReferenceSet.from_dataconfigs(self.dcs[name])
            self._rs[name] = rs
            self._retr[name] = make_retrievers(rs, self.k)
        return self._rs[name], self._retr[name]

    def stream(self, params, n, seed, chunk: int = 64):
        """Yield up to `n` examples, uniformly sampling a config, generating a chunk from it, repeating.
        `allow_curatable` is on because many IMGT references include pseudogene/ORF alleles whose
        anchors fail strict validation; a config that still can't build is dropped with a warning."""
        import warnings
        rng = random.Random(seed)
        exp: dict = {}
        produced = 0
        while produced < n and self.names:
            name = rng.choice(self.names)
            try:
                rs, retr = self._prepare(name)
                if name not in exp:
                    exp[name] = build_experiment(self.dcs[name], params, allow_curatable=True)
                recs = list(exp[name].stream_records(n=chunk, seed=rng.randint(0, 2 ** 30)))
            except Exception as e:                             # drop a config that can't build — visibly
                warnings.warn(f"AIRRistotle corpus: dropping config {name}: {e}")
                self.names.remove(name); exp.pop(name, None)
                continue
            self.sampled.add(name)
            for rec in recs:
                yield build_v2_example(rec, rs, self.tok, retr, self.v_shortlist, rs.has_d)
                produced += 1
                if produced >= n:
                    break
