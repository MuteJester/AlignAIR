"""End-to-end AIRRistotle v2 inference: query + reference -> called germline sequence(s) per gene.

Coarse-filter the reference -> build the char-level prompt -> grammar-constrained decode (guaranteed
exact copies) -> parse. The output is DNA sequence(s); `called_names` maps them back to reference
entries for evaluation. Downstream, a deterministic pairwise alignment of each called allele to the
read yields coordinates / trims / the AIRR record (not implemented here).
"""
from __future__ import annotations

from .prompt import build_prompt
from .decode import constrained_decode, parse_output
from .data import make_retrievers


def align(model, query: str, reference_set, tok, retrievers=None, v_shortlist: int = 16,
          has_d: bool = True) -> dict:
    """-> {gene: [called germline DNA seq(s)]}. Every returned seq is an exact reference sequence."""
    query = query.upper()
    retrievers = retrievers or make_retrievers(reference_set)
    sl = retrievers["V"].shortlist(query, v_shortlist)
    ref = {"V": [reference_set.gene("V").sequences[i] for i in sl]}
    if has_d:
        ref["D"] = list(reference_set.gene("D").sequences)
    ref["J"] = list(reference_set.gene("J").sequences)
    prompt = build_prompt(query, ref, tok, has_d)
    out = constrained_decode(model, prompt, ref, tok, has_d)
    return parse_output(out, tok)


def called_names(called: dict, reference_set) -> dict:
    """Map each called sequence back to ALL reference allele names sharing it. Immunogenetics DBs
    assign several names to one identical sequence (common for D genes), so one emitted sequence
    legitimately recovers the full ambiguous name set."""
    out = {}
    for G, seqs in called.items():
        gref = reference_set.gene(G)
        seq2names: dict = {}
        for n, s in zip(gref.names, gref.sequences):
            seq2names.setdefault(s, []).append(n)
        out[G] = [n for s in seqs for n in seq2names.get(s, [])]
    return out
