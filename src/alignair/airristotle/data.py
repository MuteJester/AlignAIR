"""Gym-backed example stream for AIRRistotle v2.

Each GenAIRR record -> a (prompt, target) example: the query read, the coarse-filtered V shortlist
(+ all D/J) as the prompt reference, and the true allele sequence(s) as the copy target. Genuinely
ambiguous reads (GenAIRR comma-lists several equally-valid alleles) list all of them in the target,
in prompt order. The true alleles are always force-included in the shortlist so they are copyable.
"""
from __future__ import annotations

import torch

from .prompt import build_example
from .retriever import KmerFilter


def make_retrievers(reference_set, k: int = 11) -> dict:
    """Fit the coarse filter on each gene's germlines (only V is actually shortlisted)."""
    return {"V": KmerFilter(k).fit(list(reference_set.gene("V").sequences))}


def _true_indices(rec, reference_set, G):
    names = [n for n in str(rec.get(f"{G.lower()}_call", "")).split(",") if n]
    idx, seen = [], set()
    for n in names:
        i = reference_set.gene(G).index.get(n)
        if i is not None and i not in seen:
            seen.add(i); idx.append(i)
    return idx


def _in_prompt_order(seqs_by_idx, ref_seqs):
    """Dedup + order true sequences by first appearance in the prompt reference (canonical)."""
    pos = {s: i for i, s in enumerate(ref_seqs)}
    uniq = list(dict.fromkeys(seqs_by_idx))
    return sorted(uniq, key=lambda s: pos.get(s, len(ref_seqs)))


def build_v2_example(rec, reference_set, tokenizer, retrievers, v_shortlist: int, has_d: bool) -> dict:
    query = str(rec["sequence"]).upper()
    genes = ["V", "D", "J"] if has_d else ["V", "J"]
    true_idx = {G: _true_indices(rec, reference_set, G) for G in genes}

    prim_v = true_idx["V"][0] if true_idx["V"] else None
    sl = retrievers["V"].shortlist(query, v_shortlist, force_include=prim_v)
    for t in true_idx["V"]:                                 # keep every true V copyable
        if t not in sl:
            sl.append(t)

    ref = {"V": [reference_set.gene("V").sequences[i] for i in sl]}
    if has_d:
        ref["D"] = list(reference_set.gene("D").sequences)
    ref["J"] = list(reference_set.gene("J").sequences)

    true = {}
    for G in genes:
        seqs = [reference_set.gene(G).sequences[i] for i in true_idx[G]]
        true[G] = _in_prompt_order(seqs, ref[G])

    ids, mask, plen = build_example(query, ref, true, tokenizer, has_d)
    return {"input_ids": ids, "loss_mask": mask, "prompt_len": plen}


def stream_examples(reference_set, tokenizer, params, n, seed, v_shortlist=16, max_len=None):
    from ..gym.gym import build_experiment
    import GenAIRR.data as gdata
    has_d = reference_set.has_d
    retrievers = make_retrievers(reference_set)
    exp = build_experiment(gdata.HUMAN_IGH_OGRDB, params)
    for rec in exp.stream_records(n=n, seed=seed):
        ex = build_v2_example(rec, reference_set, tokenizer, retrievers, v_shortlist, has_d)
        if max_len is not None and len(ex["input_ids"]) > max_len:
            continue                                        # skip the rare over-long prompt
        yield ex


def collate(examples, pad_id):
    L = max(len(e["input_ids"]) for e in examples)

    def T(key, val):
        return torch.tensor([e[key] + [val] * (L - len(e[key])) for e in examples], dtype=torch.long)

    return {"input_ids": T("input_ids", pad_id), "loss_mask": T("loss_mask", 0),
            "prompt_len": torch.tensor([e["prompt_len"] for e in examples], dtype=torch.long)}
