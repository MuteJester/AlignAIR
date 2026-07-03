"""Build the AIRRistotle v2 prompt + target from a query and a (shortlisted) reference.

Prompt  : <REF> <V> v1 <SEP> v2 … <D> d1 <SEP> … <J> j1 … <QUERY> «read» <ALIGN>
Target  : <V> «true V» [<SEP> «true V2» …] <D> «true D» (or <NONE>) <J> «true J» <END>

input_ids = prompt ++ target; loss_mask is 1 on the target span only (SFT-style). The target
sequences are copied verbatim from the reference, so every one is guaranteed to appear in the prompt
(constrained decoding later enforces this at generation time). Genes are laid out V, D, J; the D
section is omitted entirely for light chains.
"""
from __future__ import annotations

GENES = ("V", "D", "J")


def _join(seqs, tok) -> list[int]:
    """Encode a list of germline sequences, <SEP>-separated (no trailing separator)."""
    out: list[int] = []
    for i, s in enumerate(seqs):
        if i:
            out.append(tok.id(tok.SEP))
        out += tok.encode_seq(s)
    return out


def _gene_tokens(tok):
    return {"V": tok.id(tok.V), "D": tok.id(tok.D), "J": tok.id(tok.J)}


def build_prompt(query: str, ref: dict, tok, has_d: bool = True) -> list[int]:
    """ref[G] = list of germline seqs to place in the prompt (shortlisted V + all D/J). -> prompt ids."""
    genes = [g for g in GENES if g != "D" or has_d]
    gid = _gene_tokens(tok)
    ids = [tok.id(tok.REF)]
    for g in genes:
        ids.append(gid[g])
        ids += _join(ref[g], tok)
    ids.append(tok.id(tok.QUERY))
    ids += tok.encode_seq(query)
    ids.append(tok.id(tok.ALIGN))
    return ids


def build_target(true: dict, tok, has_d: bool = True) -> list[int]:
    """true[G] = list of true allele seqs (empty -> <NONE>). -> target ids ending in <END>."""
    genes = [g for g in GENES if g != "D" or has_d]
    gid = _gene_tokens(tok)
    ids: list[int] = []
    for g in genes:
        ids.append(gid[g])
        ids += _join(true[g], tok) if true.get(g) else [tok.id(tok.NONE)]
    ids.append(tok.id(tok.END))
    return ids


def build_example(query: str, ref: dict, true: dict, tok, has_d: bool = True):
    """-> (input_ids, loss_mask, prompt_len). loss_mask is 1 on the target span only."""
    prompt = build_prompt(query, ref, tok, has_d)
    target = build_target(true, tok, has_d)
    input_ids = prompt + target
    loss_mask = [0] * len(prompt) + [1] * len(target)
    return input_ids, loss_mask, len(prompt)
