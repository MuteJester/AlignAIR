"""Grammar-constrained decoding for AIRRistotle v2.

Guarantees the output is a valid copy of reference alleles present in the prompt — hallucination is
structurally impossible. The output grammar is:

    <V> ( <NONE> | seq (<SEP> seq)* )  <D> ( … )  <J> ( … )  <END>

where each `seq` must be a full germline allele of that gene (a path in a per-gene trie built from the
prompt's reference sequences). At every step only grammar-valid tokens are allowed; the model chooses
among them. So the model only ever *decides* where alleles diverge — everything else is forced.

Uses a KV cache: the prompt is prefilled once, then each generated token is a single incremental
forward (positioned after the cache), so decoding is O(output) forwards over one token rather than
O(output) forwards over the whole growing sequence.
"""
from __future__ import annotations

import torch

from .prompt import GENES


class _Trie:
    __slots__ = ("children", "is_end")

    def __init__(self):
        self.children: dict[int, "_Trie"] = {}
        self.is_end = False

    def add(self, ids: list[int]):
        node = self
        for i in ids:
            node = node.children.setdefault(i, _Trie())
        node.is_end = True


def _gene_tries(ref: dict, tok, genes) -> dict:
    tries = {}
    for g in genes:
        t = _Trie()
        for s in ref.get(g, []):
            t.add(tok.encode_seq(s))
        tries[g] = t
    return tries


@torch.no_grad()
def constrained_decode(model, prompt_ids: list[int], ref: dict, tok, has_d: bool = True,
                       device=None, max_new: int = 4000) -> list[int]:
    """Greedy grammar-constrained generation. Returns the generated output ids (after the prompt),
    ending in <END>."""
    device = device or next(model.parameters()).device
    model.eval()
    genes = [g for g in GENES if g != "D" or has_d]
    marker = {g: tok.id(getattr(tok, g)) for g in genes}
    SEP, END, NONE = tok.id(tok.SEP), tok.id(tok.END), tok.id(tok.NONE)
    tries = _gene_tries(ref, tok, genes)
    end_of = lambda gi: (marker[genes[gi + 1]] if gi + 1 < len(genes) else END)  # noqa: E731

    import contextlib
    use_amp = str(getattr(device, "type", device)).startswith("cuda")

    def fwd(ids, past):
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
        with ctx:
            return model(ids, past=past, return_past=True)

    out: list[int] = []
    logits, past = fwd(torch.tensor(prompt_ids, device=device)[None], None)   # prefill (KV cache)
    last = logits[0, -1]

    def step(allowed: list[int]) -> int:
        nonlocal past, last
        mask = torch.full_like(last, float("-inf"))
        mask[torch.tensor(sorted(set(allowed)), device=device)] = 0.0
        chosen = int((last + mask).argmax())
        out.append(chosen)
        logits, past = fwd(torch.tensor([[chosen]], device=device), past)     # one cached step
        last = logits[0, -1]
        return chosen

    step([marker[genes[0]]])                                   # forced first gene marker
    gi, node, fresh = 0, tries[genes[0]], True
    for _ in range(max_new):
        allowed = list(node.children)
        if fresh:
            allowed.append(NONE)
        if node.is_end:
            allowed += [SEP, end_of(gi)]
        c = step(allowed)
        if c in node.children:
            node, fresh = node.children[c], False
        elif c == SEP:
            node, fresh = tries[genes[gi]], True
        elif c == NONE:
            nxt = end_of(gi)
            step([nxt])
            if nxt == END:
                break
            gi += 1; node, fresh = tries[genes[gi]], True
        elif c == END:
            break
        else:                                                  # c was the next gene marker
            gi += 1; node, fresh = tries[genes[gi]], True
    return out


def parse_output(out_ids: list[int], tok) -> dict:
    """Parse generated output ids into {gene: [allele DNA seqs]} ([] if <NONE>)."""
    marker = {tok.id(getattr(tok, g)): g for g in GENES}
    SEP, END, NONE = tok.id(tok.SEP), tok.id(tok.END), tok.id(tok.NONE)
    result: dict[str, list[str]] = {}
    cur, buf = None, []

    def flush():
        if cur is not None and buf:
            result[cur].append(tok.decode_seq(buf))

    for i in out_ids:
        if i in marker:
            flush(); cur = marker[i]; result.setdefault(cur, []); buf = []
        elif i == SEP:
            flush(); buf = []
        elif i == NONE:
            buf = []
        elif i == END:
            flush(); break
        else:
            buf.append(i)
    return result
