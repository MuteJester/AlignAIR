"""Decode an AIRRistotle record from a genotype-in-prompt. The record grammar is FIXED, and the copy
placeholders are the field markers themselves, so the whole annotation input sequence is deterministic
given the template — we build it, run ONE forward pass, and read each call/coordinate off the copy
logits at the corresponding marker position (the shift means a field's copy is predicted from its
marker's hidden state). MVP eval only: the genotype is built (true alleles + distractors) from a record
we hold the truth for."""
from __future__ import annotations
import torch

from .prompt import build_prompt_core

_GENES = ["v", "d", "j"]
_COORD = ["S", "E", "GS", "GE"]


@torch.no_grad()
def decode_record(model, tokenizer, record, reference_set, n_distractors=6, rng=None, device=None):
    tok = tokenizer
    device = device or next(model.parameters()).device
    ids, meta = build_prompt_core(record, reference_set, tok, n_distractors=n_distractors, rng=rng)
    seq = list(ids)
    reads = []                                        # (kind, gene_or_field, marker_position)

    def marker(m):                                    # append a marker; return its position
        p = len(seq); seq.append(tok.id(m)); return p

    def placeholder(m):                               # append the copy/gen placeholder token
        seq.append(tok.id(m))

    marker(tok.ANNOT)
    p = marker(tok.ORI); reads.append(("gen_ori", None, p)); placeholder("+")
    for g in _GENES:
        if g not in meta.call_marker_pos:
            continue
        G = g.upper()
        p = marker(getattr(tok, G)); reads.append(("call", g, p)); placeholder(getattr(tok, G))
        for c in _COORD:
            fm = getattr(tok, G + c)
            p = marker(fm); reads.append((("germ" if c.startswith("G") else "read"), g + "_" + c, p))
            placeholder(fm)
    p = marker(tok.JNS); reads.append(("read", "jn_S", p)); placeholder(tok.JNS)
    p = marker(tok.JNE); reads.append(("read", "jn_E", p)); placeholder(tok.JNE)
    p = marker(tok.PROD); reads.append(("gen_prod", None, p)); placeholder("0")

    x = torch.tensor([seq], device=device)
    hid, lm = model(x)
    cp = model.copy_logits(hid, meta.prompt_len)      # (1, L, prompt_len)
    lm, cp = lm[0], cp[0]

    def blocks_of(g):
        return [b for b in meta.blocks if b.gene == g]

    def resolve_call(pos, g):
        bs = blocks_of(g)
        return min(bs, key=lambda b: abs(b.marker_pos - pos)).name if bs else None

    def resolve_germ(pos, g):                          # offset within the containing allele block
        for b in blocks_of(g):
            if b.seq_start <= pos <= b.seq_start + b.seq_len:
                return pos - b.seq_start
        b = min(blocks_of(g), key=lambda b: abs(b.seq_start - pos))
        return max(0, pos - b.seq_start)

    def read_off(pos):
        return max(0, min(meta.read_len, pos - meta.read_block_start))

    out = {"sequence": str(record["sequence"]).upper()}
    for kind, name, pos in reads:
        if kind == "gen_ori":
            out["orientation_id"] = 0 if tok.decode([int(lm[pos].argmax())])[0] == "+" else 1
        elif kind == "gen_prod":
            out["productive"] = tok.decode([int(lm[pos].argmax())])[0] == "1"
        elif kind == "call":
            out[f"{name}_call"] = resolve_call(int(cp[pos].argmax()), name)
        elif kind == "read":
            ppos = int(cp[pos].argmax())
            if name == "jn_S":
                out["junction_start"] = read_off(ppos)
            elif name == "jn_E":
                out["junction_end"] = read_off(ppos)
            else:
                g, c = name.split("_")
                out[f"{g}_sequence_{'start' if c == 'S' else 'end'}"] = read_off(ppos)
        elif kind == "germ":
            g, c = name.split("_")
            out[f"{g}_germline_{'start' if c == 'GS' else 'end'}"] = resolve_germ(int(cp[pos].argmax()), g)
    return out
