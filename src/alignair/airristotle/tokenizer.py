"""Char-level tokenizer for AIRRistotle v2.

A token is a single DNA base or a structure marker. Tiny vocab (~15), long sequences — the
deliberate trade for single-SNP exactness: a 1-base difference is a 1-token difference, so attention
can compare query-base to reference-base directly (BPE/k-mers would hide it). No names, no digits.
"""
from __future__ import annotations

_BASES = list("ACGTN")
# PAD must be id 0 (padding_idx). Structure tokens frame the prompt and the output.
_SPECIALS = ["<PAD>", "<REF>", "<V>", "<D>", "<J>", "<SEP>", "<QUERY>", "<ALIGN>", "<END>", "<NONE>"]


class AIRRTokenizer:
    def __init__(self):
        toks = _SPECIALS + _BASES
        self.vocab = {t: i for i, t in enumerate(toks)}
        self.inv = {i: t for t, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        for s in _SPECIALS:                         # expose PAD, REF, V, D, J, SEP, QUERY, ALIGN, END, NONE
            setattr(self, s.strip("<>"), s)
        self.base_ids = [self.vocab[b] for b in _BASES]

    def id(self, tok: str) -> int:
        return self.vocab[tok]

    def encode(self, tokens: list[str]) -> list[int]:
        """Encode a list of tokens (structure markers and/or single bases) to ids."""
        return [self.vocab[t] for t in tokens]

    def encode_seq(self, dna: str) -> list[int]:
        """Encode a DNA string to base ids (unknown chars -> N)."""
        n = self.vocab["N"]
        return [self.vocab.get(c, n) for c in dna.upper()]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.inv[int(i)] for i in ids]

    def decode_seq(self, ids: list[int]) -> str:
        """Decode ids back to a DNA string, keeping only base tokens."""
        return "".join(t for t in self.decode(ids) if t in _BASES)
