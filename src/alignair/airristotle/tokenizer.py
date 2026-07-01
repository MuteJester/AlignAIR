"""Char-level DNA tokenizer for AIRRistotle. A 'token' is a single DNA base, a digit, a flag, an
allele-name character, or a named special/field marker. Small vocab (~60), long sequences — the
deliberate tradeoff for exact copy-pointer coordinates over a genotype-in-prompt."""
from __future__ import annotations

_DNA = list("ACGTN")
_DIGITS = list("0123456789")
_FLAGS = list("+-")
_NAMECHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ*")   # allele-name chars (e.g. IGHVF1-G1*01); DNA dedups
_SPECIALS = ["<PAD>", "<GENO>", "<READ>", "<ANNOT>", "<S>", "<EOS>", "<ORI>", "<PROD>",
             "<V>", "<D>", "<J>",
             "<VS>", "<VE>", "<VGS>", "<VGE>",
             "<DS>", "<DE>", "<DGS>", "<DGE>",
             "<JS>", "<JE>", "<JGS>", "<JGE>",
             "<JNS>", "<JNE>"]


class AIRRTokenizer:
    def __init__(self):
        seen, toks = set(), []
        for t in ["<PAD>"] + [s for s in _SPECIALS if s != "<PAD>"] + _NAMECHARS + _DNA + _DIGITS + _FLAGS:
            if t not in seen:
                seen.add(t); toks.append(t)
        self.vocab = {t: i for i, t in enumerate(toks)}
        self.inv = {i: t for t, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        for s in _SPECIALS:                        # expose PAD, GENO, ..., VS, JNE as attributes
            setattr(self, s.strip("<>"), s)

    def id(self, tok: str) -> int:
        return self.vocab[tok]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.vocab[t] for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.inv[int(i)] for i in ids]
