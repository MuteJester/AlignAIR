"""Stage-3 Task 3.1 — AA tokenizer (23 residues + CLS) with a HARD no-truncation contract.

Silent truncation is unsafe for an applicability guard: truncating a full heavy chain or a VH+constant input
down to `max_len` could turn it into an apparently-valid VH and let it be ACCEPTED. So:

  * A raw-input length prefilter runs BEFORE tokenization and WITHHOLDS as ``insufficient_evidence`` when the
    input exceeds the supported length contract, preserving the raw length + rejection reason in provenance.
  * ``AaTokenizer`` pads ONLY — it never truncates. Feeding it an over-length sequence raises ``InputTooLong``
    (a backstop), never silently drops residues.

No biological residue is ever silently discarded.
"""
from __future__ import annotations

from alignair.errors import InputError

from dataclasses import dataclass

import torch

from alignair.aa.guard.c0_rules import GuardVerdict

MAX_LEN = 192                                  # supported residue-length contract (excludes the CLS slot)
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYXBZ"           # 23 residue tokens (20 standard + X + B + Z); unknown -> X
CLS_ID = 0
PAD_ID = 0                                     # pad reuses index 0; pad positions are MASKED, so it is inert
_AA2ID = {a: i + 1 for i, a in enumerate(AA_VOCAB)}   # residues -> 1..23
_X_ID = _AA2ID["X"]
VOCAB_SIZE = 24                                # 23 residues + CLS


class InputTooLong(InputError):
    """Raised by AaTokenizer when asked to encode a sequence longer than max_len — truncation is forbidden."""
    code = "ALIGNAIR_INPUT_TOO_LONG"


class AaTokenizer:
    def __init__(self, max_len: int = MAX_LEN):
        self.max_len = max_len
        self.vocab_size = VOCAB_SIZE

    def encode(self, seq: str):
        """(ids[max_len+1], mask[max_len+1]) with a leading [CLS]. Pads only; raises on over-length."""
        seq = seq.strip().upper()
        if len(seq) > self.max_len:
            raise InputTooLong(f"len={len(seq)} > max_len={self.max_len}; truncation is forbidden — "
                               f"reject as insufficient_evidence before tokenizing")
        ids = [CLS_ID] + [_AA2ID.get(a, _X_ID) for a in seq]
        mask = [True] * len(ids)
        pad = (self.max_len + 1) - len(ids)         # pad ONLY, never truncate
        ids += [PAD_ID] * pad
        mask += [False] * pad
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


@dataclass(frozen=True)
class InputContract:
    ok: bool
    raw_length: int
    reason: str
    verdict: GuardVerdict | None       # a withholding verdict when ok is False, else None


def enforce_length_contract(seq: str, max_len: int = MAX_LEN) -> InputContract:
    """Raw-input prefilter (run BEFORE tokenization): withhold as insufficient_evidence when the input exceeds
    the length contract, preserving raw length + reason. Never truncates, never discards residues."""
    n = len(seq.strip())
    if n > max_len:
        reason = f"input_exceeds_length_contract:raw_len={n}>max_len={max_len}"
        return InputContract(ok=False, raw_length=n, reason=reason,
                             verdict=GuardVerdict("insufficient_evidence", reason, float(n)))
    return InputContract(ok=True, raw_length=n, reason="within_length_contract", verdict=None)
