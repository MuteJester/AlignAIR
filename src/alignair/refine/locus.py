"""One locus per read: with a reference of several loci, a read's V, D and J calls come from one locus (the owner's
decision 12: J and D restricted to the called V's locus).

A read's locus is the locus of its most probable same-locus V-J pair under the scans' posteriors (``read_loci``):
``max_{v in L} log p(v) + max_{j in L} log p(j)``, over the loci whose bank holds a V and a J allele. The V and J
evidence decide together, so a read whose V is short or heavily mutated takes its locus from its J as much as from
its V. The read's V, D and J log posteriors are then restricted to that locus and renormalized
(``restrict_to_locus``), before any step that reads them (the boundary refinements, the local D, the V re-score), so
each of those works on the read's own locus's germlines; a read of a locus without D is called D-absent. The chip runs
it at inference only, for a bank of two loci or more (``BankCompiler.locus_count``); a one-locus bank never runs it.
"""
from __future__ import annotations

import torch

from ..core.d_presence import ABSENT, D_PRESENCE_LOGITS
from ..network.scan import OUT

GENES = ("v", "d", "j")
NOT_ABSENT = -1e4             # a read of a locus without D: every D-presence class but ABSENT this far behind


def read_loci(out: dict, bank) -> torch.Tensor:
    """(B,) each read's locus index: the locus maximizing its best V's and its best J's log posterior in that locus
    (``bank.locus_alleles_v`` and ``_j``, (L, A) bool). A locus without a V or a J allele scores -inf and is never a
    read's locus while another locus holds both (a bound bank of two loci or more always has one that does); a read
    for which every locus scores -inf gets locus 0."""
    score = 0
    for gene in ("v", "j"):
        log_posterior = out[f"{gene}_log_posterior"]                                    # (B, A)
        allowed = getattr(bank, f"locus_alleles_{gene}")                                # (L, A)
        score = score + log_posterior[:, None, :].masked_fill(~allowed[None], float("-inf")).amax(-1)
    return score.argmax(-1)


def restrict_to_locus(out: dict, loci: torch.Tensor, bank) -> None:
    """Restrict each read's V, D and J log posteriors in ``out`` to its locus ``loci`` (B,), renormalized, and set
    each ``{gene}_allele`` from them (the read's best allowed allele scores 1); keep the D posterior of a read whose
    locus has no D (``bank.locus_has_d``) and set its D-presence logits to ABSENT; set ``out["read_locus"]``."""
    has_d = bank.locus_has_d[loci]                                                      # (B,)
    for gene in GENES:
        key = f"{gene}_log_posterior"
        if key not in out:
            continue
        allowed = getattr(bank, f"locus_alleles_{gene}")[loci]                          # (B, A)
        restricted = torch.log_softmax(out[key].masked_fill(~allowed, OUT), -1)
        if gene == "d":                                     # a read of a locus without D keeps its D posterior
            restricted = torch.where(has_d[:, None], restricted, out[key])
        out[key] = restricted
        out[f"{gene}_allele"] = torch.exp(restricted - restricted.amax(-1, keepdim=True))
    if D_PRESENCE_LOGITS in out:
        logits = out[D_PRESENCE_LOGITS]
        absent = torch.full_like(logits, NOT_ABSENT)
        absent[:, ABSENT] = 0.
        out[D_PRESENCE_LOGITS] = torch.where(has_d[:, None], logits, absent)
    out["read_locus"] = loci
