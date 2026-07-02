"""Bridge the GenAIRR gym + ReferenceSet into OpenVocabVDJDetector inputs and targets.

The gym's `gym_collate` already emits everything the detector needs; this module reshapes it:
  - `CandidateBank` tokenizes each allele's germline sequence once per reference (the open
    vocabulary). Token ids are fixed; the detector re-embeds them each step (weights change).
  - `detector_inputs` turns one collated batch into (read tokens/mask, per-gene candidates,
    per-gene targets). Spans are normalized by per-sample read length; germline trims by the
    primary allele's germline length. The primary-allele index is passed as retrieval
    `force_include`, so the contrastive positive is always in the top-k shortlist.

Presence: a gene is "present" iff its multi-hot target has a positive (collate zeroes the
multi-hot for absent genes and for inverted-D), so span/trim/allele are supervised only there.
"""
import torch

from ...data.tokenizer import pad_tokenize
from .query_decoder import GENES


class CandidateBank:
    """Fixed tokenized germline reference (per gene). Build once per ReferenceSet."""

    def __init__(self, reference_set, genes=GENES):
        self.genes = tuple(genes)
        self.tokens, self.mask, self.lengths, self.sizes = {}, {}, {}, {}
        for G in self.genes:
            gref = reference_set.gene(G)
            tok, msk = pad_tokenize(gref.sequences)
            self.tokens[G] = tok
            self.mask[G] = msk
            self.lengths[G] = msk.sum(-1).clamp(min=1)      # germline length per allele (Kg,)
            self.sizes[G] = len(gref.names)

    def to(self, device):
        for G in self.genes:
            self.tokens[G] = self.tokens[G].to(device)
            self.mask[G] = self.mask[G].to(device)
            self.lengths[G] = self.lengths[G].to(device)
        return self


def detector_inputs(collated: dict, bank: CandidateBank, device=None, force_positive: bool = True):
    """collated: one `gym_collate` output. -> (read_tokens, read_mask, candidates, targets).

    force_positive: at TRAIN time, pass the true (primary) allele as retrieval `force_include` so the
    contrastive positive is always in the top-k shortlist. At EVAL/inference time set this False —
    you don't know the answer, so retrieval must find the allele on its own (forcing it in leaks)."""
    device = device or collated["tokens"].device
    read_tokens = collated["tokens"].to(device)
    read_mask = collated["mask"].to(device)
    L = read_mask.sum(-1).clamp(min=1).float()              # per-sample read length (B,)

    candidates, targets = {}, {}
    for G in bank.genes:
        g = G.lower()
        allele = collated[f"{g}_allele"].to(device).float()          # (B, Kg) multi-hot
        present = (allele.sum(-1) > 0).float()
        primary = collated[f"{g}_primary_idx"].to(device)            # (B,) long, -100 = absent

        s = collated[f"{g}_start"].to(device).float()
        e = collated[f"{g}_end"].to(device).float()
        span = torch.stack([s / L, e / L], dim=-1).clamp(0, 1)

        gl = bank.lengths[G][primary.clamp(min=0)].float()           # germline length of primary
        gs = collated[f"{g}_germline_start"].to(device).float()
        ge = collated[f"{g}_germline_end"].to(device).float()
        trim = torch.stack([gs / gl, (gl - ge) / gl], dim=-1).clamp(0, 1)   # (5' , 3') trims

        candidates[G] = {"tokens": bank.tokens[G], "mask": bank.mask[G]}
        if force_positive:
            candidates[G]["force_include"] = primary
        targets[G] = {"span": span, "present": present, "allele": allele, "trim": trim}
    return read_tokens, read_mask, candidates, targets
