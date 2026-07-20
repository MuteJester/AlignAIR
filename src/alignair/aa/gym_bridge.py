"""Bridge the REAL nucleotide training gym to the amino-acid model.

The nucleotide model, loss, and gym stream are reused unchanged. Only two things differ, both
forced by feeding amino acids instead of nucleotides:
1. the input tokens are the AA-tokenized ``sequence_aa`` (not the nt ``sequence``);
2. the segment coordinates are projected into AA residue space (the masked soft-argmax can only
   point inside the AA-length input).

Everything else is identical to the DNA targets — crucially the allele targets are the **full
nucleotide allele multi-hot** (not collapsed protein classes), so synonymous-allele ambiguity
must EMERGE in the sigmoid heads rather than being hand-encoded. mutation/indel/productive targets
are kept as-is (the nucleotide mutation rate is deliberately a hard target from AA input)."""
from __future__ import annotations

import numpy as np
import torch

from .tokenizer import AAProteinTokenizer
from .translation import project_codons

_GENES = ("v", "d", "j")


def aa_usable(rec: dict) -> bool:
    """A gym record is AA-usable iff GenAIRR produced a translation — i.e. the V→J junction
    survived (``sequence_aa`` is only computed when both anchors are present). Junction-trimmed
    amplicons (V-anchored) have no ``sequence_aa`` and are skipped."""
    return rec.get("junction_start") is not None and bool(rec.get("sequence_aa"))


def _segments(rec: dict) -> dict:
    segs = {}
    for g in _GENES:
        s, e = rec.get(f"{g}_sequence_start"), rec.get(f"{g}_sequence_end")
        if s is not None and e is not None:
            segs[g.upper()] = (int(s), int(e))
    return segs


def build_batch_aa(records, reference_set, cfg, device: str = "cpu", tokenizer=None):
    """(gym records, reference) -> (model_input, targets) for the REAL AlignAIR + hierarchical_loss.

    No orientation target is emitted (the AA model has ``use_orientation=False``); the loss skips
    that term. Records must already be ``aa_usable``."""
    tok = tokenizer or AAProteinTokenizer()
    genes = ["v", "j"] + (["d"] if cfg.has_d else [])
    L = cfg.max_seq_length
    B = len(records)

    tokens = torch.zeros(B, L, dtype=torch.long)
    starts = {g: torch.zeros(B, 1) for g in genes}
    ends = {g: torch.zeros(B, 1) for g in genes}
    multihot = {g: torch.zeros(B, len(reference_set.gene(g.upper()))) for g in genes}
    mut = torch.zeros(B, 1)
    ind = torch.zeros(B, 1)
    prod = torch.zeros(B, 1)

    for i, rec in enumerate(records):
        enc = tok.encode(rec["sequence_aa"])
        n = min(len(enc), L)
        tokens[i, :n] = torch.from_numpy(np.asarray(enc[:n], dtype=np.int64))

        frame = int(rec["junction_start"]) % 3
        proj = project_codons(len(rec["sequence"]), frame, _segments(rec))
        d_inverted = bool(rec.get("d_inverted", False))
        for g in genes:
            G = g.upper()
            b = proj.aa_bounds.get(G)
            if b is not None:
                starts[g][i, 0] = b[0]
                ends[g][i, 0] = b[1]
            # full nucleotide allele multi-hot; inverted-D is not supervised (matches the DNA gym)
            if rec.get(f"{g}_call") and not (g == "d" and d_inverted):
                gref = reference_set.gene(G)
                for nm in str(rec[f"{g}_call"]).split(","):
                    j = gref.index.get(nm.strip())
                    if j is not None:
                        multihot[g][i, j] = 1.0
        mut[i, 0] = float(rec["mutation_rate"])
        ind[i, 0] = float(rec.get("n_indels", 0))
        prod[i, 0] = 1.0 if rec.get("productive") else 0.0

    batch_in = {"tokenized_sequence": tokens.to(device)}
    targets: dict = {}
    for g in genes:
        targets[f"{g}_start"] = starts[g].to(device)
        targets[f"{g}_end"] = ends[g].to(device)
        targets[f"{g}_allele"] = multihot[g].to(device)
    targets["mutation_rate"] = mut.to(device)
    targets["indel_count"] = ind.to(device)
    targets["productive"] = prod.to(device)
    return batch_in, targets
