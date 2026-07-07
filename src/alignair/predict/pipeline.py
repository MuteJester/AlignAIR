"""The prediction pipeline orchestrator: model + reference + config -> AIRR-contract records.

Reads top-to-bottom as the full flow. Each call is a pure stage from :mod:`alignair.predict`.
Phase A emits the benchmark contract (calls + set + read/germline coords + CIGAR); Phase B will add
the full ``airr/`` assembly (sequence_alignment, junction, regions, quality).
"""
from __future__ import annotations

import numpy as np

from .clean import clean
from .config import PredictConfig
from .forward import run_model
from .genotype import adjust_for_genotype
from .germline import align_germline
from .segment import correct_segments
from .threshold import select_alleles


def _genes(cfg: PredictConfig):
    return ("v", "d", "j") if cfg.has_d else ("v", "j")


def predict(model, sequences, reference, cfg: PredictConfig, device: str = "cpu", aligner=None):
    genes = _genes(cfg)
    preds = clean(run_model(model, sequences, cfg, device), genes)
    if cfg.genotype:
        preds = adjust_for_genotype(preds, cfg.genotype, reference)
    seq_lens = np.array([len(s) for s in sequences])
    segs = correct_segments(preds.start, preds.end, seq_lens, cfg.max_seq_length, cfg.pad_mode)
    names = {g: list(reference.gene(g.upper()).names) for g in genes}
    calls = select_alleles(preds.allele, names, cfg.threshold_pct, cfg.cap)
    alignments = align_germline(sequences, segs, calls, reference, aligner)
    return _to_records(sequences, calls, alignments, genes)


def _to_records(sequences, calls, alignments, genes) -> list:
    records = []
    for i, seq in enumerate(sequences):
        rec = {"sequence": seq, "orientation_id": 0}
        for g in genes:
            call, aln = calls[g][i], alignments[g][i]
            rec[f"{g}_call"] = call.names[0] if call.names else ""
            rec[f"{g}_calls"] = list(call.names)
            rec[f"{g}_likelihoods"] = list(call.likelihoods)
            if aln is not None:
                rec[f"{g}_sequence_start"] = aln.seq_start
                rec[f"{g}_sequence_end"] = aln.seq_end
                rec[f"{g}_germline_start"] = aln.germ_start
                rec[f"{g}_germline_end"] = aln.germ_end
                rec[f"{g}_cigar"] = aln.cigar
        records.append(rec)
    return records
