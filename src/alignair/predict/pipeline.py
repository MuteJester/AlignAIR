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
from .germline import align_germline
from .segment import correct_segments
from .threshold import select_alleles


def _genes(cfg: PredictConfig):
    return ("v", "d", "j") if cfg.has_d else ("v", "j")


def _assert_finite(allele: dict, stage: str) -> None:
    """Fail loudly if any post-transform allele probability is non-finite (NaN/Inf) — a silently
    NaN'd probability would otherwise propagate into calls/confidence (see the pre-launch audit)."""
    for g, a in allele.items():
        if not np.all(np.isfinite(a)):
            raise ValueError(f"non-finite allele probabilities after {stage} (gene {g!r})")


def apply_input_policy(sequences, max_len: int) -> tuple[list, list]:
    """The single input-length/content gate for prediction (P0-8): uppercase, reject empty reads, and
    crop over-length reads to the model window **consistently** — the cropped string is what the
    tokenizer, coordinates, germline reader, and AIRR assembly all see, so an over-length read is never
    *silently* truncated with mismatched downstream coordinates. Returns ``(sequences, was_cropped)``."""
    sequences = [str(s).upper() for s in sequences]
    empties = [i for i, s in enumerate(sequences) if not s.strip()]
    if empties:
        shown = empties[:10]
        raise ValueError(f"empty input sequence(s) at index {shown}"
                         f"{'...' if len(empties) > 10 else ''}; every read must be non-empty")
    was_cropped = [len(s) > max_len for s in sequences]
    sequences = [s[:max_len] for s in sequences]
    return sequences, was_cropped


_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _canonicalize(seq: str, orientation_id: int) -> str:
    """Re-orient a read into the model's forward frame using the predicted orientation (the transforms
    are involutions, so re-applying recovers forward): 0=identity, 1=revcomp, 2=complement, 3=reverse.
    The model predicts coordinates in the forward frame, so the germline reader, coords, and AIRR
    assembly must all operate on this canonical sequence."""
    if orientation_id == 1:
        return seq.translate(_COMPLEMENT)[::-1]
    if orientation_id == 2:
        return seq.translate(_COMPLEMENT)
    if orientation_id == 3:
        return seq[::-1]
    return seq


def predict(model, sequences, reference, cfg: PredictConfig, device: str = "cpu", aligner=None):
    genes = _genes(cfg)
    # single input gate (P0-8): uppercase once (GenAIRR/FASTA mark bases with case; the germline reader
    # + AIRR assembly consume the raw string and case-mixed input mis-anchors the junction — DNA
    # alignment is case-insensitive), reject empty reads, and crop over-length reads to the model window
    # CONSISTENTLY so coords/AIRR refer to the same string (never a silent truncation).
    sequences, was_cropped = apply_input_policy(sequences, cfg.max_seq_length)
    preds = clean(run_model(model, sequences, cfg, device), genes)
    allowed = None
    if cfg.allele_temperatures:                    # post-hoc allele-confidence calibration
        from .calibrate import apply_temperature
        for g, T in cfg.allele_temperatures.items():
            if g in preds.allele:
                preds.allele[g] = apply_temperature(preds.allele[g], T)
        _assert_finite(preds.allele, "temperature calibration")
    if cfg.genotype:
        from ..genotype.constraint import adjust_for_genotype, genotype_allowed_mask
        allowed = genotype_allowed_mask(cfg.genotype, reference, genes=set(preds.allele))  # validates
        preds = adjust_for_genotype(preds, cfg.genotype, reference, method=cfg.genotype_method)
        _assert_finite(preds.allele, "genotype constraint")
    # canonicalize each read to the model's forward frame so coords / germline / AIRR all agree
    orient = preds.orientation
    seqs = [_canonicalize(s, int(orient[i]) if orient is not None else 0)
            for i, s in enumerate(sequences)]
    seq_lens = np.array([len(s) for s in seqs])
    segs = correct_segments(preds.start, preds.end, seq_lens, cfg.max_seq_length, cfg.pad_mode)
    names = {g: list(reference.gene(g.upper()).names) for g in genes}
    calls = select_alleles(preds.allele, names, cfg.threshold, cfg.cap, cfg.selector, allowed=allowed)
    alignments = align_germline(seqs, segs, calls, reference, aligner,
                                reader=cfg.germline_reader, indel_counts=preds.indel_count)
    records = _to_records(seqs, calls, alignments, genes, preds, cfg.chain_types, segs.low_quality,
                          was_cropped)
    if cfg.airr:
        from .airr import build_airr
        records = build_airr(records, reference, chain=("heavy" if cfg.has_d else "light"))
    return records


def _to_records(sequences, calls, alignments, genes, preds, chain_types=None, low_quality=None,
                was_cropped=None) -> list:
    orientation = preds.orientation
    records = []
    for i, seq in enumerate(sequences):
        oid = int(orientation[i]) if orientation is not None else 0
        rec = {"sequence": seq, "orientation_id": oid,
               "mutation_rate": float(preds.mutation_rate[i]),
               "indel_count": float(preds.indel_count[i]),
               "productive": bool(preds.productive[i])}
        if low_quality is not None:
            rec["segmentation_low_quality"] = bool(low_quality[i])
        if was_cropped is not None and was_cropped[i]:
            rec["length_cropped"] = True
        if preds.chain_type is not None:                       # multi-chain: predicted locus
            ct = int(preds.chain_type[i])
            rec["chain_type_id"] = ct
            if chain_types is not None and 0 <= ct < len(chain_types):
                rec["locus"] = chain_types[ct]
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
