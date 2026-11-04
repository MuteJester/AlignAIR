"""The prediction pipeline orchestrator: model + reference + config -> AIRR-contract records.

Reads top-to-bottom as the full flow. Each call is a pure stage from :mod:`alignair.predict`.
It emits the calls + equivalence set + read/germline coordinates + CIGAR, and the full ``airr/``
assembly (sequence_alignment, junction, regions, quality).
"""
from __future__ import annotations

from ..core.anchors import ANCHOR_GENES, ANCHOR_KEYS
from ..core.d_presence import ABSENT, CALLABLE, D_CLASSES, D_PRESENCE
from ..errors import ConfigurationError, NumericalError
from ..data.inputs import prepare_nucleotide_inputs

import numpy as np

from .clean import NOT_OBSERVED, clean
from .config import PredictConfig
from .forward import run_model
from .germline import align_germline
from .junction import derive_junction
from .segment import correct_segments
from .threshold import select_alleles
from .ties import tied_alleles
from .v_presence import V_ABSENT, V_PRESENCE, v_absent


def _genes(cfg: PredictConfig):
    return ("v", "d", "j") if cfg.has_d else ("v", "j")


def _assert_finite(allele: dict, stage: str) -> None:
    """Fail loudly if any post-transform allele probability is non-finite (NaN/Inf) — a silently
    NaN'd probability would otherwise propagate into calls/confidence."""
    for g, a in allele.items():
        if not np.all(np.isfinite(a)):
            raise NumericalError(f"non-finite allele probabilities after {stage} (gene {g!r})",
                                 context={"stage": stage, "gene": g, "field": "allele_probabilities"})


def _validate_predictions(preds, stage: str = "model forward") -> None:
    """Fail loudly if ANY raw model output is non-finite: a damaged checkpoint emitting NaN
    would otherwise be silently turned into a plausible call (argmax fallback) and clipped coordinates."""
    _assert_finite(preds.allele, stage)
    for name, d in (("start", preds.start), ("end", preds.end)):
        for g, a in d.items():
            if not np.all(np.isfinite(a)):
                raise NumericalError(f"non-finite {name} coordinates after {stage} (gene {g!r})",
                                     context={"stage": stage, "gene": g, "field": name})
    for name in ("mutation_rate", "indel_count"):
        a = getattr(preds, name)
        if a is not None and not np.all(np.isfinite(a)):
            raise NumericalError(f"non-finite {name} after {stage}", context={"stage": stage, "field": name})


def _locus_allowed(chain_type, reference, genes, genotype_allowed=None) -> dict:
    """Per-read allele masks restricting each read to its predicted locus. Returns
    ``{gene: (N, C) bool}``: row ``i`` is the locus mask for the locus at ``chain_type[i]``. When a
    genotype constraint is also active, the per-read locus mask is AND-ed with the (broadcast) genotype
    mask so both hold. A read's predicted locus with no ``gene`` (e.g. D on a light chain) yields an
    all-False row -> an explicit no-call downstream."""
    loci = reference.locus_names()
    out: dict = {}
    for g in genes:
        rows = []
        for c in chain_type:
            ci = int(c)
            locus = loci[ci] if 0 <= ci < len(loci) else (loci[0] if loci else "")
            rows.append(reference.locus_mask(locus, g))
        mask = np.stack(rows) if rows else np.zeros((0, 0), dtype=bool)
        if genotype_allowed and genotype_allowed.get(g) is not None:
            mask = mask & np.asarray(genotype_allowed[g], dtype=bool)[None, :]
        out[g] = mask
    return out


def apply_input_policy(sequences, max_len: int) -> tuple[list, list]:
    """Compatibility entry point for shared nucleotide prediction preparation."""
    return prepare_nucleotide_inputs(sequences, max_len)


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
    model_config = getattr(model, 'cfg', None)
    if (getattr(model_config, 'input_space', 'nucleotide') != 'nucleotide'
            or getattr(model_config, 'classification_mode', 'multilabel') != 'multilabel'):
        raise ConfigurationError('AIRR prediction requires the nucleotide multilabel allele profile; use Aligner.predict_raw for shared model outputs')
    if getattr(model_config, 'boundary_mode', 'residue') != cfg.boundary_mode:
        raise ConfigurationError('the prediction config and the model disagree on boundary_mode',
                                 context={'field': 'boundary_mode', 'model': getattr(model_config, 'boundary_mode', 'residue'),
                                          'config': cfg.boundary_mode})
    genes = _genes(cfg)
    # Tokenization, orientation, alignment and reported coordinates share the same normalized read.
    sequences, was_cropped = prepare_nucleotide_inputs(sequences, cfg.window)
    preds = clean(run_model(model, sequences, cfg, device), genes)
    _validate_predictions(preds)  # raw outputs must be finite before any use
    allowed = None
    if cfg.allele_temperatures:                    # post-hoc allele-confidence calibration
        from ..calibration import apply_temperature
        for g, T in cfg.allele_temperatures.items():
            if g in preds.allele:
                preds.allele[g] = apply_temperature(preds.allele[g], T)
        _assert_finite(preds.allele, "temperature calibration")
    if cfg.genotype:
        from ..genotype.constraint import adjust_for_genotype, genotype_allowed_mask
        allowed = genotype_allowed_mask(cfg.genotype, reference, genes=set(preds.allele))  # validates
        preds = adjust_for_genotype(preds, cfg.genotype, reference, method=cfg.genotype_method)
        _assert_finite(preds.allele, "genotype constraint")
    # multi-chain locus masking: each read's predicted locus restricts its callable alleles to
    # that locus's index range, so a cross-locus call (e.g. an IGKV allele on an IGL read, or any D on a
    # light-chain read) is impossible by construction. Produces a per-read (N, C) mask per gene, AND-ed
    # with any genotype constraint.
    if preds.chain_type is not None and getattr(reference, "loci", None):
        allowed = _locus_allowed(preds.chain_type, reference, genes, allowed)
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
    # a read without a V (a D-J rearrangement, a read starting inside the junction) gets no V call
    no_v = v_absent(seqs, alignments["v"], reference)
    # the V alleles each read cannot tell from its call, among those it may be called with
    v_ties = tied_alleles(seqs, alignments["v"], reference, allowed=None if allowed is None else allowed.get("v"))
    # `sequences` here is the post-crop, PRE-orientation read; `seqs` is its canonical (forward) frame.
    # The record must OWN the pre-orientation read (not a parallel external list) so the AIRR writer emits
    # the correct `sequence`/`rev_comp` identically for the Python API and the CLI, and crop-consistently.
    records = _to_records(seqs, calls, alignments, genes, preds, cfg.chain_types, segs.low_quality,
                          was_cropped, input_sequences=sequences, no_v=no_v, v_ties=v_ties)
    if cfg.airr:
        from .airr import build_airr
        records = build_airr(records, reference, chain=("heavy" if cfg.has_d else "light"))
    return records


def _anchor(position) -> int | None:
    """A decoded anchor as a record holds it: the read position, or None when it is not observed."""
    return None if position == NOT_OBSERVED else int(position)


def _to_records(sequences, calls, alignments, genes, preds, chain_types=None, low_quality=None,
                was_cropped=None, input_sequences=None, no_v=None, v_ties=None) -> list:
    orientation = preds.orientation
    anchors = getattr(preds, "anchors", None)
    records = []
    for i, seq in enumerate(sequences):
        absent_v = no_v is not None and no_v[i]
        oid = int(orientation[i]) if orientation is not None else 0
        rec = {"sequence": seq, "orientation_id": oid,
               "mutation_rate": float(preds.mutation_rate[i]),
               "indel_count": float(preds.indel_count[i]),
               "productive": bool(preds.productive[i]) if anchors is None else None}
        if anchors is not None:
            # a model with anchor heads: its record carries the anchors (forward frame, like `seq`) and the
            # productivity they give over the read's V start and J end (predict.junction), not a learned call
            rec.update({ANCHOR_KEYS[g]: _anchor(anchors[g][i]) for g in ANCHOR_GENES})
            if absent_v:                                       # no V, so no V anchor to open a junction
                rec[ANCHOR_KEYS["v"]] = None
            v, j = (None if absent_v else alignments["v"][i]), alignments["j"][i]
            rec["productive"] = derive_junction(seq, *(rec[ANCHOR_KEYS[g]] for g in ANCHOR_GENES), coding_span=(
                None if v is None else v.seq_start, None if j is None else j.seq_end)).productive
        if input_sequences is not None:
            # the post-crop, pre-orientation read (== `seq` when the read was already forward). The AIRR
            # writer uses THIS as the emitted `sequence` for a reverse-complement read, so `RC(sequence)`
            # equals the coordinate frame (`seq`) — regardless of any external sequence list.
            rec["input_sequence"] = input_sequences[i]
        if low_quality is not None:
            rec["segmentation_low_quality"] = bool(low_quality[i])
        if was_cropped is not None and was_cropped[i]:
            rec["length_cropped"] = True
        if preds.chain_type is not None:                       # multi-chain: predicted locus
            ct = int(preds.chain_type[i])
            rec["chain_type_id"] = ct
            if chain_types is not None and 0 <= ct < len(chain_types):
                rec["locus"] = chain_types[ct]
        elif chain_types is not None and len(chain_types) == 1:  # single-chain: the model's one locus
            rec["locus"] = chain_types[0]
        for g in genes:
            call, aln = calls[g][i], alignments[g][i]
            # the V call is listed with the alleles the read cannot tell from it (predict.ties)
            listed = v_ties[i] if g == "v" and v_ties is not None and v_ties[i] else call.names[:1]
            rec[f"{g}_call"] = ",".join(listed)
            rec[f"{g}_calls"] = list(call.names)
            rec[f"{g}_likelihoods"] = list(call.likelihoods)
            if aln is not None:
                rec[f"{g}_sequence_start"] = aln.seq_start
                rec[f"{g}_sequence_end"] = aln.seq_end
                rec[f"{g}_germline_start"] = aln.germ_start
                rec[f"{g}_germline_end"] = aln.germ_end
                rec[f"{g}_cigar"] = aln.cigar
        d_presence = getattr(preds, "d_presence", None)
        if d_presence is not None and "d" in genes:
            _gate_d(rec, int(d_presence[i]))
        if absent_v:
            _gate_v(rec)
        records.append(rec)
    return records


def _gate_v(rec: dict) -> None:
    """Drop the V of a read that holds none (``predict.v_presence``) and record it: no call, no coordinates; the AIRR
    assembly then gives the read no junction or alignment, and its D and J calls stand."""
    rec[V_PRESENCE] = V_ABSENT
    rec["v_call"], rec["v_calls"], rec["v_likelihoods"] = "", [], []
    for key in ("v_sequence_start", "v_sequence_end", "v_germline_start", "v_germline_end", "v_cigar"):
        rec.pop(key, None)


def _gate_d(rec: dict, klass: int) -> None:
    """Gate a record's D on the D presence head's class (``core.d_presence``) and record the class: an absent D is
    dropped (no call, no coordinates: the AIRR assembly then treats the read as one without a D), a short one keeps
    its span and alignment without a call (an abstention), a callable one is left as it is."""
    rec[D_PRESENCE] = D_CLASSES[klass]
    if klass == CALLABLE:
        return
    rec["d_call"], rec["d_calls"], rec["d_likelihoods"] = "", [], []
    if klass == ABSENT:
        for key in ("d_sequence_start", "d_sequence_end", "d_germline_start", "d_germline_end", "d_cigar"):
            rec.pop(key, None)
