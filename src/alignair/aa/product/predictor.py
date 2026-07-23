"""``AAPredictor.predict`` core (Task 5 + Task 7): one forward pass -> per-gene top call (plural
genes/families via the Task-1 one-to-many ASC->IMGT expansion) + hardened §5 AA coordinates +
junction_aa + §7 calibrated-set output.

**Calibrated sets (§7, Task 7).** Per gene, per sequence: resolve the calibration group once per
sequence (Task-6 ``resolve_group``), pick that gene's ``FrozenConformal`` (pooled, or the matching
Mondrian group), run ``predict_set`` on the SAME per-gene softmax probabilities used for the top call
(one forward pass total — see ``_forward``), and expand the resulting class-id set to
alleles/genes/families via the Task-1 one-to-many ``class_alleles``/``class_genes``/``class_families``
(D bypasses the class map entirely: its "classes" already ARE D allele-head indices, so its allele set
is built directly from ``d_allele_order`` — see ``_calibrated_fields``).

``applied_calibration_group`` is one field per *sequence* (§6 schema — not per gene), so group
selection/availability is resolved once per sequence and applied uniformly across V/J/D: if the
supplied group is available in every active gene's Mondrian table the group is used as given; if it is
missing from ANY of them, the whole sequence falls back to pooled (never a half-applied group) and an
extra warning is appended. In practice this fallback path never triggers on a real bundle (all genes
share the same fitted group set), but it is exercised directly with a crafted predictor in
``test_calibrated_sets.py``.

``_predict`` takes a duck-typed ``predictor`` (the ``AAPredictor`` dataclass in ``loader.py``) rather
than importing that class, so this module has ZERO dependency on ``loader.py`` — ``loader.py`` imports
``_predict`` from here instead, not the other way around, which is what keeps
``AAPredictor.predict(...)`` (defined in ``loader.py``) and this module free of a circular import.
"""
from __future__ import annotations

import torch

from .asc import class_alleles, class_families, class_genes, family_of, gene_of
from .calibration_group import resolve_group
from .schema import AAPrediction
from .validate import normalize_inputs

_OVERLAP_WARNING = "segment coordinates overlap/out-of-order"


def _split_batch(sequences):
    """Mirror ``validate._normalize_input_format``'s top-level dispatch (bare str -> one item; bare
    dict -> reject; anything else iterable -> a list of items) — NOT a reimplementation of any actual
    validation (every alphabet/length/whitespace/empty check still happens exactly once per item,
    inside ``normalize_inputs`` below). This exists only so ``_predict`` can walk the batch one input
    record at a time and interleave `AAPrediction`/`ErrorResult` outputs in the caller's original order
    (§9.1: "in place order") — something a single whole-batch ``normalize_inputs`` call cannot recover,
    since it returns valid records and invalid-record errors as two separate lists with no shared
    positional index back to the original input.
    """
    if isinstance(sequences, str):
        return [sequences]
    if isinstance(sequences, dict):
        raise TypeError(
            "sequences must be a str, an iterable of str, or an iterable of "
            "{sequence,sequence_id} dicts — not a single dict")
    return list(sequences)


def _item_id_and_sequence(item, idx: int) -> tuple:
    """Resolve the ``(sequence_id, raw_sequence)`` pair for one batch item using the exact same
    auto-ID rule ``normalize_inputs`` applies internally (explicit non-empty string id wins; otherwise
    ``seq_<index>``) — duplicated here (2 lines, not real validation) purely so that pre-resolving the
    id and handing it back to ``normalize_inputs`` per-item reproduces byte-identical ids/behavior to
    one whole-batch call, while every actual validation decision still happens inside
    ``normalize_inputs`` itself (see ``_split_batch``)."""
    if isinstance(item, dict):
        provided = item.get("sequence_id")
        seq_id = provided if isinstance(provided, str) and provided else f"seq_{idx}"
        return seq_id, item.get("sequence", "")
    return f"seq_{idx}", item


def _forward(predictor, sequences: list) -> dict:
    """One batched forward pass through the bundle's own verified tokenizer (``predictor.tokenizer``
    — reconstructed from the bundle's tokenizer section, not ``alignair.aa.infer.class_probs``'s
    module-default tokenizer instance) producing every head's raw output (class logits + start/end
    soft-argmax) in a single pass, so `class` probabilities AND coordinates both come from exactly one
    forward call per `predict()` invocation. Probabilities are computed the same way
    ``alignair.aa.infer.class_probs`` does (plain softmax over the raw per-gene class logits) — kept as
    a private single-pass forward here, instead of 2-3 separate ``class_probs`` calls (one per gene
    head, each re-running the whole model), purely to avoid that redundant recompute; the math is
    identical."""
    cfg = predictor.cfg
    L = cfg.max_seq_length
    toks = torch.zeros(len(sequences), L, dtype=torch.long)
    for i, s in enumerate(sequences):
        encoded = predictor.tokenizer.encode(s)
        if len(encoded) > L:
            raise ValueError(f"_forward: sequence length {len(encoded)} exceeds max_seq_length {L}")
        toks[i, :len(encoded)] = torch.from_numpy(encoded)
    predictor.model.eval()
    with torch.no_grad():
        return predictor.model({"tokenized_sequence": toks.to(predictor.device)})


def _select_conformal(predictor, genes: tuple, group: str):
    """Resolve, for ONE sequence, which ``FrozenConformal`` each gene actually uses.

    ``group == "pooled"`` always uses each gene's pooled conformal (never a fallback in itself). For a
    supplied Mondrian group, the group is used only if it is present in EVERY active gene's
    ``mondrian`` table; if it is missing from any of them, ALL genes fall back to pooled together, so
    ``applied_calibration_group`` (one field per sequence, §6) never describes a half-applied group.

    Returns ``(fc_by_gene, actual_group, fell_back)``.
    """
    if group == "pooled":
        return {g: predictor.conformal[g]["pooled"] for g in genes}, "pooled", False
    available = all(group in predictor.conformal[g]["mondrian"] for g in genes)
    if available:
        return {g: predictor.conformal[g]["mondrian"][group] for g in genes}, group, False
    return {g: predictor.conformal[g]["pooled"] for g in genes}, "pooled", True


def _calibrated_fields(gene: str, class_set: list, predictor) -> dict:
    """§6 ``calibrated_*``/``set_size`` for one gene, given its already-computed
    ``calibrated_class_set`` (sorted list of int class ids — for D these are D allele-head indices,
    not protein-equivalence class ids, per ``schema.AAPrediction``'s D-segment convention)."""
    if gene == "D":
        allele_set = sorted({predictor.d_allele_order[c] for c in class_set})
        gene_set = sorted({gene_of(a) for a in allele_set})
        family_set = sorted({family_of(a) for a in allele_set})
    else:
        cmap = predictor.class_maps[gene]
        allele_set = sorted({a for c in class_set for a in class_alleles(c, cmap, predictor.asc_multi)})
        gene_set = sorted({g for c in class_set for g in class_genes(c, cmap, predictor.asc_multi)})
        family_set = sorted({f for c in class_set for f in class_families(c, cmap, predictor.asc_multi)})
    return {
        "calibrated_class_set": class_set,
        "calibrated_allele_set": allele_set,
        "calibrated_gene_set": gene_set,
        "calibrated_family_set": family_set,
        "set_size": {"classes": len(class_set), "alleles": len(allele_set), "genes": len(gene_set)},
    }


def _coordinates(start_val: float, end_val: float, seq_len: int, gene: str, warnings: list):
    """§5: round (half-to-even, i.e. plain `round()` on a genuine Python float) the soft-argmax
    start/end, clip to ``[0, seq_len]``, and void (``None``) + warn if ``start > end`` after
    clipping. Coordinates are otherwise reported as the model produced them — never repaired."""
    start = max(0, min(seq_len, round(start_val)))
    end = max(0, min(seq_len, round(end_val)))
    if start > end:
        warnings.append(f"{gene}: incoherent start>end")
        return None
    return {"start": start, "end": end}


def _provenance(card: dict) -> dict:
    """§8 provenance block, pulled straight off the verified bundle card (``AAPredictor.card`` ==
    ``container.read_header`` sans ``_sections_base``, per ``bundle.py``'s header shape)."""
    integrity = card.get("integrity") or {}
    ref_prov = card.get("reference_provenance") or {}
    return {
        "model_id": card.get("model_id"),
        "model_version": card.get("model_version"),
        "weights_sha256": integrity.get("weights_sha256"),
        "calibration_sha256": integrity.get("calibration_sha256"),
        "class_map_db_hash": ref_prov.get("class_map_db_hash"),
        "alignair_version": card.get("alignair_version"),
        "calibration_scope": card.get("calibration_scope"),
    }


def _top_call_v_or_j(gene: str, class_id: int, probability: float, predictor) -> dict:
    cmap = predictor.class_maps[gene]
    return {
        "class_id": class_id,
        "class_name": f"{gene.lower()}_class_{class_id}",
        "top_alleles": class_alleles(class_id, cmap, predictor.asc_multi),
        "genes": class_genes(class_id, cmap, predictor.asc_multi),
        "families": class_families(class_id, cmap, predictor.asc_multi),
        "probability": probability,
    }


def _top_call_d(class_id: int, probability: float, predictor) -> dict:
    """D's ``class_id`` is a D allele-head index (``predictor.d_allele_order[class_id]``), not a
    protein-equivalence class id — see ``schema.AAPrediction``'s docstring for the full rationale.
    ``class_name`` is the D allele name itself (the natural, already-unique label for this index)."""
    allele = predictor.d_allele_order[class_id]
    return {
        "class_id": class_id,
        "class_name": allele,
        "top_alleles": [allele],
        "genes": [gene_of(allele)],
        "families": [family_of(allele)],
        "probability": probability,
    }


def _predict_valid(predictor, records: list, indices: list, coverage_group) -> list:
    """Run the model once on ``records`` (already-validated ``Record``s, in order) and build one
    `AAPrediction` per record. ``indices[i]`` is record ``i``'s position in the CALLER's original
    input batch (not its position among only the valid records) — required so a per-sequence
    ``coverage_group`` list/dict (§7) resolves against the same index space the caller supplied it
    in, even when ``on_invalid="skip"`` has dropped some earlier items."""
    sequences = [r.sequence for r in records]
    out = _forward(predictor, sequences)
    genes = ("V", "J", "D") if getattr(predictor.cfg, "has_d", False) else ("V", "J")

    probs = {g: torch.softmax(out[f"{g.lower()}_class_logits"], dim=-1).cpu().numpy() for g in genes}
    starts = {g: out[f"{g.lower()}_start"].detach().cpu().numpy()[:, 0] for g in genes}
    ends = {g: out[f"{g.lower()}_end"].detach().cpu().numpy()[:, 0] for g in genes}

    provenance = _provenance(predictor.card)
    results = []
    for i, rec in enumerate(records):
        seq_len = len(rec.sequence)
        warnings: list = []
        segments: dict = {}
        coords_by_gene: dict = {}

        # §7: resolve the calibration group ONCE per sequence (it is not gene-specific), then pick
        # each gene's FrozenConformal consistently — see `_select_conformal`'s docstring.
        group, source, group_warning = resolve_group(coverage_group, rec.sequence_id, indices[i])
        fc_by_gene, actual_group, fell_back = _select_conformal(predictor, genes, group)
        if fell_back:
            source = "pooled_fallback"
        if group_warning and not fell_back:
            warnings.append(group_warning)
        if fell_back:
            warnings.append(
                f"calibration group {group!r} unavailable in this model's calibration; "
                "used pooled fallback instead")
        applied_calibration_group = {"group": actual_group, "source": source}

        for gene in genes:
            class_id = int(probs[gene][i].argmax())
            probability = float(probs[gene][i, class_id])
            top_call = (_top_call_d(class_id, probability, predictor) if gene == "D"
                        else _top_call_v_or_j(gene, class_id, probability, predictor))

            coords = _coordinates(float(starts[gene][i]), float(ends[gene][i]), seq_len, gene, warnings)
            coords_by_gene[gene] = coords

            class_set = sorted(fc_by_gene[gene].predict_set(probs[gene][i]))
            segments[gene] = {"top_call": top_call,
                              **_calibrated_fields(gene, class_set, predictor),
                              "coordinates": coords}

        v_coords, j_coords, d_coords = (coords_by_gene.get("V"), coords_by_gene.get("J"),
                                        coords_by_gene.get("D"))
        v_j_overlap = bool(v_coords and j_coords and v_coords["end"] > j_coords["start"])
        d_out_of_range = bool(d_coords and v_coords and j_coords
                              and (d_coords["start"] < v_coords["end"]
                                   or d_coords["end"] > j_coords["start"]))
        if v_j_overlap or d_out_of_range:
            warnings.append(_OVERLAP_WARNING)

        junction_aa = None
        if v_coords and j_coords and v_coords["end"] <= j_coords["start"]:
            junction_aa = rec.sequence[v_coords["end"]:j_coords["start"]]

        results.append(AAPrediction(
            sequence_id=rec.sequence_id,
            input_length=seq_len,
            segments=segments,
            junction_aa=junction_aa,
            applied_calibration_group=applied_calibration_group,
            provenance=provenance,
            warnings=warnings,
        ))
    return results


def _predict(predictor, sequences, *, coverage_group=None, on_invalid="error") -> list:
    """``AAPredictor.predict`` core. Returns a list of `AAPrediction` / `ErrorResult`, one per input
    record, in the caller's original order (§9.1). ``coverage_group`` (§7) is resolved per sequence in
    ``_predict_valid``; ``on_invalid`` is exactly the §9.1 contract, delegated to ``normalize_inputs``
    per item (see ``_split_batch``'s docstring for why per-item)."""
    if on_invalid not in ("error", "skip"):
        raise ValueError(f"on_invalid must be 'error' or 'skip', got {on_invalid!r}")
    max_len = getattr(predictor.cfg, "max_seq_length", 192)

    items = _split_batch(sequences)
    ordered: list = []
    valid_records: list = []
    valid_slots: list = []

    for idx, item in enumerate(items):
        seq_id, raw = _item_id_and_sequence(item, idx)
        recs, errs = normalize_inputs([{"sequence_id": seq_id, "sequence": raw}],
                                      on_invalid=on_invalid, max_len=max_len)
        if errs:
            ordered.append(errs[0])
        else:
            ordered.append(None)          # filled in below once the batched forward pass runs
            valid_slots.append(len(ordered) - 1)
            valid_records.append(recs[0])

    if valid_records:
        predictions = _predict_valid(predictor, valid_records, valid_slots, coverage_group)
        for slot, pred in zip(valid_slots, predictions):
            ordered[slot] = pred

    return ordered
