from __future__ import annotations

from typing import Dict, List, Tuple

from .temperature import Row


def collect_calibration_rows(model, reference_set, records, *, predict_fn=None,
                             topk: int = 32, genes=("v", "d", "j"), **predict_kwargs
                             ) -> Tuple[Dict[str, List[Row]], List[float]]:
    """Run the model over labeled `records` and build per-gene calibration rows from the
    emitted candidate scores + each record's true call set. Also returns the per-read
    out-of-scope gate scores (for fitting the contaminant tau on these real reads)."""
    if predict_fn is None:
        from ..dnalignair_infer import predict_reads as predict_fn
    reads = [r["sequence"] for r in records]
    has_d = reference_set.has_d
    use_genes = [g for g in genes if g != "d" or has_d]
    preds = predict_fn(model, reference_set, reads, topk=topk, rerank="learned",
                       emit_scores=True, **predict_kwargs)
    rows: Dict[str, List[Row]] = {g.upper(): [] for g in use_genes}
    gate_scores: List[float] = []
    for rec, p in zip(records, preds):
        if p.get("contaminant_score") is not None:
            gate_scores.append(p["contaminant_score"])
        for g in use_genes:
            scored = p.get(f"{g}_scores")
            if not scored:
                continue
            names = [nm for nm, _ in scored]
            scores = [sc for _, sc in scored]
            truth = {c.strip() for c in str(rec.get(f"{g}_call", "")).split(",") if c.strip()}
            pos = [j for j, nm in enumerate(names) if nm in truth]
            rows[g.upper()].append((scores, pos))
    return rows, gate_scores
