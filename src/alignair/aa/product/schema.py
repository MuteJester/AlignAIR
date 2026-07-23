"""§6 result schema: ``AAPrediction`` — distinct from nucleotide AIRR (AA-only; no nt coordinates,
CIGAR strings, productivity calls, or full IMGT numbering).

This module owns only the top-level result *shape* + JSON serialization. ``AAPredictor.predict``
(``predictor.py``) fills it in; the calibrated-set fields it writes here are Task-7 placeholders
(empty lists / zero counts) until the §7 calibration-group contract is wired in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AAPrediction:
    """One input sequence's §6 result.

    ``segments`` is ``{"V": {...}, "J": {...}}`` plus a ``"D"`` entry iff the loaded model has a D
    branch (``predictor.cfg.has_d``). Each segment dict has the shape::

        {
            "top_call": {
                "class_id": int,
                "class_name": str,
                "top_alleles": [str, ...],   # IMGT allele names, one-to-many ASC-expanded (ALL)
                "genes": [str, ...],          # PLURAL — a class can span >1 IMGT gene
                "families": [str, ...],       # PLURAL — and potentially >1 family
                "probability": float,
            },
            "calibrated_class_set": [...],    # Task-7 placeholder: [] here
            "calibrated_allele_set": [...],   # Task-7 placeholder: [] here
            "calibrated_gene_set": [...],     # Task-7 placeholder: [] here
            "calibrated_family_set": [...],   # Task-7 placeholder: [] here
            "set_size": {"classes": 0, "alleles": 0, "genes": 0},   # Task-7 placeholder
            "coordinates": {"start": int, "end": int} | None,       # §5
        }

    **D-segment convention** (documented here per §6): D has no protein-equivalence class, so
    ``top_call.class_id`` for the ``"D"`` segment is directly a **D allele-head index**
    (``predictor.d_allele_order[class_id]``), not a class id into a class map. Its ``top_alleles`` /
    ``genes`` / ``families`` are therefore always single-element lists derived from that one D allele
    name — still lists (never a bare string), for API-shape uniformity with V/J.
    """
    sequence_id: str
    input_length: int
    segments: dict
    junction_aa: Optional[str]
    applied_calibration_group: dict          # {"group": "pooled"|"full"|"trunc"|"short", "source": ...}  (§7)
    provenance: dict                         # {model_id, model_version, weights_sha256,
                                              #  calibration_sha256, class_map_db_hash,
                                              #  alignair_version, calibration_scope}                     (§8)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serializable dict — every field here is already made of plain str/int/float/list/dict/
        None, so this is a shallow, defensive copy (never returns the live mutable containers)."""
        return {
            "sequence_id": self.sequence_id,
            "input_length": self.input_length,
            "segments": {gene: dict(seg) for gene, seg in self.segments.items()},
            "junction_aa": self.junction_aa,
            "applied_calibration_group": dict(self.applied_calibration_group),
            "provenance": dict(self.provenance),
            "warnings": list(self.warnings),
        }
