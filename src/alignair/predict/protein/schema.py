"""§6 result schema: ``AAPrediction`` — distinct from nucleotide AIRR (AA-only; no nt coordinates,
CIGAR strings, productivity calls, or full IMGT numbering).

This module owns only the top-level result *shape* + JSON serialization. ``AAPredictor.predict``
(``predictor.py``) fills it in, including the §7 calibrated-set fields (conformal class/allele/gene/
family sets + ``set_size`` + the resolved ``applied_calibration_group``).
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
                "softmax_score": float,       # raw (uncalibrated) softmax value — NOT a calibrated probability
            },
            "calibrated_class_set": [...],     # §7 conformal prediction set (class/D-allele-index ids)
            "calibrated_classes": [            # SAME set, ranked by the model's softmax class score
                {"class_id": int, "rank": int, "softmax_score": float,
                 "alleles": [str, ...], "genes": [str, ...], "families": [str, ...]},
                ...
            ],
            "calibrated_allele_set": [...],    # one-to-many ASC-expanded IMGT alleles of the class set
            "calibrated_gene_set": [...],      # every IMGT gene any covered class maps to
            "calibrated_family_set": [...],    # every IMGT family any covered class maps to
            "set_size": {"classes": int, "alleles": int, "genes": int},
            "coordinates": {"start": int, "end": int} | None,       # §5
        }

    **``calibrated_classes`` — scores are class-level, and only class-level.** This is the same
    calibrated set as ``calibrated_class_set``, re-presented as a list ranked by the model's own
    softmax CLASS score (rank 1 = highest score). ``softmax_score`` is that class's raw softmax value
    copied *verbatim* — never renormalized over the set, and (for V/J) never divided among the class's
    one-to-many alleles. When a class expands to several IMGT alleles they are indistinguishable under
    the bundled ASC/protein representation and share the single class-level score; the model has no
    evidence for a within-class split, so none is invented:

        Scores belong to model classes. All alleles expanded from one class share that class-level
        evidence; no within-class allele score is identifiable.

    The field is named ``softmax_score``, not ``probability``, on purpose: the split-conformal
    calibration calibrates the *set's* coverage, not the individual softmax value as an empirical
    probability of correctness. The score is an uncalibrated model output — use the calibrated *set*
    (and its ``set_size``), not the raw score, when a coverage guarantee is what you need.

    ``calibrated_allele_set`` is retained as a flat convenience view and is exactly the de-duplicated
    union of the per-class ``alleles`` lists (no allele added or dropped). **D is the one exception:**
    its head predicts allele indices directly, so each D entry maps to exactly one allele and carries a
    defensible allele-specific ``softmax_score`` (still an uncalibrated softmax value, not a calibrated
    allele probability).

    **``inter_vj_aa`` is method-defined — it is NOT the AIRR ``junction``.** It is simply the input
    residues between the predicted V-segment end and J-segment start (``sequence[V_end:J_start]``, or
    ``None`` when V/J coordinates are missing/incoherent). The AIRR standard defines ``junction`` as the
    CDR3 plus its conserved flanking residues, anchored on conserved motifs; this product performs no
    such anchoring, so reusing the name ``junction_aa`` would assert a definition it has not validated.
    The field is named ``inter_vj_aa`` precisely to avoid that collision. Interpreting it as a CDR3 or
    an AIRR junction is unsupported until anchor-based equivalence is separately validated.

    **D-segment convention** (documented here per §6): D has no protein-equivalence class, so
    ``top_call.class_id`` for the ``"D"`` segment is directly a **D allele-head index**
    (``predictor.d_allele_order[class_id]``), not a class id into a class map. Its ``top_alleles`` /
    ``genes`` / ``families`` are therefore always single-element lists derived from that one D allele
    name — still lists (never a bare string), for API-shape uniformity with V/J.
    """
    sequence_id: str
    input_length: int
    segments: dict
    inter_vj_aa: Optional[str]               # method-defined V_end..J_start substring; NOT an AIRR junction/CDR3
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
            "inter_vj_aa": self.inter_vj_aa,
            "applied_calibration_group": dict(self.applied_calibration_group),
            "provenance": dict(self.provenance),
            "warnings": list(self.warnings),
        }
