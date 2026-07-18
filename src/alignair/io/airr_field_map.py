"""Machine-readable field map for the AIRR TSV AlignAIR emits.

For every emitted column it records a ``source`` — how the value is produced — plus its null policy.
This is the single place that answers "what is this column, and is it a real value or a placeholder?".
A completeness test (``tests/alignair/io/test_field_map.py``) asserts every emitted column has an entry,
so an advertised-but-unpopulated column can never silently ship.

source categories:
  neural      — a model head output (v/d/j allele probs, mutation rate, orientation, chain type)
  derived     — computed from other fields (identity, productive, vj_in_frame, stop_codon, junction)
  germline    — from the reference alignment / called allele (calls, germline coords, alignments)
  read        — the read itself or coordinates into it (sequence, sequence coords, cigar, regions)
  extension   — an AlignAIR extension, NOT a standard AIRR field
"""
from __future__ import annotations

import re

from .airr import COLUMNS

NEURAL, DERIVED, GERMLINE, READ, EXTENSION = "neural", "derived", "germline", "read", "extension"

# Semantically important / irregular fields spelled out explicitly.
_EXPLICIT = {
    "sequence_id": (READ, "input read id (deduplicated)"),
    "sequence": (READ, "canonical (forward) read the coordinates refer to"),
    "input_sequence": (EXTENSION, "original pre-orientation read (only when reoriented)"),
    "rev_comp": (DERIVED, "T iff the read was reverse-complemented (orientation id 1)"),
    "orientation": (EXTENSION, "full orientation transform: forward/reverse_complement/complement/reverse"),
    "locus": (NEURAL, "predicted locus (multi-chain) or the model's single locus"),
    "productive": (DERIVED, "AIRR productive = in-frame AND no stop codon"),
    "productive_prediction": (EXTENSION, "the model's neural productivity prediction (not derived)"),
    "vj_in_frame": (DERIVED, "junction length divisible by 3 / frame-consistent"),
    "stop_codon": (DERIVED, "a stop codon appears in the translated alignment"),
    "mutation_rate": (NEURAL, "predicted SHM rate"),
    "is_contaminant": (NEURAL, "predicted contamination flag"),
    "segmentation_low_quality": (EXTENSION, "V anchor collapsed -> no feasible segmentation"),
    "length_cropped": (EXTENSION, "read exceeded the model window and was cropped"),
    "airr_assembly_status": (EXTENSION, "complete / partial / failed"),
    "airr_assembly_reason": (EXTENSION, "machine-readable reason a record is partial"),
    "airr_assembly_error": (EXTENSION, "reason when assembly failed (exception)"),
    "np1": (READ, "N/P nucleotides between V and D(/J)"), "np2": (READ, "N/P nucleotides between D and J"),
    "np1_length": (DERIVED, "len(np1)"), "np2_length": (DERIVED, "len(np2)"),
    "junction": (DERIVED, "CDR3 + conserved flanks (nt)"), "junction_aa": (DERIVED, "junction (aa)"),
    "junction_length": (DERIVED, "len(junction) nt"), "junction_aa_length": (DERIVED, "len(junction_aa)"),
    "sequence_alignment": (GERMLINE, "IMGT-gapped read alignment"),
    "germline_alignment": (GERMLINE, "IMGT-gapped germline alignment"),
    "sequence_alignment_aa": (DERIVED, "translated sequence_alignment"),
    "germline_alignment_aa": (DERIVED, "translated germline_alignment"),
}


def _category(field: str):
    if field in _EXPLICIT:
        return {"source": _EXPLICIT[field][0], "note": _EXPLICIT[field][1]}
    if field.endswith(("_call", "_calls", "_call_set", "_resolved_call")):
        return {"source": GERMLINE, "note": "called allele name(s)"}
    if field.endswith(("_likelihoods", "_set_confidence", "_call_level")):
        return {"source": NEURAL, "note": "calibrated allele-set confidence"}
    if field.endswith("_identity"):
        return {"source": DERIVED, "note": "matching nt fraction over non-gap germline positions"}
    if field.endswith("_cigar"):
        return {"source": READ, "note": "per-segment CIGAR"}
    if re.search(r"_sequence_(start|end)$", field):
        return {"source": READ, "note": "1-based coordinate into `sequence`"}
    if re.search(r"_germline_(start|end)$", field):
        return {"source": GERMLINE, "note": "1-based coordinate into the germline"}
    if re.search(r"_(alignment_start|alignment_end)$", field):
        return {"source": GERMLINE, "note": "IMGT alignment position"}
    if field.endswith(("_sequence_alignment", "_sequence_alignment_aa")):
        return {"source": READ, "note": "per-segment read alignment"}
    if field.endswith(("_germline_alignment", "_germline_alignment_aa")):
        return {"source": GERMLINE, "note": "per-segment germline alignment"}
    if re.match(r"^(fwr|cdr)\d", field):
        return {"source": READ, "note": "IMGT FWR/CDR region"}
    return None


FIELD_MAP = {c: _category(c) for c in COLUMNS}
