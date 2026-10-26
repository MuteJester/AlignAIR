"""Frozen Phase-3 label schema + row/cohort validators. No model, no alignment, no raw sequences committed.

vh_aa lives ONLY in a git-ignored hydrated map; committed rows carry exact_vh_id = sha256(vh_aa).
See Notes and Plans/docs/superpowers/specs/2026-07-21-aa-phase3-realprotein-design.md.
"""
from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass

LABEL_FIELDS = [
    "name", "arm_id", "exact_vh_id", "lineage_id", "lineage_representative", "chain_class",
    "seq_source", "seq_source_version", "retrieval_date", "raw_file_sha256", "true_v_family",
    "true_v_gene", "true_v_allele", "label_resolution", "in_catalog", "tier", "label_source",
    "label_citation", "alias_source", "evidence_locator", "humanization_status", "true_j",
    "exclusion_reason",
]
_BOOL = {"true", "false"}
_TIER = {"gold", "concordance"}
_CHAIN = {"in_scope", "murine", "chimeric", "vhh", "synthetic", "unresolved"}
_RES = {"exact_allele", "exact_gene", "subgroup_only", "ambiguous", "unresolved"}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ALIGN = ("alignment", "anarci", "imgt", "v-quest", "vquest")


@dataclass
class LabelRow:
    name: str
    arm_id: str
    exact_vh_id: str
    lineage_id: str
    lineage_representative: str
    chain_class: str
    seq_source: str
    seq_source_version: str
    retrieval_date: str
    raw_file_sha256: str
    true_v_family: str
    true_v_gene: str
    true_v_allele: str
    label_resolution: str
    in_catalog: str
    tier: str
    label_source: str
    label_citation: str
    alias_source: str
    evidence_locator: str
    humanization_status: str
    true_j: str
    exclusion_reason: str


def sha256_seq(vh_aa: str) -> str:
    return hashlib.sha256(vh_aa.encode()).hexdigest()


def validate_row(d: dict) -> list[str]:
    v: list[str] = []
    missing = [f for f in LABEL_FIELDS if f not in d]
    if missing:
        return [f"missing fields: {missing}"]
    if d["lineage_representative"] not in _BOOL:
        v.append("lineage_representative must be boolean true|false")
    if d["in_catalog"] not in _BOOL:
        v.append("in_catalog must be boolean true|false")
    if d["tier"] not in _TIER:
        v.append(f"tier must be {_TIER}")
    if d["chain_class"] not in _CHAIN:
        v.append(f"chain_class must be in {_CHAIN}")
    if d["label_resolution"] not in _RES:
        v.append("label_resolution invalid")
    if not _HEX64.match(d["exact_vh_id"]):
        v.append("exact_vh_id must be a 64-char hex hash")
    if not _HEX64.match(d["raw_file_sha256"]):
        v.append("raw_file_sha256 must be a 64-char hex hash")
    if not _ISO.match(d["retrieval_date"]):
        v.append("retrieval_date must be ISO YYYY-MM-DD")
    if not d["label_citation"]:
        v.append("every admitted row needs a label_citation")
    if not d["evidence_locator"]:
        v.append("every admitted row needs an evidence_locator")
    if d["tier"] == "gold":
        if not d["true_v_family"]:
            v.append("gold row needs a true_v_family")
        if d["label_resolution"] not in ("subgroup_only", "exact_gene", "exact_allele"):
            v.append("gold design-ancestry label must resolve to a family")
    if d["tier"] == "concordance" and not d["true_v_gene"]:
        v.append("concordance row needs a true_v_gene")
    return v


def validate_cohort(rows: list, hydrated: dict, crosswalk: dict) -> list[str]:
    v: list[str] = []
    for r in rows:
        v += [f"{r.name}: {e}" for e in validate_row(r.__dict__)]
        seq = hydrated.get(r.exact_vh_id)
        if seq is None:
            v.append(f"{r.name}: no hydrated sequence for exact_vh_id")
        elif sha256_seq(seq) != r.exact_vh_id:
            v.append(f"{r.name}: exact_vh_id != sha256(vh_aa)")
    # one deterministic representative per lineage
    by_lin: dict[str, list] = {}
    for r in rows:
        by_lin.setdefault(r.lineage_id, []).append(r)
    for lin, members in by_lin.items():
        reps = [m for m in members if m.lineage_representative == "true"]
        smallest = min(m.exact_vh_id for m in members)
        if len(reps) != 1:
            v.append(f"lineage {lin}: exactly one representative required, got {len(reps)}")
        elif reps[0].exact_vh_id != smallest:
            v.append(f"lineage {lin}: representative must be the smallest exact_vh_id (deterministic)")
    # no conflicting labels for one VH
    by_vh: dict[str, set] = {}
    for r in rows:
        by_vh.setdefault(r.exact_vh_id, set()).add((r.true_v_family, r.true_v_gene, r.tier))
    for vid, labels in by_vh.items():
        if len({(f, g) for f, g, _ in labels}) > 1:
            v.append(f"vh {vid[:8]}: conflicting labels for one sequence")
    # gold alias in crosswalk; concordance marked alignment-derived
    for r in rows:
        if r.tier == "gold" and r.label_source not in crosswalk:
            v.append(f"{r.name}: gold label_source not in frozen crosswalk")
        if r.tier == "concordance":
            marker = (r.label_source + " " + r.alias_source).lower()
            if not any(k in marker for k in _ALIGN):
                v.append(f"{r.name}: concordance row must be marked alignment-derived (IMGT/ANARCI/…)")
    return v


def load_labels(path: str) -> tuple[list, list]:
    admitted, audit = [], []
    with open(path, newline="") as fh:
        for d in csv.DictReader(fh, delimiter="\t"):
            errs = validate_row(d)
            # excluded/out-of-scope rows are audited, never admitted
            if d.get("exclusion_reason") or d.get("chain_class") not in ("in_scope",):
                audit.append({**d, "_reasons": d.get("exclusion_reason") or "out_of_scope"})
            elif errs:
                audit.append({**d, "_reasons": "; ".join(errs)})
            else:
                admitted.append(LabelRow(**{f: d[f] for f in LABEL_FIELDS}))
    return admitted, audit


def load_crosswalk(path: str) -> dict:
    """alias -> (imgt_family, imgt_gene_or_empty, citation). Every row MUST carry a citation."""
    out = {}
    with open(path, newline="") as fh:
        for d in csv.DictReader(fh, delimiter="\t"):
            if not d.get("citation"):
                raise ValueError(f"crosswalk row for {d.get('alias')!r} has no citation")
            out[d["alias"]] = (d["imgt_family"], d.get("imgt_gene", ""), d["citation"])
    return out


def load_hydrated(path: str) -> dict:
    """exact_vh_id -> vh_aa (from the git-ignored hydrated file)."""
    out = {}
    with open(path, newline="") as fh:
        for d in csv.DictReader(fh, delimiter="\t"):
            out[d["exact_vh_id"]] = d["vh_aa"]
    return out


def cohort_sha256(rows: list, hydrated: dict) -> str:
    h = hashlib.sha256()
    for line in sorted(
        "\t".join(str(getattr(r, f)) for f in LABEL_FIELDS) + "\t" + hydrated.get(r.exact_vh_id, "")
        for r in rows
    ):
        h.update(line.encode())
        h.update(b"\n")
    return h.hexdigest()
