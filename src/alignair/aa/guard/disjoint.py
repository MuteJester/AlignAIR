"""Fresh-cohort disjointness auditor.

Establishes that fresh validation records are disjoint from the ENTIRE frozen Gate-3 corpus WITHOUT
touching the seal. The denylist is derived from the PRE-SPLIT cleaned Gate-3 corpus, which contains fit,
val, AND sealed test records; excluding all of it excludes the sealed records without ever reading the
sealed partition file or revealing any record's split assignment.

Before the corpus is trusted, its committed global canonical digest is reproduced by replaying the public
freeze procedure (Phase-3 exclusion -> per-group sequence dedup -> deterministic split assignment ->
canonical digest). The historical freeze policy is preserved below and checked against captured fixture
assignments and digests. A mismatch means the corpus is not the
exact frozen one and the denylist is refused. This module imports only the pure canonical helpers; it has
no path or call capable of reading the sealed partition.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from alignair.errors import (
    AlignAIRError, ArtifactIntegrityError, DataContractError, InternalError,
    OperationCancelledError, OperationIOError, report_error,
)
from alignair.aa.product.applicability import canonical

_PHASE3_GOLD = "Notes and Plans/docs/superpowers/artifacts/2026-07-21-aa-phase3/gold_labels.tsv"


class Gate3DigestMismatch(ArtifactIntegrityError):
    """Raised when the pre-split corpus does not reproduce the committed global canonical digest."""
    code = "ALIGNAIR_COHORT_DIGEST_MISMATCH"


@dataclass(frozen=True)
class DisjointReport:
    clean: bool
    overlaps: dict


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _rank(keys):
    return sorted(keys, key=_sha)


def _load_phase3_gold(path) -> tuple[set, dict]:
    names: set = set()
    vh_by_name: dict = {}
    p = Path(path)
    if not p.exists():
        return names, vh_by_name
    with open(p) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            nm = (row.get("name") or "").strip().lower()
            if nm:
                names.add(nm)
                if row.get("exact_vh_id"):
                    vh_by_name[nm] = row["exact_vh_id"].strip()
    return names, vh_by_name


def _replay_splits(kept: list[dict]) -> dict:
    """Reproduce the frozen Gate-3 split assignment from the deduped pre-split records.

    Historical Gate-3 allocation policy: positives split
    fit/val/test .50/.25/.25 (structural-shared lineages restricted to val/test), structural inherits the
    positive_real assignment where shared else val/test .50/.50, all other groups val/test .50/.50 —
    all by deterministic sha256(lineage_key) rank.
    """
    struct_lineages = {e["lineage_key"] for e in kept if e["group"] == "structural"}
    group_lineages: dict = {}
    for e in kept:
        group_lineages.setdefault(e["group"], set()).add(e["lineage_key"])

    lk_split: dict = {}
    for g in ("positive_real", "positive_sim"):
        keys = group_lineages.get(g, set())
        fit_elig = _rank([k for k in keys if k not in struct_lineages])
        restricted = _rank([k for k in keys if k in struct_lineages])
        f, v = round(len(fit_elig) * 0.5), round(len(fit_elig) * 0.75)
        for i, k in enumerate(fit_elig):
            lk_split[(g, k)] = "fit" if i < f else "val" if i < v else "test"
        vv = round(len(restricted) * 0.5)
        for i, k in enumerate(restricted):
            lk_split[(g, k)] = "val" if i < vv else "test"
    posreal = {k: sp for (g, k), sp in lk_split.items() if g == "positive_real"}
    struct_ordered = _rank(group_lineages.get("structural", set()))
    sv = round(len(struct_ordered) * 0.5)
    for i, k in enumerate(struct_ordered):
        lk_split[("structural", k)] = posreal.get(k, "val" if i < sv else "test")
    for g, keys in group_lineages.items():
        if g in ("positive_real", "positive_sim", "structural"):
            continue
        ordered = _rank(keys)
        vv = round(len(ordered) * 0.5)
        for i, k in enumerate(ordered):
            lk_split[(g, k)] = "val" if i < vv else "test"
    return lk_split


def build_gate3_global_denylist(cleaned_cohort_path, split_fingerprint, *,
                                phase3_gold_path=_PHASE3_GOLD) -> dict:
    """Verified conservative denylist over the ENTIRE pre-split Gate-3 corpus.

    Refuses (raises ``Gate3DigestMismatch``) unless the reproduced global canonical digest equals the
    committed ``split_fingerprint['record_digest']['global']``. Never reads the sealed partition; the seal
    is covered because the pre-split corpus is the union of fit + val + sealed test.
    """
    committed = (split_fingerprint or {}).get("record_digest", {}).get("global")

    p = Path(cleaned_cohort_path)
    entries: list[dict] = []
    if p.exists():
        entries = [json.loads(line) for line in open(p) if line.strip()]

    # (1) Historical Phase-3 exclusion: drop therapeutics by INN name or VH hash.
    p3_names, vh_by_name = _load_phase3_gold(phase3_gold_path)
    name_by_vh = {v: k for k, v in vh_by_name.items()}
    kept0 = [e for e in entries
             if e["lineage_key"].strip().lower() not in p3_names and _sha(e["aa"]) not in name_by_vh]

    # (2) Per-group sequence dedup, first occurrence wins (file order preserved).
    seen: set = set()
    kept: list[dict] = []
    for e in kept0:
        key = (e["group"], _sha(e["aa"]))
        if key in seen:
            continue
        seen.add(key)
        kept.append(e)

    # (3) Reproduce and verify the committed global canonical digest.
    lk_split = _replay_splits(kept)
    canon = [canonical.canonical_record(e, lk_split[(e["group"], e["lineage_key"])]) for e in kept]
    reproduced = canonical.digest(canon)
    if committed is None or reproduced != committed:
        raise Gate3DigestMismatch(
            f"pre-split corpus digest {reproduced[:16]} != committed {str(committed)[:16]}; "
            "the cleaned Gate-3 corpus is not the exact frozen one — denylist refused")

    # (4) Conservative denylist over the full pre-split corpus.
    return {
        "sequence_hashes": {_sha(e["aa"]) for e in kept},
        "lineages": {e["lineage_key"] for e in kept},
        "donors": {e.get("donor") for e in kept if e.get("donor")},
        "studies": {e.get("study") for e in kept if e.get("study")},
    }


def audit_disjoint(fresh_records, *, gate3_denylist, phase3) -> DisjointReport:
    """Flag any fresh record overlapping the Gate-3 pre-split denylist or the Phase-3 therapeutics."""
    p3_names, p3_vh = phase3
    p3_names = p3_names or {}
    p3_vh = p3_vh or {}
    dl_seq = gate3_denylist.get("sequence_hashes", set())
    dl_lin = gate3_denylist.get("lineages", set())
    dl_don = gate3_denylist.get("donors", set())
    dl_std = gate3_denylist.get("studies", set())

    overlaps: dict = {"gate3_seqhash": [], "gate3_lineage": [], "phase3_inn": [],
                      "phase3_vh": [], "donor": [], "study": []}
    for r in fresh_records:
        rid = r["id"]
        h = _sha(r["aa"])
        nm = (r.get("lineage_key") or "").strip().lower()
        if h in dl_seq:
            overlaps["gate3_seqhash"].append(rid)
        if r.get("lineage_key") in dl_lin:
            overlaps["gate3_lineage"].append(rid)
        if nm in p3_names:
            overlaps["phase3_inn"].append(rid)
        if h in p3_vh:
            overlaps["phase3_vh"].append(rid)
        if r.get("donor") and r["donor"] in dl_don:
            overlaps["donor"].append(rid)
        if r.get("study") and r["study"] in dl_std:
            overlaps["study"].append(rid)

    clean = all(not v for v in overlaps.values())
    return DisjointReport(clean=clean, overlaps=overlaps)


def _fresh_records(path):
    """Read JSONL incrementally and identify malformed records by their line number."""
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataContractError("Invalid JSON in fresh cohort.",
                                        context={"path": path, "line": line_number}) from exc
            if (not isinstance(record, dict) or
                    any(not isinstance(record.get(key), str) for key in ("id", "aa")) or
                    any(record.get(key) is not None and not isinstance(record[key], str)
                        for key in ("lineage_key", "donor", "study"))):
                raise DataContractError("Fresh records require text id/aa and optional text lineage/donor/study fields.",
                                        context={"path": path, "line": line_number})
            yield record


def main(argv=None) -> int:
    """Audit explicit input files, returning failure for overlaps or invalid inputs."""
    parser = argparse.ArgumentParser(description="Audit fresh-cohort overlap against verified pre-split and Phase-3 data.")
    parser.add_argument("fresh_cohort", type=Path, help="Fresh records in JSONL format")
    for option in ("cleaned", "fingerprint", "phase3-gold"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--error-format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    try:
        if not args.phase3_gold.is_file():
            raise OperationIOError("Phase-3 digest replay labels are missing.", context={"path": args.phase3_gold})
        fingerprint = json.loads(args.fingerprint.read_text(encoding="utf-8"))
        if not isinstance(fingerprint, dict) or not isinstance(fingerprint.get("record_digest", {}), dict):
            raise DataContractError("Pre-split fingerprint requires a record_digest mapping.")
        denylist = build_gate3_global_denylist(args.cleaned, fingerprint, phase3_gold_path=args.phase3_gold)
        names, vh_by_name = _load_phase3_gold(args.phase3_gold)
        report = audit_disjoint(_fresh_records(args.fresh_cohort), gate3_denylist=denylist,
                                phase3=(names, set(vh_by_name.values())))
        print(json.dumps({
            "clean": report.clean,
            "overlap_counts": {key: len(values) for key, values in report.overlaps.items()},
            "denylist_sizes": {key: len(values) for key, values in denylist.items()},
        }, indent=2, sort_keys=True))
        return 0 if report.clean else 1
    except AlignAIRError as exc:
        error = exc
    except KeyboardInterrupt:
        error = OperationCancelledError("Disjointness audit interrupted.")
    except OSError as exc:
        error = OperationIOError("Cannot access disjointness audit inputs.", context={"reason": str(exc)})
        error.__cause__ = exc
    except (json.JSONDecodeError, UnicodeError) as exc:
        error = DataContractError("Invalid JSON or text encoding in disjointness audit input.")
        error.__cause__ = exc
    except Exception as exc:
        error = InternalError("Unexpected disjointness audit failure.", context={"cause_type": type(exc).__name__})
        error.__cause__ = exc
    return report_error(error, debug=args.debug, json_output=args.error_format == "json")


if __name__ == "__main__":
    raise SystemExit(main())
