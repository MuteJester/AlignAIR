"""Assembly and audit of the AA guard validation cohort."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

from alignair.errors import (
    AlignAIRError, ArtifactIntegrityError, DataContractError, InternalError,
    OperationCancelledError, OperationIOError, report_error,
)
from .hydration import atomic_write_text
from .manifest import MANIFEST_HASH_FIELD, seal_manifest
from .stage1_audit import audit_stage1, combine_allocation, global_dedup
from .structural import structural_records

SOURCES = ("sabdab", "igsf", "genairr_sim", "oas", "oas_paired", "oas_paired_diverse", "tenx_tcr")


def _load_sources(store, census_dir):
    records, present, checksums, snapshots = [], [], {}, {}
    for source in SOURCES:
        path = store / f"{source}.normalized.json"
        if not path.exists():
            continue
        raw = path.read_bytes()
        try:
            values = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DataContractError("Invalid normalized cohort source JSON.", context={"path": path}) from exc
        if not isinstance(values, list):
            raise DataContractError("Normalized cohort source must contain a record list.", context={"path": path})
        for index, record in enumerate(values):
            if not isinstance(record, dict) or any(not isinstance(record.get(key), str)
                                                   for key in ("aa", "group", "category", "lineage_key")):
                raise DataContractError("Cohort record requires text sequence, group, category, and lineage fields.",
                                        context={"source": source, "record_index": index})
        records.extend(values)
        present.append(f"{source}({len(values)})")
        snapshots[source] = {"records": len(values), "normalized_sha256": hashlib.sha256(raw).hexdigest()}
        census_path = census_dir / f"{source}.census.json"
        if census_path.exists():
            census = json.loads(census_path.read_text())
            if not isinstance(census, dict) or not isinstance(census.get("provenance", {}), dict):
                raise DataContractError("Source census requires a provenance mapping.", context={"path": census_path})
            provenance = census.get("provenance", {})
            checksums[source] = (provenance.get("raw_sha256") or provenance.get("cache_sha256") or
                                 provenance.get("uniprot_raw_sha256") or "n/a")
    return records, present, checksums, snapshots


def run_gate(*, store, census_dir, output_dir, split_fingerprint):
    """Audit current source snapshots, writing a frozen manifest only on GO.

    Source collectors perform pre-split denylist filtering. This gate binds their
    supplied normalized bytes and the declared fingerprint, then checks allocation,
    diversity, and post-deduplication floors. It never opens a sealed dataset.
    A directory containing an existing frozen manifest is refused.
    """
    store, census_dir, output_dir = Path(store), Path(census_dir), Path(output_dir)
    fingerprint_path = Path(split_fingerprint)
    manifest_path = output_dir / "FROZEN_COHORT_MANIFEST.json"
    if manifest_path.exists():
        raise ArtifactIntegrityError("A frozen cohort manifest already exists.", context={"path": manifest_path},
                                     hint="Choose a new output directory to audit a different cohort.")
    fingerprint_sha = hashlib.sha256(fingerprint_path.read_bytes()).hexdigest()
    records, present, checksums, snapshots = _load_sources(store, census_dir)

    selection, loco, study_alloc = combine_allocation(records)

    # Derive negatives only after each real-IGH parent has its final role.
    real = [r for r in records if r["group"] == "positive_real"]
    role_of_lineage = {}
    for lk in selection.get("positive_real", set()):
        role_of_lineage[lk] = "selection"
    for lk in loco.get("positive_real", set()):
        role_of_lineage[lk] = "loco"
    structural = []
    for parent in real:
        role = role_of_lineage.get(parent["lineage_key"])
        if role is None:
            continue
        structural.extend(structural_records(parent, role_of_parent=role))
    for r in structural:
        (selection if r["inherited_role"] == "selection" else loco).setdefault("structural", set()).add(r["lineage_key"])

    all_records = records + structural

    # The audit measures the records that remain after assignment and deduplication.
    kept, dropped = global_dedup(all_records)

    report = audit_stage1(records=kept, selection=selection, loco=loco, study_alloc=study_alloc,
                          source_checksums=checksums)

    census = {
        "artifact": "stage1_gate",
        "sources_present": present,
        "source_checksums": checksums,
        "total_records": len(all_records),
        "global_dedup_dropped": dropped,
        "kept": len(kept),
        "decision": report.decision,
        "role_overlap": report.role_overlap,
        "counts_by_role": report.counts_by_role,
        "counts_by_locus": report.counts_by_locus,
        "counts_by_study": report.counts_by_study,
        "counts_by_donor": report.counts_by_donor,
        "floors_ok": report.floors_ok,
        "locus_balance_37_37": report.locus_balance_37_37,
        "locus_diversity": report.locus_diversity,
        "powered_real_igh": report.powered_real_igh,
        "policy_hash_reproduced": report.policy_hash_reproduced,
        "allocator_policy_hash": study_alloc.policy_hash if study_alloc else None,
        "failures": report.failures,
    }
    if report.decision == "GO":
        identities = [(fingerprint_path, fingerprint_sha)] + [
            (store / f"{source}.normalized.json", snapshot["normalized_sha256"])
            for source, snapshot in snapshots.items()
        ]
        for path, expected in identities:
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ArtifactIntegrityError("Cohort input changed during the audit.", context={"path": path},
                                             hint="Rerun from stable source files before freezing the cohort.")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_dir / "STAGE1_RESULT.json", json.dumps(census, indent=2, sort_keys=True))
    if report.decision == "GO":
        _freeze_manifest(report, study_alloc, census, snapshots, fingerprint_path, fingerprint_sha, manifest_path)
    return census


def _freeze_manifest(report, study_alloc, census, snapshots, fingerprint_path, fingerprint_sha, output_path):
    """Seal the source snapshots used by this audit; never replace an existing seal."""
    manifest = {
        "artifact": "FROZEN_COHORT_MANIFEST",
        "frozen_date": datetime.date.today().isoformat(),
        "decision": report.decision,
        "sources": snapshots,
        "totals": {k: census[k] for k in ("total_records", "global_dedup_dropped", "kept")},
        "allocation": {"policy_hash": study_alloc.policy_hash if study_alloc else None,
                       "role_of_study": study_alloc.role_of_study if study_alloc else {}},
        "audit": {k: census[k] for k in ("role_overlap", "counts_by_role", "counts_by_locus",
                                         "counts_by_study", "counts_by_donor", "floors_ok",
                                         "locus_balance_37_37", "locus_diversity", "powered_real_igh",
                                         "policy_hash_reproduced")},
        "seal_provenance": {
            "split_fingerprint": str(fingerprint_path),
            "split_fingerprint_sha256": fingerprint_sha,
            "note": "source collectors perform pre-split denylist filtering; this gate binds the normalized "
                    "source snapshots and declared fingerprint used for its audit. No sealed dataset is read.",
        },
        "manifest_sha256_rule": (
            f"sha256(json.dumps(manifest_without_'{MANIFEST_HASH_FIELD}', sort_keys=True, "
            "separators=(',',':'))) — non-recursive; see alignair.aa.guard.manifest.compute_manifest_hash"),
    }
    manifest = seal_manifest(manifest)
    serialized = json.dumps(manifest, indent=2, sort_keys=True)
    try:
        with output_path.open("x") as stream:
            stream.write(serialized)
    except FileExistsError as exc:
        raise ArtifactIntegrityError("Refusing to replace an existing frozen cohort manifest.",
                                     context={"path": output_path}) from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--error-format", choices=("text", "json"), default="text")
    for option in ("store", "census-dir", "out", "fingerprint"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        census = run_gate(store=args.store, census_dir=args.census_dir,
                          output_dir=args.out, split_fingerprint=args.fingerprint)
        print(json.dumps(census, indent=2, sort_keys=True))
        return 0 if census["decision"] == "GO" else 1
    except AlignAIRError as exc:
        error = exc
    except KeyboardInterrupt:
        error = OperationCancelledError("Cohort audit interrupted.")
    except OSError as exc:
        error = OperationIOError("Cannot access cohort inputs or outputs.", context={"reason": str(exc)})
        error.__cause__ = exc
    except json.JSONDecodeError as exc:
        error = DataContractError("Invalid JSON in cohort input.", context={"line": exc.lineno})
        error.__cause__ = exc
    except Exception as exc:
        error = InternalError("Unexpected cohort audit failure.", context={"cause_type": type(exc).__name__})
        error.__cause__ = exc
    return report_error(error, debug=args.debug, json_output=args.error_format == "json")


if __name__ == "__main__":
    raise SystemExit(main())
