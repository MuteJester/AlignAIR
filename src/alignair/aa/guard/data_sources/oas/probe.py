"""Resumable species and completeness catalogs for unpaired OAS studies."""
from __future__ import annotations

import collections
import json
from pathlib import Path

from alignair.aa.guard.hydration import atomic_write_text
from alignair.errors import DataContractError, OperationIOError
from ..http import HttpClient
from .collectors import _studies, _units
from .records import UNPAIRED_BASE, normalize_study, parse_unit

CONFIRMED = frozenset({"eligible_human_complete", "confirmed_ineligible", "unknown_metadata"})


def sample_chain(client, study, units, minimum):
    if not units:
        return {"status": "no_units"}
    try:
        raw = client.prefix(f"{UNPAIRED_BASE}{study}/csv/{units[0]}", byte_limit=262144, rate_s=0.3)
    except OperationIOError as exc:
        return {"status": "temporarily_inaccessible" if exc.retryable else "unknown_metadata",
                "reason": "source_unavailable"}
    try:
        metadata, rows = parse_unit(raw, row_limit=249)
    except DataContractError:
        return {"status": "unknown_metadata"}
    lengths = sorted(len((row.get("sequence_alignment_aa") or "").replace(".", "").replace("-", ""))
                     for row in rows if row.get("productive") == "T")
    median = lengths[len(lengths) // 2] if lengths else 0
    species = metadata.get("Species")
    if not isinstance(species, str) or not species or species == "?":
        return {"status": "unknown_metadata", "median_aa": median}
    if species.lower() != "human":
        return {"status": "confirmed_ineligible", "species": species, "median_aa": median, "reason": "non_human"}
    if median == 0:
        return {"status": "unknown_metadata", "species": species, "median_aa": median, "reason": "no_measurable_reads"}
    if median < minimum:
        return {"status": "confirmed_ineligible", "species": species, "median_aa": median, "reason": "partial_reads"}
    return {"status": "eligible", "species": species, "median_aa": median}


def study_status(heavy, light):
    if heavy.get("status") == "eligible" or light.get("status") == "eligible":
        return "eligible_human_complete"
    statuses = {heavy.get("status"), light.get("status")}
    if "temporarily_inaccessible" in statuses:
        return "temporarily_inaccessible"
    if "unknown_metadata" in statuses:
        return "unknown_metadata"
    return "confirmed_ineligible"


def _summary(fresh, per, listed):
    counts = collections.Counter(entry.get("status") for entry in per.values())
    eligible = [name for name, entry in per.items() if entry.get("status") == "eligible_human_complete"]
    return {
        "total_studies_listed": listed, "fresh_studies": len(fresh), "by_status": dict(counts),
        "eligible_human_complete": len(eligible), "confirmed_ineligible": counts["confirmed_ineligible"],
        "unknown_metadata": counts["unknown_metadata"], "temporarily_inaccessible": counts["temporarily_inaccessible"],
        "confirmed_total": sum(counts[status] for status in CONFIRMED), "eligible_studies": sorted(eligible),
        "feasibility_note": "OAS feasibility uses CONFIRMED outcomes only; temporarily_inaccessible are retried, never counted ineligible",
    }


def probe_species(excluded_studies, *, catalog_path, client=None):
    """Probe fresh studies, retaining confirmed chain results and retrying failed chains."""
    client = client or HttpClient()
    catalog_path = Path(catalog_path)
    previous = {}
    if catalog_path.exists():
        try:
            previous = json.loads(catalog_path.read_text())
        except json.JSONDecodeError as exc:
            raise DataContractError("Invalid cached OAS species catalog.", context={"path": catalog_path}) from exc
        if not isinstance(previous, dict) or not isinstance(previous.get("per_study"), dict):
            raise DataContractError("Cached OAS catalog requires a per_study mapping.")
        previous = previous["per_study"]
    studies = _studies(client, UNPAIRED_BASE)
    fresh = [study for study in studies if normalize_study(study) not in excluded_studies]
    per = {}
    for study in fresh:
        cached = previous.get(study, {})
        if not isinstance(cached, dict):
            raise DataContractError("Cached OAS study entry must be a mapping.")
        chains = {chain: cached.get(chain) or {} for chain in ("heavy", "light")}
        if any(not isinstance(value, dict) for value in chains.values()):
            raise DataContractError("Cached OAS chain entry must be a mapping.")
        retry_chain = any(value.get("status") == "temporarily_inaccessible" for value in chains.values())
        if cached.get("status") in CONFIRMED and not retry_chain:
            per[study] = cached
        else:
            try:
                units = _units(client, UNPAIRED_BASE, study)
            except OperationIOError as exc:
                per[study] = {"status": "temporarily_inaccessible" if exc.retryable else "unknown_metadata",
                              "reason": "listing_unavailable"}
            else:
                sampled = {}
                for chain, marker, minimum in (("heavy", "Heavy", 100), ("light", "Light", 95)):
                    old = chains[chain]
                    sampled[chain] = (old if old and old.get("status") != "temporarily_inaccessible" else
                                      sample_chain(client, study, [unit for unit in units if marker in unit], minimum))
                per[study] = {"status": study_status(sampled["heavy"], sampled["light"]), **sampled}
        summary = _summary(fresh, per, len(studies))
        atomic_write_text(catalog_path, json.dumps({"per_study": per, "summary": summary}, indent=2, sort_keys=True))
    summary = _summary(fresh, per, len(studies))
    atomic_write_text(catalog_path, json.dumps({"per_study": per, "summary": summary}, indent=2, sort_keys=True))
    return summary
