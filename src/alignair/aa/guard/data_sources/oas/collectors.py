"""OAS collection workflows with separate baseline and diversity policies."""
from __future__ import annotations

import collections
import json
import re
from dataclasses import dataclass
from pathlib import Path

from alignair.aa.guard.hydration import SourceResult, atomic_write_text, filter_disjoint, sha256_bytes
from alignair.errors import ConfigurationError, DataContractError, OperationIOError
from ..http import HttpClient
from ..numbering import verify_antibody_domains
from .records import PAIRED_BASE, UNPAIRED_BASE, candidates, normalize_study, parse_unit


@dataclass(frozen=True)
class UnpairedPolicy:
    target_per_category: int = 120
    max_units: int = 80
    max_units_per_study: int = 8
    per_unit_candidates: int = 40


@dataclass(frozen=True)
class PairedPolicy:
    target_per_category: int = 120
    max_units: int = 60
    max_units_per_study: int = 8


@dataclass(frozen=True)
class DiversityPolicy:
    per_donor_per_chain: int = 45
    max_donors_per_study: int = 2
    max_units_per_study: int = 6
    max_studies: int = 10
    target_donors: int = 14
    excluded_studies: frozenset = frozenset({"vander_2017", "zhou_2013", "eccles_2020"})


def _validate_policy(policy):
    for name, value in vars(policy).items():
        if name != "excluded_studies" and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ConfigurationError("OAS sampling limits must be positive integers.", context={"field": name})


def _studies(client, base):
    return re.findall(r'href="([A-Z][^"/]+)/"', client.get(base).decode("utf-8", "replace"))


def _units(client, base, study, subdir="csv"):
    body = client.get(f"{base}{study}/{subdir}/").decode("utf-8", "replace")
    return re.findall(r'href="([^"]+\.csv\.gz)"', body)


def _unique(records):
    first = {}
    for record in records:
        first.setdefault(record["lineage_key"], record)
    return list(first.values())


def _finish(source, records, exclusions, store, census_dir, census):
    kept, dropped = filter_disjoint(_unique(records), denylist=exclusions.denylist, phase3=({}, {}))
    census.update({
        "kept_after_denylist": len(kept), "denylist_dropped": dropped,
        "by_category": {category: sum(r["category"] == category for r in kept)
                        for category in (("igk", "igl", "pos_igh_repertoire") if source == "oas" else ("igk", "igl"))},
        "distinct_donors": len({(r["study"], r["donor"]) for r in kept}),
    })
    if source != "oas":
        per_study = collections.defaultdict(collections.Counter)
        donors = collections.defaultdict(set)
        for record in kept:
            study = normalize_study(record["study"])
            per_study[study][record["category"]] += 1
            donors[study].add(record["donor"])
        census["per_study_locus"] = {study: dict(counts) for study, counts in per_study.items()}
        if source == "oas_paired_diverse":
            census["donors_by_study"] = {study: sorted(subjects) for study, subjects in donors.items()}
    atomic_write_text(Path(store) / f"{source}.normalized.json", json.dumps(kept))
    atomic_write_text(Path(census_dir) / f"{source}.census.json", json.dumps(census, indent=2, sort_keys=True))
    return SourceResult(source, kept, census)


def collect_unpaired(exclusions, *, store, census_dir, catalog_bytes, client=None, numberer=None,
                     policy=UnpairedPolicy()):
    """Collect complete human IGH/IGK/IGL domains from catalog-eligible studies."""
    if not isinstance(policy, UnpairedPolicy):
        raise ConfigurationError("Unknown OAS unpaired sampling policy.")
    _validate_policy(policy)
    client = client or HttpClient()
    try:
        catalog = json.loads(catalog_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DataContractError("Invalid OAS species/completeness catalog.") from exc
    if not isinstance(catalog, dict) or not isinstance(catalog.get("per_study"), dict):
        raise DataContractError("OAS catalog requires a per_study mapping.")
    per = catalog["per_study"]

    def eligible(entry, chain):
        if not isinstance(entry, dict) or not isinstance(entry.get(chain, {}) or {}, dict):
            raise DataContractError("OAS catalog study and chain entries must be mappings.")
        return (entry.get(chain, {}) or {}).get("status") == "eligible"

    fresh = [s for s in _studies(client, UNPAIRED_BASE) if normalize_study(s) not in exclusions.studies]
    selected = [s for s in fresh if s in per and (eligible(per[s], "heavy") or eligible(per[s], "light"))]
    ordered = ([s for s in sorted(selected) if eligible(per[s], "light")] +
               [s for s in sorted(selected) if not eligible(per[s], "light")])
    targets = {key: policy.target_per_category for key in ("igk", "igl", "pos_igh_repertoire")}
    raw_have = dict.fromkeys(targets, 0)

    def full(categories):
        return all(raw_have[key] >= 2 * targets[key] for key in categories)

    rejection = dict.fromkeys(("non_human_unit", "unknown_species_unit", "partial_ineligible_chain",
                              "category_already_full", "untouched_donor_audit_drop", "anarcii_rejected",
                              "temporarily_inaccessible", "unavailable_units", "malformed_unit"), 0)
    raw, downloaded = [], 0
    cache = Path(store) / "oas_units"
    for study in ordered:
        if downloaded >= policy.max_units or full(targets):
            break
        try:
            units = _units(client, UNPAIRED_BASE, study)
        except OperationIOError as exc:
            rejection["temporarily_inaccessible" if exc.retryable else "unavailable_units"] += 1
            continue
        study_units = 0
        for unit in sorted(units, key=lambda name: 0 if "Light" in name else 1):
            if downloaded >= policy.max_units or study_units >= policy.max_units_per_study:
                break
            chain = "Heavy" if "Heavy" in unit else "Light" if "Light" in unit else None
            if chain is None:
                continue
            if not eligible(per[study], chain.lower()):
                rejection["partial_ineligible_chain"] += 1
                continue
            categories = ["pos_igh_repertoire"] if chain == "Heavy" else ["igk", "igl"]
            if full(categories):
                rejection["category_already_full"] += 1
                continue
            try:
                raw_bytes = client.prefix(f"{UNPAIRED_BASE}{study}/csv/{unit}", byte_limit=1 << 20,
                                          cache_dir=cache, rate_s=0.4)
                metadata, rows = parse_unit(raw_bytes)
            except OperationIOError as exc:
                rejection["temporarily_inaccessible" if exc.retryable else "unavailable_units"] += 1
                continue
            except DataContractError:
                rejection["malformed_unit"] += 1
                continue
            species = metadata.get("Species")
            species = species.lower() if isinstance(species, str) else ""
            if species != "human":
                rejection["non_human_unit" if species and species != "?" else "unknown_species_unit"] += 1
                continue
            downloaded += 1
            study_units += 1
            found = candidates(metadata, rows, chain=chain, cap=policy.per_unit_candidates)
            untouched = [r for r in found if (normalize_study(r["study"]), r["donor"]) not in exclusions.donors]
            rejection["untouched_donor_audit_drop"] += len(found) - len(untouched)
            for record in untouched:
                raw_have[record["category"]] += 1
            raw.extend(untouched)
    verified = verify_antibody_domains(raw, numberer=numberer)
    rejection["anarcii_rejected"] = len(raw) - len(verified)
    census = {
        "source": "oas_shared", "fresh_studies_available": len(fresh),
        "catalog_eligible_human_studies": len(selected), "rejection_reasons": rejection,
        "studies_contributing": sorted({normalize_study(r["study"]) for r in verified}),
        "provenance": {"base": UNPAIRED_BASE, "draw": "catalog-eligible human studies only (explicit species)",
                       "verification": "ANARCII chain/locus/completeness on the numbered variable domain",
                       "catalog_sha256": sha256_bytes(catalog_bytes), "human_units_downloaded": downloaded,
                       "unit_cache": str(cache)},
    }
    return _finish("oas", verified, exclusions, store, census_dir, census)


def collect_paired(exclusions, *, store, census_dir, client=None, numberer=None, policy=PairedPolicy()):
    """Collect paired light chains using the baseline or donor-diversity policy."""
    if not isinstance(policy, (PairedPolicy, DiversityPolicy)):
        raise ConfigurationError("Unknown OAS paired sampling policy.")
    _validate_policy(policy)
    client = client or HttpClient()
    diverse = isinstance(policy, DiversityPolicy)
    source = "oas_paired_diverse" if diverse else "oas_paired"
    excluded = exclusions.studies | (policy.excluded_studies if diverse else frozenset())
    fresh = sorted(s for s in _studies(client, PAIRED_BASE) if normalize_study(s) not in excluded)
    rejection = dict.fromkeys(("non_human_unit", "unknown_species_unit", "temporarily_inaccessible",
                              "untouched_donor_audit_drop", "anarcii_rejected", "unavailable_units", "malformed_unit"), 0)
    raw, downloaded, studies_hit, donors_seen = [], 0, set(), set()
    per_donor = collections.Counter()
    have = {"igk": 0, "igl": 0}
    cache = Path(store) / ("oas_paired_diverse_units" if diverse else "oas_paired_units")
    for study in fresh:
        if diverse:
            if len(studies_hit) >= policy.max_studies or len(donors_seen) >= policy.target_donors:
                break
        elif downloaded >= policy.max_units or all(count >= 2 * policy.target_per_category for count in have.values()):
            break
        subdir, units = "csv", []
        for directory in (("csv", "csv_paired") if diverse else ("csv",)):
            try:
                units = _units(client, PAIRED_BASE, study, directory)
            except OperationIOError as exc:
                rejection["temporarily_inaccessible" if exc.retryable else "unavailable_units"] += 1
                continue
            subdir = directory
            if units:
                break
        study_donors, study_units = set(), 0
        for index, unit in enumerate(sorted(units) if diverse else units):
            if diverse:
                if len(study_donors) >= policy.max_donors_per_study or index >= policy.max_units_per_study:
                    break
            elif downloaded >= policy.max_units or study_units >= policy.max_units_per_study:
                break
            try:
                raw_bytes = client.prefix(f"{PAIRED_BASE}{study}/{subdir}/{unit}", byte_limit=1 << 20,
                                          cache_dir=cache, rate_s=0.3 if diverse else 0.4)
                metadata, rows = parse_unit(raw_bytes)
            except OperationIOError as exc:
                rejection["temporarily_inaccessible" if exc.retryable else "unavailable_units"] += 1
                continue
            except DataContractError:
                rejection["malformed_unit"] += 1
                continue
            species = metadata.get("Species")
            species = species.lower() if isinstance(species, str) else ""
            if species != "human":
                rejection["non_human_unit" if species and species != "?" else "unknown_species_unit"] += 1
                continue
            downloaded += 1
            study_units += 1
            found = candidates(metadata, rows, paired=True, diverse=diverse)
            untouched = [r for r in found if (normalize_study(r["study"]), r["donor"]) not in exclusions.donors]
            rejection["untouched_donor_audit_drop"] += len(found) - len(untouched)
            for record in untouched:
                donor = (normalize_study(record["study"]), record["donor"])
                if diverse:
                    if donor not in study_donors and len(study_donors) >= policy.max_donors_per_study:
                        continue
                    cap_key = (record["study"], record["donor"], record["category"])
                    if per_donor[cap_key] >= policy.per_donor_per_chain:
                        continue
                    per_donor[cap_key] += 1
                    donors_seen.add(donor)
                    study_donors.add(donor)
                studies_hit.add(normalize_study(study))
                have[record["category"]] += 1
                raw.append(record)
    unique = _unique(raw)
    verified = verify_antibody_domains(unique, numberer=numberer, light_only=True)
    rejection["anarcii_rejected"] = len(unique) - len(verified)
    provenance = {"base": PAIRED_BASE, "draw": "fresh human paired studies (pre-split cohort studies excluded)",
                  "verification": "ANARCII light chain/locus/completeness (K/L) on numbered V domain"}
    if diverse:
        provenance.update({"excluded_studies": sorted(policy.excluded_studies),
                           "diversity_policy": f"<= {policy.max_donors_per_study} donors/study, <= {policy.per_donor_per_chain}/chain/donor"})
    else:
        provenance["units_downloaded"] = downloaded
    census = {"source": source, "provenance": provenance, "rejection_reasons": rejection,
              "studies_contributing": sorted(studies_hit)}
    if not diverse:
        census["fresh_studies_available"] = len(fresh)
    return _finish(source, verified, exclusions, store, census_dir, census)
